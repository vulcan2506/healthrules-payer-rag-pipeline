"""
redis_cache.py
──────────────
Response cache in front of retrieve() + answer generation, backed by the
local Redis instance (already running as a systemd service — no setup
needed). Two jobs:

  1. gate_and_retrieve(query, best_of=3) — a DYNAMIC routing gate, not a
     static per-category lookup table. Earlier version of this file used
     select_method(query), a fixed if/else keyed off keyword classifiers
     and a one-time offline composite score per category — which only ever
     "knows how to route" the shapes of query it was tuned against, and
     silently has nothing to fall back on once real traffic drifts outside
     the 13-53 queries it was calibrated on.

     This version instead mirrors retriever.py's own confidence ladder
     (config.CONFIDENCE_THRESHOLD + _top_rerank_score(), see
     _retrieve_specific()): run pipeline first (cheap — it's also the first
     leg of merged/merged_all anyway), check ITS OWN live rerank confidence,
     and only escalate to a richer/pooled method if that confidence is
     actually low for THIS query, right now. That generalizes to any future
     query for free, because the gate reacts to a runtime measurement of the
     query's own retrieval quality — not a keyword match against a fixed set
     of categories. select_method() is kept below only as a fallback labeler
     for reporting/inspection, not as the thing that decides routing anymore.

     "Cheap first pass" is now best-of-`best_of` (default 3) reformulations
     of the query, not a single retrieve() call — reuses retriever.
     retrieve_best_of_n(), which reformulates the query 3 ways and runs all
     3 through the pipeline in parallel, keeping whichever scores highest.
     The confidence check and any escalation below then operate on that
     winning reformulation's result, same as before. Pass best_of=0 (or 1)
     to fall back to a single plain retrieve() call, e.g. for a quick
     latency-sensitive path. detailed mode (retrieve_detailed) deliberately
     does NOT get best-of-N by default — it already pools 3 methods per
     call; stacking 3 reformulations on top of that would be 9 retrieval
     calls per query for a use case that already trades cost for coverage.

  2. answer_query(query, mode) — cache-checked entry point with two modes,
     both dynamic, neither a static lookup:

       mode="concise"  (default) — gate_and_retrieve(): stays on cheap
                         pipeline unless its own confidence is low, then
                         escalates. "Doesn't drift" — tight by default.
       mode="detailed"           — retrieve_detailed(): always the fullest
                         pool (merged_all), no confidence check — thorough
                         by default.

     Per the full 53-query x 6-method eval (deep_eval_full_6method.md),
     merged_all is actually the single best method on Correctness, Recall,
     AND Conciseness at that scale — it is not "detailed but bloated". Its
     one consistent weakness across both eval runs is Faithfulness (worst of
     all six methods) — pooling occasionally states irrelevant content as if
     related. So "detailed" mode trades a small faithfulness risk for
     thoroughness, not thoroughness for verbosity.

     Cache hit → return instantly from Redis (no LLM call at all); miss →
     retrieve per the selected mode + generate an answer, same generation
     call as deep_eval.py's generate_answer(), and return it (NOT
     auto-cached — caching is a deliberate, separate seeding step via
     build_cache_preset.py, so the demo's "cached vs uncached" split stays
     exactly as curated). Each mode is cached under its own key, so the same
     question can hold two different cached answers, one per mode.

Usage:
    import redis_cache
    result = redis_cache.answer_query("How does claim adjudication work?")               # concise
    result = redis_cache.answer_query("How does claim adjudication work?", mode="detailed")
    result["from_cache"]   # True/False
    result["answer"]       # str
    result["method"]       # which method actually served it (post-hoc, not pre-decided)
"""

import hashlib
import json
import logging
import time
from typing import Dict, Optional

import redis

import config
import retriever
import llm_client

log = logging.getLogger(__name__)

_client: Optional["redis.Redis"] = None


def _get_client() -> "redis.Redis":
    global _client
    if _client is None:
        if config.REDIS_URL:
            _client = redis.Redis.from_url(config.REDIS_URL, decode_responses=True)
        else:
            _client = redis.Redis(
                host=config.REDIS_HOST, port=config.REDIS_PORT, db=config.REDIS_DB,
                decode_responses=True,
            )
    return _client


def _normalize(query: str) -> str:
    return " ".join(query.strip().lower().split())


def cache_key(query: str, mode: str = "concise") -> str:
    digest = hashlib.sha256(f"{mode}:{_normalize(query)}".encode("utf-8")).hexdigest()[:24]
    return f"{config.REDIS_KEY_PREFIX}{digest}"


# ── Dynamic gate (confidence-based, not category-based) ────────────────────────

METHOD_FNS = {
    "pipeline":    lambda q: retriever.retrieve(q),
    "naive":       lambda q: retriever.retrieve_naive(q),
    "traditional": lambda q: retriever.retrieve_overlap(q),
    "merged":      lambda q: retriever.retrieve_merged(q),
    "merged_all":  lambda q: retriever.retrieve_merged_all(q),
}


def select_method(query: str) -> str:
    """
    Offline/reporting-only label for "what category would this query hit,
    and which method did the deep-eval pilot favor for that category" —
    kept for build_cache_preset.py's manifest and for anyone wanting a
    quick classification without running retrieval. NOT used by
    gate_and_retrieve()/answer_query() to decide routing anymore — see the
    module docstring for why a static category lookup was replaced.
    """
    if retriever._is_cross_corpus_query(query):
        return "pipeline"
    if retriever._is_relationship_query(query):
        return "merged_all"
    if retriever._is_evolution_query(query):
        return "pipeline"
    intent = retriever.classify(query)
    if intent == "delta":
        return "merged"
    if intent == "intelligence":
        return "pipeline"
    return "traditional"


def gate_and_retrieve(query: str, best_of: int = 3) -> Dict:
    """
    Run pipeline first — cheap, and also the first leg of merged/merged_all
    anyway — and check its own live rerank confidence via the SAME mechanism
    retriever.py's confidence ladder already uses (config.CONFIDENCE_THRESHOLD,
    _top_rerank_score()). Confident -> use pipeline's answer directly (fast,
    cheap, and per the deep-eval pilot the most concise/faithful method when
    it's actually finding good matches). Not confident -> escalate.

    Which rung to escalate to (once escalation is already triggered by low
    confidence) still uses retriever's own relationship/delta classifiers —
    same as retriever.py's ladder does to pick its own escalation order —
    but that only decides WHICH pooled method to try, not WHETHER to escalate
    at all. The whether is 100% dynamic, so a query nobody anticipated still
    gets routed sensibly: high confidence -> stays cheap, low confidence ->
    gets more pooled context, regardless of its category.

    best_of (default 3): the "cheap pipeline" pass is itself a best-of-N
    reformulation (retriever.retrieve_best_of_n) rather than one retrieve()
    call — 3 parallel reformulations of the query, keeping whichever one's
    top rerank score is highest. The confidence check and escalation above
    then run against that winning result. Pass 0 or 1 to skip reformulation
    and use a single plain retrieve() call instead.
    """
    if best_of and best_of > 1:
        pipeline_result = retriever.retrieve_best_of_n(query, n=best_of)
    else:
        pipeline_result = retriever.retrieve(query)
    top_score = retriever._top_rerank_score(pipeline_result)

    if top_score >= config.CONFIDENCE_THRESHOLD:
        pipeline_result = dict(pipeline_result)
        pipeline_result["_gated_method"] = "pipeline"
        pipeline_result["_gated_confidence"] = top_score
        return pipeline_result

    if retriever._is_relationship_query(query):
        method = "merged_all"
    elif retriever.classify(query) == "delta":
        method = "merged"
    else:
        method = "merged_all"   # generic escalation: pool everything available

    result = dict(METHOD_FNS[method](query))
    result["_gated_method"] = method
    result["_gated_confidence"] = top_score
    return result


def retrieve_detailed(query: str) -> Dict:
    """
    "Detailed/explanatory" mode — always uses merged_all, no confidence
    check at all. Per the full 53-query x 6-method eval (deep_eval_full_6method.md),
    merged_all is the strongest single method on Correctness (0.998), Recall
    (0.976), Precision@5 (0.917), and even Conciseness (0.826, best of all six —
    it is NOT the "bloated" method at this scale). Its one real, consistent
    weakness across both the 13-query pilot and the full 53-query run is
    Faithfulness (0.895 here, the worst of the six) — pooling occasionally
    pulls in irrelevant content and states it as if related. Use this mode
    when the caller explicitly wants the most thorough answer available and
    accepts that faithfulness risk, as opposed to gate_and_retrieve()'s
    concise/"doesn't drift" mode, which only reaches for merged_all when
    pipeline's own confidence says it's actually needed.
    """
    result = dict(retriever.retrieve_merged_all(query))
    result["_gated_method"] = "merged_all"
    return result


# ── Answer generation (mirrors deep_eval.py's generate_answer) ────────────────

def _generate_answer(query: str, result: Dict, max_tokens: int = 900) -> str:
    return llm_client.chat(
        f"{result['context']}\n\nQuestion: {query}",
        system_prompt=result["system_prompt"],
        max_tokens=max_tokens,
    )


# ── Cache read/write ────────────────────────────────────────────────────────────

def get_cached(query: str, mode: str = "concise") -> Optional[Dict]:
    raw = _get_client().get(cache_key(query, mode))
    return json.loads(raw) if raw else None


def set_cached(query: str, method: str, answer: str, latency_s: float, mode: str = "concise") -> None:
    payload = {
        "query": query, "method": method, "answer": answer, "mode": mode,
        "latency_s_at_seed": latency_s, "cached_at": time.time(),
    }
    key = cache_key(query, mode)
    if config.REDIS_TTL_SECONDS:
        _get_client().setex(key, config.REDIS_TTL_SECONDS, json.dumps(payload))
    else:
        _get_client().set(key, json.dumps(payload))


def answer_query(
    query: str,
    mode: str = "concise",
    prefer_method: Optional[str] = None,
    best_of: int = 3,
) -> Dict:
    """
    Cache-checked entry point, now with two modes (each cached separately —
    the same question can hold two different cached answers, one per mode):

      mode="concise"  (default) — gate_and_retrieve()'s dynamic gate: best-of-
                        `best_of` cheap pipeline reformulations first (see
                        gate_and_retrieve's docstring), escalates only when
                        that winning result's confidence is still low.
                        "Doesn't drift" — stays tight unless the query
                        actually needs more.
      mode="detailed"           — retrieve_detailed(): always the fullest
                        pool (merged_all), no confidence check, no best-of-N
                        reformulation. Thorough by default, at merged_all's
                        one measured cost (Faithfulness 0.895, its weakest
                        metric).

    best_of (default 3): forwarded to gate_and_retrieve() for mode="concise"
    only — pass 0 to skip reformulation and use a single retrieve() call.
    Ignored for mode="detailed" (see gate_and_retrieve's docstring for why).

    prefer_method, if given, overrides both and forces a specific method —
    unchanged behavior, still bypasses caching-by-mode distinction (cached
    under "concise"'s key, since forcing a method isn't really either mode).

    Cache hit -> instant return, no LLM call at all. Cache miss -> retrieval
    per the selected mode + generation — NOT auto-written back to cache;
    seeding is a deliberate, separate step (build_cache_preset.py) so the
    demo's cached/uncached split stays exactly as curated.
    """
    t0 = time.time()
    cached = get_cached(query, mode)
    if cached:
        return {**cached, "from_cache": True, "latency_s": time.time() - t0}

    if prefer_method:
        method = prefer_method
        result = METHOD_FNS[method](query)
    elif mode == "detailed":
        result = retrieve_detailed(query)
        method = result["_gated_method"]
    else:
        result = gate_and_retrieve(query, best_of=best_of)
        method = result["_gated_method"]

    answer = _generate_answer(query, result)
    return {
        "query": query, "method": method, "mode": mode, "answer": answer,
        "confidence": result.get("_gated_confidence"),
        # Passthrough only — result["chunks"] is already the exact list
        # retriever.retrieve()/retrieve_merged()/retrieve_merged_all() produced
        # for whichever method the gate actually picked; no new retrieval call,
        # no re-ranking, nothing recomputed. Cached answers (from_cache=True,
        # above) don't carry this — caching is seeded separately by
        # build_cache_preset.py, which doesn't store chunks today.
        "chunks": result.get("chunks", []),
        "from_cache": False, "latency_s": time.time() - t0,
    }
