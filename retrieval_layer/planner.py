"""
planner.py
──────────
Adaptive Evidence Escalation for Detailed/Research Mode — a real, multi-hop
Evidence Gap Exploration loop, invoked as an EXCEPTION HANDLER, not a richer
default.

    The extended planner is invoked only when the baseline retrieval
    pipeline cannot confidently assemble sufficient evidence. Its purpose
    is to acquire missing evidence — not to retrieve more evidence.

Detailed Mode (redis_cache.retrieve_detailed) always pools pipeline + naive +
overlap once (retriever.retrieve_merged_all) — thorough, but still a single
retrieval pass. This module adds a planner assessment on top of that, on
EVERY Detailed Mode query (as of 2026-07-10) — not gated by confidence
anymore. What confidence still gates is FETCHING more evidence: the planner
call itself is cheap-ish due diligence ("is what we have enough, and if not,
what specifically is missing"); actually escalating into hierarchical
re-retrieval only happens if that assessment says evidence is incomplete OR
the raw confidence score is still low (see _needs_expansion — now a per-hop
escalation decision, not a pre-call gate). A prior version gated the planner
call itself on query shape (relationship/evolution-phrased); that was dropped
after a quick eval showed it cost real latency for zero measured
retrieval-quality gain on already-confident queries — same reasoning now
applies to gating the ASSESSMENT differently from gating the FETCH.

When fetching does fire, it's modeled on how a human reads documentation:
read what's there, notice a referenced concept isn't actually covered, go
look it up, repeat until satisfied (or out of budget). Each hop, the planner
LLM reads the evidence gathered SO FAR (not just the original pool) and
returns a structured verdict — enough evidence, or a list of {concept, query}
gap pairs. It never answers the question itself.

Design constraints (see config.py's PLANNER_* block):
  - Bounded, not recursive: config.PLANNER_MAX_HOPS hops max, each hop at
    most config.PLANNER_SUBQUERIES_PER_HOP gaps. Termination also has a
    deterministic backstop (config.PLANNER_SCORE_EPSILON — stop if a hop's
    rerank top-score gain over the last one is negligible) so the loop can't
    be talked into running to the cap by an LLM that keeps saying "not
    enough yet" for no real gain.
  - The planner call reasons ONLY about what's missing, never about the
    answer itself — enforced by prompt + schema (structured JSON: enough
    evidence yes/no, a list of {concept, query} gaps, one-line reason —
    never prose, never an answer).
  - Each gap's query is re-routed FRESH against its own text (router.route),
    not scoped to the parent(s) the ORIGINAL query routed to — the whole
    point of looping is to reach topics the original query's own wording
    never pointed at (e.g. "Provider Query Guidance" referenced by, but not
    filed under, a "Severe Sepsis" section). Each newly-routed parent's
    children/siblings are then pulled via router.Router.expand_children and
    reranked against the gap's query.
  - A novelty filter (reranker score vs. the current evidence bundle) drops
    gaps that are already well-covered before spending a retrieval round on
    them — keeps the loop from re-fetching near-duplicate evidence.
  - Gap queries are retrieval-only and never shown to the user. Only gaps
    that actually YIELDED new evidence get surfaced downstream (result
    ["_filled_concepts"], consumed by redis_cache._generate_answer to
    scaffold the answer around them) — a gap whose query came back empty
    never gets named to the answer generator, so the model is never invited
    to write about a concept it has no retrieved support for. That asymmetry
    (fetch broadly, but only name what was actually found) is the guardrail
    against the exact hallucination/drift risk this whole feature was built
    to avoid.
  - config.PLANNER_ENABLED = True as of 2026-07-10, evaluated against this
    codebase's standing "measure before keeping" rule (project_healthrules_
    pipeline memory) on the 53-query HealthRules golden set — see config.py's
    comment for the numbers. Re-verify if retrieval/reranking changes
    upstream, or if a broader eval later shows regressions.

Usage:
    import planner
    result = planner.explore(query, base_result)   # base_result from
                                                     # retriever.retrieve_merged_all()
    result["_explored"]         # True if fetching actually ran
    result["_plan"]             # list of per-hop plan dicts (debug/eval only)
    result["_filled_concepts"]  # concepts that got real evidence — safe to
                                 # scaffold an answer around; never includes
                                 # a gap whose retrieval came back empty
"""

import hashlib
import json
import logging
import re
from typing import Dict, List, Optional

import config
import retriever
import router as hnsw_router
import chroma_store as cs
import reranker
import llm_client

log = logging.getLogger(__name__)


# ── Domain adaptation ───────────────────────────────────────────────────────
# Same disk-cache-only read as retriever._load_domain_description() — reads
# whatever Stage 1's context_profiler.py already cached, no cross-directory
# import. Pulls specialist_role + key_terminology (richer than retriever's
# domain/document_purpose blend) since the planner prompt benefits from
# knowing the analyst persona and vocabulary, not just the corpus topic.

def _load_planner_domain_context() -> str:
    profiles_dir = config.STAGE1_OUTPUT / "profiles"
    if not profiles_dir.exists():
        return ""

    parts = []
    for f in sorted(profiles_dir.glob("*.json")):
        if f.name.startswith("_"):  # skip _bridge.json
            continue
        try:
            with open(f, encoding="utf-8") as fh:
                profile = json.load(fh)
        except Exception:
            continue
        role = profile.get("specialist_role")
        if not role:
            continue
        terms = profile.get("key_terminology", {})
        term_str = ", ".join(f"{k}={v}" for k, v in list(terms.items())[:8])
        parts.append(role + (f" (terminology: {term_str})" if term_str else ""))

    return " / ".join(parts)


_DOMAIN_CONTEXT = _load_planner_domain_context()


# ── Adaptive Evidence Escalation gate ───────────────────────────────────────
# Per-HOP decision on whether to actually fetch more evidence — NOT a
# pre-call gate on whether to ask the planner at all (the planner is now
# asked on every Detailed Mode query; see module docstring). Escalate only
# if there's something to search for AND EITHER the planner itself judged
# the evidence incomplete OR the raw confidence signal is still low — an OR
# of two independent signals (mechanical score + LLM judgment) rather than
# trusting either alone, since a fallible planner call could misjudge either
# way (e.g. claim "enough_evidence" on a query the raw score says is weak).

def _needs_expansion(plan: Dict, current_score: float) -> bool:
    if not plan["gaps"]:
        return False
    return (not plan["enough_evidence"]) or (current_score < config.CONFIDENCE_THRESHOLD)


# ── Planner call — structured gap analysis only, never an answer ───────────
# Schema: {"enough_evidence": bool,
#          "gaps": [{"concept": str, "query": str}, ...], "reason": str}
# Each gap PAIRS a human-readable concept label with the retrieval-only query
# for it — deliberately not two parallel lists — so downstream code can
# track exactly which concept a given fetch was for, and only surface a
# concept to answer generation (result["_filled_concepts"]) if its OWN query
# actually returned evidence. "reason" is for debug/eval logging
# (result["_plan"]) — never shown to the user and never fed back into the
# prompt as free text.

_PLAN_FAIL_OPEN = {
    "enough_evidence": True,
    "gaps":            [],
    "reason":          "planner call failed — stopping expansion",
}

_PLANNER_SYSTEM_TEMPLATE = (
    "You are a retrieval planner for a domain question-answering system{domain_clause}. "
    "You do NOT answer questions and you must never write an explanation of the answer. "
    "You ONLY assess whether the evidence already retrieved is enough to answer the "
    "question completely, and if not, what's still missing.\n\n"
    "RULES:\n"
    "- Output ONLY a single JSON object, no markdown, no preamble, no explanation.\n"
    "- Schema: {{\"enough_evidence\": bool, \"gaps\": [{{\"concept\": string, \"query\": "
    "string}}, ...], \"reason\": string}}\n"
    "- If the evidence already covers the question fully, set enough_evidence=true and "
    "leave gaps empty.\n"
    "- Otherwise set enough_evidence=false. List at most {max_n} gaps. Each gap pairs ONE "
    "missing concept (a short human-readable label) with a short retrieval-only search "
    "query for it — not a question, not an answer.\n"
    "- \"reason\" is ONE short sentence explaining the gap (or why evidence is sufficient)."
)

_PLANNER_USER_TEMPLATE = """\
Question: {query}

Evidence retrieved so far:
{sections}

Is this evidence sufficient to answer the question completely? If not, what's missing?"""


def _summarize_sections(chunks: List[Dict], limit: int = 8) -> str:
    seen: List[str] = []
    for c in chunks:
        h = c.get("section_header", "").strip()
        if h and h not in seen:
            seen.append(h)
        if len(seen) >= limit:
            break
    return "\n".join(f"- {h}" for h in seen) if seen else "(none)"


def _generate_plan(query: str, evidence_chunks: List[Dict], max_n: int) -> Dict:
    domain_clause = f" for {_DOMAIN_CONTEXT}" if _DOMAIN_CONTEXT else ""
    system = _PLANNER_SYSTEM_TEMPLATE.format(domain_clause=domain_clause, max_n=max_n)
    user = _PLANNER_USER_TEMPLATE.format(
        query=query,
        sections=_summarize_sections(evidence_chunks),
    )
    try:
        raw = llm_client.chat(
            user, system_prompt=system, max_tokens=400, temperature=0.2, timeout=12.0,
        )
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("` \n")
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError("planner did not return a JSON object")
        gaps = []
        for g in obj.get("gaps", []):
            if not isinstance(g, dict):
                continue
            concept = str(g.get("concept", "")).strip()
            gquery = str(g.get("query", "")).strip()
            if concept and gquery:
                gaps.append({"concept": concept, "query": gquery})
        return {
            "enough_evidence": bool(obj.get("enough_evidence", True)),
            "gaps": gaps[:max_n],
            "reason": str(obj.get("reason", "")).strip(),
        }
    except Exception as e:
        log.warning("Planner gap-analysis failed (%s) — stopping expansion", e)
        return dict(_PLAN_FAIL_OPEN)


# ── Plan cache — minimal exact-match, v2 (see module docstring) ────────────
# Caches the structured plan dict, never an answer or retrieved chunks — a
# fresh retrieval always runs for whatever plan comes back, cached or not.
# Keyed on (query, hop, evidence-bundle signature): under the multi-hop loop,
# hop 2's plan depends on hop 2's grown evidence bundle, not just the query
# text — a query-only key would replay hop 1's cached plan forever and the
# loop would never see new evidence or terminate on its own logic.
# "plan2:" (bumped from "plan:") because the cached VALUE's schema changed
# (missing_concepts/retrieval_queries -> gaps) — reusing the old prefix would
# return old-shaped dicts that KeyError downstream instead of just missing
# the cache and regenerating.
# Local import of redis_cache (not a top-level import) because redis_cache
# imports THIS module from retrieve_detailed() — a top-level import here
# would be circular.

def _plan_cache_key(query: str, hop: int, evidence_chunks: List[Dict]) -> str:
    norm = " ".join(query.strip().lower().split())
    headers = sorted({
        c.get("section_header", "").strip()
        for c in evidence_chunks if c.get("section_header", "").strip()
    })
    evidence_sig = hashlib.sha256("|".join(headers).encode("utf-8")).hexdigest()[:12]
    digest = hashlib.sha256(f"{norm}::{hop}::{evidence_sig}".encode("utf-8")).hexdigest()[:24]
    return f"{config.REDIS_KEY_PREFIX}plan2:{digest}"


def _get_plan(query: str, hop: int, evidence_chunks: List[Dict], max_n: int) -> Dict:
    import redis_cache  # local import — breaks the redis_cache <-> planner cycle

    key = _plan_cache_key(query, hop, evidence_chunks)
    try:
        cached = redis_cache._get_client().get(key)
        if cached is not None:
            return json.loads(cached)
    except Exception as e:
        log.warning("Plan cache read failed (%s) — generating fresh", e)

    plan = _generate_plan(query, evidence_chunks, max_n)
    try:
        redis_cache._get_client().set(key, json.dumps(plan))
    except Exception as e:
        log.warning("Plan cache write failed (%s)", e)
    return plan


# ── Novelty filter ──────────────────────────────────────────────────────────

def _filter_novel(gaps: List[Dict], evidence_chunks: List[Dict]) -> List[Dict]:
    """
    Drops gaps whose query the reranker scores as already well-covered by
    the current evidence bundle — reuses the same cross-encoder every other
    gate in this module uses, no new embedding call. See config.PLANNER_
    NOVELTY_THRESHOLD for the (untuned) cutoff.
    """
    if not evidence_chunks:
        return gaps

    novel: List[Dict] = []
    for gap in gaps:
        ranked = reranker.rerank(gap["query"], evidence_chunks)
        top = (ranked[0].get("rerank_score", ranked[0].get("score", 0.0)) if ranked else 0.0)
        if top < config.PLANNER_NOVELTY_THRESHOLD:
            novel.append(gap)
        else:
            log.info("Planner gap %r dropped as redundant (score %.2f)", gap["concept"], top)
    return novel


# ── Hierarchical retrieval per sub-query ────────────────────────────────────

def _relationship_chunks(subquery: str, n: int = 2) -> List[Dict]:
    chunks = []
    for r in cs.get_store().query_relationships(subquery, n=n):
        chunks.append({
            "chunk_id":       f"relationship_{abs(hash(r['text'])) % 100000}",
            "text":           r["text"],
            "section_header": f"{r.get('members', '').replace(' | ', ' × ')} — Relationship",
            "source_doc":     "Cross-Domain Synthesis",
            "version":        "",
            "score":          r.get("score", 0.0),
        })
    return chunks


def _hierarchical_retrieve(
    subquery: str,
    exclude_chunk_ids: set,
) -> List[Dict]:
    """
    One sub-query's worth of expansion — routed FRESH against the sub-query's
    own text (router.route), NOT scoped to the ORIGINAL query's parent(s).
    This is what lets a hop reach a topic the original broad query's own
    wording never pointed at at all (e.g. a "Provider Query Guidance"
    sub-query fired from a "Severe Sepsis" question, where the two live
    under different parents in the taxonomy). Pulls the newly-routed
    parent's full children/sibling set via Router.expand_children (richer
    than a plain top-k hit), reranked against THIS sub-query, plus any
    precomputed cross-topic relationship doc.
    """
    chunks: List[Dict] = []

    try:
        route_result = hnsw_router.route(subquery)
    except Exception as e:
        log.warning("Re-routing sub-query %r failed (%s) — skipping", subquery, e)
        route_result = {}

    parent_indices = sorted({p for p, s in route_result.get("_chosen_pairs", [])})
    if parent_indices:
        expansion = hnsw_router.get_router().expand_children(
            parent_indices, exclude_chunk_ids=exclude_chunk_ids,
        )
        if expansion["chunks"]:
            ranked = reranker.rerank(subquery, expansion["chunks"])
            chunks += ranked[: config.PLANNER_CHILDREN_TOP_N]

    chunks += _relationship_chunks(subquery)
    return chunks


# ── Main entry point ────────────────────────────────────────────────────────

def explore(query: str, base_result: Dict, max_hops: Optional[int] = None) -> Dict:
    """
    Runs the planner's gap assessment on EVERY call (no pre-call gate — see
    module docstring for why), then optionally expands base_result (from
    retriever.retrieve_merged_all()) via a multi-hop Evidence Gap Exploration
    loop if that assessment says fetching is actually warranted.

    Returns base_result unchanged if:
      - config.PLANNER_ENABLED is False (no planner call at all), or
      - hop 0's assessment says _needs_expansion is False — i.e. the planner
        found nothing missing AND the raw confidence score isn't low either.

    Otherwise loops up to config.PLANNER_MAX_HOPS times. Each hop:
      1. Ask the planner about the evidence gathered SO FAR (grows every hop).
      2. _needs_expansion(plan, current_score): stop unless there are gaps
         AND (the planner says evidence is incomplete OR confidence is still
         low) — an OR of the LLM's own judgment and the mechanical score, so
         one fallible signal alone can't over- or under-trigger fetching.
      3. Drop redundant gaps via the novelty filter; stop if none left.
      4. Re-route + retrieve each remaining gap's query fresh (_hierarchical_
         retrieve), merge into the pool, rerank against the ORIGINAL query.
         Track which gaps' queries actually returned new chunks —
         result["_filled_concepts"] ends up containing ONLY those concepts,
         never a gap whose query came back empty (see module docstring on
         why this asymmetry is the hallucination guardrail).
      5. Stop if this hop's top rerank score barely improved over the last
         one (config.PLANNER_SCORE_EPSILON) — deterministic backstop so
         termination doesn't rest solely on the planner LLM's self-report.

    Never raises — any LLM/retrieval failure inside just falls back to
    base_result (or whatever was accumulated so far), same fail-open
    behavior as retriever._reformulate(). A failed planner call returns
    _PLAN_FAIL_OPEN (enough_evidence=True, no gaps), which _needs_expansion
    reads as "nothing to do" — same as a genuinely confident, complete result.
    """
    if not config.PLANNER_ENABLED:
        return base_result

    max_hops = max_hops if max_hops is not None else config.PLANNER_MAX_HOPS

    pool = list(base_result.get("chunks", []))
    exclude_ids = {c["chunk_id"] for c in pool if c.get("chunk_id")}
    prev_top_score = retriever._top_rerank_score(base_result)
    plans: List[Dict] = []
    filled_concepts: List[str] = []
    explored = False

    for hop in range(max_hops):
        plan = _get_plan(query, hop, pool, config.PLANNER_SUBQUERIES_PER_HOP)
        plans.append(plan)

        if not _needs_expansion(plan, prev_top_score):
            break

        gaps = _filter_novel(plan["gaps"], pool)
        if not gaps:
            break

        fetched_new = False
        for gap in gaps:
            gap_filled = False
            for c in _hierarchical_retrieve(gap["query"], exclude_ids):
                cid = c.get("chunk_id")
                if not cid or cid not in exclude_ids:
                    pool.append(c)
                    if cid:
                        exclude_ids.add(cid)
                    fetched_new = True
                    gap_filled = True
            if gap_filled:
                filled_concepts.append(gap["concept"])

        if not fetched_new:
            break  # nothing new to show the planner — re-asking would be a wasted round-trip

        explored = True

        # Score the FULL accumulated pool in place, without truncating it —
        # reranker.rerank() mutates 'rerank_score' onto every dict it's given
        # before sorting+truncating its *return value*, so calling it here and
        # ignoring the (TOP_K_FINAL-truncated) return keeps every chunk gathered
        # so far available to the next hop's planner and novelty filter. Using
        # _merge_pools (rerank + truncate) mid-loop, like the single-shot
        # version did, would silently drop anything that didn't crack the top
        # TOP_K_FINAL against the ORIGINAL query — often exactly the
        # planner-requested evidence, since it's evidence the plain query
        # itself doesn't rank highly. Truncation happens ONCE, after the loop.
        reranker.rerank(query, pool)
        new_top_score = retriever._top_rerank_score({"chunks": pool})
        if new_top_score - prev_top_score < config.PLANNER_SCORE_EPSILON:
            break
        prev_top_score = new_top_score

    if not explored:
        return base_result

    merged = retriever._merge_pools(query, pool)
    result = dict(base_result)
    result["chunks"] = merged
    result["context"] = (
        retriever.format_context_versioned(merged) if base_result.get("versioned")
        else retriever.format_context(merged)
    )
    result["_explored"] = True
    result["_plan"] = plans
    result["_filled_concepts"] = filled_concepts
    return result
