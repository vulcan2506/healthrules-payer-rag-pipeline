"""
retriever.py
────────────
Single entry point for the chatbot. Handles:

  1. Intent classification  — which collection(s) to query
  2. Retrieval              — HNSW / ChromaDB corpus / intelligence / delta
  3. Conflict detection     — multi-version chunks for the same section
  4. Delta injection        — auto-pulls delta report when conflict detected
  5. Context formatting     — normal or version-split layout
  6. System prompt          — returns the right LLM instruction set

Smart routing logic
───────────────────
  specific    →  HNSW hierarchical (fastest, most precise)
                 → if empty, widen to corpus
                 → if reranked confidence < CONFIDENCE_THRESHOLD, pull in
                   sibling sub-parents under the SAME routed parent(s)
                   (chords in the same key, not a jump to another song) —
                   see router.py:Router.expand_siblings
                 → if STILL below threshold, bridge to a different parent
                   family entirely (the "key change") — see
                   router.py:Router.expand_cross_parent
  factual     →  ChromaDB corpus
  intelligence →  ChromaDB intelligence (summaries/overviews)
  delta       →  ChromaDB delta

  POST-RETRIEVAL (any path):
    if chunks from multiple versions cover the same section
      → conflict detected
      → query delta collection with the conflicting section name
      → format context in versioned layout
      → return versioned system prompt
"""

import concurrent.futures
import json
import logging
import re
from collections import defaultdict
from typing import Dict, List, Optional

import httpx
import router as hnsw_router
import reranker
import chroma_store as cs
import config

log = logging.getLogger(__name__)


# ── System prompts ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT_STANDARD = (
    "You are a precise technical assistant for HealthRules Payer release documentation. "
    "Answer the user's question using ONLY the provided context. "
    "Cite the source document and section for each key claim. "
    "If the context does not contain enough information, say so clearly."
)

SYSTEM_PROMPT_VERSIONED = (
    "You are a precise technical assistant for HealthRules Payer release documentation. "
    "The context below contains information from MULTIPLE VERSIONS of the document "
    "(for example, an older and a newer release). "
    "RULES:\n"
    "- Treat each version section independently — do NOT merge them into one statement.\n"
    "- If behaviour differs between versions, explicitly state which version does what "
    "  using the version label shown in the context header (e.g. 'Version X says …').\n"
    "- If a later version changed something from an earlier version, present it as a "
    "  version difference, not as a single fact.\n"
    "- Never present contradictory information from different versions as one consistent fact.\n"
    "- Use the Delta Report section (if present) as the authoritative source on what changed.\n"
    "Answer the user's question using ONLY the provided context."
)


# ── Intent classifier ──────────────────────────────────────────────────────────

_DELTA_SIGNALS = {
    "changed", "change", "compare", "comparison", "difference", "differences",
    "new in", "added", "removed", "updated", "what's new", "whats new",
    "between versions", "vs", "versus", "upgrade", "migration",
    "improvement over", "previously", "before and after",
}
# Matches any version-like token: v1.0, v25.2, 26.1, 2.0, etc.
_VERSION_RE = re.compile(r"\bv?\d+\.\d+\b")

_INTELLIGENCE_SIGNALS = {
    "overview", "summarize", "summary", "what topics", "what areas",
    "what does this cover", "coverage", "domains", "list all",
    "list of", "categories", "high level", "broadly",
}


def classify(query: str) -> str:
    """Returns: 'delta' | 'intelligence' | 'specific' """
    q      = query.lower()
    tokens = set(re.findall(r"[\w']+", q))
    if (
        _DELTA_SIGNALS & tokens
        or any(p in q for p in _DELTA_SIGNALS if " " in p)
        or _VERSION_RE.search(q)   # any version number triggers delta path
    ):
        return "delta"
    if _INTELLIGENCE_SIGNALS & tokens or any(p in q for p in _INTELLIGENCE_SIGNALS if " " in p):
        return "intelligence"
    return "specific"


# Requirement-wise gate for the confidence ladder (config.ENABLE_CONFIDENCE_
# LADDER stays the global off-switch — False by default, per the measured
# 4x latency cost with zero eval benefit on ordinary factual queries). This
# set auto-triggers the ladder ONLY for queries actually shaped like a
# cross-domain relationship question — verified to swing top rerank score
# from negative (normal pipeline, no chunk answers the relationship itself)
# to strongly positive (ladder retrieves the synthesized grandparent doc)
# on exactly this query shape. None of the 19 eval golden queries match this
# shape, so eval behavior is unaffected.
_RELATIONSHIP_SIGNALS = {
    "relate to", "relates to", "related to", "relationship between",
    "connect to", "connects to", "connection between", "what connects",
    "tie into", "ties into", "interact with", "interacts with",
    "interaction between", "link between", "linked to",
}


def _is_relationship_query(query: str) -> bool:
    q = query.lower()
    return any(p in q for p in _RELATIONSHIP_SIGNALS)


# Requirement-wise gate for evolution/value-add cards (evolution_analyzer.py).
# Note: most naturally-phrased evolution queries ("how has X evolved since
# 25.2", "what changed to improve X") already contain a version token or a
# _DELTA_SIGNALS word and get classified as "delta" intent above — for those,
# query_evolution() results are merged directly into _retrieve_delta rather
# than relying on this gate. This signal set exists for the remaining case:
# a relationship/evolution-shaped query with NO version token and NO delta
# keyword (e.g. "what value did the accumulator scheduler add"), which would
# otherwise fall through to the plain "specific" path and never see the
# evolution collection at all.
_EVOLUTION_SIGNALS = {
    "evolved", "evolve", "evolution", "value did", "value added",
    "built on", "builds on", "foundation for", "value-add",
}


def _is_evolution_query(query: str) -> bool:
    q = query.lower()
    return any(p in q for p in _EVOLUTION_SIGNALS)


# Gate for the cross-corpus relationship rung (cross_corpus_relationship.py).
# Extends _RELATIONSHIP_SIGNALS (which already catches "interact with" etc.)
# with phrasings more specific to "does X in one document affect Y in
# another" cross-module questions — "how does Connector authentication
# affect Payer configuration" style. Empty result set today (no second
# corpus exists — query_cross_corpus() has nothing to return), but the gate
# and rung are wired so this activates the moment a real second corpus does.
_CROSS_CORPUS_SIGNALS = {
    "affect", "affects", "impact on", "impacts", "influence", "influences",
}


def _is_cross_corpus_query(query: str) -> bool:
    q = query.lower()
    return any(p in q for p in _CROSS_CORPUS_SIGNALS)


# ── Conflict detection ─────────────────────────────────────────────────────────

def _extract_version(chunk: Dict) -> str:
    src = chunk.get("source_doc", "") or chunk.get("version", "")
    m   = re.search(r"(\d+\.\d+)", src)
    return m.group(1) if m else "unknown"


def detect_version_conflict(chunks: List[Dict]) -> List[str]:
    """
    Returns list of section_headers that appear in chunks from more than one version.
    Empty list = no conflict.
    """
    by_section: Dict[str, set] = defaultdict(set)
    for c in chunks:
        section = c.get("section_header", "").strip()
        version = _extract_version(c)
        if section and version != "unknown":
            by_section[section].add(version)
    return [sec for sec, versions in by_section.items() if len(versions) > 1]


# ── Delta injection ────────────────────────────────────────────────────────────

def _inject_delta(conflicting_sections: List[str], original_query: str) -> List[Dict]:
    """
    For each conflicting section, query the delta collection with the section
    name (more targeted than the original query).
    Returns combined delta findings, deduplicated.
    """
    store   = cs.get_store()
    seen    = set()
    results = []

    for section in conflicting_sections:
        # Use section name as query — much more targeted than original query
        findings = store.query_delta(section, n=3)
        for f in findings:
            key = f["text"][:80]
            if key not in seen:
                seen.add(key)
                results.append(f)

    # Fallback: if section queries returned nothing, try original query
    if not results:
        log.debug("Section-targeted delta query empty — falling back to original query")
        findings = store.query_delta(original_query, n=4)
        for f in findings:
            key = f["text"][:80]
            if key not in seen:
                seen.add(key)
                results.append(f)

    return results


# ── Context formatting ─────────────────────────────────────────────────────────

def format_context(chunks: List[Dict]) -> str:
    """Standard single-version context block."""
    parts = []
    for i, c in enumerate(chunks, 1):
        source  = c.get("source_doc", "")
        section = c.get("section_header", "")
        header  = f"[{i}] {source} — {section}" if section else f"[{i}] {source}"
        parts.append(f"{header}\n{c['text'].strip()}")
    return "\n\n".join(parts)


def format_context_versioned(
    chunks: List[Dict],
    delta_findings: Optional[List[Dict]] = None,
) -> str:
    """
    Version-split context block.  Groups chunks by version with clear headers.
    Appends delta findings as a separate section if provided.
    """
    by_version: Dict[str, List[Dict]] = defaultdict(list)
    for c in chunks:
        by_version[_extract_version(c)].append(c)

    parts = []
    counter = 1
    for version in sorted(by_version.keys()):
        parts.append(f"{'─'*40}\n📄 VERSION {version}\n{'─'*40}")
        for c in by_version[version]:
            section = c.get("section_header", "")
            source  = c.get("source_doc", "")
            header  = f"[{counter}] {source} — {section}" if section else f"[{counter}] {source}"
            parts.append(f"{header}\n{c['text'].strip()}")
            counter += 1

    if delta_findings:
        # Derive version labels from the actual findings, not hardcoded strings
        v_from = next((f.get("version_from") for f in delta_findings if f.get("version_from")), "")
        v_to   = next((f.get("version_to")   for f in delta_findings if f.get("version_to")),   "")
        delta_header = f"DELTA REPORT ({v_from} → {v_to})" if v_from and v_to else "DELTA REPORT"
        parts.append(f"{'─'*40}\n📊 {delta_header}\n{'─'*40}")
        for i, f in enumerate(delta_findings, 1):
            change  = f.get("change_type", "").upper()
            section = f.get("section_header", f.get("section", ""))
            prefix  = f"[Δ{i}] {change} — {section}" if section else f"[Δ{i}] {change}"
            parts.append(f"{prefix}\n{f['text'].strip()}")

    return "\n\n".join(parts)


# ── Traditional RAG baseline ───────────────────────────────────────────────────

def retrieve_naive(query: str, n_candidates: int = 20) -> Dict:
    """
    Flat vector search baseline — no routing, no conflict detection, no delta.

    Steps: embed query → cosine search entire corpus collection → rerank.
    Same embedding model and cross-encoder as the full pipeline so the only
    variable is the routing / conflict-detection logic.

    Use this to compare against retrieve() to demonstrate the value of
    hierarchical routing and version-conflict handling to clients.
    """
    store   = cs.get_store()
    results = store.query_corpus(query, n=n_candidates)
    chunks  = reranker.rerank(query, _chroma_to_chunks(results))
    return {
        "query":         query,
        "intent":        "naive",
        "path":          {"collection": config.CORPUS_COLLECTION, "mode": "flat"},
        "chunks":        chunks,
        "context":       format_context(chunks),
        "system_prompt": SYSTEM_PROMPT_STANDARD,
        "versioned":     False,
        "delta_used":    False,
    }


def retrieve_overlap(query: str, n_candidates: int = 20) -> Dict:
    """
    Traditional RAG baseline — sliding-window chunks from raw markdown files.

    This is the closest to how most off-the-shelf RAG systems work:
      - Raw document text split into overlapping fixed-size windows
      - No taxonomy, no pre-filtering, no version-conflict logic
      - Same embedding model and cross-encoder reranker as the pipeline

    Useful for demonstrating the value added by the full pipeline (taxonomy
    routing, QnA-enriched chunks, version conflict detection, delta injection).
    """
    store   = cs.get_store()
    results = store.query_overlap(query, n=n_candidates)
    chunks  = reranker.rerank(query, _chroma_to_chunks(results))
    return {
        "query":         query,
        "intent":        "overlap",
        "path":          {"collection": config.OVERLAP_COLLECTION, "mode": "sliding_window"},
        "chunks":        chunks,
        "context":       format_context(chunks),
        "system_prompt": SYSTEM_PROMPT_STANDARD,
        "versioned":     False,
        "delta_used":    False,
    }


# ── Best-of-N retrieval ────────────────────────────────────────────────────────

_REFORMULATE_PROMPT = """\
Rephrase the following question {n} different ways for document retrieval.
Variation 1: literal restatement (same meaning, different words).
Variation 2: abstract/conceptual (broader terms, what the user really wants to understand).
Variation 3: keyword-focused (drop stop-words, keep only the core technical terms).
Return ONLY a JSON array of {n} strings. No explanation, no markdown fences."""


def _reformulate(query: str, n: int = 3, timeout: float = 12.0, enable_thinking: bool = False) -> List[str]:
    """
    Ask the local LLM for n reformulations of query.
    Falls back to [query] * n if the call fails or response is unparseable.
    """
    prompt = _REFORMULATE_PROMPT.format(n=n) + f"\n\nQuestion: {query}"
    try:
        resp = httpx.post(
            f"{config.LLAMA_SERVER_URL}/v1/chat/completions",
            json={
                "model":       config.LLAMA_MODEL_NAME,
                "messages":    [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens":  256,
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        # Strip <think> blocks (Qwen3 reasoning traces)
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        # Strip optional markdown fences
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("` \n")
        arr = json.loads(raw)
        if isinstance(arr, list) and len(arr) >= n:
            return [str(q).strip() for q in arr[:n]]
        log.warning("Reformulation returned fewer than %d items — using originals", n)
    except Exception as e:
        log.warning("Reformulation failed (%s) — falling back to original query", e)
    return [query] * n


def _top_rerank_score(result: Dict) -> float:
    scores = [c.get("rerank_score", c.get("score", 0.0)) for c in result["chunks"]]
    return max(scores) if scores else 0.0


def _dedup_key(c: Dict) -> str:
    """
    Dedup key for pool-merging across retrieval methods. chunk_id is only
    populated for chunks that came from the 'corpus' collection (pipeline's
    routed chunks, naive's flat search) — ingest_corpus() stores it in
    metadata. Traditional/overlap chunks (ingest_overlap()) never get a
    chunk_id at all, so _chroma_to_chunks() always sets it to "" for them.
    Falling back to chunk_id alone would collide every overlap chunk onto
    the same empty-string key and silently drop all but the first one from
    any merged pool — fall back to a text prefix instead, which is unique
    per chunk regardless of source collection.
    """
    cid = c.get("chunk_id", "")
    return cid if cid else c.get("text", "")[:200]


def _merge_pools(query: str, *chunk_lists: List[Dict]) -> List[Dict]:
    """Dedup (by _dedup_key) and rerank the union of any number of chunk lists."""
    pool: List[Dict] = []
    seen: set = set()
    for chunks in chunk_lists:
        for c in chunks:
            key = _dedup_key(c)
            if key not in seen:
                pool.append(c)
                seen.add(key)
    return reranker.rerank(query, pool)


def retrieve_merged(
    query: str,
    version: Optional[str] = None,
    verbose: bool           = False,
) -> Dict:
    """
    Merges pipeline's routed candidates with naive's flat full-corpus
    candidates into ONE pool, reranks the combined pool, and returns the
    winners — a genuine ensemble across methods, not a pick-one-winner
    comparison like retrieve_best_of_n (which reformulates the SAME query
    N times through the SAME pipeline; it never sees naive's candidates
    at all).

    Reuses the exact "merge extra candidates into the pool, rerank once"
    pattern already proven by the confidence ladder's rungs (expand_siblings,
    expand_cross_parent, query_relationships, query_evolution) — naive's
    full-corpus search is just one more candidate source merged in, here
    unconditionally rather than gated behind a low-confidence threshold,
    since the goal is coverage on every query, not a fallback for rare ones.

    Keeps the pipeline's intent classification, conflict detection, and
    delta injection untouched (naive has none of that) — only the final
    candidate pool and rerank are affected.
    """
    pipeline_result = retrieve(query, version=version, verbose=verbose)
    naive_result    = retrieve_naive(query)

    merged_chunks = _merge_pools(query, pipeline_result["chunks"], naive_result["chunks"])

    result = dict(pipeline_result)
    result["chunks"]            = merged_chunks
    result["context"]           = (
        format_context_versioned(merged_chunks) if pipeline_result["versioned"]
        else format_context(merged_chunks)
    )
    result["merged"]            = True
    result["_pipeline_only_top"] = _top_rerank_score(pipeline_result)
    result["_naive_only_top"]    = _top_rerank_score(naive_result)
    return result


def retrieve_merged_all(
    query: str,
    version: Optional[str] = None,
    verbose: bool           = False,
) -> Dict:
    """
    Same idea as retrieve_merged(), extended to a THIRD candidate source:
    Traditional RAG's sliding-window chunks (retrieve_overlap — a different
    ChromaDB collection, corpus_overlap, built from raw markdown with no
    taxonomy). Pools all three methods' candidates and reranks once.

    This is why _dedup_key() falls back to a text prefix rather than
    chunk_id alone — overlap chunks never carry a chunk_id, so a naive
    union would only ever keep the FIRST overlap chunk and silently drop
    the rest as false "duplicates" of the empty string.
    """
    pipeline_result = retrieve(query, version=version, verbose=verbose)
    naive_result    = retrieve_naive(query)
    overlap_result  = retrieve_overlap(query)

    merged_chunks = _merge_pools(
        query, pipeline_result["chunks"], naive_result["chunks"], overlap_result["chunks"]
    )

    result = dict(pipeline_result)
    result["chunks"]             = merged_chunks
    result["context"]            = (
        format_context_versioned(merged_chunks) if pipeline_result["versioned"]
        else format_context(merged_chunks)
    )
    result["merged_all"]         = True
    result["_pipeline_only_top"] = _top_rerank_score(pipeline_result)
    result["_naive_only_top"]    = _top_rerank_score(naive_result)
    result["_overlap_only_top"]  = _top_rerank_score(overlap_result)
    return result


def retrieve_best_of_n(
    query: str,
    n: int = 3,
    intent: Optional[str]       = None,
    version: Optional[str]      = None,
    verbose: bool                = False,
    enable_thinking: bool       = False,
) -> Dict:
    """
    Best-of-N retrieval: generate n query reformulations, retrieve for each in
    parallel, return the result whose top rerank score is highest.

    The winner's "query" is set back to the original so the LLM sees what the
    user actually asked.  The "reformulations" key carries all n variants for
    transparency / debugging.
    """
    reformulations = _reformulate(query, n, enable_thinking=enable_thinking)
    log.info("Best-of-%d reformulations: %s", n, reformulations)

    # Pre-warm the lazy singletons (router's embedder, cross-encoder reranker,
    # ChromaDB's embedder) on THIS (main) thread before handing work to the
    # pool below. Loading a SentenceTransformer/CrossEncoder for the first
    # time from a worker thread was observed to crash with a torch "Cannot
    # copy out of meta tensor" error in this environment — the double-checked
    # locks in router.py/reranker.py/chroma_store.py prevent two threads from
    # loading the SAME model twice, but don't help if the very first load of
    # a fresh process happens to be kicked off on a non-main thread at all.
    # No-op (cheap attribute check) once already loaded.
    hnsw_router.get_router()
    reranker.warm()
    cs.warm()

    def _attempt(q: str) -> Dict:
        return retrieve(q, intent=intent, version=version, verbose=verbose)

    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(_attempt, q) for q in reformulations]
        results = [f.result() for f in futures]

    winner = max(results, key=_top_rerank_score)
    winner["query"]          = query          # restore original question
    winner["reformulations"] = reformulations
    winner["best_of_n"]      = True
    winner["best_of_scores"] = [_top_rerank_score(r) for r in results]
    return winner


# ── Retrieval paths ────────────────────────────────────────────────────────────

def _chroma_to_chunks(results: List[Dict]) -> List[Dict]:
    """Normalise ChromaDB result dicts to the standard chunk shape."""
    out = []
    for r in results:
        out.append({
            "chunk_id":       r.get("chunk_id", ""),
            "text":           r.get("text", ""),
            "section_header": r.get("section_header", r.get("master_label", r.get("section", ""))),
            "source_doc":     r.get("source_doc", r.get("source_docs", "")),
            "version":        r.get("version", ""),
            "score":          r.get("score", 0.0),
        })
    return out


def _retrieve_specific(query: str, verbose: bool) -> Dict:
    result     = hnsw_router.route(query)
    raw_chunks = result["chunks"]
    final      = reranker.rerank(query, raw_chunks)

    expanded_subs    = []
    expanded_parents = []
    pool = list(raw_chunks)

    ladder_enabled = (
        config.ENABLE_CONFIDENCE_LADDER
        or _is_relationship_query(query)
        or _is_evolution_query(query)
        or _is_cross_corpus_query(query)
    )
    if ladder_enabled and _top_rerank_score({"chunks": final}) < config.CONFIDENCE_THRESHOLD:
        exclude_ids = {c["chunk_id"] for c in pool}
        expansion   = hnsw_router.get_router().expand_siblings(result, exclude_ids)
        if expansion["chunks"]:
            pool         += expansion["chunks"]
            final         = reranker.rerank(query, pool)
            expanded_subs = expansion["expanded_subs"]
            log.info("Low confidence — pulled in sibling sub-parent(s): %s", expanded_subs)

        # Still low after sibling expansion — try a different parent family
        # entirely (the "key change" tier) before giving up on structure.
        if _top_rerank_score({"chunks": final}) < config.CONFIDENCE_THRESHOLD:
            exclude_ids   = {c["chunk_id"] for c in pool}
            bridge        = hnsw_router.get_router().expand_cross_parent(result, exclude_ids)
            if bridge["chunks"]:
                pool            += bridge["chunks"]
                final            = reranker.rerank(query, pool)
                expanded_parents = bridge["expanded_parents"]
                log.info("Still low — bridged to parent(s): %s", expanded_parents)

        # Still low after raw cross-parent chunks — try the precomputed
        # Parent×Parent relationship doc (Tier B): a genuinely synthesized
        # "how do these relate" object, not just pooled raw chunks. Last
        # rung before falling back further.
        if _top_rerank_score({"chunks": final}) < config.CONFIDENCE_THRESHOLD:
            rel_hits = cs.get_store().query_relationships(query, n=3)
            rel_chunks = [{
                "chunk_id":       f"relationship_{i}",
                "text":           r["text"],
                "section_header": f"{r.get('members','').replace(' | ', ' × ')} — Relationship",
                "source_doc":     "Cross-Domain Synthesis",
                "version":        "",
                "score":          r.get("score", 0.0),
            } for i, r in enumerate(rel_hits)]
            if rel_chunks:
                pool  += rel_chunks
                final  = reranker.rerank(query, pool)
                log.info("Still low — pulled in relationship doc(s): %s",
                          [c["section_header"] for c in rel_chunks])

        # Still low — try the precomputed cross-corpus relationship doc
        # (cross_corpus_relationship.py): links parents across TWO SEPARATE
        # document corpora (e.g. payer <-> connector), unlike the Tier B
        # rung above which only links parents within one taxonomy. Returns
        # nothing today since no second corpus exists yet — a no-op until
        # one does.
        if _top_rerank_score({"chunks": final}) < config.CONFIDENCE_THRESHOLD:
            cc_hits = cs.get_store().query_cross_corpus(query, n=3)
            cc_chunks = [{
                "chunk_id":       f"cross_corpus_{i}",
                "text":           r["text"],
                "section_header": f"{r.get('members','').replace(' | ', ' × ')} — Cross-Corpus",
                "source_doc":     "Cross-Corpus Synthesis",
                "version":        "",
                "score":          r.get("score", 0.0),
            } for i, r in enumerate(cc_hits)]
            if cc_chunks:
                pool  += cc_chunks
                final  = reranker.rerank(query, pool)
                log.info("Still low — pulled in cross-corpus doc(s): %s",
                          [c["section_header"] for c in cc_chunks])

        # Still low — try the precomputed evolution/value-add card
        # (evolution_analyzer.py): a synthesized "what value did this
        # release add" narrative, for the evolution-shaped queries that
        # reach this path (no version token / delta keyword, so intent
        # classification didn't already route them to _retrieve_delta).
        if _top_rerank_score({"chunks": final}) < config.CONFIDENCE_THRESHOLD:
            evo_hits = cs.get_store().query_evolution(query, n=3)
            evo_chunks = [{
                "chunk_id":       f"evolution_{i}",
                "text":           e["text"],
                "section_header": f"{e.get('feature_name','')} — Evolution",
                "source_doc":     f"{e.get('vA','')} → {e.get('vB','')}",
                "version":        "",
                "score":          e.get("score", 0.0),
            } for i, e in enumerate(evo_hits)]
            if evo_chunks:
                pool  += evo_chunks
                final  = reranker.rerank(query, pool)
                log.info("Still low — pulled in evolution card(s): %s",
                          [c["section_header"] for c in evo_chunks])

    path = dict(result["path"])
    if expanded_subs:
        path["expanded_subs"] = expanded_subs
    if expanded_parents:
        path["expanded_parents"] = expanded_parents

    return {
        "path":     path,
        "chunks":   final,
        "_source":  "hnsw",
        "_debug":   result.get("candidates", []) if verbose else [],
    }


def _retrieve_corpus(query: str, version: Optional[str], source_doc: Optional[str]) -> Dict:
    store   = cs.get_store()
    results = store.query_corpus(query, version=version, source_doc=source_doc)
    chunks  = reranker.rerank(query, _chroma_to_chunks(results))
    return {"path": {"collection": config.CORPUS_COLLECTION}, "chunks": chunks, "_source": "corpus"}


def _retrieve_intelligence(query: str, version: Optional[str]) -> Dict:
    store   = cs.get_store()
    results = store.query_intelligence(query, version=version)
    chunks  = reranker.rerank(query, _chroma_to_chunks(results))
    return {"path": {"collection": config.INTELLIGENCE_COLLECTION}, "chunks": chunks, "_source": "intelligence"}


def _retrieve_delta(query: str, version_from: Optional[str], version_to: Optional[str]) -> Dict:
    store   = cs.get_store()
    results = store.query_delta(query, version_from=version_from, version_to=version_to)
    chunks  = []
    for r in results:
        v_from = r.get("version_from", "")
        v_to   = r.get("version_to", "")
        src    = " → ".join(filter(None, [v_from, v_to])) or r.get("source_file", "")
        chunks.append({
            "chunk_id":       r.get("chunk_id", ""),
            "text":           r.get("text", ""),
            "section_header": r.get("section_header", r.get("section", "")),
            "source_doc":     src,
            "version":        r.get("version", ""),
            "score":          r.get("score", 0.0),
        })

    # Merge in evolution/value-add cards (evolution_analyzer.py). Most
    # naturally-phrased evolution queries ("how has X evolved since 25.2",
    # "what changed to improve X") contain a version token or a delta
    # keyword and land in THIS intent, not "specific" — so the constructive
    # value-add narrative needs to compete here, not just via the
    # confidence-ladder rung in _retrieve_specific (which only fires for
    # evolution-shaped queries with neither signal present).
    for e in store.query_evolution(query, n=config.CHROMA_N_EVOLUTION):
        chunks.append({
            "chunk_id":       f"evolution_{e.get('feature_name','')}",
            "text":           e.get("text", ""),
            "section_header": f"{e.get('feature_name','')} — Evolution",
            "source_doc":     f"{e.get('vA','')} → {e.get('vB','')}",
            "version":        "",
            "score":          e.get("score", 0.0),
        })

    chunks = reranker.rerank(query, chunks)
    return {"path": {"collection": config.DELTA_COLLECTION}, "chunks": chunks, "_source": "delta"}


# ── Main entry point ───────────────────────────────────────────────────────────

def retrieve(
    query: str,
    intent: Optional[str]       = None,
    version: Optional[str]      = None,
    source_doc: Optional[str]   = None,
    version_from: Optional[str] = None,
    version_to: Optional[str]   = None,
    verbose: bool                = False,
) -> Dict:
    """
    Full retrieval pipeline with smart conflict handling.

    Returns:
        {
          "query":         str,
          "intent":        str,
          "path":          dict,
          "chunks":        list of chunk dicts,
          "context":       str  — formatted context block ready for LLM,
          "system_prompt": str  — standard or versioned instruction set,
          "versioned":     bool — True if multi-version conflict was found,
          "delta_used":    bool — True if delta report was injected,
          "debug":         dict — only if verbose=True
        }
    """
    resolved_intent = intent or classify(query)
    log.info(f"Intent: {resolved_intent!r} — '{query[:70]}'")

    # ── Step 1: primary retrieval ──────────────────────────────────────────────
    if resolved_intent == "specific":
        raw = _retrieve_specific(query, verbose)
        # Widen to corpus if HNSW came up empty
        if not raw["chunks"]:
            log.info("HNSW empty — widening to corpus")
            raw = _retrieve_corpus(query, version, source_doc)

    elif resolved_intent == "factual":
        raw = _retrieve_corpus(query, version, source_doc)

    elif resolved_intent == "intelligence":
        raw = _retrieve_intelligence(query, version)

    elif resolved_intent == "delta":
        raw = _retrieve_delta(query, version_from, version_to)

    else:
        raw = _retrieve_specific(query, verbose)

    chunks  = raw["chunks"]
    source  = raw["_source"]

    # ── Step 2: conflict detection ─────────────────────────────────────────────
    conflicting_sections = detect_version_conflict(chunks)
    versioned            = bool(conflicting_sections)
    delta_findings       = []
    delta_used           = False

    if versioned:
        log.info(f"Version conflict in {len(conflicting_sections)} section(s): "
                 f"{conflicting_sections[:3]}")

        # Only inject delta if we didn't already come from the delta path
        if source != "delta":
            delta_findings = _inject_delta(conflicting_sections, query)
            delta_used     = bool(delta_findings)
            if delta_used:
                log.info(f"Delta injected: {len(delta_findings)} finding(s)")
            else:
                log.warning("Conflict detected but no delta findings found — "
                            "versioned format only")

    # ── Step 3: format context + pick system prompt ────────────────────────────
    if versioned:
        context       = format_context_versioned(chunks, delta_findings or None)
        system_prompt = SYSTEM_PROMPT_VERSIONED
    else:
        context       = format_context(chunks)
        system_prompt = SYSTEM_PROMPT_STANDARD

    # ── Step 4: assemble output ────────────────────────────────────────────────
    out = {
        "query":         query,
        "intent":        resolved_intent,
        "path":          raw["path"],
        "chunks":        chunks,
        "context":       context,
        "system_prompt": system_prompt,
        "versioned":     versioned,
        "delta_used":    delta_used,
    }

    if verbose:
        out["debug"] = {
            "source":                source,
            "conflicting_sections":  conflicting_sections,
            "delta_findings_count":  len(delta_findings),
            "hnsw_candidates":       raw.get("_debug", []),
        }

    return out
