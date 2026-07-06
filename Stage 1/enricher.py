"""
enricher.py
───────────
Post-processing pass on chunks.json.

Problem: some chunks that survive noise filtering are still content-poor —
typically legal citation fragments or brief transitional paragraphs. 
Their descriptions end up hollow and generic.

Solution — LLM Quality Judge + Two-Route Enrichment:
  0. LLM Judge: Rapidly evaluates text to decide if it actually needs context.
  1. Route 1 (Internal RAG): Find similar chunks from the SAME document.
  2. Route 2 (Web search fallback): DuckDuckGo search for case names/concepts.

Run after ingest + labeling, before registry building:
  python enricher.py
"""

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import config

log = logging.getLogger(__name__)

_embedder: Optional[SentenceTransformer] = None


def unload():
    global _embedder
    if _embedder is not None:
        del _embedder
        _embedder = None
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")
    return _embedder


import llm_client


# ── Text Quality Math (For Analytics) ──────────────────────────────────────────

"""
1. The Regex Detectives (What it looks for)

The first part of the code uses Regular Expressions (Regex) to hunt for specific patterns:

    _CITATION_RE: This hunts for standard Indian legal citations. It will instantly flag text that looks like:

        "Somebody v Somebody (1973) 4 SCC 225"

        "AIR 1951 SC 332"

    _PAGE_REF_RE: This hunts for the "dots and page numbers" you see at the end of an Index or Table of Contents. It flags text like:

        ". . . . . 14, 15, xiv"

2. The Math (The 3 Ingredients of the Score)

The function _calculate_text_quality splits the text into lines and calculates three ingredients:

    Noise Ratio (Weight: 50%): How many lines contain a legal citation or index page number? If every line is a court case, the noise is 100%. (We want this to be low).

    Prose Score (Weight: 30%): How many actual, complete sentences are there that have more than 8 words? (We want this to be high).

    Alpha Score (Weight: 20%): What percentage of the text is made of letters (A-Z) versus numbers and symbols (1, 2, 3, (, ), &)? (We want this to be high).

The Final Formula:
Quality = (Lack of Noise * 0.50) + (Prose * 0.30) + (Alpha * 0.20)

"""

_CITATION_RE = re.compile(
    r"(\bv\b.{3,60}\(\d{4}\).{0,30}(SCC|SCR|AIR|All|Bom|Cal|Mad))"
    r"|(\bAIR\s+\d{4}\s+SC)"
    r"|(\(\d{4}\)\s+\d+\s+SCC\s+\d+)",
    re.IGNORECASE,
)
_PAGE_REF_RE = re.compile(r"\.\s*[\dxivlc]{1,5}(?:\s*,\s*[\dxivlc]{1,5})+\s*$")

def _calculate_text_quality(text: str) -> float:
    """
    Mathematical score from 0.0 (pure citation noise) to 1.0 (rich prose).
    Useful for backend data analysis and auditing.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0

    citation_lines  = sum(1 for ln in lines if _CITATION_RE.search(ln))
    page_ref_lines  = sum(1 for ln in lines if _PAGE_REF_RE.search(ln))
    noise_ratio     = (citation_lines + page_ref_lines) / max(len(lines), 1)

    sentences       = re.split(r"[.!?]+", text)
    prose_sentences = sum(1 for s in sentences if len(s.split()) > 8)
    prose_score     = min(prose_sentences / max(len(lines), 1), 1.0)

    alpha  = sum(c.isalpha() for c in text)
    total  = max(len(text.replace(" ", "")), 1)
    alpha_score = alpha / total

    quality = ((1 - noise_ratio) * 0.50) + (prose_score * 0.30) + (alpha_score * 0.20)
    return round(quality, 3)


# ── LLM Quality Judge ──────────────────────────────────────────────────────────

_EVAL_PROMPT = (
    "Read this text passage. Does it contain enough substantive context and "
    "information to be summarized on its own, or is it a fragmented list of citations, "
    "names, or incomplete sentences that requires outside context to understand?\n\n"
    "Text: {text}\n\n"
    "Reply EXACTLY with one word: ADEQUATE or INCOMPLETE"
)

def _evaluate_chunks_with_llm(chunks: List[Dict]) -> List[Dict]:
    """
    Uses heuristic quality score to judge if chunks need RAG enrichment.
    Replaces the LLM judge — same signal, zero LLM calls.
    """
    poor_chunks = []

    chunks_to_check = [
        c for c in chunks
        if len(str(c.get("description", "")).split()) < config.DESCRIPTION_MIN_LENGTH
    ]

    if not chunks_to_check:
        return []

    log.info(f"Heuristic judge evaluating {len(chunks_to_check)} potentially poor chunks...")

    for chunk in chunks_to_check:
        score = _calculate_text_quality(chunk.get("text", ""))
        if score < config.ENRICHMENT_QUALITY_THRESHOLD:
            chunk["_quality_status"] = "INCOMPLETE"
            poor_chunks.append(chunk)
        else:
            chunk["_quality_status"] = "ADEQUATE"

    log.info(f"Heuristic judge: {len(poor_chunks)}/{len(chunks_to_check)} flagged for enrichment")
    return poor_chunks


# ── Route 1: Internal RAG ──────────────────────────────────────────────────────

def _build_source_index(
    chunks: List[Dict],
    poor_chunk_ids: set,
) -> Dict[str, Tuple[List[Dict], np.ndarray]]:
    from collections import defaultdict
    from tqdm import tqdm

    model = _get_embedder()
    by_doc: Dict[str, List[Dict]] = defaultdict(list)
    
    for c in chunks:
        if c.get("chunk_id") not in poor_chunk_ids and len(c.get("text", "").split()) > 30:
            by_doc[c["source_doc"]].append(c)

    index: Dict[str, Tuple[List[Dict], np.ndarray]] = {}
    log.info("Building per-document embedding index for RAG...")
    
    for doc, doc_chunks in tqdm(by_doc.items(), desc="Indexing docs", leave=False):
        texts = [c["text"] for c in doc_chunks]
        embs  = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        index[doc] = (doc_chunks, embs)

    return index


def _internal_rag(chunk: Dict, source_index: Dict) -> List[str]:
    doc = chunk.get("source_doc", "")
    if doc not in source_index:
        return []

    doc_chunks, doc_embs = source_index[doc]
    model  = _get_embedder()
    
    q_emb  = model.encode([chunk["text"]], normalize_embeddings=True, show_progress_bar=False)
    sims   = cosine_similarity(q_emb, doc_embs)[0]

    top_k  = np.argsort(sims)[::-1]
    top_k  = [i for i in top_k if sims[i] < 0.999][: config.ENRICHMENT_INTERNAL_TOP_K]

    return [doc_chunks[i]["text"] for i in top_k]


# ── Route 2: Web search ────────────────────────────────────────────────────────

def _extract_search_terms(chunk: Dict) -> str:
    keywords = chunk.get("keywords", [])
    text     = chunk.get("text", "")
    case_names = re.findall(r"([A-Z][a-zA-Z ]{2,30})\s+v\s+([A-Z][a-zA-Z ]{2,30})", text)
    case_query = " ".join(f"{a} v {b}" for a, b in case_names[:2])
    kw_query  = " ".join(keywords[:3])
    label     = chunk.get("master_label", "")
    return f"{label} {case_query} {kw_query}".strip()[:200]


def _web_search(query: str) -> List[str]:
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=config.ENRICHMENT_WEB_MAX_RESULTS))
        return [r.get("body", "") for r in results if r.get("body")]
    except ImportError:
        log.warning("duckduckgo_search not installed. pip install duckduckgo-search")
        return []
    except Exception as e:
        log.warning(f"Web search failed: {e}")
        return []


# ── Description generation with context ───────────────────────────────────────

_ENRICHMENT_SYSTEM_STATIC = """You are a Healthcare based assistant.
You are given a text passage (which may be sparse or citation-heavy) and
additional context from related sources.

Using BOTH the passage and the context, write a 2-3 sentence description
that explains what this passage is about in substantive terms.
Be specific and informative — do not be generic.
Reply with ONLY the description text. No JSON, no preamble."""


def _build_dynamic_enrichment_system(profile: dict) -> str:
    """Build an enrichment system prompt from a domain profile."""
    role = profile.get("specialist_role", "a technical assistant")
    domain = profile.get("domain", "technical documentation")
    term_block = ""
    terms = profile.get("key_terminology", {})
    if terms:
        term_lines = [f"  {k} = {v}" for k, v in list(terms.items())[:8]]
        term_block = "\nKey terminology:\n" + "\n".join(term_lines) + "\n"
    return (
        f"You are {role}.\n"
        f"The documents cover: {domain}.\n"
        f"{term_block}\n"
        f"You are given a text passage (which may be sparse or citation-heavy) and "
        f"additional context from related sources.\n\n"
        f"Using BOTH the passage and the context, write a 2-3 sentence description "
        f"that explains what this passage is about in substantive terms.\n"
        f"Be specific and informative — do not be generic.\n"
        f"Reply with ONLY the description text. No JSON, no preamble."
    )


def _get_enrichment_system(source_doc: str = "") -> str:
    """Return domain-adapted system prompt — saved > dynamic > static."""
    if source_doc:
        try:
            import context_profiler
            profile = context_profiler.get_profile(source_doc)
            if profile and profile.get("domain"):
                type_key = profile.get("type_key", "")
                saved = context_profiler.load_prompt(type_key, "enricher")
                if saved:
                    return saved
                prompt = _build_dynamic_enrichment_system(profile)
                context_profiler.save_prompt(type_key, "enricher", prompt)
                return prompt
        except ImportError:
            pass
    return _ENRICHMENT_SYSTEM_STATIC


def _generate_enriched_description(
    chunk: Dict,
    context_texts: List[str],
    source: str,
) -> str:
    context_block = "\n\n---\n\n".join(
        f"[Context {i+1}]: {ctx[:500]}"
        for i, ctx in enumerate(context_texts[:3])
    )

    user = (
        f"MASTER LABEL: {chunk.get('master_label', '')}\n\n"
        f"PASSAGE (may be sparse):\n{chunk['text'][:600]}\n\n"
        f"RELATED CONTEXT (from {source}):\n{context_block}\n\n"
        "Write a specific, informative 2-3 sentence description of what "
        "this passage covers, using the context to fill gaps."
    )

    system_prompt = _get_enrichment_system(chunk.get("source_doc", ""))

    try:
        result = llm_client.generate_local(user, max_tokens=config.ENRICHMENT_MAX_TOKENS, system_prompt=system_prompt, stop=llm_client.STOP_TEXT)
        result = re.sub(r"```.*?```", "", result, flags=re.DOTALL).strip()
        if len(result.split()) >= 8:
            return result
    except Exception as e:
        log.warning(f"Enrichment generation failed: {e}")

    return chunk.get("description", "")


# ── Main enrichment pass ───────────────────────────────────────────────────────

def enrich_chunks(chunks: List[Dict]) -> List[Dict]:
    from tqdm import tqdm

    # 1. Base Metadata: Score all chunks and set default source to "internal"
    for c in chunks:
        c["text_quality_score"] = _calculate_text_quality(c.get("text", ""))
        c["enrichment_source"] = "internal" # Base assumption

    # 2. Use the LLM to strictly find the chunks that ACTUALLY need RAG
    poor = _evaluate_chunks_with_llm(chunks)
    log.info(f"Enrichment: LLM Judge determined {len(poor)}/{len(chunks)} chunks truly need RAG/Web Search")

    if not poor:
        log.info("No chunks require enrichment — skipping.")
        return chunks

    poor_chunk_ids = {c.get("chunk_id") for c in poor}
    source_index = _build_source_index(chunks, poor_chunk_ids)

    route1_count = route2_count = skipped = 0

    for chunk in tqdm(poor, desc="Enriching descriptions via RAG/Web"):
        # ── Route 1: Internal RAG ──────────────────────────────────────────────
        context = _internal_rag(chunk, source_index)

        if len(context) >= 2:
            new_desc = _generate_enriched_description(chunk, context, "internal PDF")
            chunk["description"] = new_desc
            chunk["enrichment_source"] = "traversed further and generated"
            route1_count += 1
            continue

        # ── Route 2: Web search ────────────────────────────────────────────────
        if config.USE_WEB_ENRICHMENT:
            query   = _extract_search_terms(chunk)
            results = _web_search(query)

            if results:
                new_desc = _generate_enriched_description(chunk, results, "web search")
                chunk["description"] = new_desc
                chunk["enrichment_source"] = "web scraped"
                route2_count += 1
                continue

        skipped += 1
        chunk["enrichment_source"] = "enrichment_failed"

    log.info(
        f"Enrichment complete — "
        f"Route 1 (traversed further): {route1_count} | "
        f"Route 2 (web scraped): {route2_count} | "
        f"Skipped: {skipped}"
    )
    return chunks


# ── Entry point ────────────────────────────────────────────────────────────────

def run(chunks_path: Path = config.CHUNKS_CACHE) -> List[Dict]:
    log.info(f"Loading chunks from {chunks_path}")
    with open(chunks_path) as f:
        chunks = json.load(f)

    chunks = enrich_chunks(chunks)

    with open(chunks_path, "w") as f:
        json.dump(chunks, f, indent=2)
    log.info(f"Enriched chunks saved back to {chunks_path}")
    return chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run()