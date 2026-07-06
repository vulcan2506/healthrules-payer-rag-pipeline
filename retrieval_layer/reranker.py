"""
reranker.py
───────────
Cross-encoder reranker. Takes the router's candidate chunks and re-scores
them against the raw query — much more precise than embedding cosine similarity.
"""

import logging
import threading
from typing import List, Dict, Optional

from sentence_transformers import CrossEncoder

import config

log = logging.getLogger(__name__)

_model: Optional[CrossEncoder] = None
_model_lock = threading.Lock()


def _get_model() -> CrossEncoder:
    # Double-checked locking — see router.py's get_router() for why this
    # matters now that retrieve_best_of_n's parallel reformulations are the
    # default cheap-first-pass (redis_cache.gate_and_retrieve), not just an
    # opt-in debug flag: multiple threads can race to lazy-load this model on
    # the first query of a fresh process.
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                log.info(f"Loading reranker: {config.RERANKER_MODEL}")
                _model = CrossEncoder(config.RERANKER_MODEL)
    return _model


def warm() -> None:
    """
    Force-load the reranker model now, on whichever thread calls this.
    retrieve_best_of_n() (retriever.py) calls this on the MAIN thread before
    spinning up its ThreadPoolExecutor — loading a CrossEncoder for the
    first time from a worker thread has been observed to crash with a torch
    "Cannot copy out of meta tensor" error in this environment. No-op once
    already loaded.
    """
    _get_model()


def rerank(query: str, chunks: List[Dict]) -> List[Dict]:
    """
    Re-score chunks against the query using a cross-encoder.

    Args:
        query:  raw user query string
        chunks: list of chunk dicts with 'text' field (from router)

    Returns:
        Same list, sorted by reranker score descending, limited to TOP_K_FINAL.
        Each dict gains a 'rerank_score' field.
    """
    if not chunks:
        return []

    model  = _get_model()
    pairs  = [(query, c["text"]) for c in chunks]
    scores = model.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    ranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:config.TOP_K_FINAL]
