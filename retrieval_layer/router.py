"""
router.py
─────────
Hierarchical query router.

Flow:
  query
    → embed
    → ONE unrestricted HNSW knn over topic+QnA vectors (QnA hints included)
    → derive parent/sub route directly from the winning hits' own metadata
      (ambiguity-gap keeps multiple close sections instead of hard-committing
      to a single noisy argmax — same hits ARE the candidates, so routing
      can never exclude the evidence it was decided from)
    → reranker re-scores candidates
    → return top chunk_ids + routing path

Previously routed top-down by comparing the query against parent/sub NAME
embeddings first, then searching within that (possibly wrong) filter. That
meant QnA hints — which are genuinely useful, specific signal — only ever
got to influence which chunk won inside an already-chosen section, never
which section to look in. It also meant a single-point argmax over noisy,
often near-tied sub-category name scores could silently route a query to
the wrong section with no recovery. Both are fixed by deciding the route
from real content-hit scores instead of category-name proximity.
"""

import pickle
import logging
import threading
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import hnswlib
from sentence_transformers import SentenceTransformer

import config

log = logging.getLogger(__name__)


class Router:
    def __init__(self):
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        log.info("Loading retrieval index...")

        def _load(name):
            with open(config.INDEX_DIR / name, "rb") as f:
                return pickle.load(f)

        self.parent_meta  = _load("parent_meta.pkl")   # list of dicts (names only, now)
        self.sub_meta     = _load("sub_meta.pkl")       # list of dicts (names only, now)
        self.hnsw_meta    = _load("hnsw_meta.pkl")      # sidecar for HNSW ids
        self.chunk_lookup = _load("chunk_lookup.pkl")   # str(id) → chunk dict

        # HNSW
        self.hnsw = hnswlib.Index(space="cosine", dim=config.EMBED_DIM)
        self.hnsw.load_index(str(config.INDEX_DIR / "hnsw.bin"),
                             max_elements=config.HNSW_MAX_ELEMENTS)
        self.hnsw.set_ef(config.HNSW_EF_QUERY)

        # Embedder
        self.model = SentenceTransformer(config.EMBED_MODEL, device="cpu")

        self._loaded = True
        log.info(f"Index ready — {len(self.parent_meta)} parents, "
                 f"{len(self.sub_meta)} subs, "
                 f"{self.hnsw.get_current_count()} HNSW vectors")

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=True, convert_to_numpy=True)

    def _content_hits(self, query_emb: np.ndarray) -> List[Dict]:
        """
        One unrestricted HNSW search over ALL topic + QnA vectors. This is
        the only search — routing and candidate selection both come from it.
        """
        labels, distances = self.hnsw.knn_query(
            query_emb.reshape(1, -1), k=config.TOP_K_HNSW
        )
        hits = []
        seen_chunk_sets = set()
        for hnsw_id, dist in zip(labels[0], distances[0]):
            meta = self.hnsw_meta[hnsw_id]
            key = tuple(sorted(meta["chunk_ids"]))
            if key in seen_chunk_sets:
                continue
            seen_chunk_sets.add(key)
            hits.append({
                **meta,
                "score": float(1 - dist),   # cosine distance → similarity
            })
        return hits

    def _route_from_hits(self, hits: List[Dict]) -> Dict[tuple, float]:
        """
        Derive candidate (parent_idx, sub_idx) routes directly from content
        hits — a single strong topic or QnA match is a strong routing signal,
        so we take the max hit score per (parent, sub) pair rather than
        comparing the query against a separate, coarser category-name
        embedding. Ambiguity-gap keeps every pair within ROUTE_AMBIGUITY_GAP
        of the best, instead of hard-committing to a single argmax.
        """
        pair_scores: Dict[tuple, float] = {}
        for h in hits:
            key = (h["parent_idx"], h["sub_idx"])
            pair_scores[key] = max(pair_scores.get(key, 0.0), h["score"])
        return pair_scores

    # ── Public API ─────────────────────────────────────────────────────────────

    def route(self, query: str) -> Dict:
        """
        Route a query through the hierarchy and return matching chunks.

        Returns:
            {
              "path": {"parent": str, "sub": str},
              "candidates": [{"label", "type", "score", "chunk_ids"}, ...],
              "chunks": [{"chunk_id", "text", "section_header", "source_doc"}, ...]
            }
        """
        self.load()

        query_emb = self._embed(query)

        # One unrestricted search — includes topic AND QnA vectors.
        hits = self._content_hits(query_emb)

        # Route derived from the hits themselves (content-first, not category
        # names) — a strong QnA match now gets a real vote on which section
        # this query belongs to, and ambiguity-gap keeps close sections
        # instead of hard-committing to a single noisy argmax.
        pair_scores = self._route_from_hits(hits)
        if pair_scores:
            best_score = max(pair_scores.values())
            chosen_pairs = {
                pair for pair, score in pair_scores.items()
                if best_score - score < config.ROUTE_AMBIGUITY_GAP
            }
        else:
            chosen_pairs = set()

        parent_indices = sorted({p for p, s in chosen_pairs})
        sub_indices    = sorted({s for p, s in chosen_pairs})
        parent_names   = [self.parent_meta[i]["name"] for i in parent_indices]
        sub_names      = [self.sub_meta[i]["name"] for i in sub_indices]

        log.info(f"Routed → parent: {parent_names}  sub: {sub_names}")

        # Candidates are the SAME hits used to decide the route — the pairs
        # that won routing are, by construction, present in `hits`, so this
        # can never come back empty the way the old filter-after-search
        # design could.
        candidates = [h for h in hits if (h["parent_idx"], h["sub_idx"]) in chosen_pairs]

        # Level 4 — collect unique chunk_ids from top candidates
        all_chunk_ids = []
        seen = set()
        for c in sorted(candidates, key=lambda x: x["score"], reverse=True):
            for cid in c["chunk_ids"]:
                if cid not in seen:
                    seen.add(cid)
                    all_chunk_ids.append(cid)
            if len(all_chunk_ids) >= config.TOP_K_FINAL * 3:
                break

        # Fetch chunk texts
        chunks = []
        for cid in all_chunk_ids:
            chunk = self.chunk_lookup.get(str(cid))
            if chunk:
                chunks.append({"chunk_id": cid, **chunk})

        return {
            "path": {
                "parents": parent_names,
                "subs":    sub_names,
            },
            "candidates": candidates[:config.TOP_K_HNSW],
            "chunks":     chunks[:config.TOP_K_FINAL * 3],
            # Internal — consumed by expand_siblings() if reranked confidence
            # on the chosen route turns out low. Not part of the public shape.
            "_hits":         hits,
            "_chosen_pairs": list(chosen_pairs),
        }

    def _collect_pair_chunks(
        self,
        hits: List[Dict],
        pairs: List[tuple],
        exclude_chunk_ids: set,
    ) -> List[Dict]:
        """Shared by expand_siblings/expand_cross_parent: fetch chunk texts
        for the given (parent_idx, sub_idx) pairs, ranked by hit score,
        deduped against exclude_chunk_ids, capped at TOP_K_FINAL * 3."""
        pair_set = set(pairs)
        pair_hits = [h for h in hits if (h["parent_idx"], h["sub_idx"]) in pair_set]

        chunk_ids = []
        seen = set(exclude_chunk_ids)
        for h in sorted(pair_hits, key=lambda x: x["score"], reverse=True):
            for cid in h["chunk_ids"]:
                if cid not in seen:
                    seen.add(cid)
                    chunk_ids.append(cid)
            if len(chunk_ids) >= config.TOP_K_FINAL * 3:
                break

        chunks = []
        for cid in chunk_ids:
            chunk = self.chunk_lookup.get(str(cid))
            if chunk:
                chunks.append({"chunk_id": cid, **chunk})
        return chunks

    def expand_siblings(
        self,
        route_result: Dict,
        exclude_chunk_ids: Optional[set] = None,
        max_new_subs: int = 2,
    ) -> Dict:
        """
        Pull in chunks from sibling sub-parents under the SAME routed parent(s)
        — sub-parents that scored below ROUTE_AMBIGUITY_GAP and so weren't
        chosen, but are still the next-best content hits within the same
        family. Used when reranked confidence on the chosen sub-parent is
        low, before ever falling back to an unrestricted corpus search.

        This is a free lookup, not a new search — it reuses the single HNSW
        query's hits and pair scores already computed by route().

        Returns:
            {"chunks": [...], "expanded_subs": [sub_name, ...]}
        """
        hits         = route_result.get("_hits", [])
        chosen_pairs = set(tuple(p) for p in route_result.get("_chosen_pairs", []))
        exclude_chunk_ids = exclude_chunk_ids or set()

        if not hits or not chosen_pairs:
            return {"chunks": [], "expanded_subs": []}

        chosen_parent_indices = {p for p, s in chosen_pairs}
        pair_scores = self._route_from_hits(hits)

        sibling_pairs = sorted(
            (pair for pair in pair_scores
             if pair[0] in chosen_parent_indices and pair not in chosen_pairs),
            key=lambda pair: pair_scores[pair],
            reverse=True,
        )[:max_new_subs]

        if not sibling_pairs:
            return {"chunks": [], "expanded_subs": []}

        chunks        = self._collect_pair_chunks(hits, sibling_pairs, exclude_chunk_ids)
        expanded_subs = [self.sub_meta[s]["name"] for p, s in sibling_pairs]
        log.info(f"Expanded into sibling sub-parent(s): {expanded_subs}")

        return {"chunks": chunks, "expanded_subs": expanded_subs}

    def expand_cross_parent(
        self,
        route_result: Dict,
        exclude_chunk_ids: Optional[set] = None,
        max_new_parents: int = 1,
    ) -> Dict:
        """
        Bridge to a DIFFERENT parent family entirely — the "key change" tier,
        tried only when confidence is still low after expand_siblings
        couldn't resolve it within the same family. Same free lookup as
        expand_siblings: reuses the hits/pair scores already computed by
        route(), just drops the same-parent constraint.

        Returns:
            {"chunks": [...], "expanded_parents": [parent_name, ...]}
        """
        hits         = route_result.get("_hits", [])
        chosen_pairs = set(tuple(p) for p in route_result.get("_chosen_pairs", []))
        exclude_chunk_ids = exclude_chunk_ids or set()

        if not hits or not chosen_pairs:
            return {"chunks": [], "expanded_parents": []}

        chosen_parent_indices = {p for p, s in chosen_pairs}
        pair_scores = self._route_from_hits(hits)

        cross_pairs = sorted(
            (pair for pair in pair_scores if pair[0] not in chosen_parent_indices),
            key=lambda pair: pair_scores[pair],
            reverse=True,
        )[:max_new_parents]

        if not cross_pairs:
            return {"chunks": [], "expanded_parents": []}

        chunks           = self._collect_pair_chunks(hits, cross_pairs, exclude_chunk_ids)
        expanded_parents = sorted({self.parent_meta[p]["name"] for p, s in cross_pairs})
        log.info(f"Still low after sibling expansion — bridged to parent(s): {expanded_parents}")

        return {"chunks": chunks, "expanded_parents": expanded_parents}


# Singleton — load once, reuse across queries.
#
# Double-checked locking: retrieve_best_of_n() (retriever.py) runs N query
# reformulations through retrieve() concurrently via ThreadPoolExecutor, and
# that's now the DEFAULT cheap-first-pass for the gated CLI path (see
# redis_cache.gate_and_retrieve), not just an opt-in debug flag — so this
# singleton WILL be hit by multiple threads racing on the very first query of
# a fresh process. Without the lock, two threads can both see _router is None,
# both construct a Router() and call .load() concurrently, and both start
# loading the SAME SentenceTransformer model at once — observed to crash with
# "Cannot copy out of meta tensor; no data!" (a torch/transformers lazy-init
# race, not this codebase's model logic).
_router: Optional[Router] = None
_router_lock = threading.Lock()

def get_router() -> Router:
    global _router
    if _router is None:
        with _router_lock:
            if _router is None:          # re-check inside the lock
                _router = Router()
                _router.load()
    return _router


def route(query: str) -> Dict:
    return get_router().route(query)
