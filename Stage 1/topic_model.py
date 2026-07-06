"""
topic_model.py
──────────────
Thin orchestration layer between ingest and registry.

Previously: BERTopic + HDBSCAN cluster-level labeling
Now: delegates directly to labeler for per-chunk master labeling

Includes cross-version fuzzy matching via embedding similarity
so that the same feature in v25.2 and v26.1 gets grouped together
even if the LLM-generated labels differ slightly.
"""

import logging
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import config
import labeler

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


def run_labeling(chunks: List[Dict]) -> List[Dict]:
    """
    Assign master_label + description to every chunk.
    Returns enriched chunk list.
    """
    return labeler.label_all_chunks(chunks)


def group_by_master_label(chunks: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Two-phase grouping:
      Phase 1 — Exact string match on master_label (fast, precise).
      Phase 2 — Cross-version embedding merge for single-doc groups
                that describe the same feature but got different labels.

    Returns { master_label: [chunk, chunk, ...] }
    """
    # ── Phase 1: exact match ─────────────────────────────────────────────────
    groups: Dict[str, List[Dict]] = {}
    for chunk in chunks:
        label = chunk.get("master_label") or "Uncategorized"
        groups.setdefault(label, []).append(chunk)

    # ── Phase 2: cross-version fuzzy matching ────────────────────────────────
    threshold = getattr(config, 'CROSS_VERSION_MATCH_THRESHOLD', 0.75)

    # Identify single-doc groups and which doc they come from
    single_doc_groups: Dict[str, str] = {}
    for label, group_chunks in groups.items():
        docs = {c.get("source_doc", "") for c in group_chunks}
        if len(docs) == 1:
            single_doc_groups[label] = next(iter(docs))

    # Bucket by source doc
    doc_buckets: Dict[str, List[str]] = defaultdict(list)
    for label, doc in single_doc_groups.items():
        doc_buckets[doc].append(label)

    doc_names = sorted(doc_buckets.keys())
    if len(doc_names) < 2:
        log.info("[Cross-Version] Only one document — skipping fuzzy matching.")
        return groups

    # For each pair of docs, compute cross-version matches
    embedder = _get_embedder()
    total_merges = 0

    for di in range(len(doc_names)):
        for dj in range(di + 1, len(doc_names)):
            labels_a = doc_buckets[doc_names[di]]
            labels_b = doc_buckets[doc_names[dj]]

            if not labels_a or not labels_b:
                continue

            # Build embedding text: label + first description for richer signal
            def _embed_text(label: str) -> str:
                desc = ""
                if label in groups:
                    for c in groups[label]:
                        d = c.get("description", "")
                        if d:
                            desc = str(d)[:200]
                            break
                return f"{label}. {desc}".strip()

            texts_a = [_embed_text(l) for l in labels_a]
            texts_b = [_embed_text(l) for l in labels_b]

            emb_a = embedder.encode(texts_a, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
            emb_b = embedder.encode(texts_b, batch_size=64, normalize_embeddings=True, show_progress_bar=False)

            sim = cosine_similarity(emb_a, emb_b)

            # Greedy bipartite matching (descending similarity)
            pairs = []
            for i in range(len(labels_a)):
                for j in range(len(labels_b)):
                    if sim[i][j] >= threshold:
                        pairs.append((sim[i][j], i, j))
            pairs.sort(reverse=True)

            matched_a: set = set()
            matched_b: set = set()

            for score, i, j in pairs:
                if i in matched_a or j in matched_b:
                    continue
                label_a = labels_a[i]
                label_b = labels_b[j]

                # Pick the more descriptive label as canonical
                canonical = label_a if len(label_a) >= len(label_b) else label_b
                other = label_b if canonical == label_a else label_a

                # Merge other group into canonical
                for c in groups[other]:
                    c["master_label"] = canonical
                    c["_cross_version_match"] = {
                        "original_label": other,
                        "matched_with": canonical,
                        "similarity": round(float(score), 3),
                    }
                groups[canonical].extend(groups.pop(other))

                matched_a.add(i)
                matched_b.add(j)
                total_merges += 1

                log.info(
                    f"  Cross-version merge (sim={score:.3f}): "
                    f"'{other}' → '{canonical}'"
                )

    if total_merges:
        log.info(f"[Cross-Version] Merged {total_merges} label pairs across documents.")
    else:
        log.info("[Cross-Version] No cross-version matches found above threshold.")

    return groups


def aggregate_keywords_for_group(group_chunks: List[Dict]) -> List[str]:
    """
    Pool all keywords from chunks in a group, return top-10 by frequency.
    Used to populate the keywords column in the registry.
    """
    counter = Counter()
    for chunk in group_chunks:
        for kw in chunk.get("keywords", []):
            counter[kw] += 1
    return [kw for kw, _ in counter.most_common(10)]