"""
label_normalizer.py
───────────────────
Post-labeling normalization and deduplication pass.

Runs after labeler.label_all_chunks() and before registry.build_registry().
Clusters near-duplicate labels via embedding similarity and assigns a
canonical label to each cluster.

Problems solved:
  - "Billing & Financials" vs "Billing and Financials" (symbol variant)
  - "Runtime Property" vs "Runtime Properties" (plural)
  - "COB Savings Adjustment" vs "COB Saving Adjustment" (typo)
  - DB table names used as labels (HCFA1500_SERVICE_LINE_FACT)
  - Single-word garbage labels ("Bug", "Modify", "Amend")
  - Keyword-concatenation fallback labels
"""

import re
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

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


# ── Text normalization (for clustering, not final output) ─────────────────────

def _normalize_for_clustering(label: str) -> str:
    """Deterministic normalization to help embedding similarity."""
    s = label.lower().strip()
    s = s.replace('&', 'and')
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    words = s.split()
    words = [w for w in words if w not in ('the', 'a', 'an')]
    if words and words[-1].endswith('s') and len(words[-1]) > 5:
        words[-1] = words[-1][:-1]
    return ' '.join(words)


# ── Garbage detection ─────────────────────────────────────────────────────────

_DB_TABLE_RE = re.compile(r'^[A-Z][A-Z0-9_]+$')
_KEYWORD_FALLBACK_RE = re.compile(r'^([A-Z][a-z]+ ){2,}[A-Z]')
_TABLE_WORDS = {'fact', 'table', 'column', 'nullable', 'varchar', 'number', 'datatype'}


def _is_garbage_label(label: str) -> bool:
    stripped = label.strip()
    if not stripped:
        return True
    if _DB_TABLE_RE.match(stripped) and '_' in stripped:
        return True
    words = stripped.split()
    if len(words) < config.LABEL_MIN_WORDS:
        return True
    if len(words) >= 6 and _KEYWORD_FALLBACK_RE.match(stripped):
        return True
    if re.match(r'^\d+$', stripped):
        return True
    lower_words = {w.lower() for w in words}
    if len(lower_words & _TABLE_WORDS) >= 2:
        return True
    return False


# ── Union-Find for label clustering ──────────────────────────────────────────

class _LabelUF:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


# ── Canonical label selection ─────────────────────────────────────────────────

def _pick_canonical(labels: List[str], garbage_set: Set[str]) -> str:
    """
    Pick the best label from a cluster.
    Prefers: non-garbage > longer > title-cased.
    """
    non_garbage = [l for l in labels if l not in garbage_set]
    candidates = non_garbage if non_garbage else labels

    def score(label: str) -> Tuple[int, int, int]:
        words = label.split()
        is_good_length = 1 if 2 <= len(words) <= 6 else 0
        is_title = 1 if label[0].isupper() else 0
        return (is_good_length, len(words), is_title)

    return max(candidates, key=score)


# ── Main entry point ─────────────────────────────────────────────────────────

def normalize_labels(chunks: List[Dict]) -> List[Dict]:
    """
    Cluster near-duplicate master_labels and assign canonical labels.
    Modifies chunks in place and returns them.
    """
    unique_labels = list({c.get("master_label", "") for c in chunks if c.get("master_label")})
    if len(unique_labels) < 2:
        log.info("[Label Normalizer] Fewer than 2 unique labels — nothing to normalize.")
        return chunks

    log.info(f"[Label Normalizer] Analyzing {len(unique_labels)} unique labels...")

    # Identify garbage labels
    garbage_set = {l for l in unique_labels if _is_garbage_label(l)}
    if garbage_set:
        log.info(f"[Label Normalizer] Flagged {len(garbage_set)} garbage labels")

    # Compute embeddings on normalized forms
    normalized_forms = [_normalize_for_clustering(l) for l in unique_labels]
    embedder = _get_embedder()
    embeddings = embedder.encode(normalized_forms, batch_size=64, normalize_embeddings=True, show_progress_bar=False)

    # Build similarity matrix and cluster
    sim_matrix = cosine_similarity(embeddings)
    threshold = config.LABEL_MERGE_THRESHOLD

    uf = _LabelUF(len(unique_labels))
    merge_count = 0

    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            if sim_matrix[i][j] >= threshold:
                uf.union(i, j)
                merge_count += 1

    # Build clusters
    clusters: Dict[int, List[int]] = defaultdict(list)
    for i in range(len(unique_labels)):
        clusters[uf.find(i)].append(i)

    # Build label mapping: old label -> canonical label
    label_map: Dict[str, str] = {}
    n_clusters_merged = 0

    for indices in clusters.values():
        cluster_labels = [unique_labels[i] for i in indices]
        canonical = _pick_canonical(cluster_labels, garbage_set)

        if len(cluster_labels) > 1:
            n_clusters_merged += 1
            log.info(
                f"  Merged cluster ({len(cluster_labels)} labels) → '{canonical}': "
                f"{cluster_labels}"
            )

        for l in cluster_labels:
            label_map[l] = canonical

    # Apply mapping to chunks
    updated = 0
    for c in chunks:
        old = c.get("master_label", "")
        if old in label_map and label_map[old] != old:
            c["master_label"] = label_map[old]
            updated += 1

    new_unique = len({c.get("master_label", "") for c in chunks if c.get("master_label")})
    log.info(
        f"[Label Normalizer] Done. "
        f"Merged {n_clusters_merged} clusters, updated {updated} chunks. "
        f"Unique labels: {len(unique_labels)} → {new_unique}"
    )

    return chunks
