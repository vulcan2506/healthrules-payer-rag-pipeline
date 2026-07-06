"""
build_index.py
──────────────
One-time index builder. Run after the Stage 1 pipeline AND nested.py complete.

Reads:
  - enterprise_nested_topics.json  (3-level taxonomy)
  - filtered_chunks.json           (chunk texts)

Writes to index/:
  - parent_meta.pkl    list of {name, desc, emb, sub_indices}
  - sub_meta.pkl       list of {name, desc, emb, parent_idx}
  - hnsw.bin           HNSW index over topic + QnA embeddings
  - hnsw_meta.pkl      sidecar: hnsw_id → {parent_idx, sub_idx, chunk_ids,
                                            type, label, qna_q}
  - chunk_lookup.pkl   str(chunk_id) → {text, section_header, source_doc}
"""

import json
import pickle
import re
import logging
from pathlib import Path
from typing import List

import numpy as np
import hnswlib
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import config

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_chunk_ids(raw) -> list:
    """Parse nested-JSON chunk_ids into a flat list of strings.

    Handles both bracket-format strings like '[93,118]' and plain integers.
    """
    ids = []
    for item in (raw if isinstance(raw, list) else [raw]):
        ids.extend(str(x) for x in re.findall(r'[\w]+', str(item)) if x.isdigit() or '_' in x)
    return ids


def parse_description_brackets(raw: str) -> List[List[str]]:
    """
    Parse the registry's bracketed description string into one list of
    descriptions per source-doc bracket:
      "[desc1 | desc2] | [desc3]"  ->  [["desc1", "desc2"], ["desc3"]]

    Bracket order matches source_docs/chunk_ids (registry.py builds all
    three from the same sorted-by-doc grouping), and within a bracket, the
    Nth description corresponds to the Nth chunk_id group — same positional
    convention as the QnA-to-chunk-group mapping.
    """
    if not raw or not raw.strip():
        return []
    brackets = raw.split("] | [")
    brackets[0] = brackets[0].lstrip("[")
    brackets[-1] = brackets[-1].rstrip("]")
    return [
        [d.strip() for d in bracket.split(" | ") if d.strip()]
        for bracket in brackets
    ]


def embed_batch(model, texts: list, desc: str) -> np.ndarray:
    log.info(f"Embedding {len(texts)} items: {desc}")
    return model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)


# ── Chunk lookup ───────────────────────────────────────────────────────────────

def build_chunk_lookup(filtered_chunks_path: Path) -> dict:
    log.info("Building chunk lookup from filtered_chunks.json...")
    with open(filtered_chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    lookup = {}
    for c in chunks:
        cid = str(c.get("chunk_id", ""))
        lookup[cid] = {
            "text":           c.get("text", ""),
            "section_header": c.get("section_header", ""),
            "source_doc":     c.get("source_doc", ""),
        }
    log.info(f"Chunk lookup: {len(lookup)} entries")
    return lookup


# ── Main ───────────────────────────────────────────────────────────────────────

def build():
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load taxonomy ──────────────────────────────────────────────────────────
    log.info(f"Loading {config.NESTED_JSON}")
    with open(config.NESTED_JSON, encoding="utf-8") as f:
        taxonomy = json.load(f)["taxonomy"]

    # ── Load embedding model ───────────────────────────────────────────────────
    log.info(f"Loading embedder: {config.EMBED_MODEL}")
    model = SentenceTransformer(config.EMBED_MODEL, device="cpu")

    # ── Build parent + sub embeddings ──────────────────────────────────────────
    parent_texts, parent_recs = [], []
    sub_texts,    sub_recs    = [], []

    for p_idx, parent in enumerate(taxonomy):
        p_name = parent["parent_category_name"]
        p_desc = parent.get("parent_category_description", "")
        parent_texts.append(f"{p_name}. {p_desc}")
        parent_recs.append({"name": p_name, "desc": p_desc, "sub_indices": []})

        for s_idx, sub in enumerate(parent.get("sub_categories", [])):
            s_name = sub["sub_category_name"]
            s_desc = sub.get("sub_category_description", "")
            sub_texts.append(f"{s_name}. {s_desc}")
            sub_recs.append({"name": s_name, "desc": s_desc, "parent_idx": p_idx})
            parent_recs[p_idx]["sub_indices"].append(len(sub_recs) - 1)

    parent_embs = embed_batch(model, parent_texts, "parents")
    sub_embs    = embed_batch(model, sub_texts,    "sub-categories")

    for i, rec in enumerate(parent_recs):
        rec["emb"] = parent_embs[i]
    for i, rec in enumerate(sub_recs):
        rec["emb"] = sub_embs[i]

    # ── Build chunk lookup (needed below for chunk-level vectors too) ─────────
    chunk_lookup = build_chunk_lookup(config.FILTERED_CHUNKS)

    # ── Build HNSW over topic descriptions + QnA questions + chunk text ───────
    # Chunk-level vectors give narrow/specific queries something to match at
    # their own granularity — topic summaries are the right resolution for
    # broad "overview" queries, but dilute specific terminology (e.g. a topic
    # covering 5 chunks won't necessarily surface "coordination of benefits"
    # in its summary even if one of its chunks is directly about it). QnA
    # partially compensates for this today, but only for the specific facts
    # someone wrote a question about — chunk vectors cover everything.
    hnsw_texts, hnsw_meta = [], []

    sub_cursor = 0
    for p_idx, parent in enumerate(taxonomy):
        for s_idx_local, sub in enumerate(parent.get("sub_categories", [])):
            s_idx_global = sub_cursor
            sub_cursor  += 1

            for topic in sub.get("topics", []):
                label    = topic.get("master_label", "")
                keywords = topic.get("keywords", "")
                chunk_ids = parse_chunk_ids(topic.get("chunk_ids", []))

                # Topic vector(s). Prefer `grounded_summary` (topic_summarizer.py
                # output) when present — it's chunk-text-grounded and scoped per
                # source-doc bracket, so a v25.2-only summary can't surface
                # v26.1 chunks and vice versa, same scoping already used below
                # for description/QnA vectors. `summarized_description` is a
                # blend-of-short-descriptions that never reads chunk text, so it
                # only remains a fallback for topics not yet processed.
                grounded_brackets = parse_description_brackets(topic.get("grounded_summary", ""))
                chunk_id_brackets = topic.get("chunk_ids", [])

                if grounded_brackets:
                    for bracket_idx, texts in enumerate(grounded_brackets):
                        if bracket_idx >= len(chunk_id_brackets):
                            break
                        bracket_chunk_ids = parse_chunk_ids([chunk_id_brackets[bracket_idx]])
                        bracket_text = " ".join(texts).strip()
                        if bracket_text:
                            hnsw_texts.append(f"{label}. {bracket_text} {keywords}")
                            hnsw_meta.append({
                                "parent_idx": p_idx,
                                "sub_idx":    s_idx_global,
                                "chunk_ids":  bracket_chunk_ids,
                                "type":       "topic",
                                "label":      label,
                                "qna_q":      None,
                            })
                else:
                    summary = topic.get("summarized_description", "")
                    hnsw_texts.append(f"{label}. {summary} {keywords}")
                    hnsw_meta.append({
                        "parent_idx": p_idx,
                        "sub_idx":    s_idx_global,
                        "chunk_ids":  chunk_ids,
                        "type":       "topic",
                        "label":      label,
                        "qna_q":      None,
                    })

                # Map each source doc to its OWN chunk_ids group. A topic can
                # span multiple chunk groups (e.g. a v25.2 group + a v26.1
                # group for cross-version topics) — both source_docs and
                # chunk_ids are built in the same doc-sorted order by
                # registry.py, so zipping them gives an exact correspondence.
                topic_source_docs = topic.get("source_docs", [])
                topic_chunk_groups = topic.get("chunk_ids", [])
                doc_to_chunk_ids = {
                    doc: parse_chunk_ids([group])
                    for doc, group in zip(topic_source_docs, topic_chunk_groups)
                }

                # One vector per QnA question — scoped to the specific chunk
                # group it was generated from, not the topic's full (possibly
                # cross-version) chunk set. Without this, a QnA hit about a
                # v26.1-only feature would also pull in unrelated v25.2
                # chunks just because they shared the same topic.
                qna_by_doc = topic.get("qna", {})
                if not isinstance(qna_by_doc, dict):
                    qna_by_doc = {}
                for doc, pairs in qna_by_doc.items():
                    doc_chunk_ids = doc_to_chunk_ids.get(doc, chunk_ids)
                    for pair in pairs:
                        q = pair.get("q", "").strip()
                        if q:
                            hnsw_texts.append(q)
                            hnsw_meta.append({
                                "parent_idx": p_idx,
                                "sub_idx":    s_idx_global,
                                "chunk_ids":  doc_chunk_ids,
                                "type":       "qna",
                                "label":      label,
                                "qna_q":      q,
                            })

                # One vector per granular per-chunk description — sits between
                # the topic-level summary (broad, blended across every chunk
                # in the topic, including unrelated ones when a topic is
                # over-merged) and raw chunk text (noisy, truncated). Each
                # description is already a short, clean, single-chunk
                # distillation; the Nth description in a source-doc bracket
                # maps to the Nth chunk_id group in that same bracket.
                desc_brackets = parse_description_brackets(topic.get("description", ""))
                chunk_id_brackets = topic.get("chunk_ids", [])
                for bracket_idx, descs in enumerate(desc_brackets):
                    if bracket_idx >= len(chunk_id_brackets):
                        break
                    bracket_chunk_groups = parse_chunk_ids([chunk_id_brackets[bracket_idx]])
                    for desc, cid_group in zip(descs, bracket_chunk_groups):
                        if desc:
                            hnsw_texts.append(desc)
                            hnsw_meta.append({
                                "parent_idx": p_idx,
                                "sub_idx":    s_idx_global,
                                "chunk_ids":  [cid_group],
                                "type":       "description",
                                "label":      label,
                                "qna_q":      None,
                            })

                # One vector per chunk's own raw text — points to ONLY that
                # chunk (not the topic's full chunk_ids set), so a chunk hit
                # is a precise match, complementing the topic vector's breadth.
                for cid in chunk_ids:
                    chunk_text = chunk_lookup.get(cid, {}).get("text", "").strip()
                    if chunk_text:
                        hnsw_texts.append(chunk_text[:1000])
                        hnsw_meta.append({
                            "parent_idx": p_idx,
                            "sub_idx":    s_idx_global,
                            "chunk_ids":  [cid],
                            "type":       "chunk",
                            "label":      label,
                            "qna_q":      None,
                        })

    log.info(f"HNSW corpus: {len(hnsw_texts)} vectors "
             f"({sum(1 for m in hnsw_meta if m['type']=='topic')} topic, "
             f"{sum(1 for m in hnsw_meta if m['type']=='qna')} QnA, "
             f"{sum(1 for m in hnsw_meta if m['type']=='description')} description, "
             f"{sum(1 for m in hnsw_meta if m['type']=='chunk')} chunk)")

    hnsw_embs = embed_batch(model, hnsw_texts, "HNSW corpus")

    # ── Build HNSW index ───────────────────────────────────────────────────────
    log.info("Building HNSW index...")
    index = hnswlib.Index(space="cosine", dim=config.EMBED_DIM)
    index.init_index(
        max_elements=max(len(hnsw_embs), config.HNSW_MAX_ELEMENTS),
        M=config.HNSW_M,
        ef_construction=config.HNSW_EF_CONSTRUCTION,
    )
    index.add_items(hnsw_embs, list(range(len(hnsw_embs))))
    index.set_ef(config.HNSW_EF_QUERY)

    hnsw_path = config.INDEX_DIR / "hnsw.bin"
    index.save_index(str(hnsw_path))
    log.info(f"HNSW saved → {hnsw_path}")

    # ── Persist everything ─────────────────────────────────────────────────────
    def save(obj, name):
        path = config.INDEX_DIR / name
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        log.info(f"Saved → {path}")

    save(parent_recs,  "parent_meta.pkl")
    save(sub_recs,     "sub_meta.pkl")
    save(hnsw_meta,    "hnsw_meta.pkl")
    save(chunk_lookup, "chunk_lookup.pkl")

    log.info("Index build complete.")
    log.info(f"  Parents      : {len(parent_recs)}")
    log.info(f"  Sub-categories: {len(sub_recs)}")
    log.info(f"  HNSW vectors : {len(hnsw_embs)}")
    log.info(f"  Chunk lookup : {len(chunk_lookup)}")


if __name__ == "__main__":
    build()
