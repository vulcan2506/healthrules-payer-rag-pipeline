"""
registry.py
───────────
Builds the topic registry DataFrame from labeled chunks.

One row per unique master_label.
Description is a concatenation of all chunk descriptions grouped by their source PDF.
Format: "[desc_1 | desc_2] | [desc_3]" (where bracket 1 = PDF 1, bracket 2 = PDF 2)
Chunk IDs: "[id1,id2] | [id3]"

This approach preserves quality, keeps descriptions clean and readable, and enables
easy chunk tracing and PDF boundary identification.
"""

import re
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict

import pandas as pd

import config
import topic_model as tm
import llm_client

log = logging.getLogger(__name__)


def _group_descriptions_by_pdf(group_chunks: List[Dict]) -> str:
    """
    Group descriptions by their source PDF.
    Format: "[desc_1 | desc_2] | [desc_3]"
    Where each bracket = all descriptions from one specific PDF.
    """
    by_doc = defaultdict(list)
    for c in group_chunks:
        desc = c.get("description", "").strip()
        if desc:
            by_doc[c["source_doc"]].append(desc)

    # Sort by source_doc for deterministic output
    grouped_parts = []
    for doc, descs in sorted(by_doc.items()):
        if descs:
            grouped_parts.append(f"[{' | '.join(descs)}]")
            
    return " | ".join(grouped_parts)


def _group_chunk_ids_by_pdf(group_chunks: List[Dict]) -> str:
    """
    Group chunk IDs by their source PDF.
    Format: "[id1,id2,id3]|[id4,id5]"
    Where each bracket = all chunks from one specific PDF.
    """
    by_doc = defaultdict(list)
    for c in group_chunks:
        by_doc[c["source_doc"]].append(str(c["chunk_id"]))

    # Sort by source_doc for deterministic output
    return " | ".join(
        f"[{','.join(ids)}]"
        for doc, ids in sorted(by_doc.items())
    )


# ── Summarized description via LLM ────────────────────────────────────────────

_SUMMARY_SYSTEM = (
    "You are a precise research summarizer. "
    "Given multiple descriptions of text chunks sharing the same topic, "
    "write a single 2-3 sentence summary that captures every key point. "
    "Be specific — do NOT be generic. "
    "Reply with ONLY the summary text. No preamble, no JSON."
)


def _summarize_descriptions(master_label: str, flat_raw_desc: str) -> str:
    """
    Use LLM to produce a concise 2-3 sentence summary of the flat pipe-separated
    raw descriptions. Reuses the labeler pipeline — no second model load.
    Falls back to first 300 chars of raw_desc on any failure.
    """
    if not flat_raw_desc or not flat_raw_desc.strip():
        return ""

    # Split pipe-separated descriptions and take first MERGE_MAX_DESCRIPTIONS
    parts = [p.strip() for p in flat_raw_desc.split("|") if p.strip()]
    parts = parts[:config.MERGE_MAX_DESCRIPTIONS]

    if len(parts) == 1:
        # Single description — truncate to 3 sentences instead of calling LLM
        sents = re.split(r"(?<=[.!?])\s+", parts[0])
        return " ".join(sents[:3])

    try:
        numbered = "\n".join(f"{i+1}. {p}" for i, p in enumerate(parts))
        user     = (f"Topic: {master_label}\n\n"
                    f"Descriptions:\n{numbered}\n\n"
                    "Write a 2-3 sentence summary capturing all key points.")

        result = llm_client.generate(user, max_tokens=config.SUMMARY_MAX_TOKENS, system_prompt=_SUMMARY_SYSTEM, enable_thinking=False)
        result = re.sub(r"```.*?```", "", result, flags=re.DOTALL).strip()
        if len(result.split()) >= 5:
            return result
    except Exception as e:
        log.warning(f"Summarization failed for '{master_label}': {e}")

    # Fallback — first 300 chars of raw
    return flat_raw_desc[:300].rsplit(" ", 1)[0] + "..."


# ── QnA aggregation ───────────────────────────────────────────────────────────

def _group_qna_by_pdf(group_chunks: List[Dict], max_pairs_per_doc: int = 6) -> str:
    """
    Group Q&A pairs by source PDF, deduplicate within each doc, cap per doc.

    Stored format (JSON dict keyed by source_doc):
      {"doc_A.pdf": [{"q": "...", "a": "..."}, ...],
       "doc_B.pdf": [{"q": "...", "a": "..."}, ...]}

    This mirrors how chunk_ids and descriptions are grouped by PDF,
    letting downstream consumers (retrieval layer, delta analysis) scope
    Q&A pairs to a specific document version.
    """
    by_doc: Dict[str, List[Dict]] = defaultdict(list)
    seen_per_doc: Dict[str, set] = defaultdict(set)

    for c in group_chunks:
        doc = c.get("source_doc", "unknown")
        qna = c.get("qna", [])
        if isinstance(qna, str):
            try:
                qna = json.loads(qna)
            except Exception:
                continue
        if not isinstance(qna, list):
            continue
        for pair in qna:
            q_norm = str(pair.get("q", "")).lower().strip()
            a      = str(pair.get("a", "")).strip()
            if not q_norm or not a:
                continue
            seen = seen_per_doc[doc]
            if len(by_doc[doc]) >= max_pairs_per_doc:
                continue
            # Skip near-duplicate questions within same doc
            if not any(q_norm in s or s in q_norm for s in seen):
                seen.add(q_norm)
                by_doc[doc].append({"q": pair["q"], "a": a})

    if not by_doc:
        return "{}"
    # Sort by source_doc for deterministic output
    return json.dumps(dict(sorted(by_doc.items())), ensure_ascii=False)


# ── Registry builder ───────────────────────────────────────────────────────────

def build_registry(chunks: List[Dict]) -> pd.DataFrame:
    """
    Group labeled chunks by master_label.
    For each group: collect descriptions grouped by PDF inside brackets.
    Description format: "[desc_A | desc_B] | [desc_C]"

    Columns:
      master_label  — LLM-assigned broad category
      description   — bracketed pipe-separated descriptions grouped by PDF
      summarized_description — LLM 2-3 sentence summary
      keywords      — top-10 pooled keywords across group
      source_docs   — unique PDFs contributing to this label
      chunk_ids     — bracketed comma-separated chunk IDs e.g. "[93,118] | [153]"
      n_chunks      — how many chunks in this group
      status        — pending / in_progress / done / skipped
      priority      — int, manually reorderable
      created_at
      completed_at
    """
    from tqdm import tqdm

    groups   = tm.group_by_master_label(chunks)
    rows     = []

    log.info(f"Merging descriptions for {len(groups)} master label groups (bracket format)...")

    sorted_groups = sorted(groups.items())

    # Pre-compute all row metadata
    meta = []
    llm_prompts   = []   # prompt or None if single-desc (no LLM needed)
    llm_fallbacks = []   # fallback text for single-desc rows

    for label, group_chunks in sorted_groups:
        chunk_ids   = [c["chunk_id"] for c in group_chunks]
        source_docs = sorted(set(c["source_doc"] for c in group_chunks))
        keywords    = tm.aggregate_keywords_for_group(group_chunks)
        grouped_desc      = _group_descriptions_by_pdf(group_chunks)
        grouped_chunk_ids = _group_chunk_ids_by_pdf(group_chunks)
        flat_descs    = [c.get("description", "").strip() for c in group_chunks if c.get("description", "").strip()]
        flat_desc_str = " | ".join(flat_descs)

        parts = [p.strip() for p in flat_desc_str.split("|") if p.strip()]
        parts = parts[:config.MERGE_MAX_DESCRIPTIONS]

        if not flat_desc_str or not flat_desc_str.strip() or len(parts) <= 1:
            # Single/empty — no LLM needed
            if len(parts) == 1:
                sents = re.split(r"(?<=[.!?])\s+", parts[0])
                fallback = " ".join(sents[:3])
            else:
                fallback = flat_desc_str[:300].rsplit(" ", 1)[0] + "..." if flat_desc_str else ""
            llm_prompts.append(None)
            llm_fallbacks.append(fallback)
        else:
            numbered = "\n".join(f"{i+1}. {p}" for i, p in enumerate(parts))
            prompt   = (f"Topic: {label}\n\nDescriptions:\n{numbered}\n\n"
                        "Write a 2-3 sentence summary capturing all key points.")
            llm_prompts.append(prompt)
            llm_fallbacks.append(flat_desc_str[:300].rsplit(" ", 1)[0] + "...")

        meta.append({
            "master_label":  label,
            "description":   grouped_desc,
            "keywords":      ", ".join(keywords),
            "source_docs":   " | ".join(source_docs),
            "chunk_ids":     grouped_chunk_ids,
            "n_chunks":      len(chunk_ids),
            "qna":           _group_qna_by_pdf(group_chunks),
        })

    # Batch all LLM summarizations in parallel
    real_prompts = [p for p in llm_prompts if p is not None]
    real_indices = [i for i, p in enumerate(llm_prompts) if p is not None]
    summarized   = [""] * len(llm_prompts)

    if real_prompts:
        log.info(f"Summarizing {len(real_prompts)} multi-description groups in parallel...")
        results = llm_client.generate_batch(
            real_prompts,
            max_tokens=config.SUMMARY_MAX_TOKENS,
            system_prompt=_SUMMARY_SYSTEM,
            desc="Summarizing registry",
            stop=llm_client.STOP_TEXT,
            enable_thinking=False,
        )
        for i, raw in zip(real_indices, results):
            cleaned = re.sub(r"```.*?```", "", raw, flags=re.DOTALL).strip()
            summarized[i] = cleaned if len(cleaned.split()) >= 5 else llm_fallbacks[i]

    # Fill single-desc fallbacks
    for i, prompt in enumerate(llm_prompts):
        if prompt is None:
            summarized[i] = llm_fallbacks[i]

    for m, s in tqdm(zip(meta, summarized), total=len(meta), desc="Building registry"):
        rows.append({
            "master_label":           m["master_label"],
            "description":            m["description"],
            "summarized_description": s,
            "keywords":               m["keywords"],
            "source_docs":            m["source_docs"],
            "chunk_ids":              m["chunk_ids"],
            "n_chunks":               m["n_chunks"],
            "qna":                    m["qna"],
            "status":                 "pending",
            "priority":               0,
            "created_at":             datetime.now().isoformat(timespec="minutes"),
            "completed_at":           None,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("n_chunks", ascending=False).reset_index(drop=True)
    df["priority"] = range(1, len(df) + 1)
    return df


# ── Persistence ────────────────────────────────────────────────────────────────

def save_registry(df: pd.DataFrame, path: Path = config.REGISTRY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info(f"Registry saved → {path}  ({len(df)} master labels)")


def load_registry(path: Path = config.REGISTRY_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Registry not found: {path}")
    return pd.read_csv(path)


# ── State helpers ──────────────────────────────────────────────────────────────

def mark_done(master_label: str, path: Path = config.REGISTRY_PATH) -> None:
    df = load_registry(path)
    mask = df["master_label"] == master_label
    df.loc[mask, "status"]       = "done"
    df.loc[mask, "completed_at"] = datetime.now().isoformat(timespec="minutes")
    save_registry(df, path)
    log.info(f"Marked done: {master_label}")


def mark_in_progress(master_label: str, path: Path = config.REGISTRY_PATH) -> None:
    df = load_registry(path)
    df.loc[df["master_label"] == master_label, "status"] = "in_progress"
    save_registry(df, path)


def next_pending(path: Path = config.REGISTRY_PATH) -> Optional[pd.Series]:
    df = load_registry(path)
    pending = df[df["status"] == "pending"].sort_values("priority")
    return pending.iloc[0] if not pending.empty else None


def queue_exhausted(path: Path = config.REGISTRY_PATH) -> bool:
    df = load_registry(path)
    return df[df["status"].isin(["pending", "in_progress"])].empty


def print_summary(path: Path = config.REGISTRY_PATH) -> None:
    df = load_registry(path)
    counts = df["status"].value_counts()
    print("\n── Topic Registry Summary ──────────────────────")
    print(f"  Master labels : {len(df)}")
    for s in ["pending", "in_progress", "done", "skipped"]:
        print(f"  {s:<14}: {counts.get(s, 0)}")
    print(f"  Saved at      : {config.REGISTRY_PATH}")
    print("────────────────────────────────────────────────\n")