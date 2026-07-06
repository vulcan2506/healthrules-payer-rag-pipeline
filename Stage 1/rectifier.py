"""
rectifier.py
────────────
Post-ingestion table rectification pass.
"""

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import config

log = logging.getLogger(__name__)


def unload():
    pass


# ── Helpers ────────────────────────────────────────────────────────────────────

def _is_table_chunk(chunk: Dict) -> bool:
    text  = chunk.get("text", "").strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return (
        bool(lines)
        and all(ln.startswith("{") and ln.endswith("}") for ln in lines[:3])
    )

def _parse_rows(text: str) -> List[Dict]:
    rows = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line: continue
        try: rows.append(json.loads(line))
        except json.JSONDecodeError: pass
    return rows

def _rows_to_text(rows: List[Dict]) -> str:
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)


def _strip_markup(rows: List[Dict]) -> List[Dict]:
    clean = []
    for row in rows:
        new_row = {}
        for k, v in row.items():
            k2 = re.sub(r"\*\*([^*]+)\*\*", r"\1", k)
            k2 = re.sub(r"__([^_]+)__",    r"\1", k2).strip()
            v2 = str(v or "")
            v2 = re.sub(r"\*\*([^*]+)\*\*", r"\1", v2)
            v2 = re.sub(r"__([^_]+)__",    r"\1", v2)
            v2 = re.sub(r'\b(See|see)(["\'A-Z])', r'\1 \2', v2)
            v2 = re.sub(r"[\n\r\t]+", " ", v2)
            v2 = re.sub(r"\s{2,}", " ", v2).strip()
            new_row[k2] = v2
        clean.append(new_row)
    return clean

def _drop_empty_cols(rows: List[Dict]) -> List[Dict]:
    if not rows: return rows
    all_cols = []
    for r in rows:
        for k in r.keys():
            if k not in all_cols: all_cols.append(k)
    valid_cols = set()
    for col in all_cols:
        if any(str(r.get(col, "")).strip() for r in rows): valid_cols.add(col)
    cleaned_rows = []
    for r in rows:
        new_row = {k: r[k] for k in all_cols if k in r and k in valid_cols}
        cleaned_rows.append(new_row)
    return cleaned_rows


_SECTION_KW_RE = re.compile(
    r"^(release summary|this release includes|new features|issues resolved|"
    r"changes to|mandatory migration|release checklist|release notes|"
    r"overview|highlights|summary|checklist|notes|remarks)",
    re.IGNORECASE,
)

def _is_section_marker_row(row: Dict) -> Tuple[bool, str]:
    values  = list(row.values())
    primary = str(values[0]).strip() if values else ""
    rest    = [str(v).strip() for v in values[1:]]

    if _SECTION_KW_RE.match(primary) and all(v == "" for v in rest): return True, primary
    if primary.lower().startswith("this release") and all(v == "" for v in rest): return True, primary
    return False, ""

def _split_at_markers(rows: List[Dict], base_header: str) -> List[Tuple[str, List[Dict]]]:
    groups:         List[Tuple[str, List[Dict]]] = []
    current_name:   str                          = base_header
    current_rows:   List[Dict]                   = []
    marker_buffer:  List[str]                    = []
    seen_marker                                  = False

    def _clean(text: str) -> str: return re.sub(r"\*\*|__", "", text).strip()

    def _flush_markers():
        if not marker_buffer: return
        section_name = _clean(marker_buffer[0])
        content      = " ".join(_clean(m) for m in marker_buffer[1:])
        synthetic    = {"Section": section_name, "Content": content}
        groups.append((section_name + " (Summary)", [synthetic]))

    for row in rows:
        is_marker, marker_text = _is_section_marker_row(row)
        if is_marker:
            seen_marker = True
            if current_rows:
                groups.append((current_name, current_rows))
                current_rows = []
                marker_buffer = []
            marker_buffer.append(marker_text)
        else:
            if marker_buffer:
                _flush_markers()
                current_name  = _clean(marker_buffer[-1])
                marker_buffer = []
            current_rows.append(row)

    if marker_buffer: _flush_markers()
    if current_rows: groups.append((current_name, current_rows))
    if not seen_marker: return [(base_header, rows)]
    return groups if groups else [(base_header, rows)]


def _heuristic_rename(col: str, sample_vals: List[str]) -> str:
    c    = col.lower().strip()
    vals = [str(v).strip().lower() for v in sample_vals if str(v).strip()]
    if re.search(r"_\d+$", col):
        yn = sum(1 for v in vals if v in ("yes", "no", "n/a", "true", "false"))
        if vals and yn / len(vals) > 0.6: return "Status"
        base = re.sub(r"_\d+$", "", col).strip()
        return base or col
    if vals and sum(1 for v in vals if "?" in v) / len(vals) > 0.5: return "Category"
    if vals and sum(1 for v in vals if v in ("yes", "no", "n/a")) / len(vals) > 0.7: return "Status"
    if vals and any("see " in v or "page " in v or "http" in v for v in vals):
        if "description" not in c and "summary" not in c: return "Reference"
    return col


def _apply_renames(rows: List[Dict], rename: Dict[str, str]) -> List[Dict]:
    if not rename: return rows
    out_rows = []
    for row in rows:
        new_row = {}
        for k, v in row.items():
            new_key = rename.get(k, k)
            if new_key in new_row:
                existing_val = str(new_row[new_key]).strip()
                new_val      = str(v).strip()
                if not existing_val and new_val: new_row[new_key] = new_val
                elif existing_val and new_val and existing_val != new_val:
                    new_row[new_key] = f"{existing_val} {new_val}"
            else:
                new_row[new_key] = v
        out_rows.append(new_row)
    return out_rows


def _heuristic_pass(rows: List[Dict]) -> Tuple[List[Dict], bool]:
    if not rows: return rows, False
    sample     = rows[:3]
    rename_map = {}
    needs_llm  = False

    all_cols = set()
    for r in rows: all_cols.update(r.keys())

    for col in all_cols:
        sample_vals = [str(r.get(col, "")) for r in sample]
        new_name    = _heuristic_rename(col, sample_vals)
        if new_name != col: rename_map[col] = new_name
        elif re.search(r"_\d+$|^col_\d+|^field_\d+", col, re.IGNORECASE): needs_llm = True

    return _apply_renames(rows, rename_map), needs_llm


# ── TOC extraction-artifact detection and conversion ───────────────────────────

def _is_toc_artifact(rows: List[Dict]) -> bool:
    """
    Detects the common PDF extraction artifact where a TOC/feature-list table's
    first row becomes the JSON keys: first key = long feature title, second key =
    a bare page number string.  All rows share the same keys by JSONL convention.
    """
    if len(rows) < 2:
        return False
    keys = list(rows[0].keys())
    if len(keys) != 2:
        return False
    feature_key, page_key = keys
    return bool(re.match(r"^\d+$", str(page_key)) and len(str(feature_key).split()) >= 4)


def _toc_rows_to_bullets(rows: List[Dict], start_n: int = 1) -> List[str]:
    """Convert parsed TOC-artifact rows to numbered bullet strings.  The key
    names form item 1 (the 'header' feature + its page); the row values fill
    the remaining items."""
    keys = list(rows[0].keys())
    feature_key, page_key = keys
    items: List[Tuple[str, str]] = [(str(feature_key), str(page_key))]
    for row in rows:
        name = str(row.get(feature_key, "")).strip()
        page = str(row.get(page_key, "")).strip()
        if name:
            items.append((name, page))
    return [f"{start_n + i}. {name} (p.{page})" for i, (name, page) in enumerate(items)]


def _base_header(section_header: str) -> str:
    """Strip trailing ' (Table)' qualifier so intro and table chunks compare equal."""
    return re.sub(r"\s*\(Table\)\s*$", "", section_header, flags=re.IGNORECASE).strip()


def _merge_toc_group(toc_chunks: List[Dict], intro_chunk: Optional[Dict]) -> Dict:
    """Merge one or more consecutive TOC-artifact table chunks (same section) into
    a single bullet-list text chunk.  Optionally prepend a section-intro chunk."""
    counter = 1
    all_lines: List[str] = []
    for chunk in toc_chunks:
        rows = _parse_rows(chunk.get("text", ""))
        rows = _strip_markup(rows)
        if _is_toc_artifact(rows):
            lines = _toc_rows_to_bullets(rows, start_n=counter)
            all_lines.extend(lines)
            counter += len(lines)
        else:
            # Unexpected: fall back to raw JSONL text
            all_lines.append(chunk.get("text", ""))

    merged_text = "\n".join(all_lines)

    base = toc_chunks[0]
    section = _base_header(base.get("section_header", "Table"))
    if intro_chunk:
        intro_text = intro_chunk.get("text", "").strip()
        merged_text = f"{intro_text}\n\n{merged_text}" if intro_text else merged_text

    return {
        **base,
        "section_header": f"{section} (Table)",
        "text": merged_text,
        "chunk_id": base["chunk_id"],
    }


def _rectify_table_chunk(chunk: Dict) -> Tuple[List[Dict], bool]:
    rows = _parse_rows(chunk.get("text", ""))
    if not rows: return [chunk], False
    base_header = chunk.get("section_header", "Table").replace(" (Table)", "").strip()

    rows = _strip_markup(rows)
    rows = _drop_empty_cols(rows)
    groups = _split_at_markers(rows, base_header)

    result_chunks: List[Dict] = []
    any_needs_llm = False

    for group_name, group_rows in groups:
        if not group_rows: continue
        group_rows, needs_llm = _heuristic_pass(group_rows)
        if needs_llm: any_needs_llm = True

        sub_header = group_name if group_name.endswith("(Summary)") or group_name.endswith("(Table)") else f"{group_name} (Table)"

        result_chunks.append({
            **chunk,
            "section_header": sub_header,
            "text":           _rows_to_text(group_rows),
            "_needs_rename":  needs_llm,
            "_rename_rows":   group_rows[:2] if needs_llm else [],
        })
    return result_chunks, any_needs_llm


def rectify(chunks: List[Dict]) -> List[Dict]:
    from tqdm import tqdm

    if not chunks:
        return chunks

    # ── Pass 1: identify and merge consecutive TOC-artifact table chunks ────────
    # Build a working list so we can collapse groups before the main loop.
    # Skip-set tracks indices that have been absorbed into a merge.
    skip: set = set()
    toc_replacements: Dict[int, Dict] = {}  # lead_idx → merged chunk

    i = 0
    while i < len(chunks):
        chunk = chunks[i]
        if not _is_table_chunk(chunk):
            i += 1
            continue

        rows = _parse_rows(chunk.get("text", ""))
        rows = _strip_markup(rows)
        if not _is_toc_artifact(rows):
            i += 1
            continue

        # Found a TOC artifact chunk.  Collect consecutive siblings with same
        # (source_doc, base section_header).
        base_hdr = _base_header(chunk.get("section_header", ""))
        src_doc  = chunk.get("source_doc", "")
        group    = [chunk]
        j        = i + 1
        while j < len(chunks):
            nxt = chunks[j]
            if not _is_table_chunk(nxt):
                break
            nxt_rows = _strip_markup(_parse_rows(nxt.get("text", "")))
            if (
                _is_toc_artifact(nxt_rows)
                and _base_header(nxt.get("section_header", "")) == base_hdr
                and nxt.get("source_doc", "") == src_doc
            ):
                group.append(nxt)
                j += 1
            else:
                break

        # Optionally absorb a preceding section-intro (same base header, same doc).
        intro_chunk: Optional[Dict] = None
        if i > 0:
            prev = chunks[i - 1]
            if (
                not _is_table_chunk(prev)
                and _base_header(prev.get("section_header", "")) == base_hdr
                and prev.get("source_doc", "") == src_doc
                and i - 1 not in skip
            ):
                intro_chunk = prev
                skip.add(i - 1)

        merged = _merge_toc_group(group, intro_chunk)
        toc_replacements[i] = merged
        # Mark absorbed chunks as skip; lead index (i) is NOT skipped — it's
        # handled via toc_replacements so the merged chunk appears in output.
        for k in range(i + 1, j):
            skip.add(k)

        log.debug(
            f"TOC merge: '{base_hdr}' — {len(group)} table chunk(s)"
            f"{' + intro' if intro_chunk else ''} → 1 bullet-list chunk"
        )
        i = j

    # ── Pass 2: standard table rectification for non-TOC table chunks ──────────
    standard_table_idxs = [
        idx for idx, c in enumerate(chunks)
        if _is_table_chunk(c) and idx not in skip and idx not in toc_replacements
    ]

    std_replacements: Dict[int, List[Dict]] = {}
    for orig_idx in tqdm(standard_table_idxs, desc="Rectifying tables"):
        sub_chunks, _ = _rectify_table_chunk(chunks[orig_idx])
        std_replacements[orig_idx] = sub_chunks

    # ── Pass 3: assemble final list ─────────────────────────────────────────────
    final: List[Dict] = []
    for idx, chunk in enumerate(chunks):
        if idx in skip:
            continue
        if idx in toc_replacements:
            final.append(toc_replacements[idx])
        elif idx in std_replacements:
            subs = std_replacements[idx]
            for sub_idx, sc in enumerate(subs):
                sc["chunk_id"] = (
                    f"{chunk['chunk_id']}_{sub_idx+1}" if len(subs) > 1 else chunk["chunk_id"]
                )
                sc.pop("_needs_rename", None)
                sc.pop("_rename_rows", None)
                final.append(sc)
        else:
            final.append(chunk)

    return final