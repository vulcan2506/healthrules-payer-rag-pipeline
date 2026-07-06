"""
re_rectifier.py
───────────────
LLM-based post-processing for tables via local llama.cpp server.
"""

import os
import re
import json
import logging
from typing import Dict, List

import config
import llm_client

log = logging.getLogger(__name__)


def unload():
    pass


# ── NO .format() TO AVOID KEY ERRORS ──────────────────────────────────────────

_FIX_PROMPT_BASE = """You are an expert data cleaner. Fix this JSONL table based on the Original Markdown Table.

CRITICAL RULES:
1. Output ONLY valid JSON Lines (JSONL). No markdown formatting blocks, no preamble.
2. DO NOT HALLUCINATE: If a cell is empty in the markdown, do NOT invent data or copy the category name to fill it.
3. RECOVER MISSING DATA: If the Markdown has values (e.g., 'Yes') that are missing from the Broken JSONL, add them back using a logical key (e.g., "Status": "Yes").
4. DROP IRRELEVANT KEYS: If a row is a simple checklist item (like "Changes to Server?"), DO NOT include keys from other tables (like "Change Description"). Just use keys that make sense for that row.
5. INFER MISSING HEADERS: If the Original Markdown table has blank headers (e.g. `| Date | | |`), you MUST invent logical header names based on the content of the columns (e.g., "Description", "Affected Section", "Reference"). Do NOT use empty strings ("") as keys.

--- EXAMPLES ---
EXAMPLE 1: Glued Checklists
ORIGINAL MARKDOWN:
| Date | Change Description |
|---|---|
| 2026-03-13 | Runtime properties enhanced |
| Release Summary | |
| New Features? | Yes |

BROKEN JSONL:
{"Date": "2026-03-13", "Change Description": "Runtime properties enhanced"}
{"Date": "New Features?", "Change Description": "New Features?", "Column_3": ""}

FIXED JSONL:
{"Date": "2026-03-13", "Change Description": "Runtime properties enhanced"}
{"Category": "New Features?", "Status": "Yes"}

EXAMPLE 2: Blank Headers
ORIGINAL MARKDOWN:
| Date | | |
|---|---|---|
| 08/28/2025 | Added Error Handling Details | See Integer Validation |

BROKEN JSONL:
{"Date": "08/28/2025", "": "Added Error Handling Details", "Column_3": "See Integer Validation"}

FIXED JSONL:
{"Date": "08/28/2025", "Description": "Added Error Handling Details", "Reference": "See Integer Validation"}"""


def _needs_fixing(jsonl_text: str) -> bool:
    """Return True only if JSONL has empty keys, Column_N keys, or duplicate keys."""
    for line in jsonl_text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
            for k in row:
                k_s = str(k).strip()
                if not k_s or re.match(r"^Column_\d+$", k_s, re.IGNORECASE):
                    return True
        except json.JSONDecodeError:
            return True
    return False


def _build_prompt(raw_markdown: str, jsonl_data: str) -> str:
    return (
        f"{_FIX_PROMPT_BASE}\n\n"
        f"--- ACTUAL TASK ---\n"
        f"ORIGINAL MARKDOWN TABLE:\n{raw_markdown}\n\n"
        f"BROKEN JSONL TABLE:\n{jsonl_data}\n\n"
        f"FIXED JSONL:\n"
    )


def _purge_empty_keys(rows: List[Dict]) -> List[Dict]:
    if not rows: return rows
        
    all_keys = set()
    for r in rows: all_keys.update(r.keys())
        
    valid_keys = set()
    for k in all_keys:
        k_clean = str(k).strip()
        if not k_clean or re.match(r"^Column_\d+$", k_clean, re.IGNORECASE):
            continue
        if any(str(r.get(k, "")).strip() for r in rows):
            valid_keys.add(k)
            
    cleaned_rows = []
    for r in rows:
        clean_row = {k: v for k, v in r.items() if k in valid_keys and str(v).strip() != ""}
        if clean_row: cleaned_rows.append(clean_row)
        
    return cleaned_rows


def run_re_rectifier(chunks: List[Dict], batch_size: int = 8) -> List[Dict]:
    target_indices = [i for i, c in enumerate(chunks) if c.get("md_file_path")]
    
    if not target_indices:
        log.info("[Re-Rectifier] No raw markdown files found. Skipping.")
        return chunks

    # Pre-filter: only send tables that actually have broken JSONL
    broken_indices = [i for i in target_indices if _needs_fixing(chunks[i].get("text", ""))]
    skipped = len(target_indices) - len(broken_indices)
    log.info(f"[Re-Rectifier] {len(broken_indices)}/{len(target_indices)} tables need fixing ({skipped} skipped — already clean)")

    if not broken_indices:
        return chunks

    prompts = []
    valid_targets = []

    for i in broken_indices:
        md_path = chunks[i]["md_file_path"]
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                raw_md = f.read()
            prompts.append(_build_prompt(raw_md, chunks[i]["text"]))
            valid_targets.append(i)
        except Exception as e:
            log.error(f"Error preparing prompt for {md_path}: {e}")

    llm_outputs = llm_client.generate_local_batch(
        prompts, max_tokens=600,
        desc="Re-Rectification", stop=llm_client.STOP_JSON,
        enable_thinking=False,
    )

    for idx, raw_out in zip(valid_targets, llm_outputs):
        clean = re.sub(r"```(?:jsonl|json)?|```", "", raw_out).strip()
        
        parsed_rows = []
        for line in clean.splitlines():
            line = line.strip().strip(",")
            if line.startswith("{") and line.endswith("}"):
                try: parsed_rows.append(json.loads(line))
                except json.JSONDecodeError: pass
                
        parsed_rows = _purge_empty_keys(parsed_rows)
        
        if parsed_rows:
            chunks[idx]["text"] = "\n".join(json.dumps(r, ensure_ascii=False) for r in parsed_rows)
            chunks[idx]["_re_rectified"] = True

        try:
            if os.path.exists(chunks[idx]["md_file_path"]):
                os.remove(chunks[idx]["md_file_path"])
        except Exception as e:
            log.warning(f"Failed to delete {chunks[idx]['md_file_path']}: {e}")
            
        del chunks[idx]["md_file_path"]

    return chunks