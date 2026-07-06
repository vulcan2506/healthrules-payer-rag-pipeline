"""
labeler.py
──────────
Per-chunk master labeling via OpenRouter API.
"""

import re
import json
import logging
from typing import Dict, List, Optional, Tuple

import config
import llm_client

log = logging.getLogger(__name__)


def unload():
    pass


def _parse_label_output(raw: str) -> Tuple[Optional[str], Optional[str]]:
    raw = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()

    try:
        d = json.loads(raw)
        lbl = d.get("master_label", "").strip()
        if "words" in lbl.lower() and len(lbl) < 15:
            return None, None
        return lbl, d.get("description", "").strip()
    except json.JSONDecodeError:
        pass

    m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group())
            lbl = d.get("master_label", "").strip()
            if "words" in lbl.lower() and len(lbl) < 15:
                return None, None
            return lbl, d.get("description", "").strip()
        except json.JSONDecodeError:
            pass

    return None, None


# ── Prompts ───────────────────────────────────────────────────────────────────
# Static fallbacks are used when no domain profile exists.
# When a profile IS available, prompts are built dynamically with
# domain-specific role, terminology, and few-shot examples.

_LABEL_PROMPT_STATIC = """You are a healthcare software and payer systems specialist.
Given a technical passage and its keywords, assign a master label and write a concise description.

RULES:
1. "master_label": Must be a specific, descriptive noun phrase (e.g., "Claims Adjudication Engine", "WebLogic Migration", "Database Schema Updates"). NEVER output the literal instructions like "2-4 words".
2. "description": 1-2 complete sentences explaining what the passage implements or changes.
3. Output ONLY valid JSON format. No markdown, no conversational text.

--- FEW-SHOT EXAMPLES ---

EXAMPLE 1:
TEXT: "The HealthRules Connector API has been updated to support new data models for claims processing."
OUTPUT: {"master_label": "Connector API Claims Updates", "description": "Updates the HealthRules Connector API to support new data models for claims processing."}

EXAMPLE 2:
TEXT: "HealthEdge is replacing the proprietary Oracle T3 protocol with the standard HTTP protocol."
OUTPUT: {"master_label": "Oracle T3 to HTTP Migration", "description": "Replaces the proprietary Oracle T3 protocol with the standard HTTP protocol to enhance interoperability."}

EXAMPLE 3 (Tables):
TEXT: {"Feature": "Changes to OLTP database?", "Included": "Yes"}
OUTPUT: {"master_label": "OLTP Database Modifications", "description": "A feature matrix indicating that changes to the OLTP database are included in this release."}"""

_DESC_PROMPT_STATIC = """You are a healthcare software technical writer.
Write a 1-2 sentence description of what this technical passage specifically implements or changes.
Be concrete — mention the feature, system component, or impact. Do NOT be vague or generic."""


def _build_dynamic_label_base(profile: dict) -> str:
    """Build a label prompt base from a domain profile (CoT + domain few-shots)."""
    role = profile.get("specialist_role", "a technical documentation specialist")
    terminology = profile.get("key_terminology", {})
    few_shots = profile.get("labeling_few_shots", [])
    entity_types = profile.get("entity_types", [])
    domain = profile.get("domain", "technical documentation")

    term_block = ""
    if terminology:
        term_lines = [f"  {k} = {v}" for k, v in list(terminology.items())[:10]]
        term_block = "\nKEY TERMINOLOGY (use these to inform your labels):\n" + "\n".join(term_lines) + "\n"

    entity_hint = ""
    if entity_types:
        entity_hint = (
            f"\nDOMAIN ENTITY TYPES (labels should reference these where appropriate):\n"
            f"  {', '.join(entity_types)}\n"
        )

    examples_block = ""
    if few_shots:
        for i, ex in enumerate(few_shots, 1):
            snippet = ex.get("text_snippet", "").replace('"', '\\"')
            label = ex.get("master_label", "")
            desc = ex.get("description", "").replace('"', '\\"')
            examples_block += (
                f'\nEXAMPLE {i}:\n'
                f'TEXT: "{snippet}"\n'
                f'OUTPUT: {{"master_label": "{label}", "description": "{desc}"}}\n'
            )

    return (
        f"You are {role}.\n"
        f"The documents being processed are about: {domain}.\n"
        f"Given a technical passage and its keywords, assign a master label and write a concise description.\n\n"
        f"THINK STEP BY STEP:\n"
        f"1. Read the passage and keywords carefully.\n"
        f"2. Identify the primary subject — what specific feature, component, or change is described?\n"
        f"3. Formulate a master_label as a specific noun phrase (2-5 words) that names this subject.\n"
        f"4. Write a 1-2 sentence description of what the passage implements or changes.\n\n"
        f"RULES:\n"
        f'1. "master_label": Must be a specific, descriptive noun phrase. '
        f'NEVER output literal instructions like "2-4 words".\n'
        f'2. "description": 1-2 complete sentences explaining what the passage implements or changes.\n'
        f"3. Output ONLY valid JSON format. No markdown, no conversational text.\n"
        f"{term_block}"
        f"{entity_hint}\n"
        f"--- FEW-SHOT EXAMPLES ---\n"
        f"{examples_block}"
    )


def _build_dynamic_desc_base(profile: dict) -> str:
    """Build a description prompt base from a domain profile."""
    role = profile.get("specialist_role", "a technical documentation specialist")
    domain = profile.get("domain", "technical documentation")
    return (
        f"You are {role} and technical writer.\n"
        f"Context: These documents cover {domain}.\n"
        f"Write a 1-2 sentence description of what this technical passage specifically "
        f"implements or changes.\n"
        f"Be concrete — mention the feature, system component, or impact. "
        f"Do NOT be vague or generic."
    )


def _get_label_base(source_doc: str = "") -> str:
    """Return the appropriate label prompt base — saved > dynamic > static."""
    if source_doc:
        try:
            import context_profiler
            profile = context_profiler.get_profile(source_doc)
            if profile:
                type_key = profile.get("type_key", "")
                saved = context_profiler.load_prompt(type_key, "labeler")
                if saved:
                    return saved
                if profile.get("labeling_few_shots"):
                    prompt = _build_dynamic_label_base(profile)
                    context_profiler.save_prompt(type_key, "labeler", prompt)
                    return prompt
        except ImportError:
            pass
    return _LABEL_PROMPT_STATIC


def _get_desc_base(source_doc: str = "") -> str:
    """Return the appropriate description prompt base — saved > dynamic > static."""
    if source_doc:
        try:
            import context_profiler
            profile = context_profiler.get_profile(source_doc)
            if profile:
                type_key = profile.get("type_key", "")
                saved = context_profiler.load_prompt(type_key, "labeler_desc")
                if saved:
                    return saved
                if profile.get("domain"):
                    prompt = _build_dynamic_desc_base(profile)
                    context_profiler.save_prompt(type_key, "labeler_desc", prompt)
                    return prompt
        except ImportError:
            pass
    return _DESC_PROMPT_STATIC


def _build_label_prompt(text: str, keywords: List[str], source_doc: str = "") -> str:
    kw_str  = ", ".join(keywords) if keywords else "none"
    prompt_base = _get_label_base(source_doc)

    return (
        f"{prompt_base}\n\n"
        f"--- ACTUAL INPUT ---\n"
        f"KEYWORDS: {kw_str}\n"
        f"TEXT:\n{text[:800]}\n\n"
        f"Respond ONLY with valid JSON:\n"
    )


def _build_desc_prompt(text: str, master_label: str, source_doc: str = "") -> str:
    prompt_base = _get_desc_base(source_doc)

    return (
        f"{prompt_base}\n\n"
        f"Topic area: {master_label}\n"
        f"Passage:\n{text[:700]}\n\n"
        f"Reply with ONLY the description text, no JSON, no preamble, no markdown."
    )


# ── Orchestration ──────────────────────────────────────────────────────────────

def fill_missing_descriptions(chunks: List[Dict], batch_size: int = 8) -> List[Dict]:
    missing_indices = [
        i for i, c in enumerate(chunks)
        if not c.get("description")
        or len(str(c.get("description", "")).split()) < config.DESCRIPTION_MIN_LENGTH
    ]

    if not missing_indices:
        log.info("All descriptions present — no fill pass needed.")
        return chunks

    log.info(f"Filling {len(missing_indices)} missing/short descriptions (Batch size: {batch_size})...")

    prompts = [
        _build_desc_prompt(
            chunks[i]["text"],
            chunks[i].get("master_label", ""),
            chunks[i].get("source_doc", ""),
        )
        for i in missing_indices
    ]
    raw_outputs = llm_client.generate_local_batch(prompts, max_tokens=config.DESCRIPTION_MAX_TOKENS, batch_size=8, desc="Filling descriptions", stop=llm_client.STOP_TEXT)

    for i, raw in zip(missing_indices, raw_outputs):
        raw_clean = re.sub(r"```.*?```", "", raw, flags=re.DOTALL).strip()
        raw_clean = re.sub(r'^\{.*?"description"\s*:\s*"([^"]+)".*\}$', r'\1', raw_clean, flags=re.DOTALL)
        
        if len(raw_clean.split()) >= 5:
            chunks[i]["description"] = raw_clean
        else:
            sents = re.split(r'(?<=[.!?])\s+', chunks[i]["text"].strip())
            chunks[i]["description"] = " ".join(sents[:2])

    return chunks


def label_all_chunks(chunks: List[Dict], batch_size: int = 8) -> List[Dict]:
    log.info(f"[Pass 1] Labeling {len(chunks)} chunks...")

    prompts = [
        _build_label_prompt(c["text"], c.get("keywords", []), c.get("source_doc", ""))
        for c in chunks
    ]
    raw_outputs = llm_client.generate_local_batch(prompts, max_tokens=config.LABEL_MAX_TOKENS, batch_size=8, desc="Labeling", stop=["```\n"])

    failed_indices = []

    for i, (chunk, raw) in enumerate(zip(chunks, raw_outputs)):
        label, desc = _parse_label_output(raw)
        if label:
            chunk["master_label"] = label
            chunk["description"]  = desc or ""
        else:
            failed_indices.append(i)

    if failed_indices:
        log.warning(f"Retrying {len(failed_indices)} failed JSON extractions...")
        retry_prompts = [prompts[i] for i in failed_indices]
        retry_outputs = llm_client.generate_local_batch(retry_prompts, max_tokens=config.LABEL_MAX_TOKENS, batch_size=8, desc="Retrying labels", stop=["```\n"])

        for i, raw in zip(failed_indices, retry_outputs):
            label, desc = _parse_label_output(raw)
            if label:
                chunks[i]["master_label"] = label
                chunks[i]["description"]  = desc or ""
            else:
                kws = chunks[i].get("keywords", [])
                fallback = " ".join(kw.replace("_", " ") for kw in kws[:3]).title() or "System Update"
                chunks[i]["master_label"] = fallback
                chunks[i]["description"]  = ""

    fill_missing_descriptions(chunks, batch_size=batch_size)

    empty = sum(1 for c in chunks if not c.get("description"))
    log.info(f"Labeling complete. Empty descriptions remaining: {empty}")
    
    return chunks