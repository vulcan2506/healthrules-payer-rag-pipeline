"""
context_profiler.py
───────────────────
Builds domain-aware profiles from document content to dynamically adapt
all downstream LLM prompts.  Runs once per unique document type.

Pipeline position:  After enrichment, before quality filter.

Architecture:
  1. Group chunks by source_doc → derive a type_key per document family
     (e.g. both '25.2_HR_Payer_Release_Notes.pdf' and
      '26.1_HR_Payer_Release_Notes.pdf' share key 'HR_Payer_Release_Notes')
  2. For each unique type_key, collect:
       - Captured TOC text (stored by ingest.py before discard)
       - Section header distribution
       - Top keywords by frequency
       - 3 high-quality, diverse sample chunks
  3. Single LLM call produces a domain profile:
       domain, document_purpose, specialist_role, analyst_role,
       entity_types, key_terminology, labeling_few_shots
  4. Profile is cached to disk — subsequent runs skip the LLM call
  5. Downstream modules call get_profile(source_doc) to fetch it

Bridge profiles (cross-document connections):
  When multiple type_keys exist in the same pipeline run, a lightweight
  bridge context is generated capturing how the document families relate.
"""

import json
import re
import logging
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional

import config
import llm_client

log = logging.getLogger(__name__)

PROFILE_CACHE_DIR = config.OUTPUT_DIR / "profiles"
PROMPT_OUTPUT_DIR = config.PROMPT_OUTPUT_DIR

_profiles: Dict[str, dict] = {}
_toc_store: Dict[str, str] = {}
_bridge_context: Optional[dict] = None


# ══════════════════════════════════════════════════════════════════════════════
# TOC CAPTURE  (called by ingest.py)
# ══════════════════════════════════════════════════════════════════════════════

def store_toc(source_doc: str, toc_text: str):
    """Called by ingest.py to preserve TOC text before it is discarded."""
    if source_doc in _toc_store:
        _toc_store[source_doc] += "\n\n" + toc_text
    else:
        _toc_store[source_doc] = toc_text


# ══════════════════════════════════════════════════════════════════════════════
# TYPE KEY DERIVATION
# ══════════════════════════════════════════════════════════════════════════════

def _derive_type_key(filename: str) -> str:
    """
    Strip version prefix to get document type.
    '25.2_HR_Payer_Release_Notes.pdf'  →  'HR_Payer_Release_Notes'
    '26.1_HR_Payer_Release_Notes.pdf'  →  'HR_Payer_Release_Notes'
    'Clinical_Trial_Report_v3.pdf'     →  'Clinical_Trial_Report_v3'
    """
    stem = Path(filename).stem
    stripped = re.sub(r'^\d+[\.\d]*[_\s-]+', '', stem)
    return stripped if stripped else stem




# ══════════════════════════════════════════════════════════════════════════════
# COLLECTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _collect_section_headers(chunks: List[Dict]) -> List[str]:
    seen = set()
    headers = []
    for c in chunks:
        h = c.get("section_header", "").strip()
        if h and h not in seen:
            seen.add(h)
            headers.append(h)
    return headers


def _collect_top_keywords(chunks: List[Dict], top_n: int = 30) -> List[str]:
    counter = Counter()
    for c in chunks:
        for kw in c.get("keywords", []):
            counter[kw.lower().strip()] += 1
    return [kw for kw, _ in counter.most_common(top_n)]


def _select_sample_chunks(chunks: List[Dict], n: int = 3) -> List[Dict]:
    scored = [c for c in chunks if c.get("text_quality_score", 0) > 0]
    scored.sort(key=lambda c: c.get("text_quality_score", 0), reverse=True)

    selected = []
    seen_headers = set()
    for c in scored:
        h = c.get("section_header", "")
        if h not in seen_headers:
            selected.append(c)
            seen_headers.add(h)
            if len(selected) >= n:
                break

    for c in scored:
        if c not in selected:
            selected.append(c)
            if len(selected) >= n:
                break

    return selected[:n]


# ══════════════════════════════════════════════════════════════════════════════
# PROFILE PROMPT
# ══════════════════════════════════════════════════════════════════════════════

_PROFILE_PROMPT = """\
You are a document analysis expert.  Analyze the document metadata and \
content samples below, then produce a domain profile JSON that will be \
used to adapt LLM prompts for processing this type of document.

DOCUMENT TYPE KEY: {type_key}
SOURCE FILES: {filenames}

{toc_section}

SECTION HEADERS ({n_headers} unique):
{section_headers}

TOP KEYWORDS (by frequency):
{top_keywords}

SAMPLE CONTENT:
{samples}

Output ONLY a JSON object — no markdown, no preamble.  Close all braces.

{{
  "domain": "<specific domain, e.g. 'healthcare claims administration'>",
  "document_purpose": "<what these documents are, e.g. 'software release notes'>",
  "specialist_role": "<prompt persona, e.g. 'a healthcare payer systems specialist'>",
  "analyst_role": "<analysis persona, e.g. 'a healthcare IT analyst'>",
  "entity_types": ["<5-8 key entity types found in the content>"],
  "key_terminology": {{
    "<ACRONYM>": "<definition>",
    "<ACRONYM>": "<definition>"
  }},
  "labeling_few_shots": [
    {{
      "text_snippet": "<1-2 sentence excerpt from the samples above>",
      "master_label": "<specific 2-5 word noun phrase>",
      "description": "<1-2 sentence description>"
    }},
    {{
      "text_snippet": "<different excerpt — prefer table/structured data if present>",
      "master_label": "<specific 2-5 word noun phrase>",
      "description": "<1-2 sentence description>"
    }},
    {{
      "text_snippet": "<different excerpt — different content type from above>",
      "master_label": "<specific 2-5 word noun phrase>",
      "description": "<1-2 sentence description>"
    }}
  ]
}}

RULES:
- Base the profile on the ACTUAL content provided — do not invent terms.
- specialist_role and analyst_role must be specific to the detected domain.
- labeling_few_shots MUST use real text from the samples, not invented text.
- key_terminology: 5-10 most important acronyms/abbreviations found.
- entity_types: the recurring nouns/concepts that chunk labels should reference.
- Output ONLY JSON.  No text before or after.  Close all braces."""


def _build_profile_prompt(
    type_key: str,
    filenames: List[str],
    toc_text: str,
    section_headers: List[str],
    top_keywords: List[str],
    sample_chunks: List[Dict],
) -> str:
    toc_section = (
        f"TABLE OF CONTENTS (captured from document):\n{toc_text[:2000]}"
        if toc_text.strip()
        else "TABLE OF CONTENTS: Not available — using section headers instead."
    )

    samples_text = ""
    for i, c in enumerate(sample_chunks, 1):
        header = c.get("section_header", "Unknown")
        text = c.get("text", "")[:500]
        kws = ", ".join(c.get("keywords", [])[:5])
        samples_text += (
            f"--- Sample {i} (Section: {header}) ---\n"
            f"{text}\n"
            f"Keywords: {kws}\n\n"
        )

    return _PROFILE_PROMPT.format(
        type_key=type_key,
        filenames=", ".join(filenames),
        toc_section=toc_section,
        n_headers=len(section_headers),
        section_headers="\n".join(f"  - {h}" for h in section_headers[:40]),
        top_keywords=", ".join(top_keywords),
        samples=samples_text,
    )


# ══════════════════════════════════════════════════════════════════════════════
# BRIDGE PROMPT  (cross-document-type connections)
# ══════════════════════════════════════════════════════════════════════════════

_BRIDGE_PROMPT = """\
You are a document analysis expert.  Two or more document families are \
being processed together.  Describe how they relate so that cross-document \
analysis can maintain context.

DOCUMENT FAMILIES:
{family_descriptions}

Output ONLY a JSON object:
{{
  "relationship": "<1-2 sentences describing how these document families connect>",
  "shared_entities": ["<entities/concepts that appear across families>"],
  "cross_reference_notes": "<guidance for analysts comparing content across families>"
}}

Output ONLY JSON.  Close all braces."""


def _build_bridge(profiles: Dict[str, dict]) -> Optional[dict]:
    if len(profiles) < 2:
        return None

    descs = ""
    for key, p in profiles.items():
        descs += (
            f"- {key}: domain={p.get('domain','?')}, "
            f"purpose={p.get('document_purpose','?')}, "
            f"entities={p.get('entity_types',[])} \n"
        )

    prompt = _BRIDGE_PROMPT.format(family_descriptions=descs)

    try:
        raw = llm_client.generate(prompt, max_tokens=500, enable_thinking=False)
        bridge = _extract_json(raw)
        if bridge and "relationship" in bridge:
            return bridge
    except Exception as e:
        log.warning(f"Bridge generation failed: {e}")

    return None


# ══════════════════════════════════════════════════════════════════════════════
# JSON EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def _extract_json(raw: str) -> Optional[dict]:
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find('{')
    if start == -1:
        return None

    stack: list = []
    in_str = escape = False
    for i, ch in enumerate(cleaned[start:], start):
        if escape:
            escape = False
            continue
        if ch == '\\' and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in '{[':
            stack.append(ch)
        elif ch in '}]':
            if stack:
                stack.pop()
                if not stack:
                    try:
                        return json.loads(cleaned[start:i + 1])
                    except json.JSONDecodeError:
                        break

    # Repair truncated JSON
    if stack:
        closings = ''.join(']' if b == '[' else '}' for b in reversed(stack))
        base = cleaned[start:].rstrip()
        candidates = [
            base,
            re.sub(r',?\s*"[^"]*$', '', base),
            re.sub(r',\s*$', '', base),
        ]
        for candidate in candidates:
            try:
                return json.loads(candidate + closings)
            except json.JSONDecodeError:
                pass

    return None


# ══════════════════════════════════════════════════════════════════════════════
# DEFAULT / FALLBACK PROFILE
# ══════════════════════════════════════════════════════════════════════════════

_DEFAULT_PROFILE = {
    "domain": "general technical documentation",
    "document_purpose": "technical documentation",
    "specialist_role": "a technical documentation specialist",
    "analyst_role": "a technical analyst",
    "entity_types": ["documents", "features", "components", "configurations"],
    "key_terminology": {},
    "labeling_few_shots": [],
}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def build_profiles(chunks: List[Dict]) -> Dict[str, dict]:
    """
    Build domain profiles from pre-quality-filter chunks.

    Chunks are grouped by source_doc → type_key.  Documents sharing a
    type_key (e.g. versioned release notes) get a single shared profile.
    Profiles are cached to disk so subsequent runs are instant.

    Returns {type_key: profile_dict}.
    """
    global _bridge_context
    PROFILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Group chunks by type key ──────────────────────────────────────────────
    type_groups: Dict[str, Dict] = {}
    for c in chunks:
        doc = c.get("source_doc", "unknown")
        key = _derive_type_key(doc)
        if key not in type_groups:
            type_groups[key] = {"filenames": set(), "chunks": []}
        type_groups[key]["filenames"].add(doc)
        type_groups[key]["chunks"].append(c)

    log.info(
        f"[Context Profiler] {len(type_groups)} document type(s) detected: "
        f"{list(type_groups.keys())}"
    )

    # ── Build / load profile per type key ─────────────────────────────────────
    for type_key, group in type_groups.items():
        cache_path = PROFILE_CACHE_DIR / f"{type_key}.json"

        if cache_path.exists():
            log.info(f"[Context Profiler] Cache hit for '{type_key}'")
            with open(cache_path, "r", encoding="utf-8") as f:
                _profiles[type_key] = json.load(f)
            continue

        filenames = sorted(group["filenames"])
        doc_chunks = group["chunks"]

        log.info(
            f"[Context Profiler] Profiling '{type_key}' "
            f"({len(doc_chunks)} chunks from {len(filenames)} file(s))..."
        )

        # Collect signals
        toc_parts = [
            f"[From {fn}]\n{_toc_store[fn]}"
            for fn in filenames if fn in _toc_store
        ]
        toc_text = "\n\n".join(toc_parts)
        section_headers = _collect_section_headers(doc_chunks)
        top_keywords = _collect_top_keywords(doc_chunks)
        samples = _select_sample_chunks(doc_chunks)

        prompt = _build_profile_prompt(
            type_key, filenames, toc_text,
            section_headers, top_keywords, samples,
        )

        try:
            raw = llm_client.generate(prompt, max_tokens=750, enable_thinking=False)
            profile = _extract_json(raw)

            if not profile or "domain" not in profile:
                log.warning(
                    f"[Context Profiler] Parse failed for '{type_key}', "
                    f"using defaults.  Raw output:\n{raw[:300]}"
                )
                profile = dict(_DEFAULT_PROFILE)
                profile["_parse_failed"] = True
        except Exception as e:
            log.error(f"[Context Profiler] LLM call failed for '{type_key}': {e}")
            profile = dict(_DEFAULT_PROFILE)
            profile["_error"] = str(e)

        for key, default in _DEFAULT_PROFILE.items():
            if key not in profile:
                profile[key] = default

        profile["type_key"] = type_key
        profile["source_files"] = filenames

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)
        log.info(f"[Context Profiler] Profile cached → {cache_path}")

        _profiles[type_key] = profile

    # ── Bridge context (multi-type runs) ──────────────────────────────────────
    if len(_profiles) >= 2:
        bridge_path = PROFILE_CACHE_DIR / "_bridge.json"
        if bridge_path.exists():
            with open(bridge_path, "r", encoding="utf-8") as f:
                _bridge_context = json.load(f)
            log.info("[Context Profiler] Loaded bridge context from cache")
        else:
            log.info("[Context Profiler] Building cross-document bridge context...")
            _bridge_context = _build_bridge(_profiles)
            if _bridge_context:
                with open(bridge_path, "w", encoding="utf-8") as f:
                    json.dump(_bridge_context, f, indent=2, ensure_ascii=False)
                log.info(f"[Context Profiler] Bridge cached → {bridge_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    for key, p in _profiles.items():
        log.info(
            f"  ├─ {key}: domain='{p.get('domain','')}', "
            f"purpose='{p.get('document_purpose','')}', "
            f"{len(p.get('labeling_few_shots',[]))} few-shots, "
            f"{len(p.get('key_terminology',{}))} terms"
        )

    return dict(_profiles)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ACCESSORS  (used by downstream modules)
# ══════════════════════════════════════════════════════════════════════════════

def get_profile(source_doc: str) -> Optional[dict]:
    """Look up the cached profile for a given source document filename."""
    key = _derive_type_key(source_doc)
    return _profiles.get(key)


def get_all_profiles() -> Dict[str, dict]:
    return dict(_profiles)


def get_bridge() -> Optional[dict]:
    return _bridge_context


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT PERSISTENCE  (save / load dynamic prompts as .md files)
# ══════════════════════════════════════════════════════════════════════════════

def save_prompt(type_key: str, name: str, content: str):
    """
    Save a generated prompt template as a readable .md file.

    Files land in:  data/output/prompts/<type_key>/<name>.md
    Modules call this once when they first generate a dynamic prompt.
    On subsequent runs the saved file is loaded instead of regenerating,
    and users can hand-edit the .md to tweak prompt style.
    """
    prompt_dir = PROMPT_OUTPUT_DIR / type_key
    prompt_dir.mkdir(parents=True, exist_ok=True)
    path = prompt_dir / f"{name}.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    log.info(f"[Prompt Saved] {path}")


def load_prompt(type_key: str, name: str) -> Optional[str]:
    """
    Load a previously saved prompt template.
    Returns None if no file exists — caller should fall back to generation.
    """
    path = PROMPT_OUTPUT_DIR / type_key / f"{name}.md"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return None


# ── Prompt fragment builders (convenience for downstream modules) ─────────

def get_role(source_doc: str, kind: str = "specialist") -> str:
    """
    Returns the domain-appropriate role string.
    kind: 'specialist' | 'analyst'
    Falls back to a generic role if no profile exists.
    """
    p = get_profile(source_doc)
    if p:
        field = "specialist_role" if kind == "specialist" else "analyst_role"
        return p.get(field, _DEFAULT_PROFILE[field])
    return _DEFAULT_PROFILE.get(
        "specialist_role" if kind == "specialist" else "analyst_role",
        "a technical specialist",
    )


def get_terminology_block(source_doc: str) -> str:
    """Returns a formatted terminology block for injection into prompts."""
    p = get_profile(source_doc)
    if not p:
        return ""
    terms = p.get("key_terminology", {})
    if not terms:
        return ""
    lines = [f"  {k} = {v}" for k, v in list(terms.items())[:10]]
    return "KEY TERMINOLOGY:\n" + "\n".join(lines) + "\n"


def get_few_shot_block(source_doc: str) -> str:
    """Returns formatted few-shot examples for labeling prompts."""
    p = get_profile(source_doc)
    if not p:
        return ""
    shots = p.get("labeling_few_shots", [])
    if not shots:
        return ""
    parts = []
    for i, ex in enumerate(shots, 1):
        snippet = ex.get("text_snippet", "")
        label = ex.get("master_label", "")
        desc = ex.get("description", "")
        parts.append(
            f'EXAMPLE {i}:\n'
            f'TEXT: "{snippet}"\n'
            f'OUTPUT: {{"master_label": "{label}", '
            f'"description": "{desc}"}}'
        )
    return "\n\n".join(parts)
