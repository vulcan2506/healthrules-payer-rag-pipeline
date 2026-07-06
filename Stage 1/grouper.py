"""
grouper.py
──────────
Two-stage Semantic Grouper for release-note style documents.

GROUPING INTENT
───────────────
Each logical unit in the source document is a "feature block":

    ## Feature Title
    Intro paragraph.
    ## Key Updates          ← sub-section, belongs to the feature above
    - bullet ...
    | Table |               ← table, belongs to the feature above
    ## NOTE                 ← sub-section, belongs to the feature above
    - note ...
    [PAYOR-XXXXX]           ← closing ticket ref, marks end of this block

Everything from the feature title to the closing PAYOR ref should be ONE
merged chunk.  A new group starts only when a new top-level feature title
appears.

STAGE 1 – Rule-Based Pre-Grouper (zero LLM calls)
    Handles patterns that are structurally obvious from header text alone:
      Rule 1  (Table)  suffix  → merge with preceding if base headers match
      Rule 2  (Cont.)  suffix  → merge with preceding unconditionally
      Rule 3  Transitional intro → merge with following (short bridge chunks)
      Rule 4  Sub-section names → merge with preceding
                 e.g. "Key Updates", "Overview", "NOTE", "Web UI Changes",
                      "DW Changes", "Prerequisites", "Benefits", "Usage" …

STAGE 2 – LLM Semantic Grouper (GPU-batched)
    Receives pre-merged chunks; decides whether adjacent chunks belong to the
    same feature block.  Each chunk's ID is remapped to a simple sequential
    integer before the prompt so that small LLMs reliably copy IDs back.

STAGE 2.5 – Cohesion Validator (embedding-based, zero LLM calls)
    After overlap deduplication stitches groups from sliding windows, checks
    each multi-chunk group for internal coherence.  Embeds constituent texts
    and splits where consecutive cosine similarity drops below threshold
    (GROUP_COHESION_THRESHOLD).  Catches cross-window stitching artifacts
    where unrelated chunks get chained through shared overlap IDs.

LLM parse cascade (stops at first success):
    1. json.loads on raw extracted 2D array
    2. json.loads after sanitizing invalid backslash escapes
    3. Structure scan  – sublist boundary scan for known IDs
    4. Header mapping  – resolve header text back to simple IDs
    5. Fallback        – no merging
"""

import re
import json
import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import config
import llm_client

log = logging.getLogger(__name__)

# ── QnA generation constants ──────────────────────────────────────────────────

_QNA_SYSTEM = (
    "You are a technical analyst who reads release notes with genuine curiosity. "
    "You want to understand WHY changes were made, HOW mechanisms actually work, "
    "and WHAT the real-world implications are — not just catalog what changed."
)

_QNA_PROMPT = """\
Read this release note text about "{topic}" and generate {n} Q&A pairs \
that explore WHY this change happened, HOW the mechanism works, or \
WHAT the implications are for real-world use.

Rules:
- Ask "why", "how", or "what does this imply" — avoid pure lookup questions.
- Answers: 2-3 sentences with brief reasoning before the conclusion.
- If the text is too short for {n} questions, generate fewer — no padding.
- Output ONLY a JSON array — nothing before or after it. Close all brackets.

Examples of the depth we want:
[
  {{"q": "Why would the system preserve only the last four digits of a card number rather than masking it completely?", \
"a": "Showing the last four digits balances security with usability — users need enough context to recognize which card is on file without exposing the full PAN. Masking everything forces a lookup that creates friction in payment disputes or verification flows."}},
  {{"q": "How does enabling WORKBASKET_RULE_OPTIMIZATION_ENABLE actually change the claim assignment flow?", \
"a": "The property likely activates a pre-filtering step that narrows the candidate pool before running full rule evaluation, reducing the number of rules the engine executes per claim. This matters most under high claim volumes where rule evaluation latency compounds across thousands of concurrent assignments."}}
]

TOPIC: {topic}
TEXT:
{text}

OUTPUT (JSON array only, max {n} pairs, close all brackets):"""


def _parse_qna(raw: str) -> List[Dict]:
    """Extract [{q, a}] list from LLM output with brace-scan and regex fallback."""
    # Strip <think>...</think> blocks — present when enable_thinking=True
    cleaned = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL)
    cleaned = re.sub(r"```(?:json)?|```", "", cleaned).strip()

    start = cleaned.find("[")
    if start != -1:
        candidate = cleaned[start:]
        depth = 0
        for j, ch in enumerate(candidate):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(candidate[: j + 1])
                        if isinstance(data, list) and data:
                            # Format 1: [{"q": ..., "a": ...}, ...]
                            if isinstance(data[0], dict):
                                return [
                                    {
                                        "q": str(item.get("q", "")).strip(),
                                        "a": str(item.get("a", "")).strip(),
                                    }
                                    for item in data
                                    if isinstance(item, dict)
                                    and item.get("q")
                                    and item.get("a")
                                ]
                            # Format 2: flat [q1, a1, q2, a2, ...] string array
                            if isinstance(data[0], str):
                                pairs = []
                                for k in range(0, len(data) - 1, 2):
                                    q = str(data[k]).strip()
                                    a = str(data[k + 1]).strip()
                                    if q and a:
                                        pairs.append({"q": q, "a": a})
                                return pairs
                    except Exception:
                        break

    # Regex fallback
    pairs = []
    for m in re.finditer(
        r'"q"\s*:\s*"([^"]+)"\s*,\s*"a"\s*:\s*"([^"]+)"', cleaned, re.DOTALL
    ):
        pairs.append({"q": m.group(1).strip(), "a": m.group(2).strip()})
    return pairs


def generate_group_qna(chunks: List[Dict], n_pairs: int = 3) -> List[Dict]:
    """
    Generate Q&A pairs for each grouped chunk and store as chunk["qna"].
    Skips chunks with fewer than 20 words. Batches all LLM calls in parallel.
    """
    prompts: List[str] = []
    indices: List[int] = []

    for i, c in enumerate(chunks):
        text = c.get("text", "")
        word_count = len(text.split())
        if word_count < 20:
            c.setdefault("qna", [])
            continue

        topic = c.get("master_label") or c.get("section_header") or "this feature"
        n = min(n_pairs, max(1, word_count // 40))
        prompts.append(_QNA_PROMPT.format(topic=topic, text=text[:2000], n=n))
        indices.append(i)

    if not prompts:
        return chunks

    log.info(f"QnA Generation: {len(prompts)} chunks to process ({len(chunks) - len(prompts)} skipped — too short)")
    raw_outputs = llm_client.generate_batch(
        prompts,
        max_tokens=500,
        system_prompt=_QNA_SYSTEM,
        desc="QnA Generation",
        stop=["```\n", "```"],
        enable_thinking=False,
    )

    for i, raw in zip(indices, raw_outputs):
        chunks[i]["qna"] = _parse_qna(raw)
        log.debug(f"Chunk {chunks[i].get('chunk_id')}: {len(chunks[i]['qna'])} Q&A pairs")

    parsed_counts = [len(chunks[i]["qna"]) for i in indices]
    log.info(
        f"QnA Generation complete — "
        f"total pairs: {sum(parsed_counts)}, "
        f"avg per chunk: {sum(parsed_counts)/len(parsed_counts):.1f}"
    )
    return chunks

_embedder: Optional[SentenceTransformer] = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")
    return _embedder


def unload():
    global _embedder
    if _embedder is not None:
        del _embedder
        _embedder = None
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── STAGE 1: RULE-BASED PRE-GROUPER ───────────────────────────────────────────

# Generic sub-section headers that are NEVER standalone features.
# These always belong to the feature block introduced by the preceding chunk.
_SUBSECTION_HEADERS: frozenset = frozenset({
    # structure / process sub-sections
    "key updates", "overview", "prerequisites", "details",
    "enhancement details", "how it works", "previous behavior",
    "new enhancement", "benefits", "usage", "example", "examples",
    "enabling this feature", "parameter descriptions",
    # document sections
    "note", "important note", "important",
    # UI / data sub-sections
    "web ui changes", "web ui - changes", "dw changes", "dw updates",
    "data warehouse updates", "two new tables",
})


def _normalize_header(header: str) -> str:
    """Strip leading bullets / list markers and collapse whitespace."""
    h = re.sub(r'^[\s\-\u2013\u2022\tn]+', '', header).strip()
    return re.sub(r'\s+', ' ', h)


def _base_header(header: str) -> str:
    """
    Canonical base: remove leading markers, trailing '(Table)',
    trailing '(Cont.)'/'(Cont)', and trailing colons.
    """
    h = _normalize_header(header)
    h = re.sub(r'\s*\(Table\)\s*$', '', h, flags=re.IGNORECASE).strip()
    h = re.sub(r'\s*\(Cont\.?\)\s*$', '', h, flags=re.IGNORECASE).strip()
    return h.rstrip(':').strip()


def _is_table_chunk(header: str) -> bool:
    return bool(re.search(r'\(Table\)\s*$', header, re.IGNORECASE))


def _is_continuation_chunk(header: str) -> bool:
    return bool(re.search(r'\(Cont\.?\)\s*$', header, re.IGNORECASE))


def _is_subsection_chunk(header: str) -> bool:
    """
    True when the header is a known generic sub-section name.
    Normalises the header before lookup so 'Key Updates:' and
    'key updates' both resolve to the same entry.
    """
    key = _normalize_header(header).lower().rstrip(':').strip()
    return key in _SUBSECTION_HEADERS


def _ends_with_payor_ref(chunk: Dict) -> bool:
    """True when chunk text ends with a [PAYOR-XXXXX] ticket reference."""
    text = chunk.get('text', '').rstrip()
    return bool(re.search(r'\[PAYOR-\d+[^\]]*\]\s*$', text))


def _is_transitional_chunk(chunk: Dict) -> bool:
    """
    True for short bridge chunks whose only purpose is to introduce the
    content that follows.
    Conditions (both must hold):
      - Text shorter than 200 characters
      - Text ends with a lead-in phrase
    """
    text = chunk.get('text', '').strip()
    if len(text) >= 200:
        return False
    return bool(re.search(
        r'(as follows|the following|are described below|'
        r'listed below|see below|illustrated below)\s*:?\s*$',
        text, re.IGNORECASE,
    ))


def rule_based_pre_grouper(
    chunks: List[Dict],
) -> Tuple[List[List[int]], int]:
    """
    Deterministically group chunks (for one document) using header patterns.

    Returns:
      grouped_indices – list of groups; each is a sorted list of positional
                        indices into `chunks`.
      n_merges        – number of merge operations performed.

    Rules (priority order for each consecutive pair i-1, i):
      0. PAYOR boundary – if chunk i-1 ends with [PAYOR-XXXXX], never merge.
      1. (Table)  – merge chunk i into chunk i-1 when base headers match.
      2. (Cont.)  – merge chunk i into the preceding group unconditionally.
      3. Transitional intro – merge chunk i-1 into chunk i when i-1 is a
                              short lead-in bridge.
      4. Sub-section name   – merge chunk i into the preceding group when
                              chunk i's header is a known generic sub-section.

    Guards:
      - Max group size cap (config.MAX_RULE_GROUP_SIZE) prevents runaway chains.
    """
    max_size = getattr(config, 'MAX_RULE_GROUP_SIZE', 8)
    n = len(chunks)
    parent = list(range(n))
    group_size = [1] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return True
        if group_size[ra] + group_size[rb] > max_size:
            return False
        group_size[ra] += group_size[rb]
        parent[rb] = ra
        return True

    n_merges = 0

    for i in range(1, n):
        h_cur  = chunks[i].get('section_header', '')
        h_prev = chunks[i - 1].get('section_header', '')

        # Rule 0: PAYOR-ref boundary — previous chunk closes a feature block
        if _ends_with_payor_ref(chunks[i - 1]):
            continue

        # Rule 1: (Table) suffix
        if _is_table_chunk(h_cur):
            if _base_header(h_cur) == _base_header(h_prev):
                if union(i - 1, i):
                    n_merges += 1
                continue

        # Rule 2: (Cont.) suffix
        if _is_continuation_chunk(h_cur):
            if union(i - 1, i):
                n_merges += 1
            continue

        # Rule 3: transitional intro chunk precedes current
        if _is_transitional_chunk(chunks[i - 1]):
            if union(i - 1, i):
                n_merges += 1
            continue

        # Rule 4: known generic sub-section header
        if _is_subsection_chunk(h_cur):
            if union(i - 1, i):
                n_merges += 1
            continue

    root_to_group: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        root_to_group[find(i)].append(i)

    grouped = sorted(root_to_group.values(), key=lambda g: g[0])
    return grouped, n_merges


# ── PROMPT ENGINEERING ────────────────────────────────────────────────────────

_GROUPER_PROMPT_STATIC = """\
You are processing chunks extracted from a software release notes document.
Your task: decide which adjacent chunks belong to the SAME feature block and
should be merged.

## WHAT IS A FEATURE BLOCK?
A feature block starts with a descriptive feature title and continues through
all of its sub-sections until the NEXT feature title begins.

SAME feature block (MERGE these):
  - A feature title chunk followed by its sub-sections such as
    "Key Updates", "Overview", "Web UI Changes", "DW Changes",
    "Prerequisites", "NOTE", "Benefits", "How It Works", "Examples", etc.
  - A sub-section followed by another sub-section of the SAME parent feature.
  - A description chunk followed by its data table.
  - A short intro sentence followed by detailed content on the same topic.

DIFFERENT feature blocks (DO NOT MERGE):
  - Two chunks that each introduce a distinct, independently named feature.
  - A chunk whose text ends with a [PAYOR-XXXXX] ticket reference followed
    by a chunk that starts a new named feature — the PAYOR ref marks the
    END of the previous feature block.
  - Chunks on completely unrelated topics.

## OUTPUT FORMAT
Think step by step (THOUGHT PROCESS), then output ONLY the JSON block.
The JSON must be a 2D array of the integer IDs shown in the input — nothing
else inside the array.

### FEW-SHOT EXAMPLE ###
INPUT CHUNKS:
[ID: 1] Header: "Ability to Hold Claim Payments" | Text: "This feature allows the system to hold claim payments when the member SL limit is exceeded..."
[ID: 2] Header: "Key Updates" | Text: "- n Hold Payments on Exceeding Member SL Limit..."
[ID: 3] Header: "NOTE" | Text: "- n This update ensures compliance... [PAYOR-165137]"
[ID: 4] Header: "Ability to View Reinsurer Details" | Text: "This feature allows ASO accounts to track reinsurer amounts separately..."
[ID: 5] Header: "Web UI Changes" | Text: "- n Claim view page: A new tab called Reinsurance Details has been added..."

THOUGHT PROCESS:
- 1 is a feature title. 2 is "Key Updates" — a sub-section of feature 1. Merge 1+2.
- 3 is "NOTE" and ends with a PAYOR ticket ref → last piece of the same feature. Merge 1+2+3.
- 4 is a NEW feature title (completely different topic). Starts a new group.
- 5 is "Web UI Changes" — a sub-section of feature 4. Merge 4+5.

JSON OUTPUT:
```json
[[1, 2, 3], [4, 5]]
```

--- ACTUAL TASK ---"""


def _build_dynamic_grouper_prompt(profile: dict) -> str:
    """Build the full grouper prompt base from a domain profile."""
    domain = profile.get("domain", "technical documentation")
    doc_purpose = profile.get("document_purpose", "technical document")
    entity_types = profile.get("entity_types", [])

    entity_hint = ""
    if entity_types:
        entity_hint = (
            f"\n## DOMAIN CONTEXT\n"
            f"These documents cover: {domain}\n"
            f"Key entity types: {', '.join(entity_types)}\n"
            f"A 'feature block' in this domain is a cohesive unit about one of these entities.\n"
        )

    e1 = entity_types[0] if entity_types else "Feature A"
    e2 = entity_types[1] if len(entity_types) > 1 else "Feature B"
    e1_title = e1.replace("_", " ").title()
    e2_title = e2.replace("_", " ").title()

    few_shot = (
        f"### FEW-SHOT EXAMPLE ###\n"
        f"INPUT CHUNKS:\n"
        f'[ID: 1] Header: "{e1_title} Configuration" '
        f'| Text: "This enhancement introduces new configuration options for {e1}..."\n'
        f'[ID: 2] Header: "Key Updates" '
        f'| Text: "The following updates have been made to the {e1} system..."\n'
        f'[ID: 3] Header: "NOTE" '
        f'| Text: "This update ensures compliance with new requirements..."\n'
        f'[ID: 4] Header: "{e2_title} Management" '
        f'| Text: "This feature enables tracking and management of {e2}..."\n'
        f'[ID: 5] Header: "Details" '
        f'| Text: "The {e2} details are now displayed in a new view..."\n\n'
        f"THOUGHT PROCESS:\n"
        f"- 1 is a feature title about {e1}. "
        f'2 is "Key Updates" — a sub-section of feature 1. Merge 1+2.\n'
        f'- 3 is "NOTE" — still part of the same feature block. Merge 1+2+3.\n'
        f"- 4 is a NEW feature title about {e2} (different topic). Starts a new group.\n"
        f'- 5 is "Details" — a sub-section of feature 4. Merge 4+5.\n\n'
        f"JSON OUTPUT:\n"
        f"```json\n"
        f"[[1, 2, 3], [4, 5]]\n"
        f"```\n"
    )

    return (
        f"You are processing chunks extracted from a {doc_purpose} document.\n"
        f"Your task: decide which adjacent chunks belong to the SAME feature block and\n"
        f"should be merged.\n"
        f"{entity_hint}\n"
        f"## WHAT IS A FEATURE BLOCK?\n"
        f"A feature block starts with a descriptive feature title and continues through\n"
        f"all of its sub-sections until the NEXT feature title begins.\n\n"
        f"SAME feature block (MERGE these):\n"
        f"  - A feature title chunk followed by its sub-sections such as\n"
        f'    "Key Updates", "Overview", "Details", "Prerequisites",\n'
        f'    "NOTE", "Benefits", "How It Works", "Examples", etc.\n'
        f"  - A sub-section followed by another sub-section of the SAME parent feature.\n"
        f"  - A description chunk followed by its data table.\n"
        f"  - A short intro sentence followed by detailed content on the same topic.\n\n"
        f"DIFFERENT feature blocks (DO NOT MERGE):\n"
        f"  - Two chunks that each introduce a distinct, independently named feature.\n"
        f"  - A chunk that clearly closes one topic followed by a chunk starting a new one.\n"
        f"  - Chunks on completely unrelated topics.\n\n"
        f"## OUTPUT FORMAT\n"
        f"Think step by step (THOUGHT PROCESS), then output ONLY the JSON block.\n"
        f"The JSON must be a 2D array of the integer IDs shown in the input — nothing\n"
        f"else inside the array.\n\n"
        f"{few_shot}\n"
        f"--- ACTUAL TASK ---"
    )


def _get_grouper_prompt_base(source_doc: str = "") -> str:
    """Return the grouper prompt base — saved > dynamic > static."""
    if not source_doc:
        return _GROUPER_PROMPT_STATIC
    try:
        import context_profiler
        profile = context_profiler.get_profile(source_doc)
        if not profile or not profile.get("domain"):
            return _GROUPER_PROMPT_STATIC
        type_key = profile.get("type_key", "")
        saved = context_profiler.load_prompt(type_key, "grouper")
        if saved:
            return saved
        prompt = _build_dynamic_grouper_prompt(profile)
        context_profiler.save_prompt(type_key, "grouper", prompt)
        return prompt
    except ImportError:
        return _GROUPER_PROMPT_STATIC


def _build_prompt(chunk_list_text: str, source_doc: str = "") -> str:
    prompt_base = _get_grouper_prompt_base(source_doc)
    return (
        f"{prompt_base}\n\n"
        f"INPUT CHUNKS:\n{chunk_list_text}\n\n"
        "THOUGHT PROCESS:"
    )


# ── STAGE 2: ID REMAPPING ─────────────────────────────────────────────────────

def _prepare_sequence(
    seq: List[Dict],
) -> Tuple[str, List[str], Dict[str, str], Dict[str, Dict]]:
    """
    Replace each chunk's original ID with a simple sequential integer
    (1, 2, 3 ...) before building the LLM prompt.

    Small LLMs reliably copy small integers; compound strings like "48_49"
    or "79_80_81" cause frequent hallucination.

    Returns:
      chunk_list_text    – formatted prompt input block
      simple_ids         – ordered list of simple string IDs ("1", "2", ...)
      simple_to_orig     – simple_id → original chunk_id string
      simple_id_to_chunk – simple_id → chunk dict (for header fallback)
    """
    simple_ids: List[str] = []
    simple_to_orig: Dict[str, str] = {}
    simple_id_to_chunk: Dict[str, Dict] = {}
    lines: List[str] = []

    for idx, c in enumerate(seq):
        sid = str(idx + 1)
        simple_ids.append(sid)
        simple_to_orig[sid] = str(c["chunk_id"])
        simple_id_to_chunk[sid] = c

        preview = str(c.get("text", ""))[:300].replace('\n', ' ')
        lines.append(
            f'[ID: {sid}] Header: "{c.get("section_header", "None")}" '
            f'| Text: "{preview}..."'
        )

    return '\n'.join(lines), simple_ids, simple_to_orig, simple_id_to_chunk


# ── STAGE 2: LLM OUTPUT PARSING ───────────────────────────────────────────────

def _sanitize_for_json(text: str) -> str:
    """Fix invalid backslash escapes from LLM output."""
    return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)


def _recover_groups_by_structure(
    array_str: str, original_ids_str: List[str]
) -> Optional[List[List[str]]]:
    """
    Fallback when json.loads fails entirely.
    Splits on ], [ boundaries and scans each region for known IDs via
    word-boundary regex.
    """
    inner = array_str.strip().lstrip('[').rstrip(']')
    sublist_texts = re.split(r'\]\s*,\s*\[', inner)

    groups: List[List[str]] = []
    for subtext in sublist_texts:
        found: List[str] = []
        for id_ in original_ids_str:
            pattern = r'(?<!\w)' + re.escape(id_) + r'(?!\w)'
            if re.search(pattern, subtext) and id_ not in found:
                found.append(id_)
        if found:
            groups.append(found)

    return groups if groups else None


def _recover_groups_by_headers(
    parsed: List[List],
    original_ids_str: List[str],
    id_to_chunk: Dict[str, Dict],
) -> Optional[List[List[str]]]:
    """
    Fallback when the LLM outputs header text instead of integer IDs.
    `id_to_chunk` maps simple IDs → chunk dicts.
    Indexes both raw and normalised header forms for better coverage.
    """
    header_to_id: Dict[str, str] = {}
    for sid, chunk in id_to_chunk.items():
        raw_hdr = str(chunk.get("section_header", "")).strip()
        if raw_hdr:
            header_to_id[raw_hdr] = sid
        norm_hdr = _normalize_header(raw_hdr)
        if norm_hdr and norm_hdr != raw_hdr:
            header_to_id[norm_hdr] = sid

    recovered: List[List[str]] = []
    for group in parsed:
        resolved: List[str] = []
        for item in group:
            key = str(item).strip()
            key_norm = _normalize_header(key)
            cid = (
                header_to_id.get(key)
                or header_to_id.get(key_norm)
                or next(
                    (v for h, v in header_to_id.items()
                     if key in h or h in key or key_norm in h),
                    None,
                )
            )
            if cid and cid not in resolved:
                resolved.append(cid)
        if resolved:
            recovered.append(resolved)

    if not recovered:
        return None

    all_found = {cid for g in recovered for cid in g}
    return recovered if all_found.issubset(set(original_ids_str)) else None


def _parse_grouping_output(
    raw_output: str,
    original_ids: List[str],
    id_to_chunk: Optional[Dict[str, Dict]] = None,
) -> List[List[str]]:
    """
    Extract a valid grouping from LLM output.
    `original_ids` and `id_to_chunk` keys are SIMPLE IDs ("1", "2", ...).
    Translation back to original chunk IDs is done by the caller.

    Parse cascade:
      1. json.loads on raw extracted 2D array
      2. json.loads after backslash sanitization
      3. Structure scan
      4. Header mapping
      5. Fallback (no merging)
    """
    original_ids_str = [str(i) for i in original_ids]
    fallback = [[i] for i in original_ids_str]

    # 1. Extract the JSON block
    match = re.search(
        r"```(?:json)?\s*(\[\s*\[.*?\]\s*\])\s*```",
        raw_output, re.DOTALL | re.IGNORECASE,
    )
    if not match:
        match = re.search(r"(\[\s*\[.*?\]\s*\])", raw_output, re.DOTALL)

    if not match:
        log.warning("Grouper LLM: no 2D array found in output. Falling back.")
        log.debug(f"Grouper LLM raw output was: {raw_output[:300]!r}")
        return fallback

    array_str = match.group(1)

    # 2 & 3. Progressive JSON parsing
    parsed = None
    last_err: Optional[Exception] = None

    for candidate in [array_str, _sanitize_for_json(array_str)]:
        try:
            parsed = json.loads(candidate)
            break
        except json.JSONDecodeError as e:
            last_err = e

    # 4. Structure scan
    if parsed is None:
        log.debug(f"Grouper LLM: json.loads failed ({last_err}); structure scan.")
        parsed = _recover_groups_by_structure(array_str, original_ids_str)
        if parsed is None:
            snippet = raw_output.replace('\n', ' ')[:120]
            log.warning(
                f"Grouper LLM: all parse attempts failed. "
                f"Last error: {last_err}. Snippet: {snippet}..."
            )
            return fallback

    # 5. Normalize
    parsed = [
        [str(item) for item in sublist]
        for sublist in parsed
        if isinstance(sublist, list)
    ]
    parsed_flat = [item for sublist in parsed for item in sublist]

    # 6. Validate; header mapping fallback
    if not set(parsed_flat).issubset(set(original_ids_str)):
        log.debug("Grouper LLM: output had text/headers; attempting header map.")
        recovered = (
            _recover_groups_by_headers(parsed, original_ids_str, id_to_chunk)
            if id_to_chunk else None
        )
        if recovered:
            parsed = recovered
            parsed_flat = [item for sublist in parsed for item in sublist]
        else:
            log.warning(
                "Grouper LLM: hallucinated invalid IDs and header mapping "
                "failed. Falling back to unmerged."
            )
            return fallback

    # 7. Restore any IDs the LLM dropped
    missing = [i for i in original_ids_str if i not in parsed_flat]
    for m in missing:
        parsed.append([m])

    parsed.sort(key=lambda x: original_ids_str.index(x[0]))
    return parsed


# ── STAGE 2.5: COHESION VALIDATOR ─────────────────────────────────────────────

def _cohesion_split(
    group_ids: List[str],
    id_to_chunk: Dict[str, Dict],
) -> List[List[str]]:
    """
    Split a merged group where embedding similarity between consecutive
    chunks drops below GROUP_COHESION_THRESHOLD.

    Uses the chunk text (header + body) to compute embeddings, then scans
    for cosine-similarity valleys between consecutive constituents.
    Groups of size 1 pass through unchanged.
    """
    if len(group_ids) <= 1:
        return [group_ids]

    threshold = getattr(config, 'GROUP_COHESION_THRESHOLD', 0.40)
    embedder = _get_embedder()

    texts = []
    valid_ids = []
    for gid in group_ids:
        chunk = id_to_chunk.get(gid)
        if not chunk:
            continue
        header = chunk.get("section_header", "")
        body = chunk.get("text", "")
        texts.append(f"{header}. {body}"[:500])
        valid_ids.append(gid)

    if len(valid_ids) <= 1:
        return [valid_ids] if valid_ids else []

    embeddings = embedder.encode(texts, convert_to_numpy=True)
    sims = [
        float(cosine_similarity(
            embeddings[i].reshape(1, -1),
            embeddings[i + 1].reshape(1, -1),
        )[0, 0])
        for i in range(len(embeddings) - 1)
    ]

    sub_groups: List[List[str]] = [[valid_ids[0]]]
    for i, sim in enumerate(sims):
        if sim < threshold:
            sub_groups.append([valid_ids[i + 1]])
        else:
            sub_groups[-1].append(valid_ids[i + 1])

    return sub_groups


def _cohesion_split_chunks(chunks_list: List[Dict]) -> List[List[Dict]]:
    """
    Split a list of chunk dicts where embedding similarity between
    consecutive chunks drops below threshold.  Operates directly on
    chunk dicts (used for post-stage1 validation on raw constituents).
    """
    if len(chunks_list) <= 1:
        return [chunks_list]

    threshold = getattr(config, 'GROUP_COHESION_THRESHOLD', 0.40)
    embedder = _get_embedder()

    texts = [
        f"{c.get('section_header', '')}. {c.get('text', '')}"[:500]
        for c in chunks_list
    ]
    embeddings = embedder.encode(texts, convert_to_numpy=True)
    sims = [
        float(cosine_similarity(
            embeddings[i].reshape(1, -1),
            embeddings[i + 1].reshape(1, -1),
        )[0, 0])
        for i in range(len(embeddings) - 1)
    ]

    sub_groups: List[List[Dict]] = [[chunks_list[0]]]
    for i, sim in enumerate(sims):
        if sim < threshold:
            sub_groups.append([chunks_list[i + 1]])
        else:
            sub_groups[-1].append(chunks_list[i + 1])

    return sub_groups


def _validate_groups(
    merged_groups: Dict[int, List[str]],
    id_to_chunk: Dict[str, Dict],
) -> List[List[str]]:
    """
    Run cohesion validation on all merged groups.
    Returns a flat list of validated (possibly split) groups.
    """
    validated: List[List[str]] = []
    n_splits = 0

    for group_ids in merged_groups.values():
        if len(group_ids) <= 1:
            validated.append(group_ids)
            continue

        sub = _cohesion_split(group_ids, id_to_chunk)
        if len(sub) > 1:
            n_splits += 1
        validated.extend(sub)

    log.info(
        f"Stage 2.5 (cohesion): validated {len(merged_groups)} groups, "
        f"split {n_splits} incoherent groups → {len(validated)} final groups."
    )
    return validated


# ── CHUNK MERGING ──────────────────────────────────────────────────────────────

def _merge_chunk_dicts(chunk_group: List[Dict]) -> Dict:
    """Combine a list of chunk dicts into a single merged chunk."""
    if len(chunk_group) == 1:
        return chunk_group[0]

    merged = dict(chunk_group[0])
    merged["chunk_id"] = "_".join(str(c["chunk_id"]) for c in chunk_group)
    merged["text"] = "\n\n".join(str(c.get("text", "")) for c in chunk_group)

    seen_kws: set = set()
    all_kws: List[str] = []
    for c in chunk_group:
        for kw in c.get("keywords", []):
            if kw not in seen_kws:
                seen_kws.add(kw)
                all_kws.append(kw)
    merged["keywords"] = all_kws

    pages: List[int] = []
    for c in chunk_group:
        for n in re.findall(r'\d+', str(c.get("page_range", ""))):
            pages.append(int(n))

    if pages:
        lo, hi = min(pages), max(pages)
        merged["page_range"] = f"{lo}-{hi}" if lo != hi else str(lo)

    return merged


# ── ORCHESTRATOR ───────────────────────────────────────────────────────────────

def run_llm_grouping(
    chunks: List[Dict],
    batch_size: int = 8,
    sequence_size: int = 10,
) -> List[Dict]:
    """
    Three-stage grouping pipeline.

    Stage 1   – Rule-based pre-grouper (zero LLM calls)
    Stage 2   – LLM semantic grouper (GPU-batched sliding windows)
    Stage 2.5 – Cohesion validator: splits merged groups where embedding
                similarity between consecutive chunks drops below threshold,
                catching cross-window stitching artifacts.
    """
    max_size = getattr(config, 'MAX_RULE_GROUP_SIZE', 8)

    docs: Dict[str, List[Dict]] = defaultdict(list)
    for c in chunks:
        docs[c["source_doc"]].append(c)

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    stage1_chunks: List[Dict] = []
    total_rule_merges = 0
    stage1_cohesion_splits = 0

    for doc, doc_chunks in docs.items():
        grouped_indices, n_merges = rule_based_pre_grouper(doc_chunks)
        total_rule_merges += n_merges
        for index_group in grouped_indices:
            constituent = [doc_chunks[i] for i in index_group]
            if len(constituent) > 2:
                sub_groups = _cohesion_split_chunks(constituent)
                if len(sub_groups) > 1:
                    stage1_cohesion_splits += 1
                for sub in sub_groups:
                    stage1_chunks.append(_merge_chunk_dicts(sub))
            else:
                stage1_chunks.append(_merge_chunk_dicts(constituent))

    log.info(
        f"Stage 1 (rules): {total_rule_merges} merges, "
        f"{stage1_cohesion_splits} cohesion splits. "
        f"Chunks after Stage 1: {len(stage1_chunks)} (was {len(chunks)})."
    )

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    docs2: Dict[str, List[Dict]] = defaultdict(list)
    for c in stage1_chunks:
        docs2[c["source_doc"]].append(c)

    all_prompts:  List[str]  = []
    all_metadata: List[Dict] = []

    log.info(
        f"Stage 2 (LLM): building prompts "
        f"(sequence window: {sequence_size} chunks)..."
    )

    overlap = getattr(config, 'STAGE2_OVERLAP', 2)
    step = max(1, sequence_size - overlap)

    for doc, doc_chunks in docs2.items():
        for i in range(0, len(doc_chunks), step):
            seq = doc_chunks[i:i + sequence_size]
            chunk_list_text, simple_ids, simple_to_orig, simple_id_to_chunk = (
                _prepare_sequence(seq)
            )
            all_prompts.append(_build_prompt(chunk_list_text, source_doc=doc))
            all_metadata.append({
                "doc":                doc,
                "seq":                seq,
                "simple_ids":         simple_ids,
                "simple_to_orig":     simple_to_orig,
                "simple_id_to_chunk": simple_id_to_chunk,
            })

    llm_outputs = llm_client.generate_batch(
        all_prompts, max_tokens=650, desc="LLM Grouper Inference", stop=["```\n"]
    )

    # Collect groups per window, enforce max group size on LLM output
    all_groups: List[List[str]] = []
    seen_chunk_ids: set = set()

    for meta, raw_out in zip(all_metadata, llm_outputs):
        grouped_simple = _parse_grouping_output(
            raw_out,
            meta["simple_ids"],
            meta["simple_id_to_chunk"],
        )

        for simple_group in grouped_simple:
            orig_ids = [
                meta["simple_to_orig"][sid]
                for sid in simple_group
                if sid in meta["simple_to_orig"]
            ]
            if not orig_ids:
                continue
            # Count total individual parts (each orig_id may be compound like "12_13_14")
            total_parts = sum(len(oid.split('_')) for oid in orig_ids)
            if total_parts <= max_size:
                all_groups.append(orig_ids)
            else:
                # Split into sub-groups that each stay under max_size parts
                sub: List[str] = []
                sub_parts = 0
                for oid in orig_ids:
                    oid_parts = len(oid.split('_'))
                    if sub_parts + oid_parts > max_size and sub:
                        all_groups.append(sub)
                        sub = []
                        sub_parts = 0
                    sub.append(oid)
                    sub_parts += oid_parts
                if sub:
                    all_groups.append(sub)

    # Deduplicate overlap: if a chunk appears in multiple groups, merge them
    # but respect max_size to prevent cascading
    chunk_to_group: Dict[str, int] = {}
    group_parent: Dict[int, int] = {}
    group_size: Dict[int, int] = {}

    def gfind(x: int) -> int:
        while group_parent[x] != x:
            group_parent[x] = group_parent[group_parent[x]]
            x = group_parent[x]
        return x

    def gunion(a: int, b: int) -> bool:
        ra, rb = gfind(a), gfind(b)
        if ra == rb:
            return True
        if group_size.get(ra, 1) + group_size.get(rb, 1) > max_size:
            return False
        group_size[ra] = group_size.get(ra, 1) + group_size.get(rb, 1)
        group_parent[rb] = ra
        return True

    for gi, group in enumerate(all_groups):
        group_parent[gi] = gi
        group_size[gi] = sum(len(oid.split('_')) for oid in group)
        for cid in group:
            if cid in chunk_to_group:
                gunion(chunk_to_group[cid], gi)
            else:
                chunk_to_group[cid] = gi

    # Collect merged groups
    merged_groups: Dict[int, List[str]] = defaultdict(list)
    for gi, group in enumerate(all_groups):
        root = gfind(gi)
        for cid in group:
            if cid not in seen_chunk_ids:
                merged_groups[root].append(cid)
                seen_chunk_ids.add(cid)

    # ── Stage 2.5: Cohesion validation ──────────────────────────────────────
    all_stage1_by_id = {str(c["chunk_id"]): c for c in stage1_chunks}
    validated_groups = _validate_groups(merged_groups, all_stage1_by_id)

    # Build final chunks
    final_chunks: List[Dict] = []

    for group_ids in validated_groups:
        constituents = [all_stage1_by_id[oid] for oid in group_ids if oid in all_stage1_by_id]
        if constituents:
            final_chunks.append(_merge_chunk_dicts(constituents))

    stage2_merges = len(stage1_chunks) - len(final_chunks)
    log.info(
        f"Stage 2 + 2.5: {stage2_merges} net merges. "
        f"Final chunk count: {len(final_chunks)} "
        f"(total reduction: {len(chunks) - len(final_chunks)} from {len(chunks)})."
    )

    return final_chunks