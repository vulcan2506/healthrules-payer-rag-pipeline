"""
delta_analyzer.py
─────────────────
Three-pass structured delta analysis using Pydantic schemas +
LangChain JsonOutputParser.

EXTRACTION ARCHITECTURE (per chunk, two sub-passes)
────────────────────────────────────────────────────
Sub-pass 1a  Holistic Read
  The full cleaned chunk text is read in one go.
  The model produces a rough BehavioralProfile covering the main feature,
  behaviors, requirements, deprecations, and new items.

Sub-pass 1b  Fine-Grained Gap Fill
  The same full text is re-read paragraph-by-paragraph WITH the rough
  profile visible.  The model outputs ONLY the facts it missed the first
  time (ProfileAdditions).  This catches edge-case properties, secondary
  features buried mid-chunk, and table-embedded details.

  Merge  rough BehavioralProfile + ProfileAdditions → final BehavioralProfile
  Deduplication is done by substring matching so near-duplicate statements
  are not added twice.

COMPARISON (Pass 2)
───────────────────
Final profiles for both versions + raw text excerpts + taxonomy descriptions
→ DeltaResult with relevance_score (LLM-assessed, 1-10), change_type,
  analysis, key_differences, and confidence.

PROMPT DESIGN NOTE
──────────────────
The extraction prompts use a concrete hand-crafted JSON template instead of
the auto-generated format_instructions string.  Small LLMs (Phi-3-mini etc.)
often echo back the schema definition verbatim when shown format_instructions;
a filled-in example template avoids that entirely.

The comparison prompt still uses delta_parser.get_format_instructions() because
the DeltaResult schema is more complex and benefits from the full description.

VERSION LABELS
──────────────
Throughout prompts and the report the versions are referred to as
  "{vA} (Older)"  and  "{vB} (New Version)"
rather than the generic "Version A / Version B".
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

import config
import llm_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

FILTERED_JSON_PATH  = config.OUTPUT_DIR / "cross_version_topics_only.json"
CHUNKS_JSON_PATH    = config.CHUNKS_CACHE
GROUPED_CHUNKS_PATH = config.OUTPUT_DIR / "filtered_chunks.json"
REPORT_MD_PATH      = config.OUTPUT_DIR / "version_delta_report.md"

def unload():
    pass


# ══════════════════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class BehavioralProfile(BaseModel):
    feature_name:     str        = Field(description="Short descriptive name of the feature.")
    key_behaviors:    List[str]  = Field(description="What the system does in this version.")
    requirements:     List[str]  = Field(description="Config flags, runtime properties, restart requirements.")
    deprecated_items: List[str]  = Field(description="Items removed, replaced, or deprecated.")
    new_items:        List[str]  = Field(description="New features, APIs, columns, UI elements introduced.")


class ProfileAdditions(BaseModel):
    """Facts missed during the holistic read — output only NEW information."""
    additional_behaviors:    List[str] = Field(description="Behaviors not in the initial profile.")
    additional_requirements: List[str] = Field(description="Requirements not in the initial profile.")
    additional_deprecated:   List[str] = Field(description="Deprecated items not in the initial profile.")
    additional_new_items:    List[str] = Field(description="New items not in the initial profile.")


class DeltaResult(BaseModel):
    relevance_score:  int        = Field(description="1-10: how well-matched this pair is as the same feature across versions.")
    relevance_reason: str        = Field(description="One sentence justifying the relevance score.")
    change_type:      str        = Field(description="Direct Contradiction | Deprecation | New Requirement | Workflow Automation | Bug Fix | Minor Enhancement | No Change Detected")
    analysis:         str        = Field(description="3-4 complete sentences explaining the delta with direct version references.")
    key_differences:  List[str]  = Field(description="3-5 diff bullets: '<vA> (Older): X → <vB> (New Version): Y'.")
    confidence:       str        = Field(description="high | medium | low")


delta_parser = JsonOutputParser(pydantic_object=DeltaResult)




# ══════════════════════════════════════════════════════════════════════════════
# TEXT CLEANING  (no truncation)
# ══════════════════════════════════════════════════════════════════════════════

def _clean_text(text: str) -> str:
    """Remove noise without truncating prose content."""
    text  = re.sub(r'<!--\s*image\s*-->', '[image]', text)
    lines = text.splitlines()
    out: List[str] = []
    jsonl_count = 0
    for line in lines:
        s = line.strip()
        if s.startswith('{') and s.endswith('}'):
            jsonl_count += 1
            if jsonl_count <= 3:
                out.append(line)
            elif jsonl_count == 4:
                out.append(f"[... +{jsonl_count - 3} more table rows omitted ...]")
            continue
        if jsonl_count > 4:
            out[-1] = f"[... +{jsonl_count - 3} more table rows omitted ...]"
        jsonl_count = 0
        out.append(line)
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(out)).strip()


def _split_paragraphs(text: str) -> List[str]:
    """Split cleaned text into logical paragraphs for fine-grained re-read."""
    paras = re.split(r'\n{2,}', text)
    return [p.strip() for p in paras if p.strip()]


# ══════════════════════════════════════════════════════════════════════════════
# SUB-PASS 1a — HOLISTIC EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
# Uses a concrete hand-crafted JSON template (NOT format_instructions).
# Small LLMs echo format_instructions schemas back verbatim; a filled-in
# example is far more reliable.

_HOLISTIC_PROMPT = """\
You are {analyst_role}. Read the {doc_purpose} text below about \
"{topic}" and extract a structured behavioral profile.

CRITICAL RULES:
- Output ONLY a single JSON object — nothing before or after it.
- Do NOT output the structure definition itself; fill in real values.
- Keep each list to a MAXIMUM of 5 items so the JSON stays concise.
- If the text covers multiple features, combine their behaviors under one \
  profile using a descriptive feature_name like "Feature A + Feature B".
- Use exact property names where mentioned (e.g. WORKBASKET_RULE_OPTIMIZATION_ENABLE).
- If a list has no items write [].
- Complete the closing braces — never leave the JSON unfinished.

Output this exact structure with real values:

{{
  "feature_name": "<short descriptive name — combine if multiple features>",
  "key_behaviors": [
    "<behavior 1>",
    "<behavior 2>",
    "<behavior 3 — max 5 total>"
  ],
  "requirements": [
    "<runtime property / config flag / restart requirement — max 5 total>"
  ],
  "deprecated_items": [
    "<technology or feature removed/replaced — max 5 total>"
  ],
  "new_items": [
    "<new UI element / API / table column / property — max 5 total>"
  ]
}}

### EXAMPLES ###

EXAMPLE A — Platform migration
TEXT: The earlier release is the final release supporting Oracle WebLogic. \
Starting the newer release, WebLogic is deprecated. Apache Tomcat replaces WebLogic. \
Apache Artemis replaces WebLogic JMS. HTTP replaces Oracle T3. \
OpenJDK 11 (Temurin) replaces Oracle JDK. Restart required for Tomcat config.

OUTPUT:
{{
  "feature_name": "Platform Migration: WebLogic to Apache Tomcat",
  "key_behaviors": [
    "The earlier release is the last version to support Oracle WebLogic",
    "The newer release adopts Apache Tomcat as the sole supported application server",
    "Apache Artemis replaces WebLogic JMS for internal messaging",
    "Standard HTTP replaces the Oracle T3 proprietary protocol",
    "OpenJDK 11 (Temurin) replaces Oracle JDK 11"
  ],
  "requirements": [
    "Deployment scripts must be updated for Apache Tomcat conventions",
    "Messaging configs must be updated for Apache Artemis compatibility"
  ],
  "deprecated_items": [
    "Oracle WebLogic application server",
    "WebLogic JMS messaging",
    "Oracle T3 protocol",
    "Oracle JDK 11"
  ],
  "new_items": [
    "Apache Tomcat application server",
    "Apache Artemis messaging",
    "OpenJDK 11 (Temurin)"
  ]
}}

EXAMPLE B — Multi-feature chunk (Workbasket + Reprocessed Claims)
TEXT: Assignment rules for Utilization Review workbaskets can now be \
configured using Authorization Input Source. Input sources such as Cigna, \
Aerial are available. [PAYOR-178412] Previously, the Reprocessed Claims \
workbasket showed only initial results. Now it dynamically updates to reflect \
subsequent events. A visual indicator appears next to the Claim ID when \
results may differ. A new Claim ID filter has been added. [PAYOR-178410]

OUTPUT:
{{
  "feature_name": "Workbasket Assignment Rules + Reprocessed Claims Updates",
  "key_behaviors": [
    "Utilization Review workbaskets can be configured using Authorization Input Source as a condition",
    "Input sources such as Cigna and Aerial are available when defining assignment rules",
    "Reprocessed Claims workbasket now dynamically updates to reflect subsequent events",
    "Visual indicator appears next to Claim ID when results may differ from reprocessing",
    "New Claim ID filter added to the Reprocessed Claims workbasket screen"
  ],
  "requirements": [],
  "deprecated_items": [
    "Static view in Reprocessed Claims workbasket that only showed initial reprocessing results"
  ],
  "new_items": [
    "Authorization Input Source condition for Utilization Review workbasket rules",
    "Visual indicator icon on Claim ID in Reprocessed Claims workbasket",
    "Claim ID filter in Reprocessed Claims screen"
  ]
}}

EXAMPLE C — Bug fix with new property
TEXT: Previously, same-confinement claims were included in \
DeliveredWithinPeriodOfConfinement evaluation causing wrong results. \
Fixed by new property EXCLUDE_LINES_FROM_CURRENT_CONFINEMENT (default: true). \
No restart needed.

OUTPUT:
{{
  "feature_name": "Confinement Claim Evaluation Fix",
  "key_behaviors": [
    "Same-confinement claims are now excluded from period-of-confinement evaluation",
    "Property defaults to true so the fix activates automatically after upgrade",
    "Episode of Care evaluation is also corrected by this change"
  ],
  "requirements": [
    "EXCLUDE_LINES_FROM_CURRENT_CONFINEMENT property controls the behavior (default: true)",
    "No server restart required"
  ],
  "deprecated_items": [
    "Previous behavior that included same-confinement claims in evaluation"
  ],
  "new_items": [
    "EXCLUDE_LINES_FROM_CURRENT_CONFINEMENT runtime property"
  ]
}}

### ACTUAL TASK ###
TOPIC: {topic}
TEXT:
{text}

OUTPUT (JSON only — close all braces, max 5 items per list):"""


# ── Sub-pass 1b — Fine-Grained Gap Fill ───────────────────────────────────────

_GAP_FILL_PROMPT = """\
You are {analyst_role} doing a careful second pass on a \
{doc_purpose} about "{topic}".

A colleague already extracted this initial profile from the text:
{rough_profile}
{qna_hints}
Re-read the FULL TEXT below and find facts MISSING from the profile above.

Output ONLY a JSON object — nothing before or after it. \
Max 5 items per list. Close all braces.

{{
  "additional_behaviors": ["<missed behavior, or empty list>"],
  "additional_requirements": ["<missed property/config/restart, or empty list>"],
  "additional_deprecated": ["<missed deprecated item, or empty list>"],
  "additional_new_items": ["<missed new item, or empty list>"]
}}

Rules:
- Add a fact ONLY if it is GENUINELY MISSING from the initial profile.
- Do NOT repeat facts already captured (even if worded differently).
- If nothing is missing, write [] for every key.
- No text before or after the JSON. Close all braces.

TOPIC: {topic}
FULL TEXT:
{text}

OUTPUT (JSON only — close all braces):"""


def _profile_to_text(p: BehavioralProfile) -> str:
    """Compact text rendering of a profile for use inside the gap-fill prompt."""
    lines = [f"Feature: {p.feature_name}"]
    if p.key_behaviors:
        lines.append("Behaviors: " + " | ".join(p.key_behaviors))
    if p.requirements:
        lines.append("Requirements: " + " | ".join(p.requirements))
    if p.deprecated_items:
        lines.append("Deprecated: " + " | ".join(p.deprecated_items))
    if p.new_items:
        lines.append("New items: " + " | ".join(p.new_items))
    return "\n".join(lines)


def _format_qna_hint(qna: list) -> str:
    """Format Q&A pairs as a prompt hint block (empty string if no pairs)."""
    if not qna:
        return ""
    lines = ["\nKnown Q&A from this version (use as extraction hints — add any property names or requirements not yet in the profile):"]
    for pair in qna[:4]:
        q = str(pair.get("q", "")).strip()
        a = str(pair.get("a", "")).strip()
        if q and a:
            lines.append(f"  Q: {q}")
            lines.append(f"  A: {a}")
    lines.append("")
    return "\n".join(lines) + "\n"


def _get_qna_for_id(qna_dict: Dict, compound_id: str) -> list:
    """
    Look up Q&A pairs for a potentially compound chunk ID.
    The registry may join atomic+grouped IDs as "136_137_192" where "136_137"
    and "192" are separate filtered-chunks entries. Uses longest-match left-to-right.
    """
    if compound_id in qna_dict:
        return qna_dict[compound_id]
    parts = compound_id.split("_")
    result: List[Dict] = []
    seen_q: set = set()
    i = 0
    while i < len(parts):
        matched = False
        for j in range(len(parts), i, -1):
            candidate = "_".join(parts[i:j])
            if candidate in qna_dict:
                for pair in qna_dict[candidate]:
                    q = pair.get("q", "")
                    if q not in seen_q:
                        seen_q.add(q)
                        result.append(pair)
                i = j
                matched = True
                break
        if not matched:
            i += 1
    return result


def _build_holistic_prompts(jobs: List[Dict], side: str) -> List[str]:
    return [
        _HOLISTIC_PROMPT.format(
            analyst_role=job.get("_analyst_role", "a healthcare IT analyst"),
            doc_purpose=job.get("_doc_purpose", "release note"),
            topic=job["topic"],
            text=_clean_text(job[f"text_{side}"]),
        )
        for job in jobs
    ]


def _build_gap_fill_prompts(jobs: List[Dict], side: str) -> List[str]:
    return [
        _GAP_FILL_PROMPT.format(
            analyst_role=job.get("_analyst_role", "a healthcare IT analyst"),
            doc_purpose=job.get("_doc_purpose", "release note"),
            topic=job["topic"],
            rough_profile=_profile_to_text(job[f"rough_profile_{side}"]),
            qna_hints=_format_qna_hint(job.get(f"qna_{side}", [])),
            text=_clean_text(job[f"text_{side}"]),
        )
        for job in jobs
    ]


# ── JSON parsing helpers ───────────────────────────────────────────────────────

def _extract_json_block(raw: str) -> Optional[dict]:
    """
    Robustly extract the first {...} JSON object from LLM output.

    Four-stage cascade:
      1. Strip markdown fences + direct json.loads
      2. Brace-depth scan to isolate the first complete {...} object
      3. JSON repair: close truncated objects that ran out of tokens
      4. Regex field extraction as last resort
    """
    # Stage 1: strip fences and try direct parse
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    for attempt in [cleaned, cleaned.split('\n\n')[0]]:
        try:
            return json.loads(attempt)
        except Exception:
            pass

    start = cleaned.find('{')
    if start == -1:
        return None

    content = cleaned[start:]

    # Stage 2: brace-depth scan for a complete object
    stack: List[str] = []
    in_str = escape = False
    complete_end = -1

    for i, ch in enumerate(content):
        if escape:
            escape = False; continue
        if ch == '\\' and in_str:
            escape = True; continue
        if ch == '"':
            in_str = not in_str; continue
        if in_str:
            continue
        if ch in '{[':
            stack.append(ch)
        elif ch in '}]':
            if stack:
                stack.pop()
                if not stack:
                    complete_end = i
                    break

    if complete_end != -1:
        try:
            return json.loads(content[:complete_end + 1])
        except Exception:
            pass

    # Stage 3: repair truncated JSON (model ran out of tokens mid-object)
    # Build the closings needed to complete the stack
    if stack:
        closings = ''.join(']' if b == '[' else '}' for b in reversed(stack))
        base = content.rstrip()

        # Candidates: remove trailing partial string / hanging comma, then close
        candidates = [
            base,
            re.sub(r',?\s*"[^"]*$', '', base),   # drop partial unclosed string
            re.sub(r',\s*$', '', base),             # drop trailing comma
            re.sub(r',?\s*"[^"]*$', '', re.sub(r',\s*$', '', base)),
        ]
        for candidate in candidates:
            for suffix in [closings, closings.rstrip('}') + '}']:
                try:
                    return json.loads(candidate + suffix)
                except Exception:
                    pass

    # Stage 4: regex field extraction (very malformed output)
    result: dict = {}
    fn = re.search(r'"feature_name"\s*:\s*"([^"]+)"', content)
    if fn:
        result["feature_name"] = fn.group(1)
    for field in ["key_behaviors", "requirements", "deprecated_items", "new_items",
                  "additional_behaviors", "additional_requirements",
                  "additional_deprecated", "additional_new_items"]:
        items = re.findall(rf'"{field}"\s*:\s*\[(.*?)\]', content, re.DOTALL)
        if items:
            result[field] = [
                m.strip().strip('"')
                for m in re.findall(r'"([^"]+)"', items[0])
                if m.strip()
            ]
    return result if result else None


def _parse_profile(raw: str) -> BehavioralProfile:
    data = _extract_json_block(raw)
    if data:
        try:
            return BehavioralProfile(**data)
        except Exception:
            pass

    lines = [ln.lstrip('-•n ').strip() for ln in raw.splitlines() if ln.strip()]
    return BehavioralProfile(
        feature_name="Could not extract",
        key_behaviors=lines[:6] or ["Extraction failed — review raw chunk."],
        requirements=[],
        deprecated_items=[],
        new_items=[],
    )


def _parse_additions(raw: str) -> ProfileAdditions:
    data = _extract_json_block(raw)
    if data:
        try:
            return ProfileAdditions(**data)
        except Exception:
            pass
    return ProfileAdditions(
        additional_behaviors=[],
        additional_requirements=[],
        additional_deprecated=[],
        additional_new_items=[],
    )


def _merge_profiles(base: BehavioralProfile, additions: ProfileAdditions) -> BehavioralProfile:
    """
    Merge a rough profile with gap-fill additions.
    Deduplicates by substring containment so near-duplicate lines are dropped.
    """
    def dedup(existing: List[str], new: List[str], cap: int = 7) -> List[str]:
        result = list(existing)
        for item in new:
            if len(result) >= cap:
                break
            item_l = item.lower().strip()
            if not item_l:
                continue
            already = any(
                item_l in e.lower() or e.lower() in item_l
                for e in result
            )
            if not already:
                result.append(item)
        return result

    return BehavioralProfile(
        feature_name     = base.feature_name,
        key_behaviors    = dedup(base.key_behaviors,    additions.additional_behaviors),
        requirements     = dedup(base.requirements,     additions.additional_requirements),
        deprecated_items = dedup(base.deprecated_items, additions.additional_deprecated),
        new_items        = dedup(base.new_items,         additions.additional_new_items),
    )


# ══════════════════════════════════════════════════════════════════════════════
# PASS 2 — DELTA COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def _build_compare_prompt(job: Dict) -> str:
    vA      = job["vA"]
    vB      = job["vB"]
    label_A = f"{vA} (Older)"
    label_B = f"{vB} (New Version)"

    analyst_role = job.get("_analyst_role", "a healthcare IT analyst")
    doc_purpose  = job.get("_doc_purpose", "software releases")

    prompt = f"""\
You are {analyst_role} comparing the feature "{job['topic']}" \
across two {doc_purpose}.

Output ONLY a single JSON object with this exact structure — fill in real \
values, do not copy the structure itself. Close all braces. \
Max 4 items in key_differences.

{{
  "relevance_score": <integer 1-10>,
  "relevance_reason": "<one sentence citing specific content from each version>",
  "change_type": "<one of the types below>",
  "analysis": "<3-4 complete sentences explaining the delta with direct version references>",
  "key_differences": [
    "{label_A}: <old behavior> -> {label_B}: <new behavior>",
    "<another difference>"
  ],
  "confidence": "<high | medium | low>"
}}

RELEVANCE SCORING RUBRIC (be strict — do NOT default to 9 or 10):
  1-3  Mismatched pair — the two chunks describe unrelated features
  4-5  Weak overlap — same broad area but different specific features
  6-7  Moderate — same feature area, but chunks focus on different aspects
  8-9  Strong — clearly the same feature, with meaningful differences to compare
  10   Perfect — identical feature, both chunks describe the same behavior in detail

CHANGE TYPE (pick the MOST SPECIFIC match — do NOT default to Direct Contradiction):
  Direct Contradiction - a behavior in {label_A} is EXPLICITLY REVERSED in {label_B} (e.g. "required" → "not required", "enabled" → "disabled"). Must cite the specific contradicting statements.
  Deprecation          - a technology/feature present in {label_A} is REMOVED or REPLACED in {label_B}
  New Requirement      - {label_B} adds a mandatory property, config, or restart requirement NOT present in {label_A}
  Workflow Automation  - a manual step in {label_A} is automated in {label_B}
  Bug Fix              - incorrect behavior in {label_A} is corrected in {label_B}
  Minor Enhancement    - same feature in both versions; {label_B} EXTENDS, REFINES, or adds OPTIONS. This is the most common type when a feature evolves between versions — use it when the old behavior still works but new capabilities are added.
  No Change Detected   - behaviors are identical, or the pair is unrelated

IMPORTANT: "Minor Enhancement" should be your default when the feature exists in both versions and {label_B} adds new options, fields, or capabilities WITHOUT removing old behavior. "Direct Contradiction" requires an explicit reversal — not just an addition or expansion.

### FEW-SHOT EXAMPLES ###

EXAMPLE 1 - Deprecation
{label_A} profile:
  Feature: Oracle WebLogic Platform
  Behaviors: WebLogic is the app server | WebLogic JMS handles messaging | T3 protocol used
  New: Apache Tomcat early-adopter support

{label_B} profile:
  Feature: Platform Migration Complete - Apache Tomcat
  Behaviors: WebLogic fully deprecated | Apache Tomcat mandatory | Artemis replaces JMS | HTTP replaces T3
  Deprecated: Oracle WebLogic, WebLogic JMS, Oracle T3, Oracle JDK 11

OUTPUT:
{{
  "relevance_score": 10,
  "relevance_reason": "Both chunks describe the same platform infrastructure; {label_A} documents WebLogic as the current stack while {label_B} confirms its full deprecation.",
  "change_type": "Deprecation",
  "analysis": "{label_A} supports Oracle WebLogic as the application server and ships early-adopter Apache Tomcat support for migration testing. {label_B} completes this migration by fully deprecating the entire Oracle stack. Customers must update deployment scripts, messaging configurations, and client protocols before upgrading. This is a breaking infrastructure change requiring careful planning.",
  "key_differences": [
    "{label_A}: Oracle WebLogic app server -> {label_B}: Apache Tomcat mandatory, WebLogic fully deprecated",
    "{label_A}: WebLogic JMS messaging -> {label_B}: Apache Artemis replaces WebLogic JMS",
    "{label_A}: Oracle T3 protocol -> {label_B}: Standard HTTP replaces Oracle T3"
  ],
  "confidence": "high"
}}

EXAMPLE 2 - Direct Contradiction
{label_A} profile:
  Feature: Runtime property with restart
  Requirements: Server restart required after property change

{label_B} profile:
  Feature: Runtime Property Hot-Swap
  Behaviors: All runtime property changes applied immediately without restart

OUTPUT:
{{
  "relevance_score": 9,
  "relevance_reason": "Both chunks describe the same runtime property but {label_A} requires a restart while {label_B} eliminates that requirement.",
  "change_type": "Direct Contradiction",
  "analysis": "{label_A} explicitly requires a server restart after every property change, making configuration updates a disruptive operation needing maintenance windows. {label_B} reverses this entirely, introducing hot-swap capability where changes take effect immediately without downtime. This is a significant operational improvement that changes deployment and maintenance procedures. Teams relying on restart procedures in {vA} will need to update their runbooks for {vB}.",
  "key_differences": [
    "{label_A}: server restart required after property change -> {label_B}: no restart required, hot-swap applied immediately",
    "{label_A}: configuration updates need maintenance windows -> {label_B}: changes can be applied live"
  ],
  "confidence": "high"
}}

EXAMPLE 3 - Minor Enhancement
{label_A} profile:
  Feature: Service Definition Split Handling
  Behaviors: Full evaluation completes before applying splits; splits only applied if they affect outcome

{label_B} profile:
  Feature: Service Definition Split Optimization
  Behaviors: Optimization override scenarios handled correctly; performance metrics available
  Requirements: SERVICE_DEFINITION_EVALUATOR_OPTIMIZATION_ENABLED (default: true); no restart needed
  New: SERVICE_DEFINITION_EVALUATOR_OPTIMIZATION_ENABLED property

OUTPUT:
{{
  "relevance_score": 8,
  "relevance_reason": "Both chunks address split handling during service definition evaluation, with {label_B} adding runtime property control and performance tooling.",
  "change_type": "Minor Enhancement",
  "analysis": "{label_A} improved split handling so full evaluation completes before splits are applied, ensuring splits only take effect when they impact the outcome. {label_B} extends this with SERVICE_DEFINITION_EVALUATOR_OPTIMIZATION_ENABLED (default: true) controlling expression reordering for better performance. The evaluation utility is also fixed to correctly handle optimization override scenarios. The core behavior from {vA} is preserved and extended.",
  "key_differences": [
    "{label_A}: no runtime control over evaluation optimization -> {label_B}: SERVICE_DEFINITION_EVALUATOR_OPTIMIZATION_ENABLED added (default: true)",
    "{label_A}: evaluation utility had gaps in override handling -> {label_B}: all override scenarios handled correctly",
    "{label_A}: no performance monitoring -> {label_B}: optional metrics collection property added (default: false)"
  ],
  "confidence": "medium"
}}

EXAMPLE 4 - Mismatched pair
{label_A} profile:
  Feature: Reinsurance Details Tab on Claim View

{label_B} profile:
  Feature: Platform Migration: WebLogic Deprecation

OUTPUT:
{{
  "relevance_score": 1,
  "relevance_reason": "{label_A} describes a UI enhancement for reinsurance tracking while {label_B} describes an infrastructure migration - unrelated topics incorrectly matched.",
  "change_type": "No Change Detected",
  "analysis": "These two chunks do not represent the same feature. {label_A} introduces a Reinsurance Details tab in the Claim View UI. {label_B} announces the deprecation of Oracle WebLogic and adoption of Apache Tomcat. The cross-version topic match appears incorrect and should be reviewed manually. No meaningful delta can be produced for mismatched pairs.",
  "key_differences": [
    "{label_A}: UI feature (Reinsurance Details tab) vs {label_B}: infrastructure change (WebLogic deprecation)"
  ],
  "confidence": "low"
}}

### ACTUAL TASK ###

Topic: {job['topic']}

{label_A} (Chunk {job['id_A']})
Taxonomy description: {job.get('desc_A', 'N/A')}
Extracted profile:
{_profile_to_text(job['profile_A'])}
Raw text excerpt:
{_clean_text(job['text_A'])[:1200]}

{label_B} (Chunk {job['id_B']})
Taxonomy description: {job.get('desc_B', 'N/A')}
Extracted profile:
{_profile_to_text(job['profile_B'])}
Raw text excerpt:
{_clean_text(job['text_B'])[:1200]}

OUTPUT (JSON only - fill in real values, close all braces):"""

    return prompt

def _parse_delta(raw: str, vA: str, vB: str) -> DeltaResult:
    data = _extract_json_block(raw)
    if data:
        try:
            return DeltaResult(**data)
        except Exception:
            pass

    # Field-by-field regex fallback
    def _re(pattern, text, default="", flags=0):
        m = re.search(pattern, text, flags)
        return m.group(1).strip() if m else default

    return DeltaResult(
        relevance_score  = int(m.group(1)) if (m := re.search(r'"relevance_score"\s*:\s*(\d+)', raw)) else 5,
        relevance_reason = _re(r'"relevance_reason"\s*:\s*"([^"]+)"', raw, "Could not parse."),
        change_type      = _re(r'"change_type"\s*:\s*"([^"]+)"', raw, "System Evolution"),
        analysis         = _re(r'"analysis"\s*:\s*"([^"]+)"', raw, raw if len(raw) > 20 else "Could not parse.", re.DOTALL),
        key_differences  = [],
        confidence       = _re(r'"confidence"\s*:\s*"([^"]+)"', raw, "low"),
    )


# ══════════════════════════════════════════════════════════════════════════════
# DATA PARSING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _extract_version(filename: str) -> str:
    m = re.search(r"(\d+\.\d+(?:\.\d+)?)", filename)
    return f"v{m.group(1)}" if m else filename.replace(".pdf", "").split("_")[0]


def _parse_source_docs(source_docs) -> Tuple[Optional[str], Optional[str]]:
    if isinstance(source_docs, list):
        docs = source_docs
    elif isinstance(source_docs, str):
        docs = [d.strip() for d in source_docs.split("|") if d.strip()]
    else:
        return None, None
    return (
        (_extract_version(docs[0]), _extract_version(docs[1]))
        if len(docs) >= 2 else (None, None)
    )


def _extract_compound_id(raw) -> str:
    s = str(raw).strip().lstrip('[').rstrip(']').strip()
    if not s:
        return ""
    if '_' in s and ',' not in s:
        return s
    if ',' in s:
        return '_'.join(p.strip() for p in s.split(',') if p.strip())
    return s


def _parse_ids_and_descs(chunk_ids, descriptions) -> Tuple[str, str, str, str]:
    id_A = id_B = desc_A = desc_B = ""

    if isinstance(chunk_ids, list) and len(chunk_ids) >= 2:
        id_A = _extract_compound_id(chunk_ids[0])
        id_B = _extract_compound_id(chunk_ids[1])
    elif isinstance(chunk_ids, str):
        sep   = "] | [" if "] | [" in chunk_ids else "|"
        parts = chunk_ids.split(sep)
        if len(parts) >= 2:
            id_A = _extract_compound_id(parts[0])
            id_B = _extract_compound_id(parts[1])

    if isinstance(descriptions, list) and len(descriptions) >= 2:
        desc_A, desc_B = str(descriptions[0]).strip(), str(descriptions[1]).strip()
    elif isinstance(descriptions, str):
        sep   = "] | [" if "] | [" in descriptions else "|"
        parts = descriptions.split(sep)
        if len(parts) >= 2:
            desc_A = parts[0].replace("[", "").replace("]", "").strip()
            desc_B = parts[1].replace("[", "").replace("]", "").strip()
        else:
            desc_A = desc_B = descriptions.strip()

    return id_A, id_B, desc_A, desc_B


def _get_chunk_text(chunk_dict: Dict[str, str], compound_id: str) -> str:
    if compound_id in chunk_dict:
        return chunk_dict[compound_id]
    parts = compound_id.split("_")
    texts = [chunk_dict[p] for p in parts if p in chunk_dict]
    if not texts:
        return ""
    log.debug(f"Reconstructed text for '{compound_id}' from {len(texts)} parts.")
    return "\n\n".join(texts)


# ══════════════════════════════════════════════════════════════════════════════
# REPORT RENDERING
# ══════════════════════════════════════════════════════════════════════════════

TYPE_EMOJI = {
    "Direct Contradiction": "⚡",
    "Deprecation":          "🗑️",
    "New Requirement":      "🔧",
    "Workflow Automation":  "🤖",
    "Bug Fix":              "🐛",
    "Minor Enhancement":    "✨",
    "No Change Detected":   "✅",
    "System Evolution":     "📈",
}
CONF_BADGE = {"high": "🟢 High", "medium": "🟡 Medium", "low": "🔴 Low"}


def _render_profile(p: BehavioralProfile) -> List[str]:
    """Render a profile as clean markdown bullets — no strikethrough."""
    lines = [f"**Feature:** {p.feature_name}\n"]

    if p.key_behaviors:
        lines.append("**Key Behaviors:**")
        lines.extend(f"- {b}" for b in p.key_behaviors)
        lines.append("")

    if p.requirements:
        lines.append("**Requirements / Properties:**")
        lines.extend(f"- {r}" for r in p.requirements)
        lines.append("")

    if p.deprecated_items:
        lines.append("**Deprecated in this version:**")
        lines.extend(f"- {d}" for d in p.deprecated_items)
        lines.append("")

    if p.new_items:
        lines.append("**New in this version:**")
        lines.extend(f"- {n}" for n in p.new_items)
        lines.append("")

    return lines


def _render_topic(idx: int, job: Dict) -> List[str]:
    d      = job["delta"]
    emoji  = TYPE_EMOJI.get(d.change_type, "📌")
    conf   = CONF_BADGE.get(d.confidence.lower(), d.confidence)
    score  = d.relevance_score
    rel    = (
        f"🟢 {score}/10" if score >= 8 else
        f"🟡 {score}/10" if score >= 5 else
        f"🔴 {score}/10 — possibly mismatched pair"
    )
    label_A = f"{job['vA']} (Older)"
    label_B = f"{job['vB']} (New Version)"

    md = []
    md.append(f"### {idx}. {job['topic']}")
    md.append(f"**Location:** {job['parent']} → {job['sub']}  ")
    md.append(f"**LLM Relevance:** {rel}  ")
    md.append(f"*{d.relevance_reason}*\n")

    # ── Older version block ────────────────────────────────────────────────────
    md.append(f"#### {label_A} — Chunk `{job['id_A']}`\n")
    md.append("> **Actual Text**\n>")
    for line in _clean_text(job["text_A"]).splitlines():
        md.append(f"> {line}" if line.strip() else ">")
    md.append("")
    md.append("**Extracted Profile:**")
    md.extend(_render_profile(job["profile_A"]))

    md.append("---")

    # ── New version block ──────────────────────────────────────────────────────
    md.append(f"#### {label_B} — Chunk `{job['id_B']}`\n")
    md.append("> **Actual Text**\n>")
    for line in _clean_text(job["text_B"]).splitlines():
        md.append(f"> {line}" if line.strip() else ">")
    md.append("")
    md.append("**Extracted Profile:**")
    md.extend(_render_profile(job["profile_B"]))

    md.append("---")

    # ── Delta verdict ──────────────────────────────────────────────────────────
    md.append(f"#### {emoji} Delta: {d.change_type}   |   Confidence: {conf}\n")
    md.append(f"{d.analysis}\n")

    if d.key_differences:
        md.append("**Key Differences:**")
        for diff in d.key_differences:
            md.append(f"- {diff}")
        md.append("")

    md.append(
        f"*Traceability: Chunk `{job['id_A']}` ({label_A}) "
        f"vs Chunk `{job['id_B']}` ({label_B})*"
    )
    md.append("\n---\n")
    return md


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_delta_analysis():
    log.info("Loading data...")

    with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        chunk_dict = {str(c["chunk_id"]): c["text"] for c in json.load(f)}

    # Load QnA from grouped chunks (produced by generate_group_qna in grouper)
    qna_dict: Dict[str, list] = {}
    if GROUPED_CHUNKS_PATH.exists():
        with open(GROUPED_CHUNKS_PATH, "r", encoding="utf-8") as f:
            for c in json.load(f):
                cid = str(c.get("chunk_id", ""))
                qna = c.get("qna", [])
                if cid and isinstance(qna, list) and qna:
                    qna_dict[cid] = qna
                    # Also index each atomic part so compound-ID lookups work
                    for part in cid.split("_"):
                        if part not in qna_dict:
                            qna_dict[part] = qna
        log.info(f"Loaded QnA hints for {len(qna_dict)} chunk IDs.")

    if not FILTERED_JSON_PATH.exists():
        log.error(f"Filtered JSON not found: {FILTERED_JSON_PATH}")
        return

    with open(FILTERED_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    jobs:     List[Dict] = []
    global_vA, global_vB = "Older", "Newer"

    for parent in data.get("taxonomy", []):
        for sub in parent.get("sub_categories", []):
            for topic in sub.get("topics", []):
                vA, vB = _parse_source_docs(topic.get("source_docs"))
                if not vA or not vB:
                    continue
                global_vA, global_vB = vA, vB

                id_A, id_B, desc_A, desc_B = _parse_ids_and_descs(
                    topic.get("chunk_ids"), topic.get("description")
                )
                if not id_A or not id_B:
                    log.warning(f"Skipped '{topic['master_label']}' — missing IDs.")
                    continue

                text_A = _get_chunk_text(chunk_dict, id_A)
                text_B = _get_chunk_text(chunk_dict, id_B)
                if not text_A or not text_B:
                    log.warning(f"Skipped '{topic['master_label']}' — chunk text missing (id_A={id_A}, id_B={id_B}).")
                    continue

                # Inject domain context from profile if available
                analyst_role = "a healthcare IT analyst"
                doc_purpose  = "release note"
                try:
                    import context_profiler
                    source_docs = topic.get("source_docs", [])
                    ref_doc = source_docs[0] if isinstance(source_docs, list) and source_docs else ""
                    profile = context_profiler.get_profile(ref_doc) if ref_doc else None
                    if profile:
                        analyst_role = profile.get("analyst_role", analyst_role)
                        doc_purpose  = profile.get("document_purpose", doc_purpose)
                except ImportError:
                    pass

                jobs.append({
                    "parent": parent["parent_category_name"],
                    "sub":    sub["sub_category_name"],
                    "topic":  topic["master_label"],
                    "vA": vA, "vB": vB,
                    "id_A": id_A,   "id_B": id_B,
                    "desc_A": desc_A, "desc_B": desc_B,
                    "text_A": text_A, "text_B": text_B,
                    "qna_A":  _get_qna_for_id(qna_dict, id_A),
                    "qna_B":  _get_qna_for_id(qna_dict, id_B),
                    "_analyst_role": analyst_role,
                    "_doc_purpose":  doc_purpose,
                })

    log.info(f"Collected {len(jobs)} analysis jobs.")
    if not jobs:
        return

    # ── Save prompt templates as .md reference files ─────────────────────────
    try:
        import context_profiler
        ref = jobs[0]
        ref_role = ref.get("_analyst_role", "a healthcare IT analyst")
        ref_purpose = ref.get("_doc_purpose", "release note")
        source_docs = []
        for parent in data.get("taxonomy", []):
            for sub in parent.get("sub_categories", []):
                for topic in sub.get("topics", []):
                    sd = topic.get("source_docs", [])
                    if isinstance(sd, list):
                        source_docs.extend(sd)
        ref_doc = source_docs[0] if source_docs else ""
        ref_profile = context_profiler.get_profile(ref_doc) if ref_doc else None
        if ref_profile:
            tk = ref_profile.get("type_key", "")
            if tk and not context_profiler.load_prompt(tk, "delta_holistic"):
                context_profiler.save_prompt(
                    tk, "delta_holistic",
                    _HOLISTIC_PROMPT.replace(
                        "{analyst_role}", ref_role
                    ).replace("{doc_purpose}", ref_purpose),
                )
                context_profiler.save_prompt(
                    tk, "delta_gapfill",
                    _GAP_FILL_PROMPT.replace(
                        "{analyst_role}", ref_role
                    ).replace("{doc_purpose}", ref_purpose),
                )
    except ImportError:
        pass

    # ── Sub-pass 1a: Holistic extraction ──────────────────────────────────────
    log.info("Sub-pass 1a: Holistic extraction (full read)...")

    raw_holistic_A = llm_client.generate_batch(
        _build_holistic_prompts(jobs, "A"), max_tokens=700, desc="1a — Older", stop=["```\n"], enable_thinking=False
    )
    raw_holistic_B = llm_client.generate_batch(
        _build_holistic_prompts(jobs, "B"), max_tokens=700, desc="1a — New", stop=["```\n"], enable_thinking=False
    )
    for job, ra, rb in zip(jobs, raw_holistic_A, raw_holistic_B):
        job["rough_profile_A"] = _parse_profile(ra)
        job["rough_profile_B"] = _parse_profile(rb)

    # ── Sub-pass 1b: Gap fill (paragraph-by-paragraph) ────────────────────────
    log.info("Sub-pass 1b: Fine-grained gap fill...")

    raw_gaps_A = llm_client.generate_batch(
        _build_gap_fill_prompts(jobs, "A"), max_tokens=500, desc="1b — Older gaps", stop=["```\n"], enable_thinking=False
    )
    raw_gaps_B = llm_client.generate_batch(
        _build_gap_fill_prompts(jobs, "B"), max_tokens=500, desc="1b — New gaps", stop=["```\n"], enable_thinking=False
    )
    for job, ga, gb in zip(jobs, raw_gaps_A, raw_gaps_B):
        job["profile_A"] = _merge_profiles(job["rough_profile_A"], _parse_additions(ga))
        job["profile_B"] = _merge_profiles(job["rough_profile_B"], _parse_additions(gb))

    # ── Pass 2: Delta comparison ───────────────────────────────────────────────
    log.info("Pass 2: Delta comparison + relevance scoring...")

    raw_deltas = llm_client.generate_batch(
        [_build_compare_prompt(job) for job in jobs],
        max_tokens=750, desc="Pass 2 — Compare", stop=["```\n"], enable_thinking=False
    )
    for job, raw in zip(jobs, raw_deltas):
        job["delta"] = _parse_delta(raw, job["vA"], job["vB"])

    # ── Markdown report ────────────────────────────────────────────────────────
    log.info("Generating Markdown report...")

    md: List[str] = []
    md.append(f"# {global_vA} vs {global_vB} — Delta Report\n")
    md.append(
        "Behavioral profiles built via two-sub-pass extraction "
        "(holistic read + fine-grained gap fill), then compared pair-wise "
        "with LLM-assessed relevance scoring. Full chunk text used.\n"
    )

    md.append("## Summary\n")
    md.append("| # | Topic | Change Type | LLM Relevance | Confidence |")
    md.append("|---|-------|-------------|---------------|------------|")

    for idx, job in enumerate(jobs, 1):
        d     = job["delta"]
        emoji = TYPE_EMOJI.get(d.change_type, "📌")
        conf  = CONF_BADGE.get(d.confidence.lower(), d.confidence)
        score = d.relevance_score
        rel   = (
            f"🟢 {score}/10" if score >= 8 else
            f"🟡 {score}/10" if score >= 5 else
            f"🔴 {score}/10"
        )
        md.append(
            f"| {idx} | **{job['topic']}** | {emoji} {d.change_type} | {rel} | {conf} |"
        )

    md.append("\n---\n")
    md.append("## Detailed Analysis\n")
    for idx, job in enumerate(jobs, 1):
        md.extend(_render_topic(idx, job))

    with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    log.info(f"✅ Delta report saved → {REPORT_MD_PATH}")

    # ── Save structured jobs cache for delta_critic ───────────────────────────
    jobs_cache_path = config.OUTPUT_DIR / "delta_jobs_cache.json"
    cache_jobs = []
    for job in jobs:
        cache_jobs.append({
            "parent":  job["parent"],
            "sub":     job["sub"],
            "topic":   job["topic"],
            "vA":      job["vA"],
            "vB":      job["vB"],
            "id_A":    job["id_A"],
            "id_B":    job["id_B"],
            "text_A":  job["text_A"],
            "text_B":  job["text_B"],
            "profile_A": job["profile_A"].model_dump() if job.get("profile_A") else None,
            "profile_B": job["profile_B"].model_dump() if job.get("profile_B") else None,
            "delta":   job["delta"].model_dump() if job.get("delta") else None,
        })
    with open(jobs_cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_jobs, f, indent=2, ensure_ascii=False)
    log.info(f"Jobs cache saved → {jobs_cache_path}")


if __name__ == "__main__":
    run_delta_analysis()