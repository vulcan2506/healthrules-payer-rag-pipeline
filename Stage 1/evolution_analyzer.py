"""
evolution_analyzer.py
──────────────────────
Value-add / evolution synthesis pass, built on top of delta_analyzer.py's
output. Where the delta report asks "what changed" (and is neutral about
whether a change is good, bad, or a contradiction), this pass asks a
narrower, constructive question: for features that genuinely evolved for
the better, what foundation did the older version lay, and what value did
the newer version build on top of it?

INPUT
─────
Reads data/output/delta_jobs_cache.json (produced by delta_analyzer.py) —
no raw chunk re-reading, no new profile extraction. Each cached job already
has profile_A, profile_B, and a scored/classified delta.

FILTER
──────
Only constructive change types are carried forward: Minor Enhancement,
New Requirement, Workflow Automation, Bug Fix. Direct Contradiction,
Deprecation, and No Change Detected are skipped — those already surface
correctly via the existing delta report and are not "value added" stories.

SCOPE NOTE
──────────
This produces one card per topic per adjacent version-pair job — today
that means a single hop (v25.2 -> v26.1), since that's the only real pair
on disk. The schema and cache are shaped so that when a third real version
document lands, the same per-pair card generation runs again on the new
pair; a follow-on "timeline compiler" that chains cards across 3+ versions
into a multi-hop narrative is deliberately NOT built here — there's nothing
to chain yet, and building it now would be speculative scaffolding.
"""

import json
import logging
import re
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

import config
import llm_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

DELTA_JOBS_CACHE_PATH = config.OUTPUT_DIR / "delta_jobs_cache.json"
EVOLUTION_CACHE_PATH  = config.OUTPUT_DIR / "evolution_cards_cache.json"
REPORT_MD_PATH        = config.OUTPUT_DIR / "version_evolution_report.md"

CONSTRUCTIVE_CHANGE_TYPES = {
    "Minor Enhancement",
    "New Requirement",
    "Workflow Automation",
    "Bug Fix",
}


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA
# ══════════════════════════════════════════════════════════════════════════════

class EvolutionCard(BaseModel):
    feature_name:    str       = Field(description="Short descriptive name of the feature.")
    foundation:      str       = Field(description="What the older version established — one sentence.")
    value_added:     List[str] = Field(description="Concrete capabilities/value the newer version builds on top, max 5.")
    narrative:       str       = Field(description="2-3 sentences: older version introduced X; newer version builds on it by Y, enabling Z.")
    change_type:     str       = Field(default="")
    builds_on_prior: bool      = Field(default=True)


def unload():
    pass


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT
# ══════════════════════════════════════════════════════════════════════════════

_EVOLUTION_PROMPT = """\
You are {analyst_role} writing a "value-add" note for the feature \
"{topic}". You already have structured profiles for both versions and a \
classified delta below — do NOT re-read raw text, just re-synthesize \
these facts into a constructive evolution narrative.

CRITICAL RULES:
- Output ONLY a single JSON object — nothing before or after it.
- Ground every statement in the profiles/delta below — do not invent facts.
- Max 5 items in value_added. Close all braces.

Output this exact structure with real values:

{{
  "feature_name": "<short descriptive name>",
  "foundation": "<one sentence: what {vA} established>",
  "value_added": [
    "<concrete capability {vB} adds on top of that foundation>",
    "<another — max 5 total>"
  ],
  "narrative": "<2-3 sentences: {vA} introduced X; {vB} builds on it by Y, enabling Z>"
}}

### EXAMPLE ###

TOPIC: Service Definition Split Optimization
{vA} profile:
  Feature: Service Definition Split Handling
  Behaviors: Full evaluation completes before applying splits; splits only applied if they affect outcome

{vB} profile:
  Feature: Service Definition Split Optimization
  Behaviors: Optimization override scenarios handled correctly; performance metrics available
  Requirements: SERVICE_DEFINITION_EVALUATOR_OPTIMIZATION_ENABLED (default: true); no restart needed

Delta analysis: {vA} improved split handling so full evaluation completes before splits are applied. \
{vB} extends this with a runtime property controlling expression reordering for performance, and fixes \
override handling gaps. The core {vA} behavior is preserved and extended.

OUTPUT:
{{
  "feature_name": "Service Definition Split Optimization",
  "foundation": "{vA} established correct split handling by completing full evaluation before applying splits.",
  "value_added": [
    "SERVICE_DEFINITION_EVALUATOR_OPTIMIZATION_ENABLED property adds configurable performance optimization on top of the existing evaluation logic",
    "Override scenarios that previously had gaps are now handled correctly",
    "Optional performance metrics collection was added for visibility into evaluation cost"
  ],
  "narrative": "{vA} introduced correct, if unoptimized, split-handling logic. {vB} builds directly on that foundation by adding a configurable optimization path and fixing override edge cases, giving teams a faster evaluator without losing any of the correctness {vA} established."
}}

### ACTUAL TASK ###
TOPIC: {topic}

{vA} profile:
{profile_A}

{vB} profile:
{profile_B}

Delta analysis: {delta_analysis}
Key differences: {key_differences}

OUTPUT (JSON only — close all braces, max 5 items in value_added):"""


def _profile_to_text(p: dict) -> str:
    lines = [f"Feature: {p.get('feature_name', '')}"]
    if p.get("key_behaviors"):
        lines.append("Behaviors: " + " | ".join(p["key_behaviors"]))
    if p.get("requirements"):
        lines.append("Requirements: " + " | ".join(p["requirements"]))
    if p.get("new_items"):
        lines.append("New items: " + " | ".join(p["new_items"]))
    return "\n  ".join(lines)


def _build_prompt(job: Dict, analyst_role: str) -> str:
    d = job["delta"]
    return _EVOLUTION_PROMPT.format(
        analyst_role=analyst_role,
        topic=job["topic"],
        vA=job["vA"],
        vB=job["vB"],
        profile_A=_profile_to_text(job["profile_A"]),
        profile_B=_profile_to_text(job["profile_B"]),
        delta_analysis=d.get("analysis", ""),
        key_differences=" | ".join(d.get("key_differences", [])),
    )


# ══════════════════════════════════════════════════════════════════════════════
# JSON PARSING (same robust cascade approach as delta_analyzer._extract_json_block)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_json_block(raw: str) -> Optional[dict]:
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

    if stack:
        closings = ''.join(']' if b == '[' else '}' for b in reversed(stack))
        base = content.rstrip()
        candidates = [
            base,
            re.sub(r',?\s*"[^"]*$', '', base),
            re.sub(r',\s*$', '', base),
            re.sub(r',?\s*"[^"]*$', '', re.sub(r',\s*$', '', base)),
        ]
        for candidate in candidates:
            for suffix in [closings, closings.rstrip('}') + '}']:
                try:
                    return json.loads(candidate + suffix)
                except Exception:
                    pass
    return None


def _parse_card(raw: str, job: Dict) -> Optional[EvolutionCard]:
    data = _extract_json_block(raw)
    if not data:
        log.warning(f"Could not parse evolution card for '{job['topic']}' — skipping.")
        return None
    try:
        return EvolutionCard(
            feature_name=data.get("feature_name", job["topic"]),
            foundation=data.get("foundation", ""),
            value_added=(data.get("value_added") or [])[:5],
            narrative=data.get("narrative", ""),
            change_type=job["delta"]["change_type"],
            builds_on_prior=True,
        )
    except Exception:
        log.warning(f"Could not construct EvolutionCard for '{job['topic']}' — skipping.")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# REPORT RENDERING
# ══════════════════════════════════════════════════════════════════════════════

def _render_card(idx: int, job: Dict, card: EvolutionCard) -> List[str]:
    md = []
    md.append(f"### {idx}. {card.feature_name}")
    md.append(f"**Location:** {job['parent']} → {job['sub']}  ")
    md.append(f"**Appears in:** {job['vA']} → {job['vB']}  ")
    md.append(f"**Change type:** {card.change_type}\n")
    md.append(f"**Foundation ({job['vA']}):** {card.foundation}\n")
    md.append(f"**Value added ({job['vB']}):**")
    for v in card.value_added:
        md.append(f"- {v}")
    md.append("")
    md.append(f"**Narrative:** {card.narrative}\n")
    md.append(f"*Traceability: Chunk `{job['id_A']}` ({job['vA']}) vs Chunk `{job['id_B']}` ({job['vB']})*")
    md.append("\n---\n")
    return md


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_evolution_analysis():
    log.info("Loading delta jobs cache...")
    if not DELTA_JOBS_CACHE_PATH.exists():
        log.error(f"Delta jobs cache not found: {DELTA_JOBS_CACHE_PATH} — run delta_analyzer.py first.")
        return

    with open(DELTA_JOBS_CACHE_PATH, "r", encoding="utf-8") as f:
        all_jobs = json.load(f)

    jobs = [
        j for j in all_jobs
        if j.get("delta") and j["delta"].get("change_type") in CONSTRUCTIVE_CHANGE_TYPES
        and j.get("profile_A") and j.get("profile_B")
    ]
    log.info(f"{len(jobs)}/{len(all_jobs)} jobs are constructive change types — building evolution cards.")
    if not jobs:
        log.warning("No constructive-change jobs found — nothing to do.")
        return

    analyst_role = "a healthcare IT analyst"
    try:
        import context_profiler
        profiles = context_profiler.get_all_profiles()
        if profiles:
            first_profile = next(iter(profiles.values()))
            analyst_role = first_profile.get("analyst_role", analyst_role)
    except ImportError:
        pass

    prompts = [_build_prompt(job, analyst_role) for job in jobs]
    raw_outputs = llm_client.generate_batch(
        prompts, max_tokens=500, desc="Evolution synthesis", stop=["```\n"], enable_thinking=False
    )

    cards: List[EvolutionCard] = []
    kept_jobs: List[Dict] = []
    for job, raw in zip(jobs, raw_outputs):
        card = _parse_card(raw, job)
        if card:
            cards.append(card)
            kept_jobs.append(job)

    log.info(f"Built {len(cards)} evolution cards.")

    # ── Markdown report ────────────────────────────────────────────────────────
    md: List[str] = []
    global_vA = jobs[0]["vA"] if jobs else "Older"
    global_vB = jobs[0]["vB"] if jobs else "Newer"
    md.append(f"# {global_vA} → {global_vB} — Feature Evolution Report\n")
    md.append(
        "Constructive value-add narratives, synthesized from delta_analyzer.py's "
        "already-extracted profiles and classified deltas. Only constructive change "
        "types are included (Minor Enhancement, New Requirement, Workflow Automation, "
        "Bug Fix) — contradictions and deprecations are intentionally excluded here; "
        "see version_delta_report.md for those.\n"
    )
    md.append("## Summary\n")
    md.append("| # | Feature | Change Type |")
    md.append("|---|---------|-------------|")
    for idx, (job, card) in enumerate(zip(kept_jobs, cards), 1):
        md.append(f"| {idx} | **{card.feature_name}** | {card.change_type} |")
    md.append("\n---\n")
    md.append("## Feature Knowledge Cards\n")
    for idx, (job, card) in enumerate(zip(kept_jobs, cards), 1):
        md.extend(_render_card(idx, job, card))

    with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    log.info(f"✅ Evolution report saved → {REPORT_MD_PATH}")

    # ── Cache for downstream ingestion ────────────────────────────────────────
    cache_out = []
    for job, card in zip(kept_jobs, cards):
        cache_out.append({
            "parent": job["parent"],
            "sub":    job["sub"],
            "topic":  job["topic"],
            "vA":     job["vA"],
            "vB":     job["vB"],
            "id_A":   job["id_A"],
            "id_B":   job["id_B"],
            "card":   card.model_dump(),
        })
    with open(EVOLUTION_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache_out, f, indent=2, ensure_ascii=False)
    log.info(f"Evolution cards cache saved → {EVOLUTION_CACHE_PATH}")


if __name__ == "__main__":
    run_evolution_analysis()
