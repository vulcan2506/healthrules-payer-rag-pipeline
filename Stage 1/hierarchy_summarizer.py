"""
hierarchy_summarizer.py
────────────────────────
Rolls up topic-level summaries (topic_summarizer.py output where available,
the original summarized_description otherwise) into sub-category and
parent-level GROUP PROFILES — structured, multi-field extraction in the
same spirit as delta_analyzer.py's BehavioralProfile, not a single soft
prose paragraph.

Each level is synthesized ONLY from its direct children's own already-
distinct content — never flattened across the whole subtree — so a
sub-category with 20 unrelated topics reads as an overview PLUS named
distinct features, not one blended paragraph. Single-pass (not delta_
analyzer's holistic+gap-fill two-pass): that two-pass exists to recover
facts missed on a first read of long RAW text; here the input is already-
condensed topic summaries, so there's no raw-text gap to fill.

Two different consumers need two different amounts of this text:
  - The bi-encoder embedding (paraphrase-multilingual-MiniLM-L12-v2,
    max_seq_length=128 tokens) wants a short, coherent, natural-language
    overview — verified empirically: front-loading a raw feature-name list
    to fit more terms in the truncation window measurably HURT relevance
    (Q9 -8.03→-11.32) versus a clean prose sentence. Bi-encoders embed
    semantic gestalt, not term density.
  - The cross-encoder reranker (ms-marco-MiniLM-L-6-v2, max_seq_length=512
    tokens) and downstream answer-generation both read whatever text is
    STORED in ChromaDB, independent of what was embedded — so the full
    granular profile (named features, specific facts) belongs there, not
    in the embedding text.

So each node gets two fields:
  - sub["sub_category_overview"] / parent["parent_category_overview"]
      Short prose — this is what gets embedded.
  - sub["sub_category_detail"] / parent["parent_category_detail"]
      Overview + bulleted key_features + notable_details — this is what
      gets stored/reranked/answered from. See retrieval_layer/
      chroma_store.py's ingest_hierarchy_summaries for the embed-vs-store
      split (passes precomputed embeddings so the two can differ).

Writes:
  - data/output/hierarchy_summaries/<parent_slug>.md — delta-report-style:
    a summary table of sub-categories, then a detailed breakdown with the
    structured profile AND every topic listed underneath its sub-category
    (not blended).
  - enterprise_nested_topics.json patched in place with the four fields
    above. Patched, not regenerated — same no-reclustering discipline as
    topic_summarizer.py.
"""

import json
import logging
from typing import Dict, List

from pydantic import BaseModel, Field

import config
import llm_client
from delta_analyzer import _extract_json_block
from topic_summarizer import _get_domain_profile, _slugify

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

NESTED_PATH = config.NESTED_OUTPUT_PATH
OUT_DIR     = config.OUTPUT_DIR / "hierarchy_summaries"

SUB_CHAR_BUDGET     = 6000   # topic summaries fed into one sub-category profile prompt
PARENT_CHAR_BUDGET  = 6000   # sub-category detail text fed into one parent profile prompt
PROFILE_MAX_TOKENS  = 650    # room for overview + up to 15 key_features + up to 8 notable_details


class GroupProfile(BaseModel):
    overview:        str       = Field(description="2-3 sentence synthesis of the connecting theme across this group's features.")
    key_features:    List[str] = Field(description="Every distinct feature/capability in this group, named specifically with its core behavior in one line each. Do not merge distinct features together. Max 15.")
    notable_details: List[str] = Field(description="Specific config properties, requirements, or facts worth surfacing that recur across this group. Max 8.")


def _parse_group_profile(raw: str) -> GroupProfile:
    data = _extract_json_block(raw)
    if data:
        try:
            return GroupProfile(**data)
        except Exception:
            pass
    lines = [ln.lstrip("-•n ").strip() for ln in raw.splitlines() if ln.strip()]
    return GroupProfile(
        overview=lines[0] if lines else "Could not extract",
        key_features=lines[1:8],
        notable_details=[],
    )


def _render_group_profile(p: GroupProfile) -> List[str]:
    lines = [p.overview, ""]
    if p.key_features:
        lines.append("**Key Features:**")
        lines.extend(f"- {f}" for f in p.key_features)
        lines.append("")
    if p.notable_details:
        lines.append("**Notable Details:**")
        lines.extend(f"- {d}" for d in p.notable_details)
        lines.append("")
    return lines


def _detail_text(p: GroupProfile) -> str:
    return "\n".join(_render_group_profile(p)).strip()


def _fit_budget(items: List[str], budget: int) -> List[str]:
    out, total = [], 0
    for item in items:
        add = len(item) + 2
        if out and total + add > budget:
            break
        out.append(item)
        total += add
    return out


_GROUP_PROMPT = """\
You are {analyst_role}. Below are {child_kind} grouped \
under "{group_name}".

{numbered}

Output ONLY a single JSON object — nothing before or after it — with this exact \
structure, filled in with real values:

{{
  "overview": "<2-3 sentence synthesis of the connecting theme across these features>",
  "key_features": [
    "<distinct feature 1, named specifically, with its core behavior in one line>",
    "<distinct feature 2 — max 15 total, do not merge distinct features together>"
  ],
  "notable_details": [
    "<specific config property, requirement, or fact worth surfacing — max 8 total>"
  ]
}}

Complete the closing braces — never leave the JSON unfinished. If a list has no \
items write []."""


def _group_prompt(group_name: str, child_kind: str, lines: List[str], analyst_role: str) -> str:
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(lines))
    return _GROUP_PROMPT.format(group_name=group_name, child_kind=child_kind, numbered=numbered, analyst_role=analyst_role)


def run_hierarchy_summarization():
    with open(NESTED_PATH, encoding="utf-8") as f:
        taxonomy_data = json.load(f)

    # Domain-aware role (see context_profiler.py) — same lookup topic_summarizer.py
    # uses, reused here instead of the hardcoded "healthcare IT documentation
    # specialist" this file previously had.
    analyst_role, _doc_purpose = _get_domain_profile(taxonomy_data)

    # ── Pass 1: sub-category group profiles, from their own topics only ──
    sub_jobs: List[Dict] = []
    for parent in taxonomy_data["taxonomy"]:
        for sub in parent["sub_categories"]:
            topic_lines = []
            for t in sub.get("topics", []):
                label = t.get("master_label", "")
                summ  = t.get("summarized_description", "") or t.get("description", "")
                if label and summ:
                    topic_lines.append(f"{label}: {summ}")
            topic_lines = _fit_budget(topic_lines, SUB_CHAR_BUDGET)
            if not topic_lines:
                continue
            sub_jobs.append({
                "sub_ref": sub,
                "prompt": _group_prompt(sub.get("sub_category_name", ""), "summaries of distinct topics", topic_lines, analyst_role),
            })

    log.info(f"Synthesizing {len(sub_jobs)} sub-category profiles...")
    responses = llm_client.generate_batch(
        [j["prompt"] for j in sub_jobs], max_tokens=PROFILE_MAX_TOKENS,
        desc="Sub-category profiles", stop=["```\n"], enable_thinking=False,
    )
    for job, raw in zip(sub_jobs, responses):
        profile = _parse_group_profile(raw)
        job["sub_ref"]["sub_category_overview"] = profile.overview
        job["sub_ref"]["sub_category_detail"]   = _detail_text(profile)

    # ── Pass 2: parent group profiles, from their sub-categories' DETAIL text
    #            (the granular key_features/notable_details just built, not
    #            just the short overview) — so granularity survives the roll-up
    #            instead of being re-compressed away one level up. ──
    parent_jobs: List[Dict] = []
    for parent in taxonomy_data["taxonomy"]:
        sub_lines = []
        for sub in parent["sub_categories"]:
            name   = sub.get("sub_category_name", "")
            detail = sub.get("sub_category_detail", "")
            if name and detail:
                sub_lines.append(f"{name}:\n{detail}")
        sub_lines = _fit_budget(sub_lines, PARENT_CHAR_BUDGET)
        if not sub_lines:
            continue
        parent_jobs.append({
            "parent_ref": parent,
            "prompt": _group_prompt(parent.get("parent_category_name", ""), "detailed profiles of sub-categories", sub_lines, analyst_role),
        })

    log.info(f"Synthesizing {len(parent_jobs)} parent profiles...")
    responses = llm_client.generate_batch(
        [j["prompt"] for j in parent_jobs], max_tokens=PROFILE_MAX_TOKENS,
        desc="Parent profiles", stop=["```\n"], enable_thinking=False,
    )
    for job, raw in zip(parent_jobs, responses):
        profile = _parse_group_profile(raw)
        job["parent_ref"]["parent_category_overview"] = profile.overview
        job["parent_ref"]["parent_category_detail"]   = _detail_text(profile)

    # ── Pass 3: render markdown per parent — profile + full topic breakdown ──
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_rendered = 0
    for parent in taxonomy_data["taxonomy"]:
        pname  = parent.get("parent_category_name", "")
        pdetail = parent.get("parent_category_detail", "")
        if not pdetail:
            continue

        md = [f"# {pname}", "", pdetail, "", "## Sub-Categories", "",
              "| Sub-Category | # Topics | Overview |", "|---|---|---|"]
        for sub in parent["sub_categories"]:
            sname = sub.get("sub_category_name", "")
            n     = len(sub.get("topics", []))
            so    = sub.get("sub_category_overview", "").replace("|", "/")
            md.append(f"| {sname} | {n} | {so[:150]} |")

        md += ["", "## Detailed Breakdown", ""]
        for sub in parent["sub_categories"]:
            sname  = sub.get("sub_category_name", "")
            sdetail = sub.get("sub_category_detail", "")
            md += [f"### {sname}", "", sdetail, "", "**Topics in this group:**", ""]
            for t in sub.get("topics", []):
                label = t.get("master_label", "")
                summ  = t.get("summarized_description", "")
                md.append(f"- **{label}**: {summ}")
            md.append("")

        (OUT_DIR / f"{_slugify(pname)}.md").write_text("\n".join(md), encoding="utf-8")
        n_rendered += 1

    with open(NESTED_PATH, "w", encoding="utf-8") as f:
        json.dump(taxonomy_data, f, indent=4)

    log.info(f"Patched {len(sub_jobs)} sub-category profiles, {len(parent_jobs)} parent profiles "
             f"in {NESTED_PATH}")
    log.info(f"Rendered {n_rendered} parent markdown reports → {OUT_DIR}")


if __name__ == "__main__":
    run_hierarchy_summarization()
