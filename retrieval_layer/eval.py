"""
eval.py
───────
Evaluation harness for the retrieval layer.

Runs a golden query set and produces a report covering:
  • Intent classification accuracy
  • Retrieval score distribution (top / mean per query)
  • Keyword hit rate (expected terms found in retrieved text)
  • Version conflict detection accuracy
  • Per-query latency

Comparison mode (--compare) runs BOTH the full pipeline and traditional flat RAG
on every query and shows a side-by-side table — useful for client demos.

Usage:
    python eval.py              # run all golden queries, print report
    python eval.py --compare    # pipeline vs naive RAG side-by-side
    python eval.py --naive      # run naive RAG only (baseline)
    python eval.py --export     # also save RAGAS-ready JSON → eval_export.json
    python eval.py --verbose    # show chunk text previews per query
    python eval.py --query "How does claim adjudication work?"  # single ad-hoc query

RAGAS integration (after pip install ragas):
    from ragas import evaluate
    from ragas.metrics import context_precision, context_recall
    from datasets import Dataset
    import json
    data = json.load(open("eval_export.json"))
    ds   = Dataset.from_list(data["ragas"])
    # Fill in "ground_truth" fields first for precision/recall.
    # Add "answer" field (LLM output) for faithfulness + answer_relevancy.
    evaluate(ds, metrics=[context_precision, context_recall])
"""

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import retriever

logging.basicConfig(level=logging.WARNING, format="%(levelname)-8s %(message)s")

# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class GoldenQuery:
    query: str
    expected_intent: str                              # specific | intelligence | delta
    expected_keywords: List[str] = field(default_factory=list)  # hit if ANY appear in retrieved text
    expect_conflict: bool = False                     # True = multi-version conflict should fire
    note: str = ""


@dataclass
class QueryResult:
    query: str
    expected_intent: str
    actual_intent: str
    intent_correct: bool
    top_score: float
    mean_score: float
    keyword_hit: bool
    conflict_detected: bool
    conflict_expected: bool
    latency_s: float
    chunks: list = field(default_factory=list)        # kept for RAGAS export
    expected_keywords: List[str] = field(default_factory=list)
    note: str = ""


# ── Golden query set ───────────────────────────────────────────────────────────
#
# Intent routing tests need no document knowledge — they validate classifier logic.
# Content tests use expected_keywords; add section-specific terms you know appear
# in the documents (e.g. exact feature names from the TOC).
# Conflict tests mark expect_conflict=True for sections known to exist in both versions.

GOLDEN_QUERIES: List[GoldenQuery] = [

    # ── Specific intent ────────────────────────────────────────────────────────
    GoldenQuery(
        query="How does claim adjudication work?",
        expected_intent="specific",
        expected_keywords=["claim", "adjudication"],
        note="Core feature — HNSW path",
    ),
    GoldenQuery(
        query="What are the configuration options for prior authorization?",
        expected_intent="specific",
        expected_keywords=["prior authorization", "configuration"],
        note="Config query — specific path",
    ),
    GoldenQuery(
        query="How do I set up coordination of benefits?",
        expected_intent="specific",
        expected_keywords=["coordination", "benefit"],
        note="Setup/how-to — specific path",
    ),
    GoldenQuery(
        query="What is the eligibility verification process?",
        expected_intent="specific",
        expected_keywords=["eligibility"],
        note="Process query — specific path",
    ),
    GoldenQuery(
        query="How are remittance advices generated?",
        expected_intent="specific",
        expected_keywords=["remittance"],
        note="Output artifact query — specific path",
    ),

    # ── Intelligence intent ────────────────────────────────────────────────────
    GoldenQuery(
        query="Give me an overview of what this document covers",
        expected_intent="intelligence",
        note="Overview → intelligence",
    ),
    GoldenQuery(
        query="What topics and areas does this release address at a high level?",
        expected_intent="intelligence",
        note="High-level survey → intelligence",
    ),
    GoldenQuery(
        query="Summarize the main categories of features in this release",
        expected_intent="intelligence",
        note="Summary request → intelligence",
    ),
    GoldenQuery(
        query="List all the domains covered in the documentation",
        expected_intent="intelligence",
        note="'list all' trigger → intelligence",
    ),
    GoldenQuery(
        query="What broadly does this system handle?",
        expected_intent="intelligence",
        note="'broadly' trigger → intelligence",
    ),

    # ── Delta intent ───────────────────────────────────────────────────────────
    GoldenQuery(
        query="What changed between versions?",
        expected_intent="delta",
        note="Generic delta trigger",
    ),
    GoldenQuery(
        query="What new features were added in the latest release?",
        expected_intent="delta",
        note="'added' keyword → delta",
    ),
    GoldenQuery(
        query="What was removed or deprecated?",
        expected_intent="delta",
        note="'removed' keyword → delta",
    ),
    GoldenQuery(
        query="Compare the claim processing rules across versions",
        expected_intent="delta",
        expected_keywords=["claim"],
        note="Explicit compare → delta",
    ),
    GoldenQuery(
        query="What improvements were made versus the previous release?",
        expected_intent="delta",
        note="'improvement over' → delta",
    ),

    # ── Version conflict detection ─────────────────────────────────────────────
    # Sections that likely exist in BOTH versions of the document.
    # Set expect_conflict=True; if conflict fires the eval counts it as a pass.
    # Add more once you know which sections appear across versions.
    GoldenQuery(
        query="How does the system handle duplicate claims?",
        expected_intent="specific",
        expected_keywords=["duplicate", "claim"],
        expect_conflict=True,
        note="Likely in both versions — conflict + delta injection expected",
    ),
    GoldenQuery(
        query="What are the rules for claim editing?",
        expected_intent="specific",
        expected_keywords=["claim", "edit"],
        expect_conflict=True,
        note="Claim editing — shared section across versions",
    ),

    # ── Edge cases ─────────────────────────────────────────────────────────────
    GoldenQuery(
        query="xyzzy nonexistent feature that absolutely does not exist anywhere",
        expected_intent="specific",
        note="Garbage query — expect very low scores",
    ),
    GoldenQuery(
        query="list all",
        expected_intent="intelligence",
        note="Minimal query — 'list all' → intelligence",
    ),

    # ── Relationship queries (cross-domain / grandparent-family) ───────────────
    # Exercise the confidence ladder's Tier A (expand_siblings/expand_cross_parent)
    # and Tier B (query_relationships) — none of the queries above are shaped to
    # trigger either tier. Written against the actual confirmed grandparent
    # family in parent_relationship_clusters.json ("Healthcare Claims &
    # Financial Operations", 13 members), plus a couple of genuine cross-family
    # (IT ops vs healthcare) phrasings.
    GoldenQuery(
        query="How does claims processing relate to financial operations and billing?",
        expected_intent="specific",
        expected_keywords=["claim", "financial"],
        note="Relationship query — within confirmed grandparent family",
    ),
    GoldenQuery(
        query="What connects healthcare data governance to claims transaction integrity?",
        expected_intent="specific",
        expected_keywords=["data", "claim"],
        note="Relationship query — within confirmed grandparent family",
    ),
    GoldenQuery(
        query="How do system configuration changes tie into claims adjudication accuracy?",
        expected_intent="specific",
        expected_keywords=["configuration", "claim"],
        note="Relationship query — within confirmed grandparent family",
    ),
    GoldenQuery(
        query="What is the relationship between Medicaid billing operations and healthcare payer configuration?",
        expected_intent="specific",
        expected_keywords=["medicaid", "payer"],
        note="Relationship query — within confirmed grandparent family",
    ),
    GoldenQuery(
        query="How does identity management interact with financial security controls?",
        expected_intent="specific",
        expected_keywords=["identity", "security"],
        note="Relationship query — cross-family (identity vs financial)",
    ),
    GoldenQuery(
        query="What links healthcare data architecture to claims interoperability?",
        expected_intent="specific",
        expected_keywords=["data", "claim"],
        note="Relationship query — within confirmed grandparent family",
    ),
    GoldenQuery(
        query="How does general IT operations connect to healthcare system configuration?",
        expected_intent="specific",
        expected_keywords=["configuration"],
        note="Relationship query — cross-family (IT ops vs healthcare config)",
    ),
    GoldenQuery(
        query="What is the connection between duplicate claim handling and data integrity governance?",
        expected_intent="specific",
        expected_keywords=["duplicate", "claim"],
        note="Relationship query — within confirmed grandparent family",
    ),
    GoldenQuery(
        query="How do claim editing rules relate to financial operations security?",
        expected_intent="specific",
        expected_keywords=["claim", "edit"],
        note="Relationship query — within confirmed grandparent family",
    ),

    # ── Evolution / value-add queries (evolution_analyzer.py) ──────────────────
    # Grounded in real cards from evolution_cards_cache.json. Queries that name
    # a version token route to "delta" intent (query_delta + query_evolution
    # merged in _retrieve_delta); queries phrased without one route to
    # "specific" intent, where _is_evolution_query() gates the confidence-
    # ladder's evolution rung as a low-confidence fallback.
    GoldenQuery(
        query="How has the Accumulator Scheduler and Recalculation Engine evolved since v25.2?",
        expected_intent="delta",
        expected_keywords=["accumulator", "scheduler"],
        note="Evolution query — version token routes via delta+evolution merge",
    ),
    GoldenQuery(
        query="What value did v26.1 add to NDC-based matching for unlisted HCPCS codes?",
        expected_intent="delta",
        expected_keywords=["ndc", "hcpcs"],
        note="Evolution query — version token routes via delta+evolution merge",
    ),
    GoldenQuery(
        query="What is the value-add of the Medicaid Billing Adjustment Engine in the newer release?",
        expected_intent="specific",
        expected_keywords=["medicaid", "billing"],
        note="Evolution query — no version token, exercises the confidence-ladder evolution rung",
    ),
    GoldenQuery(
        query="How did v25.2's foundation for the New Reinsurance Workflow Tab carry forward into v26.1?",
        expected_intent="delta",
        expected_keywords=["reinsurance", "workflow"],
        note="Evolution query — version token routes via delta+evolution merge",
    ),
    GoldenQuery(
        query="What foundation did the Secondary Editor Configuration Parameters feature build on, and what value did the update add?",
        expected_intent="specific",
        expected_keywords=["secondary", "editor"],
        note="Evolution query — no version token, exercises the confidence-ladder evolution rung",
    ),
    GoldenQuery(
        query="How has Claims Adjudication Performance Optimization evolved in the latest release?",
        expected_intent="specific",
        expected_keywords=["claims", "adjudication"],
        note="Evolution query — no version token, exercises the confidence-ladder evolution rung",
    ),

    # ── Evolution benefit probe — 19 new queries (2026-07-05) ──────────────────
    # Before/after comparison (query_evolution/query_cross_corpus monkeypatched
    # to return nothing) measured which of these are GENUINE wins vs correct
    # no-ops vs correctly-inert negative controls. The evolution card only ever
    # helps when the plain router+rerank/delta path scores below
    # CONFIDENCE_THRESHOLD on its own; when it already scores well, the ladder
    # correctly never engages and nothing changes — a "+0.0" delta here is
    # success, not evidence the feature is inert. See [[evolution-analyzer]]
    # memory for the full before/after numbers and reasoning.
    GoldenQuery(
        query="How did v26.1 build on the reinsurance contract management workflow established in v25.2?",
        expected_intent="delta",
        expected_keywords=["reinsurance", "contract"],
        note="Evolution probe — no delta (8.867 both ways): raw delta finding already outscores the card",
    ),
    GoldenQuery(
        query="How did v25.2's HealthRules Payer Hub Integration Configuration carry forward into v26.1?",
        expected_intent="delta",
        expected_keywords=["hub", "integration"],
        note="Evolution probe — verified real win, top score 6.72->7.035 without/with",
    ),
    GoldenQuery(
        query="What value did the new Reinsurance Workflow Tab implementation add?",
        expected_intent="specific",
        expected_keywords=["reinsurance", "workflow"],
        note="Evolution probe — no delta (2.668 both ways): initial confidence already above threshold, ladder correctly doesn't fire",
    ),
    GoldenQuery(
        query="How has PCP auto-assignment and waiting period rules evolved?",
        expected_intent="specific",
        expected_keywords=["pcp", "waiting period"],
        note="Evolution probe — no delta (5.261 both ways): initial confidence already above threshold",
    ),
    GoldenQuery(
        query="What value did the Workbasket Rule Optimization Feature Control add in the update?",
        expected_intent="specific",
        expected_keywords=["workbasket", "optimization"],
        note="Evolution probe — no delta (3.435 both ways): initial confidence already above threshold",
    ),
    GoldenQuery(
        query="What is the value-add of the Multi-Plan State Confirmation Display Enhancement?",
        expected_intent="specific",
        expected_keywords=["multi-plan", "confirmation"],
        note="Evolution probe — no delta (1.591 both ways): initial confidence already above threshold",
    ),
    GoldenQuery(
        query="How has the Supplier Invoice Data Warehouse Migration evolved since the older release?",
        expected_intent="delta",
        expected_keywords=["supplier", "invoice"],
        note="Evolution probe — no delta (5.586 both ways): raw delta finding already outscores the card",
    ),
    GoldenQuery(
        query="How did v26.1 build on the New York Medicaid Billing Configuration from v25.2?",
        expected_intent="delta",
        expected_keywords=["medicaid", "billing"],
        note="Evolution probe — no delta (7.728 both ways): raw delta finding already outscores the card",
    ),
    GoldenQuery(
        query="What value did the Affiliate Transfer Payment Logic gain in the update?",
        expected_intent="specific",
        expected_keywords=["affiliate", "transfer"],
        note="Evolution probe — no delta (2.968 both ways): initial confidence already above threshold",
    ),
    GoldenQuery(
        query="How has the Claims Scheduler Data Warehouse Migration evolved?",
        expected_intent="delta",
        expected_keywords=["claims scheduler", "warehouse"],
        note="Evolution probe — verified real win, top score 5.585->6.105 without/with",
    ),
    GoldenQuery(
        query="What foundation did the Member Responsibility Accumulator Configuration build on, and what value did it add?",
        expected_intent="specific",
        expected_keywords=["member responsibility", "accumulator"],
        note="Evolution probe — verified real win, top score -0.687->6.987 without/with (ladder rescue)",
    ),
    GoldenQuery(
        query="How did v25.2's PCP Auto-Assignment and Suggestion Logic carry forward into v26.1?",
        expected_intent="delta",
        expected_keywords=["pcp", "suggestion"],
        note="Evolution probe — no delta (6.592 both ways): raw delta finding already outscores the card",
    ),
    GoldenQuery(
        query="What value-add came from the Split Claim Line UDT Display Enhancement?",
        expected_intent="specific",
        expected_keywords=["split claim", "udt"],
        note="Evolution probe — no delta (5.205 both ways): initial confidence already above threshold",
    ),
    GoldenQuery(
        query="How has the askHE Chatbot Interactive Content feature evolved since the older release?",
        expected_intent="specific",
        expected_keywords=["askhe", "chatbot"],
        note="Evolution probe — no delta (3.271 both ways): initial confidence already above threshold",
    ),
    GoldenQuery(
        query="What did v26.1 build on top of the HealthRules Payer Compatibility Matrix from v25.2?",
        expected_intent="delta",
        expected_keywords=["compatibility", "matrix"],
        note="Evolution probe — verified real win, top score 6.43->7.647 without/with",
    ),

    # ── Negative controls — confirm the evolution/cross-corpus layers don't
    # leak into queries they shouldn't touch ──────────────────────────────────
    GoldenQuery(
        query="What value did v26.1 add to the Oracle WebLogic to Apache Tomcat migration?",
        expected_intent="delta",
        expected_keywords=["weblogic", "tomcat"],
        note="Negative control — Deprecation change_type, excluded from evolution cards by design, confirmed zero effect",
    ),
    GoldenQuery(
        query="How did v26.1 build on the HealthRules Additional Fields Runtime Configuration from v25.2?",
        expected_intent="delta",
        expected_keywords=["additional fields", "runtime"],
        note="Negative control — Direct Contradiction change_type, excluded from evolution cards by design, confirmed zero effect",
    ),
    GoldenQuery(
        query="What are the system requirements for running HealthRules Payer?",
        expected_intent="specific",
        expected_keywords=["healthrules", "payer"],
        note="Negative control — generic query, not tied to any specific evolution card, confirmed zero effect",
    ),
    GoldenQuery(
        query="How does connector authentication affect payer configuration?",
        expected_intent="specific",
        note="Negative control — triggers _is_cross_corpus_query() but cross_corpus_relationship_clusters.json doesn't exist yet (no 2nd corpus); confirmed zero effect, purely a no-op until one lands",
    ),
]


# ── Runner ─────────────────────────────────────────────────────────────────────

def _scores(chunks):
    return [c.get("rerank_score", c.get("score", 0.0)) for c in chunks]

def _kw_hit(chunks, keywords):
    if not keywords:
        return True
    combined = " ".join(c.get("text", "") for c in chunks).lower()
    return any(kw.lower() in combined for kw in keywords)

def _to_chunk_list(chunks):
    return [
        {"text": c.get("text", ""), "source": c.get("source_doc", ""),
         "section": c.get("section_header", ""), "score": c.get("rerank_score", c.get("score", 0))}
        for c in chunks
    ]

def _run_query(gq: GoldenQuery) -> QueryResult:
    t0     = time.perf_counter()
    result = retriever.retrieve(gq.query, verbose=False)
    t1     = time.perf_counter()
    chunks = result["chunks"]
    sc     = _scores(chunks)
    return QueryResult(
        query             = gq.query,
        expected_intent   = gq.expected_intent,
        actual_intent     = result["intent"],
        intent_correct    = result["intent"] == gq.expected_intent,
        top_score         = max(sc) if sc else 0.0,
        mean_score        = sum(sc) / len(sc) if sc else 0.0,
        keyword_hit       = _kw_hit(chunks, gq.expected_keywords),
        conflict_detected = result["versioned"],
        conflict_expected = gq.expect_conflict,
        latency_s         = round(t1 - t0, 3),
        chunks            = _to_chunk_list(chunks),
        expected_keywords = gq.expected_keywords,
        note              = gq.note,
    )

def _run_query_naive(gq: GoldenQuery) -> QueryResult:
    t0     = time.perf_counter()
    result = retriever.retrieve_naive(gq.query)
    t1     = time.perf_counter()
    chunks = result["chunks"]
    sc     = _scores(chunks)
    return QueryResult(
        query             = gq.query,
        expected_intent   = "naive",
        actual_intent     = "naive",
        intent_correct    = True,        # not graded for naive
        top_score         = max(sc) if sc else 0.0,
        mean_score        = sum(sc) / len(sc) if sc else 0.0,
        keyword_hit       = _kw_hit(chunks, gq.expected_keywords),
        conflict_detected = False,       # naive never detects conflicts
        conflict_expected = gq.expect_conflict,
        latency_s         = round(t1 - t0, 3),
        chunks            = _to_chunk_list(chunks),
        expected_keywords = gq.expected_keywords,
        note              = gq.note,
    )


def _run_query_merged(gq: GoldenQuery) -> QueryResult:
    t0     = time.perf_counter()
    result = retriever.retrieve_merged(gq.query)
    t1     = time.perf_counter()
    chunks = result["chunks"]
    sc     = _scores(chunks)
    return QueryResult(
        query             = gq.query,
        expected_intent   = gq.expected_intent,
        actual_intent     = result["intent"],
        intent_correct    = result["intent"] == gq.expected_intent,
        top_score         = max(sc) if sc else 0.0,
        mean_score        = sum(sc) / len(sc) if sc else 0.0,
        keyword_hit       = _kw_hit(chunks, gq.expected_keywords),
        conflict_detected = result["versioned"],
        conflict_expected = gq.expect_conflict,
        latency_s         = round(t1 - t0, 3),
        chunks            = _to_chunk_list(chunks),
        expected_keywords = gq.expected_keywords,
        note              = gq.note,
    )


def _run_query_merged_all(gq: GoldenQuery) -> QueryResult:
    t0     = time.perf_counter()
    result = retriever.retrieve_merged_all(gq.query)
    t1     = time.perf_counter()
    chunks = result["chunks"]
    sc     = _scores(chunks)
    return QueryResult(
        query             = gq.query,
        expected_intent   = gq.expected_intent,
        actual_intent     = result["intent"],
        intent_correct    = result["intent"] == gq.expected_intent,
        top_score         = max(sc) if sc else 0.0,
        mean_score        = sum(sc) / len(sc) if sc else 0.0,
        keyword_hit       = _kw_hit(chunks, gq.expected_keywords),
        conflict_detected = result["versioned"],
        conflict_expected = gq.expect_conflict,
        latency_s         = round(t1 - t0, 3),
        chunks            = _to_chunk_list(chunks),
        expected_keywords = gq.expected_keywords,
        note              = gq.note,
    )


def _run_query_best_of_n(gq: GoldenQuery, n: int = 3) -> QueryResult:
    t0     = time.perf_counter()
    result = retriever.retrieve_best_of_n(gq.query, n=n)
    t1     = time.perf_counter()
    chunks = result["chunks"]
    sc     = _scores(chunks)
    return QueryResult(
        query             = gq.query,
        expected_intent   = gq.expected_intent,
        actual_intent     = result["intent"],
        intent_correct    = result["intent"] == gq.expected_intent,
        top_score         = max(sc) if sc else 0.0,
        mean_score        = sum(sc) / len(sc) if sc else 0.0,
        keyword_hit       = _kw_hit(chunks, gq.expected_keywords),
        conflict_detected = result["versioned"],
        conflict_expected = gq.expect_conflict,
        latency_s         = round(t1 - t0, 3),
        chunks            = _to_chunk_list(chunks),
        expected_keywords = gq.expected_keywords,
        note              = gq.note,
    )


def _run_query_overlap(gq: GoldenQuery) -> QueryResult:
    t0     = time.perf_counter()
    result = retriever.retrieve_overlap(gq.query)
    t1     = time.perf_counter()
    chunks = result["chunks"]
    sc     = _scores(chunks)
    return QueryResult(
        query             = gq.query,
        expected_intent   = "overlap",
        actual_intent     = "overlap",
        intent_correct    = True,        # not graded for overlap
        top_score         = max(sc) if sc else 0.0,
        mean_score        = sum(sc) / len(sc) if sc else 0.0,
        keyword_hit       = _kw_hit(chunks, gq.expected_keywords),
        conflict_detected = False,       # traditional RAG has no conflict logic
        conflict_expected = gq.expect_conflict,
        latency_s         = round(t1 - t0, 3),
        chunks            = _to_chunk_list(chunks),
        expected_keywords = gq.expected_keywords,
        note              = gq.note,
    )


# ── Report ─────────────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
DIM    = "\033[2m"


def _tick(ok: bool, neutral: bool = False) -> str:
    if neutral:
        return f"{DIM}–{RESET}"
    return f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"


def _print_report(results: List[QueryResult]):
    total          = len(results)
    intent_ok      = sum(r.intent_correct for r in results)
    kw_results     = [r for r in results if r.expected_keywords]  # only graded if keywords set
    kw_hits        = sum(r.keyword_hit for r in kw_results)
    conflict_cases = [r for r in results if r.conflict_expected]
    conflict_hits  = sum(r.conflict_detected for r in conflict_cases)
    mean_latency   = sum(r.latency_s for r in results) / total if results else 0

    print(f"\n{BOLD}{'═'*72}{RESET}")
    print(f"{BOLD}  RETRIEVAL LAYER EVALUATION  ({total} queries){RESET}")
    print(f"{'═'*72}{RESET}\n")

    hdr = f"  {'#':<3} {'INTENT (exp→got)':<18} {'INT':<4} {'TOP':<6} {'MEAN':<6} {'KW':<4} {'CONF':<5} {'ms':<6}  NOTE"
    print(hdr)
    print(f"  {'─'*3} {'─'*18} {'─'*4} {'─'*6} {'─'*6} {'─'*4} {'─'*5} {'─'*6}  {'─'*35}")

    for i, r in enumerate(results, 1):
        if r.intent_correct:
            intent_label = f"{r.actual_intent:<18}"
        else:
            intent_label = f"{RED}{r.expected_intent}→{r.actual_intent}{RESET}"
            intent_label = f"{intent_label:<18}"

        conf_tick = (
            _tick(r.conflict_detected) if r.conflict_expected
            else _tick(False, neutral=True)
        )
        kw_tick   = _tick(r.keyword_hit) if r.expected_keywords else f"{DIM}–{RESET}"

        note_short = r.note[:38] + ("…" if len(r.note) > 38 else "")
        print(
            f"  {i:<3} {intent_label} {_tick(r.intent_correct):<5} "
            f"{r.top_score:<6.3f} {r.mean_score:<6.3f} {kw_tick:<5} "
            f"{conf_tick:<6} {int(r.latency_s*1000):<6}  {note_short}"
        )

    print(f"\n{BOLD}{'─'*72}{RESET}")
    print(f"{BOLD}  SUMMARY{RESET}")
    print(f"  Intent accuracy    : {intent_ok}/{total}  ({100*intent_ok/total:.0f}%)")
    if kw_results:
        print(f"  Keyword hit rate   : {kw_hits}/{len(kw_results)}  ({100*kw_hits/len(kw_results):.0f}%)  "
              f"{DIM}(only queries with expected_keywords){RESET}")
    if conflict_cases:
        print(f"  Conflict detection : {conflict_hits}/{len(conflict_cases)} expected conflicts triggered")
    scores_all = [r.top_score for r in results if r.top_score > 0]
    if scores_all:
        print(f"  Score range (top)  : {min(scores_all):.3f} – {max(scores_all):.3f}  "
              f"(mean {sum(scores_all)/len(scores_all):.3f})")
    print(f"  Mean latency       : {mean_latency*1000:.0f} ms/query  "
          f"(total {sum(r.latency_s for r in results):.1f}s)")
    print(f"{'═'*72}{RESET}\n")


def _print_verbose(r: QueryResult):
    print(f"  {DIM}Retrieved chunks:{RESET}")
    for j, c in enumerate(r.chunks, 1):
        preview = c["text"][:130].replace("\n", " ")
        print(f"    [{j}] score={c['score']:.3f}  {c['source']}  /  {c['section']}")
        print(f"         {preview}{'…' if len(c['text']) > 130 else ''}")
    print()


# ── Comparison report ──────────────────────────────────────────────────────────

def _print_comparison(
    pipeline_results: List[QueryResult],
    naive_results: List[QueryResult],
    overlap_results: Optional[List[QueryResult]] = None,
    best_of_results: Optional[List[QueryResult]] = None,
    merged_results: Optional[List[QueryResult]] = None,
    merged_all_results: Optional[List[QueryResult]] = None,
):
    """Side-by-side pipeline vs naive RAG vs traditional overlap RAG vs best-of-N vs merged vs merged-all."""
    total    = len(pipeline_results)
    has_ovlp = overlap_results is not None and len(overlap_results) == total
    has_bon  = best_of_results is not None and len(best_of_results) == total
    has_mrg  = merged_results is not None and len(merged_results) == total
    has_mga  = merged_all_results is not None and len(merged_all_results) == total

    p_kw   = sum(r.keyword_hit for r in pipeline_results if r.expected_keywords)
    n_kw   = sum(r.keyword_hit for r in naive_results    if r.expected_keywords)
    o_kw   = sum(r.keyword_hit for r in (overlap_results or []) if r.expected_keywords)
    b_kw   = sum(r.keyword_hit for r in (best_of_results or []) if r.expected_keywords)
    m_kw   = sum(r.keyword_hit for r in (merged_results or [])  if r.expected_keywords)
    g_kw   = sum(r.keyword_hit for r in (merged_all_results or []) if r.expected_keywords)
    kw_tot = sum(1 for r in pipeline_results if r.expected_keywords)

    p_scores = [r.top_score for r in pipeline_results]
    n_scores = [r.top_score for r in naive_results]
    o_scores = [r.top_score for r in (overlap_results or [])]
    b_scores = [r.top_score for r in (best_of_results or [])]
    m_scores = [r.top_score for r in (merged_results or [])]
    g_scores = [r.top_score for r in (merged_all_results or [])]

    p_conf   = sum(r.conflict_detected for r in pipeline_results if r.conflict_expected)
    b_conf   = sum(r.conflict_detected for r in (best_of_results or []) if r.conflict_expected)
    m_conf   = sum(r.conflict_detected for r in (merged_results or [])  if r.conflict_expected)
    g_conf   = sum(r.conflict_detected for r in (merged_all_results or []) if r.conflict_expected)
    conf_tot = sum(1 for r in pipeline_results if r.conflict_expected)

    p_lat = sum(r.latency_s for r in pipeline_results) / total
    n_lat = sum(r.latency_s for r in naive_results)    / total
    o_lat = (sum(r.latency_s for r in overlap_results) / total) if has_ovlp else 0
    b_lat = (sum(r.latency_s for r in best_of_results) / total) if has_bon  else 0
    m_lat = (sum(r.latency_s for r in merged_results)  / total) if has_mrg  else 0
    g_lat = (sum(r.latency_s for r in merged_all_results) / total) if has_mga else 0

    extra_cols = sum([has_bon, has_mrg, has_mga])
    W = 80 + 20 * (int(has_ovlp) + extra_cols)
    active = ["PIPELINE", "NAIVE"]
    if has_ovlp: active.append("TRADITIONAL")
    if has_bon:  active.append("BEST-OF-N")
    if has_mrg:  active.append("MERGED")
    if has_mga:  active.append("MERGED-ALL")
    title = " vs ".join(active)
    print(f"\n{BOLD}{'=' * W}{RESET}")
    print(f"{BOLD}  {title}  --  {total} queries{RESET}")
    print(f"{'=' * W}{RESET}\n")

    # Per-query table header
    q_w  = 30
    col  = f"{'TOP':>6} {'KW':>3} {'ms':>4}"
    sep  = f"{'---':->6} {'---':->3} {'----':->4}"
    hdr1 = f"  {'#':<3} {'QUERY':<{q_w}}  {'-- PIPELINE --':^14}"
    hdr2 = f"  {'':3} {'':>{q_w}}  {col}"
    hr   = f"  {'---':->3} {'---':->30}  {sep}"
    hdr1 += f"  {'-- NAIVE --':^14}"; hdr2 += f"  {col}"; hr += f"  {sep}"
    if has_ovlp:
        hdr1 += f"  {'-- TRADITIONAL --':^14}"; hdr2 += f"  {col}"; hr += f"  {sep}"
    if has_bon:
        hdr1 += f"  {'-- BEST-OF-N --':^14}"; hdr2 += f"  {col}"; hr += f"  {sep}"
    if has_mrg:
        hdr1 += f"  {'-- MERGED --':^14}"; hdr2 += f"  {col}"; hr += f"  {sep}"
    if has_mga:
        hdr1 += f"  {'-- MERGED-ALL --':^14}"; hdr2 += f"  {col}"; hr += f"  {sep}"
    print(hdr1); print(hdr2); print(hr)

    def _kw_sym(r):
        if r.keyword_hit:         return f"{GREEN}v{RESET}"
        if r.expected_keywords:   return f"{RED}x{RESET}"
        return f"{DIM}-{RESET}"

    def _col(r):
        return f"{r.top_score:>6.3f} {_kw_sym(r):<4} {int(r.latency_s*1000):>4}"

    for i, (p, n) in enumerate(zip(pipeline_results, naive_results), 1):
        o        = overlap_results[i - 1] if has_ovlp else None
        b        = best_of_results[i - 1] if has_bon  else None
        m        = merged_results[i - 1]  if has_mrg  else None
        g        = merged_all_results[i - 1] if has_mga else None
        conf_tag = f" {YELLOW}!{RESET}" if p.conflict_detected else ""
        q_short  = p.query[:q_w] + ("..." if len(p.query) > q_w else "")
        line     = f"  {i:<3} {q_short:<{q_w}}  {_col(p)}  {_col(n)}"
        if o: line += f"  {_col(o)}"
        if b: line += f"  {_col(b)}"
        if m: line += f"  {_col(m)}"
        if g: line += f"  {_col(g)}"
        print(line + conf_tag)

    # Summary table
    print(f"\n{BOLD}{'-' * W}{RESET}")
    print(f"{BOLD}  SUMMARY{RESET}\n")

    p_mean = sum(p_scores) / len(p_scores) if p_scores else 0
    n_mean = sum(n_scores) / len(n_scores) if n_scores else 0
    o_mean = sum(o_scores) / len(o_scores) if o_scores else 0
    b_mean = sum(b_scores) / len(b_scores) if b_scores else 0
    m_mean = sum(m_scores) / len(m_scores) if m_scores else 0
    g_mean = sum(g_scores) / len(g_scores) if g_scores else 0

    col_w = 16
    header = f"  {'Metric':<30}" + "".join(f" {m:>{col_w}}" for m in active) + "   Winner"
    print(header)
    print(f"  {'-'*30}" + "".join(f" {'-'*col_w}" for _ in active) + "   " + "-"*12)

    def _row(label, vals, winner_i=None):
        row = f"  {label:<30}" + "".join(f" {str(v):>{col_w}}" for v in vals)
        if winner_i is not None:
            row += f"   {active[winner_i]}"
        print(row)

    # conflict detection / delta injection route through the real pipeline
    # conflict-detection logic for PIPELINE, BEST-OF-N, MERGED, and MERGED-ALL
    # (all four call the real retrieve()/pipeline path) — NAIVE and
    # TRADITIONAL never do.
    conflict_capable = {"PIPELINE": True, "NAIVE": False, "TRADITIONAL": False,
                        "BEST-OF-N": True, "MERGED": True, "MERGED-ALL": True}

    means_list = [p_mean, n_mean] + ([o_mean] if has_ovlp else []) + ([b_mean] if has_bon else []) + ([m_mean] if has_mrg else []) + ([g_mean] if has_mga else [])
    means_str  = [f"{v:.3f}" for v in means_list]
    _row("Mean top rerank score", means_str, means_list.index(max(means_list)))

    if kw_tot:
        kw_list = [p_kw, n_kw] + ([o_kw] if has_ovlp else []) + ([b_kw] if has_bon else []) + ([m_kw] if has_mrg else []) + ([g_kw] if has_mga else [])
        kw_str  = [f"{k}/{kw_tot} ({100*k//kw_tot}%)" for k in kw_list]
        _row("Keyword hit rate", kw_str, kw_list.index(max(kw_list)))

    if conf_tot:
        conf_vals = {"PIPELINE": p_conf, "BEST-OF-N": b_conf, "MERGED": m_conf, "MERGED-ALL": g_conf}
        conf_str  = [f"{conf_vals[a]}/{conf_tot}" if conflict_capable[a] else "0 (n/a)" for a in active]
        _row("Conflict detection", conf_str, 0)
        inj_str   = ["yes (auto)" if conflict_capable[a] else "no" for a in active]
        _row("Delta injection", inj_str)

    lat_list = [p_lat, n_lat] + ([o_lat] if has_ovlp else []) + ([b_lat] if has_bon else []) + ([m_lat] if has_mrg else []) + ([g_lat] if has_mga else [])
    lat_str  = [f"{v*1000:.0f} ms" for v in lat_list]
    _row("Mean latency", lat_str, lat_list.index(min(lat_list)))

    print(f"\n  {YELLOW}! = version conflict detected + delta injected (pipeline / best-of-N / merged / merged-all only){RESET}")
    print(f"{'=' * W}{RESET}\n")


# ── RAGAS export ───────────────────────────────────────────────────────────────

def _export_ragas(results: List[QueryResult], path: str = "eval_export.json"):
    ragas_rows = [
        {
            "question":    r.query,
            "contexts":    [c["text"] for c in r.chunks],
            "ground_truth": "",   # fill manually for context_precision / context_recall
            # "answer": ""        # add LLM output here for faithfulness / answer_relevancy
        }
        for r in results
    ]
    payload = {
        "_note": (
            "Fill 'ground_truth' for context_precision/recall. "
            "Add 'answer' (LLM output) for faithfulness/answer_relevancy. "
            "Then: from ragas import evaluate; from datasets import Dataset; "
            "evaluate(Dataset.from_list(data['ragas']), metrics=[...])"
        ),
        "ragas": ragas_rows,
        "raw":   [asdict(r) for r in results],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  RAGAS export saved → {path}")
    print(f"  Next: pip install ragas datasets; fill 'ground_truth' fields;\n"
          f"  then call evaluate() with context_precision / context_recall metrics.\n")


# ── Markdown report export ─────────────────────────────────────────────────────

def _export_markdown(
    pipeline_results: List[QueryResult],
    naive_results: Optional[List[QueryResult]] = None,
    overlap_results: Optional[List[QueryResult]] = None,
    merged_results: Optional[List[QueryResult]] = None,
    merged_all_results: Optional[List[QueryResult]] = None,
    path: str = "eval_report.md",
):
    """
    Write a detailed, human-readable markdown evaluation report.
    Shows per-query scores, top retrieved chunks (with full text), and a summary table.
    """
    import datetime
    lines = []
    now   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    total = len(pipeline_results)
    has_naive = naive_results is not None and len(naive_results) == total
    has_ovlp  = overlap_results is not None and len(overlap_results) == total
    has_mrg   = merged_results is not None and len(merged_results) == total
    has_mga   = merged_all_results is not None and len(merged_all_results) == total

    lines.append(f"# Retrieval Layer Evaluation Report")
    lines.append(f"*Generated: {now} — {total} queries*\n")

    # ── Summary table ──────────────────────────────────────────────────────────
    lines.append("## Summary\n")

    p_scores  = [r.top_score for r in pipeline_results]
    p_kw_tot  = sum(1 for r in pipeline_results if r.expected_keywords)
    p_kw_hit  = sum(r.keyword_hit for r in pipeline_results if r.expected_keywords)
    p_conf    = sum(r.conflict_detected for r in pipeline_results if r.conflict_expected)
    conf_tot  = sum(1 for r in pipeline_results if r.conflict_expected)
    p_lat     = sum(r.latency_s for r in pipeline_results) / total

    header_cols = ["Metric", "Pipeline"]
    if has_naive:
        n_scores = [r.top_score for r in naive_results]
        n_kw_hit = sum(r.keyword_hit for r in naive_results if r.expected_keywords)
        n_lat    = sum(r.latency_s for r in naive_results) / total
        header_cols.append("Naive RAG")
    if has_ovlp:
        o_scores = [r.top_score for r in overlap_results]
        o_kw_hit = sum(r.keyword_hit for r in overlap_results if r.expected_keywords)
        o_lat    = sum(r.latency_s for r in overlap_results) / total
        header_cols.append("Traditional RAG")
    if has_mrg:
        m_scores = [r.top_score for r in merged_results]
        m_kw_hit = sum(r.keyword_hit for r in merged_results if r.expected_keywords)
        m_conf   = sum(r.conflict_detected for r in merged_results if r.conflict_expected)
        m_lat    = sum(r.latency_s for r in merged_results) / total
        header_cols.append("Merged (Pipeline+Naive)")
    if has_mga:
        g_scores = [r.top_score for r in merged_all_results]
        g_kw_hit = sum(r.keyword_hit for r in merged_all_results if r.expected_keywords)
        g_conf   = sum(r.conflict_detected for r in merged_all_results if r.conflict_expected)
        g_lat    = sum(r.latency_s for r in merged_all_results) / total
        header_cols.append("Merged-All (Pipeline+Naive+Traditional)")

    sep = ["---"] + ["---:"] * (len(header_cols) - 1)
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("| " + " | ".join(sep) + " |")

    def _row(label, p_val, n_val=None, o_val=None, m_val=None, g_val=None):
        parts = [label, p_val]
        if has_naive:  parts.append(n_val or "—")
        if has_ovlp:   parts.append(o_val or "—")
        if has_mrg:    parts.append(m_val or "—")
        if has_mga:    parts.append(g_val or "—")
        lines.append("| " + " | ".join(str(x) for x in parts) + " |")

    p_mean = sum(p_scores) / len(p_scores) if p_scores else 0
    n_mean = (sum(n_scores) / len(n_scores)) if has_naive and n_scores else 0
    o_mean = (sum(o_scores) / len(o_scores)) if has_ovlp and o_scores else 0
    m_mean = (sum(m_scores) / len(m_scores)) if has_mrg and m_scores else 0
    g_mean = (sum(g_scores) / len(g_scores)) if has_mga and g_scores else 0
    _row("Mean top rerank score",
         f"{p_mean:.3f}", f"{n_mean:.3f}" if has_naive else None, f"{o_mean:.3f}" if has_ovlp else None,
         f"{m_mean:.3f}" if has_mrg else None, f"{g_mean:.3f}" if has_mga else None)
    if p_kw_tot:
        _row(f"Keyword hit rate (/{p_kw_tot})",
             f"{p_kw_hit}/{p_kw_tot} ({100*p_kw_hit//p_kw_tot}%)",
             f"{n_kw_hit}/{p_kw_tot} ({100*n_kw_hit//p_kw_tot}%)" if has_naive else None,
             f"{o_kw_hit}/{p_kw_tot} ({100*o_kw_hit//p_kw_tot}%)" if has_ovlp else None,
             f"{m_kw_hit}/{p_kw_tot} ({100*m_kw_hit//p_kw_tot}%)" if has_mrg else None,
             f"{g_kw_hit}/{p_kw_tot} ({100*g_kw_hit//p_kw_tot}%)" if has_mga else None)
    if conf_tot:
        _row("Version conflict detected",
             f"{p_conf}/{conf_tot}",
             f"0/{conf_tot}" if has_naive else None,
             f"0/{conf_tot}" if has_ovlp else None,
             f"{m_conf}/{conf_tot}" if has_mrg else None,
             f"{g_conf}/{conf_tot}" if has_mga else None)
        _row("Delta auto-injection",
             "✓ yes", "✗ no" if has_naive else None, "✗ no" if has_ovlp else None,
             "✓ yes" if has_mrg else None, "✓ yes" if has_mga else None)
    _row("Mean latency",
         f"{p_lat*1000:.0f} ms",
         f"{n_lat*1000:.0f} ms" if has_naive else None,
         f"{o_lat*1000:.0f} ms" if has_ovlp else None,
         f"{m_lat*1000:.0f} ms" if has_mrg else None,
         f"{g_lat*1000:.0f} ms" if has_mga else None)

    lines.append("")

    # ── Per-query sections ─────────────────────────────────────────────────────
    lines.append("---\n")
    lines.append("## Per-Query Detail\n")

    for i, p in enumerate(pipeline_results):
        n = naive_results[i]   if has_naive else None
        o = overlap_results[i] if has_ovlp  else None
        m = merged_results[i]  if has_mrg   else None
        g = merged_all_results[i] if has_mga else None

        lines.append(f"### Query {i+1}: {p.query}\n")

        # Score comparison mini-table
        cols = ["Method", "Top Score", "Mean Score", "KW Hit", "Conflict", "Latency (ms)"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] + ["---:"] * (len(cols)-1)) + "|")

        def _qrow(label, r, show_conf=False):
            kw  = "✓" if r.keyword_hit else ("✗" if r.expected_keywords else "—")
            cf  = ("⚡ yes" if r.conflict_detected else "no") if show_conf else "—"
            lines.append(
                f"| {label} | {r.top_score:.4f} | {r.mean_score:.4f} | {kw} | {cf} | {int(r.latency_s*1000)} |"
            )

        _qrow("Pipeline",    p, show_conf=True)
        if n: _qrow("Naive RAG",   n)
        if o: _qrow("Traditional", o)
        if m: _qrow("Merged", m, show_conf=True)
        if g: _qrow("Merged-All", g, show_conf=True)
        lines.append("")

        if p.conflict_detected:
            lines.append("> ⚡ **Version conflict detected** — delta report auto-injected into context\n")

        # Expected keywords
        if p.expected_keywords:
            kw_str = ", ".join(f"`{k}`" for k in p.expected_keywords)
            lines.append(f"**Expected keywords:** {kw_str}\n")

        # Retrieved chunks — one subsection per method
        def _chunks_section(label, r, anchor_prefix):
            lines.append(f"#### {label} — Retrieved Chunks\n")
            if not r.chunks:
                lines.append("*No chunks retrieved.*\n")
                return
            for j, c in enumerate(r.chunks[:5], 1):
                src     = c.get("source", c.get("source_doc", ""))
                section = c.get("section", c.get("section_header", ""))
                score   = c.get("score", 0)
                text    = c.get("text", "").strip()
                lines.append(f"**[{j}]** `score={score:.4f}` — *{src}*"
                              + (f" — {section}" if section else ""))
                lines.append("")
                # Full chunk text in a blockquote
                for line in text.splitlines():
                    lines.append(f"> {line}" if line.strip() else ">")
                lines.append("")

        _chunks_section("Pipeline",    p, "p")
        if n: _chunks_section("Naive RAG",   n, "n")
        if o: _chunks_section("Traditional RAG", o, "o")
        if m: _chunks_section("Merged (Pipeline+Naive)", m, "m")
        if g: _chunks_section("Merged-All (Pipeline+Naive+Traditional)", g, "g")

        lines.append("---\n")

    # Write
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Markdown report saved → {path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Retrieval layer evaluation harness")
    parser.add_argument("--export",    action="store_true", help="Save RAGAS-ready JSON")
    parser.add_argument("--export-md", action="store_true", help="Save detailed markdown report")
    parser.add_argument("--verbose",   action="store_true", help="Show chunk text previews")
    parser.add_argument("--naive",     action="store_true", help="Run naive RAG only (baseline)")
    parser.add_argument("--compare",   action="store_true", help="Pipeline vs naive RAG side-by-side")
    parser.add_argument("--best-of",   type=int, default=0, metavar="N",
                        help="Include best-of-N retrieval as an extra comparison method")
    parser.add_argument("--merged",    action="store_true",
                        help="Include merged pipeline+naive retrieval as an extra comparison method")
    parser.add_argument("--merged-all", action="store_true",
                        help="Include merged pipeline+naive+traditional retrieval as an extra comparison method")
    parser.add_argument("--query",     type=str, default=None,
                        help="Run a single ad-hoc query instead of the golden set")
    args = parser.parse_args()

    best_of_n = args.best_of  # 0 = disabled

    # ── Single ad-hoc query ────────────────────────────────────────────────────
    if args.query:
        gq = GoldenQuery(query=args.query, expected_intent="specific", note="ad-hoc")
        if args.compare:
            p = _run_query(gq)
            n = _run_query_naive(gq)
            o = _run_query_overlap(gq)
            print(f"\n{'─'*60}")
            print(f"  PIPELINE     →  intent={p.actual_intent}  top={p.top_score:.3f}  {int(p.latency_s*1000)}ms")
            print(f"  NAIVE RAG    →  top={n.top_score:.3f}  {int(n.latency_s*1000)}ms")
            print(f"  TRADITIONAL  →  top={o.top_score:.3f}  {int(o.latency_s*1000)}ms")
            if best_of_n:
                b = _run_query_best_of_n(gq, n=best_of_n)
                print(f"  BEST-OF-{best_of_n}    →  intent={b.actual_intent}  top={b.top_score:.3f}  {int(b.latency_s*1000)}ms")
            if args.merged:
                m = _run_query_merged(gq)
                print(f"  MERGED       →  intent={m.actual_intent}  top={m.top_score:.3f}  {int(m.latency_s*1000)}ms")
            if args.merged_all:
                g = _run_query_merged_all(gq)
                print(f"  MERGED-ALL   →  intent={g.actual_intent}  top={g.top_score:.3f}  {int(g.latency_s*1000)}ms")
            if args.verbose:
                print(f"\n  [Pipeline chunks]");  _print_verbose(p)
                print(f"  [Naive chunks]");       _print_verbose(n)
                print(f"  [Traditional chunks]"); _print_verbose(o)
        elif args.naive:
            r = _run_query_naive(gq)
            print(f"\nMode   : NAIVE RAG")
            print(f"Scores : top={r.top_score:.3f}  mean={r.mean_score:.3f}")
            print(f"Latency: {int(r.latency_s*1000)}ms")
            if args.verbose:
                _print_verbose(r)
        elif best_of_n:
            r = _run_query_best_of_n(gq, n=best_of_n)
            print(f"\nMode   : BEST-OF-{best_of_n}")
            print(f"Intent : {r.actual_intent}")
            print(f"Scores : top={r.top_score:.3f}  mean={r.mean_score:.3f}")
            print(f"Latency: {int(r.latency_s*1000)}ms")
            if args.verbose:
                _print_verbose(r)
        elif args.merged:
            r = _run_query_merged(gq)
            print(f"\nMode   : MERGED (Pipeline+Naive)")
            print(f"Intent : {r.actual_intent}")
            print(f"Scores : top={r.top_score:.3f}  mean={r.mean_score:.3f}")
            print(f"Latency: {int(r.latency_s*1000)}ms")
            if args.verbose:
                _print_verbose(r)
        elif args.merged_all:
            r = _run_query_merged_all(gq)
            print(f"\nMode   : MERGED-ALL (Pipeline+Naive+Traditional)")
            print(f"Intent : {r.actual_intent}")
            print(f"Scores : top={r.top_score:.3f}  mean={r.mean_score:.3f}")
            print(f"Latency: {int(r.latency_s*1000)}ms")
            if args.verbose:
                _print_verbose(r)
        else:
            r = _run_query(gq)
            print(f"\nIntent : {r.actual_intent}")
            print(f"Scores : top={r.top_score:.3f}  mean={r.mean_score:.3f}")
            print(f"Latency: {int(r.latency_s*1000)}ms")
            if args.verbose:
                _print_verbose(r)
        return

    # ── Full golden set ────────────────────────────────────────────────────────
    print(f"\nLoading retrieval layer…  (first query may be slow — models loading)\n")

    pipeline_results, naive_results, overlap_results, best_of_results, merged_results, merged_all_results = [], [], [], [], [], []

    for i, gq in enumerate(GOLDEN_QUERIES, 1):
        label   = f"[{i}/{len(GOLDEN_QUERIES)}]"
        q_short = gq.query[:50] + ("…" if len(gq.query) > 50 else "")

        if args.compare:
            print(f"  {label} {q_short}", end="", flush=True)
            p = _run_query(gq)
            n = _run_query_naive(gq)
            o = _run_query_overlap(gq)
            pipeline_results.append(p)
            naive_results.append(n)
            overlap_results.append(o)
            line = f"  p={p.top_score:.3f} n={n.top_score:.3f} o={o.top_score:.3f}"
            if best_of_n:
                b = _run_query_best_of_n(gq, n=best_of_n)
                best_of_results.append(b)
                line += f" b={b.top_score:.3f}"
            if args.merged:
                m = _run_query_merged(gq)
                merged_results.append(m)
                line += f" m={m.top_score:.3f}"
            if args.merged_all:
                g = _run_query_merged_all(gq)
                merged_all_results.append(g)
                line += f" g={g.top_score:.3f}"
            conf = " ⚡" if p.conflict_detected else ""
            print(line + conf)
            if args.verbose:
                _print_verbose(p)
        elif args.naive:
            print(f"  {label} {q_short}", end="", flush=True)
            r = _run_query_naive(gq)
            naive_results.append(r)
            ok = "✓" if r.keyword_hit else "✗"
            print(f"  {ok}  {int(r.latency_s*1000)}ms")
            if args.verbose:
                _print_verbose(r)
        else:
            print(f"  {label} {q_short}", end="", flush=True)
            r = _run_query(gq)
            pipeline_results.append(r)
            ok = "✓" if (r.intent_correct and r.keyword_hit) else "✗"
            print(f"  {ok}  {int(r.latency_s*1000)}ms")
            if args.verbose:
                _print_verbose(r)

    # ── Print report ───────────────────────────────────────────────────────────
    if args.compare:
        _print_comparison(pipeline_results, naive_results, overlap_results,
                          best_of_results if best_of_results else None,
                          merged_results if merged_results else None,
                          merged_all_results if merged_all_results else None)
    elif args.naive:
        _print_report(naive_results)
    else:
        _print_report(pipeline_results)

    if args.export:
        export_results = pipeline_results if pipeline_results else naive_results
        _export_ragas(export_results)

    if args.export_md:
        if args.compare:
            _export_markdown(pipeline_results, naive_results, overlap_results,
                             merged_results if merged_results else None,
                             merged_all_results if merged_all_results else None,
                             path="eval_report.md")
        elif args.naive:
            _export_markdown(naive_results, path="eval_report.md")
        else:
            _export_markdown(pipeline_results, path="eval_report.md")


if __name__ == "__main__":
    main()
