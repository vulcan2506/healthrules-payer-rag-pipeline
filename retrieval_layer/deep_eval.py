"""
deep_eval.py
────────────
Answer-quality + retrieval-quality deep evaluation, built on top of the
golden queries already defined in eval.py.

Why this exists: eval.py measures retrieval only (rerank score, keyword hit,
intent accuracy, conflict detection) — it never generates an actual answer,
so it can't tell you whether a method's answer is faithful, correct,
complete, or concise. That gap is exactly what surfaced the concern that
merged/merged_all answers might score well on retrieval but be "bahut
zyada" (bloated) and not simply answer the core question.

Two metric families, computed together in one pass per query/method:

  Answer-quality (LLM-judged, via DeepEval, reference-free — no gold answer
  needed, since none exist for this golden set and hand-writing 53 of them
  wasn't worth it just to unlock ContextualPrecision/Recall):
    - Faithfulness   (DeepEval built-in: does the answer avoid claims not
                       supported by the retrieved context?)
    - Correctness    (custom GEval: does the answer correctly reflect what
                       the retrieved context actually says?)
    - Completeness   (custom GEval: does the answer cover the relevant
                       information present in the context?)
    - Conciseness    (custom GEval: does the answer avoid padding/repetition
                       while still answering the question?)

  Retrieval-quality (keyword-based, no LLM — reuses each GoldenQuery's
  existing `expected_keywords` field as an automatable stand-in for
  hand-labeled chunk relevance, since no manual relevance judgments exist):
    - MRR            (reciprocal rank of first chunk containing a keyword)
    - Precision@5    (fraction of top-5 chunks containing a keyword)
    - Recall         (fraction of expected keywords found anywhere in top-5)

The judge/generator model is the same local llama.cpp server the rest of
the pipeline already uses (config.LLAMA_SERVER_URL) — via DeepEval's
LocalModel, no OpenAI key involved. `enable_thinking` must be forced off:
left on, the model rambles inside <think> and blows through DeepEval's
retry/timeout budget (measured — a single GEval call timed out after
~90s+3 retries with thinking on; ~5s with it off).

Purpose: derive empirical, measured gating conditions for when
retrieve_merged/retrieve_merged_all should fire, instead of guessing.

Usage:
    python3 deep_eval.py --pilot                       # 11-query representative subset (fast)
    python3 deep_eval.py --all                         # full 53-query golden set (slow)
    python3 deep_eval.py --pilot --methods pipeline,merged
    python3 deep_eval.py --query "..." --methods pipeline,merged,merged_all
"""

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

import config
import eval as eval_mod
import retriever

logging.basicConfig(level=logging.WARNING, format="%(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

PARALLEL_SLOTS = 6   # matches Stage 1's llama-server -np 6

# Representative subset spanning every query shape in the golden set:
# specific/intelligence/delta, relationship, evolution, cross-corpus, negative
# control. Overlaps deliberately with the earlier Q1/Q5/Q12/Q19/Q22 deep-dive
# for continuity.
PILOT_INDICES = [1, 5, 6, 11, 12, 18, 19, 20, 22, 29, 30, 37, 53]


# ── Method registry ────────────────────────────────────────────────────────────

def _run_pipeline(q: str) -> Dict:
    return retriever.retrieve(q)

def _run_naive(q: str) -> Dict:
    return retriever.retrieve_naive(q)

def _run_overlap(q: str) -> Dict:
    return retriever.retrieve_overlap(q)

def _run_merged(q: str) -> Dict:
    return retriever.retrieve_merged(q)

def _run_merged_all(q: str) -> Dict:
    return retriever.retrieve_merged_all(q)

def _run_gated(q: str) -> Dict:
    """Production-shaped routing: redis_cache.gate_and_retrieve()'s dynamic confidence-based gate."""
    import redis_cache
    return redis_cache.gate_and_retrieve(q)

def _run_detailed(q: str) -> Dict:
    """merged_all + planner.explore() (only actually loops if config.PLANNER_ENABLED)."""
    import redis_cache
    return redis_cache.retrieve_detailed(q)

METHOD_FNS = {
    "pipeline":    _run_pipeline,
    "naive":       _run_naive,
    "traditional": _run_overlap,
    "merged":      _run_merged,
    "merged_all":  _run_merged_all,
    "gated":       _run_gated,
    "detailed":    _run_detailed,
}
DEFAULT_METHODS = ["pipeline", "merged", "merged_all"]


# ── Judge / generator model (local llama.cpp, no OpenAI key) ──────────────────

_judge_model = None
_gen_client: Optional[OpenAI] = None


def _get_judge_model():
    global _judge_model
    if _judge_model is None:
        from deepeval.models import LocalModel
        _judge_model = LocalModel(
            model=config.LLAMA_MODEL_NAME,
            base_url=config.LLAMA_SERVER_URL + "/v1",
            api_key="none",
            temperature=0.0,
            timeout=90.0,
            generation_kwargs={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        )
    return _judge_model


def _get_gen_client() -> OpenAI:
    global _gen_client
    if _gen_client is None:
        _gen_client = OpenAI(base_url=config.LLAMA_SERVER_URL + "/v1", api_key="none")
    return _gen_client


def generate_answer(query: str, result: Dict, max_tokens: int = 900) -> Tuple[str, bool]:
    """
    One answer-generation call using the method's own context/system_prompt.

    Returns (answer, was_truncated). 400 tokens (the original default) measurably
    cut off mid-sentence on delta/list-shaped queries across every method — that
    silently corrupted the Completeness/Conciseness scores for those queries (an
    incomplete answer reads as "not concise" for the wrong reason). Bumped to 900
    and now surfaces finish_reason so a future truncation shows up in the report
    instead of masquerading as a genuine conciseness/completeness score.
    """
    resp = _get_gen_client().chat.completions.create(
        model=config.LLAMA_MODEL_NAME,
        messages=[
            {"role": "system", "content": result["system_prompt"]},
            {"role": "user",   "content": f"{result['context']}\n\nQuestion: {query}"},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    choice = resp.choices[0]
    truncated = choice.finish_reason == "length"
    return (choice.message.content or "").strip(), truncated


# ── Answer-quality metrics (built once, reused across all test cases) ─────────

def _build_metrics():
    from deepeval.metrics import FaithfulnessMetric, GEval
    from deepeval.test_case import SingleTurnParams

    model = _get_judge_model()

    faithfulness = FaithfulnessMetric(model=model, threshold=0.5, include_reason=True)

    correctness = GEval(
        name="Correctness",
        model=model,
        threshold=0.5,
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.RETRIEVAL_CONTEXT],
        evaluation_steps=[
            "Identify the specific facts and claims made in the actual output.",
            "Check each claim against the retrieval context — is it supported, contradicted, or unverifiable?",
            "Penalize claims that contradict the retrieval context or are not present in it at all.",
            "Do not penalize the answer for omitting information — that is judged separately (Completeness).",
            "A correct answer scores high even if brief; an answer with any unsupported/contradicted claim scores low.",
        ],
    )

    completeness = GEval(
        name="Completeness",
        model=model,
        threshold=0.5,
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.RETRIEVAL_CONTEXT],
        evaluation_steps=[
            "Identify all facts in the retrieval context that are directly relevant to answering the input question.",
            "Check whether the actual output includes each of those directly-relevant facts.",
            "Penalize omission of directly-relevant facts found in the retrieval context.",
            "Do not penalize the answer for including extra context, or for style/length — that is judged separately (Conciseness).",
            "A complete answer scores high even if verbose; an answer missing directly-relevant facts scores low.",
        ],
    )

    conciseness = GEval(
        name="Conciseness",
        model=model,
        threshold=0.5,
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        evaluation_steps=[
            "Check whether the actual output answers the input question without unnecessary padding, hedging, or repetition.",
            "Penalize restating the question, generic preamble/conclusion sentences, or explaining obvious background not asked for.",
            "Do not penalize the answer for being incomplete or for factual errors — those are judged separately.",
            "A concise answer scores high even if short; a bloated answer that buries the point scores low regardless of correctness.",
        ],
    )

    return {"faithfulness": faithfulness, "correctness": correctness,
            "completeness": completeness, "conciseness": conciseness}


# ── Retrieval-quality metrics (keyword-based, no LLM) ──────────────────────────

def _is_relevant(text: str, keywords: List[str]) -> bool:
    t = text.lower()
    return any(kw.lower() in t for kw in keywords)


def retrieval_metrics(chunks: List[Dict], keywords: List[str]) -> Dict[str, Optional[float]]:
    if not keywords:
        return {"mrr": None, "precision_at_5": None, "recall": None}
    top5 = chunks[:5]
    texts = [c.get("text", "") for c in top5]
    flags = [_is_relevant(t, keywords) for t in texts]

    mrr = 0.0
    for rank, flag in enumerate(flags, 1):
        if flag:
            mrr = 1.0 / rank
            break

    precision = (sum(flags) / len(flags)) if flags else 0.0

    all_text = " ".join(texts).lower()
    found = sum(1 for kw in keywords if kw.lower() in all_text)
    recall = found / len(keywords)

    return {"mrr": mrr, "precision_at_5": precision, "recall": recall}


# ── Per query/method evaluation ────────────────────────────────────────────────

@dataclass
class DeepResult:
    query: str
    method: str
    answer: str
    routed_method: str = ""   # for "gated": which underlying method select_method() actually picked
    truncated: bool = False
    faithfulness: Optional[float] = None
    correctness: Optional[float] = None
    completeness: Optional[float] = None
    conciseness: Optional[float] = None
    mrr: Optional[float] = None
    precision_at_5: Optional[float] = None
    recall: Optional[float] = None
    top_rerank_score: float = 0.0
    latency_s: float = 0.0
    reasons: Dict[str, str] = field(default_factory=dict)


def evaluate_one(query: str, method: str, metrics: Dict, keywords: List[str]) -> DeepResult:
    from deepeval.test_case import LLMTestCase

    t0 = time.time()
    result = METHOD_FNS[method](query)
    chunks = result["chunks"]
    answer, truncated = generate_answer(query, result)
    if truncated:
        log.warning(f"[{method}] answer truncated at max_tokens for {query!r} — raise generate_answer's max_tokens")

    retrieval_context = [c.get("text", "") for c in chunks]
    tc = LLMTestCase(input=query, actual_output=answer, retrieval_context=retrieval_context)

    scores, reasons = {}, {}
    for name, metric in metrics.items():
        try:
            metric.measure(tc)
            scores[name] = metric.score
            reasons[name] = getattr(metric, "reason", "") or ""
        except Exception as e:
            log.warning(f"[{method}] {name} metric failed on {query!r}: {type(e).__name__}: {e}")
            scores[name] = None
            reasons[name] = f"ERROR: {e}"

    rmetrics = retrieval_metrics(chunks, keywords)
    top_score = max((c.get("rerank_score", c.get("score", 0)) for c in chunks), default=0.0)

    return DeepResult(
        query=query, method=method, answer=answer,
        routed_method=result.get("_gated_method", method), truncated=truncated,
        faithfulness=scores.get("faithfulness"), correctness=scores.get("correctness"),
        completeness=scores.get("completeness"), conciseness=scores.get("conciseness"),
        mrr=rmetrics["mrr"], precision_at_5=rmetrics["precision_at_5"], recall=rmetrics["recall"],
        top_rerank_score=top_score, latency_s=time.time() - t0, reasons=reasons,
    )


# ── Harness ─────────────────────────────────────────────────────────────────────

def _warmup(methods: List[str]) -> None:
    """
    Force every lazy-loaded singleton (router's SentenceTransformer, reranker,
    ChromaDB store) to initialize on the main thread before the ThreadPoolExecutor
    starts. Loading a SentenceTransformer from two threads at once races on
    PyTorch's meta-device init (measured: "Cannot copy out of meta tensor").
    """
    for m in methods:
        METHOD_FNS[m]("warmup")


def run_generate_only(queries: List[eval_mod.GoldenQuery], methods: List[str], cache_path: str) -> None:
    """
    Retrieval + answer-generation only, no judging — caches (query, method, answer,
    retrieval_context) to JSON so a later run_rejudge() can score the SAME test
    cases under a different judge model without regenerating anything. This is
    what makes a fair judge-model comparison possible: the answer/context are
    fixed, judge choice is the only variable.
    """
    _warmup(methods)
    jobs = [(gq, method) for gq in queries for method in methods]
    cases: List[Optional[Dict]] = [None] * len(jobs)

    def _work(idx_job):
        idx, (gq, method) = idx_job
        t0 = time.time()
        result = METHOD_FNS[method](gq.query)
        chunks = result["chunks"]
        answer, truncated = generate_answer(gq.query, result)
        top_score = max((c.get("rerank_score", c.get("score", 0)) for c in chunks), default=0.0)
        return idx, {
            "query": gq.query, "method": method, "answer": answer,
            "retrieval_context": [c.get("text", "") for c in chunks],
            "expected_keywords": gq.expected_keywords,
            "top_rerank_score": top_score, "truncated": truncated,
            "latency_s": time.time() - t0,
        }

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=PARALLEL_SLOTS) as pool:
        futs = {pool.submit(_work, (i, j)): i for i, j in enumerate(jobs)}
        done = 0
        for fut in as_completed(futs):
            idx, case = fut.result()
            cases[idx] = case
            done += 1
            flag = "  [TRUNCATED]" if case["truncated"] else ""
            log.warning(f"[{done}/{len(jobs)}] generated {case['method']:12s} {case['query'][:50]!r}{flag}")
    log.warning(f"Generation done in {time.time()-t0:.1f}s")

    with open(cache_path, "w") as f:
        json.dump(cases, f, indent=2)
    log.warning(f"Wrote {len(cases)} cached test case(s) -> {cache_path}")


def run_rejudge(cache_path: str, methods: Optional[List[str]] = None) -> List[DeepResult]:
    """
    Re-score cached (answer, retrieval_context) test cases with whatever judge
    model _get_judge_model() currently resolves to (set JUDGE_BACKEND / point
    config.LLAMA_SERVER_URL at a different local model server beforehand).
    Skips retrieval + generation entirely — isolates judge choice as the only
    variable between two runs over the identical test cases.
    """
    from deepeval.test_case import LLMTestCase

    with open(cache_path) as f:
        cases = json.load(f)
    if methods:
        cases = [c for c in cases if c["method"] in methods]

    metrics = _build_metrics()
    results: List[Optional[DeepResult]] = [None] * len(cases)

    def _work(idx_case):
        idx, case = idx_case
        tc = LLMTestCase(input=case["query"], actual_output=case["answer"],
                          retrieval_context=case["retrieval_context"])
        scores, reasons = {}, {}
        for name, metric in metrics.items():
            try:
                metric.measure(tc)
                scores[name] = metric.score
                reasons[name] = getattr(metric, "reason", "") or ""
            except Exception as e:
                log.warning(f"[{case['method']}] {name} metric failed on {case['query']!r}: {type(e).__name__}: {e}")
                scores[name] = None
                reasons[name] = f"ERROR: {e}"

        rmetrics = retrieval_metrics(
            [{"text": t} for t in case["retrieval_context"]], case.get("expected_keywords") or [])

        return idx, DeepResult(
            query=case["query"], method=case["method"], answer=case["answer"],
            truncated=case.get("truncated", False),
            faithfulness=scores.get("faithfulness"), correctness=scores.get("correctness"),
            completeness=scores.get("completeness"), conciseness=scores.get("conciseness"),
            mrr=rmetrics["mrr"], precision_at_5=rmetrics["precision_at_5"], recall=rmetrics["recall"],
            top_rerank_score=case.get("top_rerank_score", 0.0), reasons=reasons,
        )

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=PARALLEL_SLOTS) as pool:
        futs = {pool.submit(_work, (i, c)): i for i, c in enumerate(cases)}
        done = 0
        for fut in as_completed(futs):
            idx, res = fut.result()
            results[idx] = res
            done += 1
            log.warning(f"[{done}/{len(cases)}] {res.method:12s} {res.query[:55]!r} "
                        f"F={res.faithfulness} C={res.correctness} Co={res.completeness} Cc={res.conciseness}")
    log.warning(f"Rejudge done in {time.time()-t0:.1f}s")
    return results


def run(queries: List[eval_mod.GoldenQuery], methods: List[str]) -> List[DeepResult]:
    _warmup(methods)
    metrics = _build_metrics()
    jobs = [(gq, method) for gq in queries for method in methods]
    results: List[Optional[DeepResult]] = [None] * len(jobs)

    def _work(idx_job):
        idx, (gq, method) = idx_job
        return idx, evaluate_one(gq.query, method, metrics, gq.expected_keywords)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=PARALLEL_SLOTS) as pool:
        futs = {pool.submit(_work, (i, j)): i for i, j in enumerate(jobs)}
        done = 0
        for fut in as_completed(futs):
            idx, res = fut.result()
            results[idx] = res
            done += 1
            log.warning(f"[{done}/{len(jobs)}] {res.method:12s} {res.query[:55]!r} "
                        f"F={res.faithfulness} C={res.correctness} Co={res.completeness} Cc={res.conciseness}")
    log.warning(f"Done in {time.time()-t0:.1f}s")
    return results


# ── Reporting ────────────────────────────────────────────────────────────────────

def _mean(vals: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _fmt(v: Optional[float]) -> str:
    return f"{v:.3f}" if v is not None else "—"


def print_report(results: List[DeepResult], methods: List[str]) -> None:
    by_method = {m: [r for r in results if r.method == m] for m in methods}

    print("\n" + "=" * 100)
    print("DEEP EVAL — ANSWER-QUALITY + RETRIEVAL-QUALITY SUMMARY")
    print("=" * 100)
    header = f"{'Method':14s} {'Faithful':9s} {'Correct':9s} {'Complete':9s} {'Concise':9s} {'MRR':7s} {'Prec@5':7s} {'Recall':7s} {'TopScore':9s}"
    print(header)
    print("-" * len(header))
    for m in methods:
        rs = by_method[m]
        print(f"{m:14s} "
              f"{_fmt(_mean([r.faithfulness for r in rs])):9s} "
              f"{_fmt(_mean([r.correctness for r in rs])):9s} "
              f"{_fmt(_mean([r.completeness for r in rs])):9s} "
              f"{_fmt(_mean([r.conciseness for r in rs])):9s} "
              f"{_fmt(_mean([r.mrr for r in rs])):7s} "
              f"{_fmt(_mean([r.precision_at_5 for r in rs])):7s} "
              f"{_fmt(_mean([r.recall for r in rs])):7s} "
              f"{_fmt(_mean([r.top_rerank_score for r in rs])):9s}")
    print()

    # Per-query detail
    by_query: Dict[str, Dict[str, DeepResult]] = {}
    for r in results:
        by_query.setdefault(r.query, {})[r.method] = r

    for q, per_method in by_query.items():
        print(f"\n  Q: {q}")
        for m in methods:
            r = per_method.get(m)
            if not r:
                continue
            trunc_flag = "  [TRUNCATED]" if r.truncated else ""
            routed = f"  routed->{r.routed_method}" if r.routed_method and r.routed_method != m else ""
            print(f"    [{m:12s}] F={_fmt(r.faithfulness)} C={_fmt(r.correctness)} "
                  f"Co={_fmt(r.completeness)} Cc={_fmt(r.conciseness)} "
                  f"MRR={_fmt(r.mrr)} P@5={_fmt(r.precision_at_5)} R={_fmt(r.recall)} "
                  f"top_score={r.top_rerank_score:.3f}{trunc_flag}{routed}")
            print(f"        answer: {r.answer[:180].replace(chr(10), ' ')}")


def export_markdown(results: List[DeepResult], methods: List[str], path: str) -> None:
    by_method = {m: [r for r in results if r.method == m] for m in methods}
    by_query: Dict[str, Dict[str, DeepResult]] = {}
    for r in results:
        by_query.setdefault(r.query, {})[r.method] = r

    lines = ["# Deep Eval Report", ""]
    lines.append("## Summary (mean across queries)")
    lines.append("")
    lines.append("| Method | Faithfulness | Correctness | Completeness | Conciseness | MRR | Precision@5 | Recall | Top rerank score |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for m in methods:
        rs = by_method[m]
        lines.append(
            f"| {m} | {_fmt(_mean([r.faithfulness for r in rs]))} | {_fmt(_mean([r.correctness for r in rs]))} | "
            f"{_fmt(_mean([r.completeness for r in rs]))} | {_fmt(_mean([r.conciseness for r in rs]))} | "
            f"{_fmt(_mean([r.mrr for r in rs]))} | {_fmt(_mean([r.precision_at_5 for r in rs]))} | "
            f"{_fmt(_mean([r.recall for r in rs]))} | {_fmt(_mean([r.top_rerank_score for r in rs]))} |"
        )
    lines.append("")

    lines.append("## Per-query detail")
    for q, per_method in by_query.items():
        lines.append("")
        lines.append(f"### {q}")
        for m in methods:
            r = per_method.get(m)
            if not r:
                continue
            trunc_flag = "  **[TRUNCATED]**" if r.truncated else ""
            routed = f"  _(routed->{r.routed_method})_" if r.routed_method and r.routed_method != m else ""
            lines.append(f"\n**{m}** — F={_fmt(r.faithfulness)} C={_fmt(r.correctness)} "
                         f"Co={_fmt(r.completeness)} Cc={_fmt(r.conciseness)} "
                         f"MRR={_fmt(r.mrr)} P@5={_fmt(r.precision_at_5)} R={_fmt(r.recall)} "
                         f"top_score={r.top_rerank_score:.3f}{trunc_flag}{routed}")
            lines.append(f"\n> {r.answer}")
            if r.reasons.get("conciseness"):
                lines.append(f"\n_Conciseness reason: {r.reasons['conciseness']}_")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    log.warning(f"Wrote {path}")


# ── CLI ──────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Deep eval: answer-quality + retrieval-quality")
    ap.add_argument("--pilot", action="store_true", help=f"Run the {len(PILOT_INDICES)}-query representative subset")
    ap.add_argument("--all", action="store_true", help="Run the full golden query set")
    ap.add_argument("--query", type=str, default=None, help="Run a single ad-hoc query")
    ap.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS),
                    help=f"Comma list from {list(METHOD_FNS)}")
    ap.add_argument("--export-md", type=str, default="deep_eval_report.md")
    ap.add_argument("--cache-only", type=str, default=None, metavar="PATH",
                    help="Generate answers/context only (no judging), cache to PATH for a later --rejudge run")
    ap.add_argument("--rejudge", type=str, default=None, metavar="PATH",
                    help="Skip retrieval/generation — re-score a --cache-only PATH with the current judge model")
    ap.add_argument("--judge-model-alias", type=str, default=None,
                    help="Override the local judge model's --alias (use when a different llama-server model is live)")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in METHOD_FNS:
            raise SystemExit(f"Unknown method {m!r}, choose from {list(METHOD_FNS)}")

    if args.judge_model_alias:
        config.LLAMA_MODEL_NAME = args.judge_model_alias

    if args.rejudge:
        results = run_rejudge(args.rejudge, methods=methods if args.methods != ",".join(DEFAULT_METHODS) else None)
        print_report(results, sorted({r.method for r in results}))
        export_markdown(results, sorted({r.method for r in results}), args.export_md)
        return

    if args.query:
        queries = [eval_mod.GoldenQuery(query=args.query, expected_intent="specific")]
    elif args.all:
        queries = eval_mod.GOLDEN_QUERIES
    else:
        # default / --pilot
        queries = [eval_mod.GOLDEN_QUERIES[i - 1] for i in PILOT_INDICES]

    if args.cache_only:
        log.warning(f"Generating only (no judging): {len(queries)} queries x {len(methods)} methods = {len(queries)*len(methods)} jobs")
        run_generate_only(queries, methods, args.cache_only)
        return

    log.warning(f"Running deep eval: {len(queries)} queries x {len(methods)} methods = {len(queries)*len(methods)} jobs")
    results = run(queries, methods)
    print_report(results, methods)
    export_markdown(results, methods, args.export_md)


if __name__ == "__main__":
    main()
