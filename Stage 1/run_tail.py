"""
run_tail.py
───────────
Runs everything that follows after recover_qna.py completes:

  1. filter.py                     — filter enterprise_nested_topics.json to cross-version topics only
  2. parent_relationship.py        — Louvain-cluster parent categories into grandparent groups
                                      (Tier B sibling/grandparent relationships). Only needs the
                                      nested taxonomy, so runs early — nothing downstream depends
                                      on it except chroma_store's ingestion step.
  3. cross_corpus_relationship.py  — same clustering, but across separate document-type taxonomies
                                      (e.g. payer <-> connector). Currently a no-op: only "payer" is
                                      registered in its SOURCES list until a second corpus exists.
  4. delta_analyzer.py             — compare topic pairs across versions, save report + jobs cache
  5. topic_summarizer.py           — chunk-grounded, source-scoped topic summaries (leaf level).
                                      Reuses delta_jobs_cache.json from step 4 where available.
  6. hierarchy_summarizer.py       — rolls topic summaries up into sub-category/parent group
                                      profiles (breadth queries like "list all the domains" need
                                      this — no single topic summary can answer them). Reads
                                      step 5's output, so must run after it.
  7. convert_delta()               — transform delta_jobs_cache.json → delta_reports/*.json
                                      (format chroma_store.py expects)
  8. build_index.py                — build HNSW index from nested JSON
  9. chroma_store.py               — populate corpus / intelligence / delta ChromaDB collections
 10. eval.py --compare              — pipeline vs naive RAG comparison
 11. build_cache_preset.py          — seed the Redis hackathon-demo cache preset (samples the
                                      existing QnA bank, splits half cached/half uncached, writes
                                      the demo Word doc). Re-run standalone to regenerate the
                                      preset later (e.g. incremental weekly/monthly refresh).

Run AFTER recover_qna.py has finished (with llama server still running):
    cd "Stage 1"
    python run_tail.py
"""

import argparse
import importlib.util
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("run_tail.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger("run_tail")

import config

# Retrieval layer lives one directory up
RETRIEVAL_DIR = Path(__file__).parent.parent / "retrieval_layer"
sys.path.insert(0, str(RETRIEVAL_DIR))


# ── Step 1: Cross-version filter ───────────────────────────────────────────────

def step_filter():
    log.info("\n[1/9] Filtering nested topics to cross-version only…")
    import filter as f_mod
    f_mod.filter_json()
    out = Path(f_mod.OUTPUT_JSON_PATH)
    if not out.exists():
        log.error(f"cross_version_topics_only.json not found at {out}")
        sys.exit(1)
    with open(out) as f:
        data = json.load(f)
    log.info(f"Cross-version filter done → {out.name}  ({len(data)} entries)")


# ── Step 2: Parent relationship clustering (grandparent groups) ───────────────

def step_parent_relationship():
    log.info("\n[2/9] Running parent relationship clustering (grandparent groups)…")
    import parent_relationship
    parent_relationship.main()
    out = parent_relationship.OUT_PATH
    if not out.exists():
        log.error(f"parent_relationship_clusters.json not found at {out}")
        sys.exit(1)
    with open(out) as f:
        clusters = json.load(f)
    log.info(f"Parent relationship clustering done — {len(clusters)} grandparent group(s)")


# ── Step 3: Cross-corpus relationship linking ──────────────────────────────────

def step_cross_corpus_relationship():
    log.info("\n[3/9] Running cross-corpus relationship linking…")
    import cross_corpus_relationship
    cross_corpus_relationship.main()
    out = cross_corpus_relationship.OUT_PATH
    if out.exists():
        with open(out) as f:
            clusters = json.load(f)
        log.info(f"Cross-corpus relationship linking done — {len(clusters)} cross-corpus group(s)")
    else:
        log.info(
            "Cross-corpus relationship linking produced no groups — expected until a second "
            "document corpus is registered (only 'payer' is in cross_corpus_relationship.SOURCES today)"
        )


# ── Step 4: Delta analysis ─────────────────────────────────────────────────────

def step_delta_analyzer():
    log.info("\n[4/9] Running delta analysis…")
    import delta_analyzer
    delta_analyzer.run_delta_analysis()
    cache = config.OUTPUT_DIR / "delta_jobs_cache.json"
    if not cache.exists():
        log.error("delta_jobs_cache.json not found — delta analysis may have failed")
        sys.exit(1)
    with open(cache) as f:
        jobs = json.load(f)
    done = sum(1 for j in jobs if j.get("delta"))
    log.info(f"Delta analysis done — {done}/{len(jobs)} topics analysed")


# ── Step 5: Topic-level chunk-grounded summaries ───────────────────────────────

def step_topic_summarizer():
    log.info("\n[5/9] Running topic summarizer (leaf-level, chunk-grounded)…")
    import topic_summarizer
    topic_summarizer.run_topic_summarization()


# ── Step 6: Sub-category / parent group profiles ───────────────────────────────

def step_hierarchy_summarizer():
    log.info("\n[6/9] Running hierarchy summarizer (sub-category/parent roll-up)…")
    import hierarchy_summarizer
    hierarchy_summarizer.run_hierarchy_summarization()


# ── Step 7: Convert delta cache → chroma-ready JSON files ──────────────────────

def step_convert_delta():
    """
    Transform delta_jobs_cache.json into individual JSON report files
    under data/output/delta_reports/ in the format chroma_store expects:

    Each file = one parent-level group.
    Each finding = {section, change_type, summary, detail, version_from, version_to}
    """
    log.info("\n[7/9] Converting delta cache → delta_reports/…")

    cache_path = config.OUTPUT_DIR / "delta_jobs_cache.json"
    delta_dir  = config.OUTPUT_DIR / "delta_reports"
    delta_dir.mkdir(parents=True, exist_ok=True)

    with open(cache_path, encoding="utf-8") as f:
        jobs = json.load(f)

    # Group by parent so each file covers one top-level domain
    by_parent: dict = {}
    for job in jobs:
        d = job.get("delta")
        if not d:
            continue
        parent = job.get("parent", "unknown")
        by_parent.setdefault(parent, [])

        # Map delta fields to chroma finding format
        section     = job.get("topic", "")
        change_type = d.get("change_type", "")
        summary     = d.get("analysis", "")
        detail      = "\n".join(d.get("key_differences", []))
        v_from      = job.get("vA", "")
        v_to        = job.get("vB", "")

        by_parent[parent].append({
            "section":      section,
            "change_type":  change_type,
            "summary":      summary,
            "detail":       detail,
            "version_from": v_from,
            "version_to":   v_to,
            "relevance_score": d.get("relevance_score", 0),
            "confidence":   d.get("confidence", ""),
        })

    total_findings = 0
    for parent, findings in by_parent.items():
        safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in parent)
        safe_name = safe_name.strip().replace(" ", "_")[:60]
        out_path  = delta_dir / f"delta_{safe_name}.json"
        payload   = {"parent": parent, "findings": findings}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        total_findings += len(findings)

    log.info(f"Delta reports written → {delta_dir}  "
             f"({len(by_parent)} files, {total_findings} total findings)")


# ── Step 8: Build HNSW index ───────────────────────────────────────────────────

def step_build_index():
    log.info("\n[8/9] Building HNSW index…")
    result = subprocess.run(
        [sys.executable, str(RETRIEVAL_DIR / "build_index.py")],
        cwd=str(RETRIEVAL_DIR),
        capture_output=False,
    )
    if result.returncode != 0:
        log.error("build_index.py failed — aborting")
        sys.exit(1)
    idx_dir = RETRIEVAL_DIR / "index"
    files   = list(idx_dir.glob("*.bin")) + list(idx_dir.glob("*.pkl"))
    log.info(f"HNSW index built → {idx_dir}  ({len(files)} files)")


# ── Step 9: Populate ChromaDB ──────────────────────────────────────────────────

def step_chroma():
    log.info("\n[9/9] Populating ChromaDB collections…")
    result = subprocess.run(
        [sys.executable, str(RETRIEVAL_DIR / "chroma_store.py")],
        cwd=str(RETRIEVAL_DIR),
        capture_output=False,
    )
    if result.returncode != 0:
        log.error("chroma_store.py failed — aborting")
        sys.exit(1)
    # Read stats from chroma directly
    try:
        import sys as _sys
        _sys.path.insert(0, str(RETRIEVAL_DIR))
        import importlib, types
        # Load retrieval config without collision
        spec = importlib.util.spec_from_file_location(
            "rl_config", RETRIEVAL_DIR / "config.py"
        )
        rl_cfg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rl_cfg)
        import chromadb
        client = chromadb.PersistentClient(path=str(rl_cfg.CHROMA_DIR))
        stats = {c.name: c.count() for c in client.list_collections()}
        log.info(f"ChromaDB populated — {stats}")
    except Exception as e:
        log.info(f"ChromaDB populated (stats unavailable: {e})")


# ── Step 10: Run eval comparison ───────────────────────────────────────────────

def step_eval():
    log.info("\n[10/11] Running evaluation — pipeline vs naive RAG…")
    import subprocess
    eval_path = RETRIEVAL_DIR / "eval.py"
    result = subprocess.run(
        [sys.executable, str(eval_path), "--compare"],
        cwd=str(RETRIEVAL_DIR),
        capture_output=False,   # stream output live
    )
    if result.returncode != 0:
        log.warning("eval.py exited with non-zero code — check output above")
    else:
        log.info("Eval complete.")


# ── Step 11: Seed hackathon Redis cache demo preset ────────────────────────────

def step_cache_preset():
    log.info("\n[11/11] Seeding Redis hackathon-demo cache preset…")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "build_cache_preset.py")],
        cwd=str(Path(__file__).parent),
        capture_output=False,
    )
    if result.returncode != 0:
        log.warning("build_cache_preset.py exited with non-zero code — check output above")
    else:
        log.info("Cache preset seeded.")


# ── Entry point ────────────────────────────────────────────────────────────────

def main(skip_eval: bool = False):
    # Verify prerequisites
    nested = config.OUTPUT_DIR / "enterprise_nested_topics.json"
    filtered = config.OUTPUT_DIR / "filtered_chunks.json"
    registry = config.OUTPUT_DIR / "topic_registry.csv"

    for p in [nested, filtered, registry]:
        if not p.exists():
            log.error(f"Prerequisite missing: {p.name} — run recover_qna.py first")
            sys.exit(1)

    # Check QnA actually populated
    with open(filtered) as f:
        chunks = json.load(f)
    qna_count = sum(1 for c in chunks if c.get("qna"))
    if qna_count == 0:
        log.warning("filtered_chunks.json has no QnA pairs — run recover_qna.py first")
        sys.exit(1)
    log.info(f"Prerequisites OK — {qna_count}/{len(chunks)} chunks have QnA pairs")

    step_filter()
    step_parent_relationship()
    step_cross_corpus_relationship()
    step_delta_analyzer()
    step_topic_summarizer()
    step_hierarchy_summarizer()
    step_convert_delta()
    step_build_index()
    step_chroma()

    if skip_eval:
        log.info("\n[10-11/11] Skipping eval.py --compare and build_cache_preset.py (--skip-eval)")
    else:
        step_eval()
        step_cache_preset()

    log.info("\n✅ Full pipeline tail complete.")
    log.info("   Interactive test: cd retrieval_layer && python cli.py --compare")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip steps 10-11 (eval.py --compare, build_cache_preset.py) — for fast "
             "UI-triggered runs. Manual invocations should omit this for the full report.",
    )
    args = parser.parse_args()
    main(skip_eval=args.skip_eval)
