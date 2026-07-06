"""
recover_qna.py
──────────────
Re-runs pipeline steps 4.7 → 5 → 6 on the existing filtered_chunks.json.

Use this when QnA generation produced empty results (all qna: [])
without needing to redo the expensive chunking / labeling / grouping steps.

Run with the llama.cpp server already started:
    cd "Stage 1"
    python recover_qna.py
"""

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("recover_qna.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger("recover_qna")

import config
import grouper
import registry
import nested

FILTERED = config.OUTPUT_DIR / "filtered_chunks.json"


def main():
    if not FILTERED.exists():
        log.error(f"filtered_chunks.json not found at {FILTERED}")
        sys.exit(1)

    log.info(f"Loading {FILTERED.name}…")
    with open(FILTERED, encoding="utf-8") as f:
        chunks = json.load(f)
    log.info(f"Loaded {len(chunks)} chunks")

    # Sanity check: report current QnA state
    nonempty = sum(1 for c in chunks if c.get("qna"))
    log.info(f"QnA before re-run: {nonempty}/{len(chunks)} chunks have pairs")

    # ── Step 4.7: QnA Generation ───────────────────────────────────────────────
    log.info("\n[4.7] Re-generating synthetic Q&A pairs…")
    chunks = grouper.generate_group_qna(chunks)

    nonempty = sum(1 for c in chunks if c.get("qna"))
    total_pairs = sum(len(c.get("qna", [])) for c in chunks)
    log.info(f"QnA after re-run: {nonempty}/{len(chunks)} chunks have pairs ({total_pairs} total pairs)")

    with open(FILTERED, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    log.info("Saved filtered_chunks.json with QnA")

    # ── Step 5: Topic Registry ─────────────────────────────────────────────────
    log.info("\n[5] Building topic registry…")
    df = registry.build_registry(chunks)
    registry.save_registry(df)
    registry.print_summary()

    # ── Step 6: Nested Taxonomy ────────────────────────────────────────────────
    log.info("\n[6] Building nested taxonomy…")
    nested.run_enterprise_pipeline()
    log.info(f"Nested taxonomy saved → {config.NESTED_OUTPUT_PATH}")

    log.info("\nRecovery complete. Next steps:")
    log.info("  cd ../retrieval_layer && python build_index.py")
    log.info("  python chroma_store.py")


if __name__ == "__main__":
    main()
