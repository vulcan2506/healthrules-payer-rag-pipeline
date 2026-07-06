"""
main.py — Stage 1 pipeline orchestrator
"""

import logging
import sys
import json
from pathlib import Path

import grouper

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger("stage1")

import config
import llm_client
import ingest
import rectifier
import re_rectifier
import enricher
import context_profiler
import topic_model as tm
import label_normalizer
import registry
import nested
import quality_filter


def run():
    log.info("═" * 55)
    log.info("  STAGE 1  —  Ingest → Rectify → Label → Enrich → Filter → Registry")
    log.info("═" * 55)

    # ── 1. Per-PDF Extraction & Rectification ──────────────────────────────────
    log.info("\n[1/7] Processing PDFs (Extract -> Rectify -> Re-Rectify)")
    
    chunks = []
    cache_path = config.CHUNKS_CACHE
    
    if cache_path.exists():
        log.info(f"Cache found at {cache_path}. Loading previous progress...")
        with open(cache_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        log.info(f"Loaded {len(chunks)} chunks.")
    else:
        pdf_files = sorted(config.PDF_DIR.glob("*.pdf"))
        if not pdf_files:
            log.error(f"No PDFs found in {config.PDF_DIR}")
            sys.exit(1)
            
        global_id = {"id": 0}
        
        # Loop through one PDF at a time
        for pdf_path in pdf_files:
            log.info(f"\n--- Processing PDF: {pdf_path.name} ---")
            
            # Step A: Ingest & Cohesion
            pdf_chunks = ingest.process_single_pdf(pdf_path, global_id)
            
            # Step B: Rectify Tables (Heuristic)
            pdf_chunks = rectifier.rectify(pdf_chunks)
            
            # Step C: Re-Rectify Tables (LLM Fill & Delete Local MDs)
            pdf_chunks = re_rectifier.run_re_rectifier(pdf_chunks, batch_size=8)
            
            chunks.extend(pdf_chunks)
            
            # Save progress checkpoint after every PDF
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)
                
        log.info(f"Finished extracting all PDFs. Total chunks: {len(chunks)}")

    # Free VRAM from ingest embedder/KeyBERT + local LLM
    ingest.unload()
    llm_client.unload()
    log.info("Unloaded ingest models + local LLM.")

    # ── 2. Description enrichment & Quality Scoring ────────────────────────────
    log.info("\n[2/7] Enriching content-poor chunk descriptions and calculating quality scores")
    chunks = enricher.enrich_chunks(chunks)
    
    with open(config.CHUNKS_CACHE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    log.info("Enriched chunks saved to cache.")

    # Free VRAM from enricher embedder + local LLM
    enricher.unload()
    llm_client.unload()
    log.info("Unloaded enricher models + local LLM.")

    # ── 2.5 Context Profiling (domain-aware prompt adaptation) ──────────────────
    log.info("\n[2.5/7] Building domain profiles for dynamic prompting")
    profiles = context_profiler.build_profiles(chunks)
    log.info(f"Active profiles: {list(profiles.keys())}")

    # No local LLM to unload — context profiler uses API

    filtered_cache = config.OUTPUT_DIR / "filtered_chunks.json"

    if filtered_cache.exists():
        log.info("\n[3/7+3.5/7] filtered_chunks.json cache found — skipping quality filter + grouper")
        with open(filtered_cache, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        log.info(f"Loaded {len(chunks)} grouped chunks from cache.")
    else:
        # ── 3. Quality Filter ──────────────────────────────────────────────────
        log.info("\n[3/7] Running Quality Threshold Filter (< 70%)")
        chunks = quality_filter.run_filter(chunks, threshold=0.70)

        # ── 3.5 Semantic LLM Grouper ──────────────────────────────────────────
        log.info("\n[3.5/7] Running CoT LLM Grouper to merge related chunks")
        chunks = grouper.run_llm_grouping(chunks, batch_size=8, sequence_size=10)

        grouper.unload()
        log.info("Unloaded grouper embedder.")

        with open(filtered_cache, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

    # ── 4. Master labeling ─────────────────────────────────────────────────────
    unlabeled = [c for c in chunks if not c.get("master_label")]
    if unlabeled:
        log.info("\n[4/7] Assigning master labels (%d chunks)", len(unlabeled))
        tm.run_labeling(chunks)
        log.info("Labeling complete.")
    else:
        log.info("\n[4/7] All chunks already labeled — skipping.")

    # No local LLM to unload — labeler uses API

    # ── 4.5 Label normalization ─────────────────────────────────────────────────
    log.info("\n[4.5/7] Normalizing and deduplicating master labels")
    chunks = label_normalizer.normalize_labels(chunks)

    # Save labeled+normalized chunks to filtered_chunks (NOT chunks.json —
    # chunks.json is the raw ingest cache and must not contain grouped data)
    with open(filtered_cache, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    log.info("Labeled chunks saved to filtered_chunks.json.")

    # Free embedder from label normalizer
    label_normalizer.unload()

    # ── 4.7 QnA Generation ─────────────────────────────────────────────────────
    log.info("\n[4.7/7] Generating synthetic Q&A pairs for grouped chunks")
    chunks = grouper.generate_group_qna(chunks)
    with open(filtered_cache, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    log.info("Chunks with QnA metadata saved to filtered_chunks.json.")

    # ── 5. Build registry ──────────────────────────────────────────────────────
    log.info("\n[5/7] Building topic registry from filtered chunks")
    # Because we pass the filtered 'chunks' list, registry.py ONLY builds using high-quality data
    df = registry.build_registry(chunks)
    registry.save_registry(df)
    registry.print_summary()

    # Free cross-version matching embedder
    tm.unload()

    # ── 6. Nested taxonomy ──────────────────────────────────────────────────────
    log.info("\n[6/7] Building nested taxonomy")
    nested.run_enterprise_pipeline()
    log.info("Nested taxonomy saved → %s", config.NESTED_OUTPUT_PATH)

    log.info("\nStage 1 complete.")


if __name__ == "__main__":
    run()