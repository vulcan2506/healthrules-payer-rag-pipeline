import os as _os
from pathlib import Path

from dotenv import load_dotenv as _load_dotenv

# Loads Stage 1/.env (ANTHROPIC_API_KEY) — does not override already-set env vars.
_load_dotenv(Path(__file__).parent / ".env")

# ── Paths ──────────────────────────────────────────────────────────────────────
PDF_DIR          = Path("data/pdfs")
OUTPUT_DIR       = Path("data/output")
REGISTRY_PATH    = OUTPUT_DIR / "topic_registry.csv"
CHUNKS_CACHE     = OUTPUT_DIR / "chunks.json"

# ── Models ─────────────────────────────────────────────────────────────────────
# Multilingual handles Hindi/Sanskrit terms in Indian polisci PDFs
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# ── LLM: llama.cpp server (Qwen3.5-9B-MTP Q4_K_M, GPU) ──────────────────────
# Start server with: ./start_server.sh before running the pipeline
LLAMA_SERVER_URL    = "http://127.0.0.1:8080"
LLAMA_MODEL_NAME    = "qwen35-9b"       # matches --alias in start_server.sh
LLAMA_PARALLEL_SLOTS = 6               # 6 parallel slots (-c 4096 -np 6); ~6x batch throughput

# ── LLM: Claude API (primary backend — hackathon) ───────────────────────────
# "local" routes llm_client.generate()/generate_batch() back to the llama.cpp
# server above with zero call-site changes (same abstraction the KT doc
# describes for the gemma→Qwen swap — only this pointer changes).
LLM_BACKEND          = _os.environ.get("LLM_BACKEND", "anthropic")  # "anthropic" | "local"
ANTHROPIC_API_KEY    = _os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL      = _os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")  # account only has Sonnet 4.6 access
ANTHROPIC_PARALLEL_SLOTS = 6            # concurrent request cap for generate_batch()

# Fallback chain if Claude fails (auth, rate limit, outage, refusal):
#   1. Claude (primary)
#   2. OpenRouter — free-tier model, OpenAI-compatible endpoint
#   3. Groq — OpenAI-compatible endpoint
#   4. Local llama.cpp server (auto-started if not already running)
# NOTE: per the KT doc ("External LLM APIs are a hard block, not a soft
# preference", Section 10) and this project's own memory, OpenRouter and Groq
# were both already hard-blocked here by Claude Code's data-exfiltration
# classifier when corpus content would be sent to them. Wiring them back in
# at the user's explicit request, to demonstrate the block again.
OPENROUTER_API_KEY   = _os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL     = _os.environ.get("OPENROUTER_MODEL", "openai/gpt-oss-120b:free")
OPENROUTER_BASE_URL  = "https://openrouter.ai/api/v1"

GROQ_API_KEY         = _os.environ.get("GROQ_API_KEY")
GROQ_MODEL           = _os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL        = "https://api.groq.com/openai/v1"

LOCAL_FALLBACK_STARTUP_TIMEOUT = 90     # seconds to wait for start_server.sh to report healthy
LOCAL_FALLBACK_POLL_INTERVAL   = 2.0    # seconds between health-check polls
LOCAL_FALLBACK_HEALTH_TIMEOUT  = 2.0    # per-request timeout for the health check itself

# ── OCR: Claude vision (primary) with Docling fallback ──────────────────────
# Docling extraction (ingest.py:_extract_pdf_with_docling) remains the
# fallback path — triggered only if the Claude OCR pass raises an exception
# (API error, timeout, refusal), not on output-quality heuristics.
USE_CLAUDE_OCR       = _os.environ.get("USE_CLAUDE_OCR", "true").lower() == "true"
OCR_PAGE_DPI         = 150               # page render resolution fed to Claude vision
OCR_MAX_TOKENS       = 4096              # per-page transcription budget

# ── Context-shift chunking ─────────────────────────────────────────────────────
# Sentences are split into a new chunk when cosine similarity between
# consecutive sentence embeddings drops below this threshold.
# Lower = more sensitive to topic shifts (more, smaller chunks)
# Higher = less sensitive (fewer, larger chunks)
CONTEXT_SHIFT_THRESHOLD = 0.45

CHUNK_MIN_TOKENS = 60    # merge chunks shorter than this into the previous
CHUNK_MAX_TOKENS = 450   # hard ceiling — force split even if similarity is high

# ── Noise filtering ────────────────────────────────────────────────────────────
# Alpha ratio: paragraphs where < this fraction of chars are alphabetic
# are flagged as noise (tables, citation lists, number-heavy content).
# Was 0.55 — relaxed to 0.40 to stop dropping legitimate scholarly text
# with dates, percentages, and citations.
ALPHA_RATIO_THRESHOLD = 0.40

# TOC/index page detection: if this fraction of lines on a page are short
# (< TOC_LINE_WORD_LIMIT words), the whole page is skipped.
TOC_SHORT_LINE_RATIO  = 0.65
TOC_LINE_WORD_LIMIT   = 8

# LLM noise filter — runs after chunking (set False to skip, faster)
USE_LLM_NOISE_FILTER = False  # skips a full LLM pass — heuristic filter is enough

# ── KeyBERT ────────────────────────────────────────────────────────────────────
KEYBERT_TOP_N     = 5
KEYBERT_NGRAM_MIN = 1
KEYBERT_NGRAM_MAX = 2
KEYBERT_DIVERSITY = 0.5   # MMR: 0=redundant, 1=maximally diverse

# ── Master labeling ────────────────────────────────────────────────────────────
LABEL_MAX_TOKENS        = 200   # Qwen3-14B is concise — 350 was wasteful
DESCRIPTION_MAX_TOKENS  = 100   # reduced from 150
DESCRIPTION_MIN_LENGTH  = 20    # descriptions shorter than this are treated as missing

# ── Description merging ────────────────────────────────────────────────────────
# Max individual chunk descriptions fed to the merge prompt.
# Groups with more chunks than this get the top-N by description length.
MERGE_MAX_DESCRIPTIONS  = 8
MERGE_MAX_TOKENS        = 150   # reduced from 250

# ── Description enrichment ─────────────────────────────────────────────────────
# Chunks with quality score below this get sent through enrichment routes
ENRICHMENT_QUALITY_THRESHOLD = 0.40

# Route 1 — Internal RAG
# How many similar chunks from same source_doc to pull as context
ENRICHMENT_INTERNAL_TOP_K    = 4

# Route 2 — Web search fallback
# Set False to disable web search entirely (offline environments)
USE_WEB_ENRICHMENT           = True
ENRICHMENT_WEB_MAX_RESULTS   = 3
ENRICHMENT_MAX_TOKENS        = 150   # reduced from 250

# ── Batching ───────────────────────────────────────────────────────────────────
# KeyBERT: how many chunk texts per extract_keywords() call
# Was: 1 per chunk → N×"Batches:1/1" bars
# Now: ceil(N/32) bars total
KEYBERT_BATCH_SIZE  = 32

# LLM rectification: how many chunks sent to pipe() per call
RECTIFY_BATCH_SIZE  = 16

# ── Grouper guards ────────────────────────────────────────────────────────────
MAX_RULE_GROUP_SIZE = 8       # Stage 1: refuse merge if combined group exceeds this
STAGE2_OVERLAP      = 2       # Stage 2: overlapping chunks between consecutive LLM windows
GROUP_COHESION_THRESHOLD = 0.40  # Stage 2.5: split merged groups where cosine sim drops below this
# ── Multiprocessing ────────────────────────────────────────────────────────────
# Pages per worker batch for large PDFs (> PAGE_BATCH_SIZE pages)
# 800-page PDF at 30 pages/batch = 27 batches across N_WORKERS cores
PAGE_BATCH_SIZE = 30
# Workers for parallel page extraction (CPU-bound, no GPU needed)
# Set to os.cpu_count() // 2 for safe default, or override manually
N_WORKERS = max(2, (_os.cpu_count() or 4) // 2)

# ── Nested taxonomy ────────────────────────────────────────────────────────────
MACRO_THRESHOLD         = 0.50
MICRO_THRESHOLD         = 0.60
NESTED_OUTPUT_PATH      = OUTPUT_DIR / "enterprise_nested_topics.json"

# ── Context profiler ──────────────────────────────────────────────────────────
PROFILE_CACHE_DIR       = OUTPUT_DIR / "profiles"
PROMPT_OUTPUT_DIR       = OUTPUT_DIR / "prompts"

# ── Registry summarization ─────────────────────────────────────────────────────
SUMMARY_MAX_TOKENS      = 280  # 80 output + 200 thinking overhead

# ── Label normalization ───────────────────────────────────────────────────────
LABEL_MERGE_THRESHOLD   = 0.82  # Cosine similarity above which labels are merged
LABEL_MIN_WORDS         = 2     # Labels shorter than this are flagged as garbage

# ── Cross-version matching ────────────────────────────────────────────────────
CROSS_VERSION_MATCH_THRESHOLD = 0.75  # Embedding similarity for cross-doc label merge