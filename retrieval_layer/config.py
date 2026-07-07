import os as _os
from pathlib import Path

from dotenv import load_dotenv as _load_dotenv

# Same key as Stage 1 — BYOK, loaded from Stage 1/.env (ANTHROPIC_API_KEY),
# never hardcoded here. Does not override an already-set env var.
_load_dotenv(Path(__file__).parent.parent / "Stage 1" / ".env")

# ── Paths ──────────────────────────────────────────────────────────────────────
STAGE1_OUTPUT   = Path(__file__).parent.parent / "Stage 1" / "data" / "output"
NESTED_JSON     = STAGE1_OUTPUT / "enterprise_nested_topics.json"
FILTERED_CHUNKS = STAGE1_OUTPUT / "filtered_chunks.json"
PARENT_RELATIONSHIPS = STAGE1_OUTPUT / "parent_relationship_clusters.json"
EVOLUTION_CARDS = STAGE1_OUTPUT / "evolution_cards_cache.json"  # written by evolution_analyzer.py
CROSS_CORPUS_RELATIONSHIPS = STAGE1_OUTPUT / "cross_corpus_relationship_clusters.json"  # written by cross_corpus_relationship.py
INDEX_DIR       = Path(__file__).parent / "index"

# ── LLM server (same as Stage 1) ─────────────────────────────────────────────
LLAMA_SERVER_URL  = "http://127.0.0.1:8080"
LLAMA_MODEL_NAME  = "qwen35-9b"

# ── LLM: Claude API (primary backend — reformulation, rewriting, generation) ──
LLM_BACKEND       = _os.environ.get("LLM_BACKEND", "anthropic")  # "anthropic" | "local"
ANTHROPIC_API_KEY = _os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL   = _os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")  # account only has Sonnet 4.6 access

# Fallback chain if Claude fails (auth, rate limit, outage, refusal):
#   1. Claude (primary)
#   2. OpenRouter — free-tier model, OpenAI-compatible endpoint
#   3. Groq — OpenAI-compatible endpoint
#   4. Local llama.cpp server (auto-started if not already running, via
#      Stage 1's start_server.sh). Separate from LLM_BACKEND=local, which
#      skips Claude entirely.
OPENROUTER_API_KEY   = _os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL     = _os.environ.get("OPENROUTER_MODEL", "openai/gpt-oss-120b:free")
OPENROUTER_BASE_URL  = "https://openrouter.ai/api/v1"

GROQ_API_KEY         = _os.environ.get("GROQ_API_KEY")
GROQ_MODEL           = _os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL        = "https://api.groq.com/openai/v1"

STAGE1_DIR = Path(__file__).parent.parent / "Stage 1"
LOCAL_FALLBACK_STARTUP_TIMEOUT = 90     # seconds to wait for start_server.sh to report healthy
LOCAL_FALLBACK_POLL_INTERVAL   = 2.0    # seconds between health-check polls
LOCAL_FALLBACK_HEALTH_TIMEOUT  = 2.0    # per-request timeout for the health check itself

# ── Embedding model (same as Stage 1 pipeline) ────────────────────────────────
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM   = 384

# ── HNSW ──────────────────────────────────────────────────────────────────────
HNSW_M               = 16    # connections per node (higher = better recall, more RAM)
HNSW_EF_CONSTRUCTION = 200   # build-time accuracy (one-time cost)
HNSW_EF_QUERY        = 50    # query-time accuracy vs speed (tune up for better recall)
HNSW_MAX_ELEMENTS    = 50_000

# ── Routing ───────────────────────────────────────────────────────────────────
# Any (parent, sub) pair within this of the best content-hit score gets
# included as a candidate section, instead of hard-committing to a single
# argmax that a small score gap could flip incorrectly. Originally 0.07,
# calibrated against a 2-vector-type index (topic+QnA). Raised to 0.13 after
# adding description/chunk vectors (4 types, 6432 total) pushed the score
# ceiling higher, which made 0.07 too tight and excluded previously-good
# candidates — re-verified against the full 19-query eval, not just the
# original 2 hand-picked cases.
ROUTE_AMBIGUITY_GAP  = 0.13
# HNSW candidates before reranking. Raised 20->50->100 as the index grew
# through today's changes (5021->6432 vectors across 4 types competing for
# slots). 100 vs 50 tested near-identical on Pipeline's own win count (one
# flip was noise on the deliberate garbage-query sanity check, not a real
# loss), but meaningfully improved Best-of-N (100% keyword hit, best mean
# score of the session) at no latency cost. Tested 200 too — Pipeline's
# scores were byte-identical to 100 on every query (hard plateau), and
# Best-of-N got worse, not better. 100 is the ceiling for this index size.
TOP_K_HNSW           = 100
                               # for slots, since the earlier 50-vs-100 test predates that change
TOP_K_FINAL          = 5     # chunks returned to LLM after reranking

# ── Reranker ──────────────────────────────────────────────────────────────────
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_DIR             = Path(__file__).parent / "chroma_db"
CORPUS_COLLECTION      = "corpus"       # raw chunk texts — what things ARE
INTELLIGENCE_COLLECTION = "intelligence" # topic/doc summaries — overviews
DELTA_COLLECTION       = "delta"        # delta analysis — what CHANGED
OVERLAP_COLLECTION     = "corpus_overlap"  # traditional sliding-window chunks from raw markdown

# Stage 1 outputs consumed by ChromaDB
REGISTRY_CSV  = STAGE1_OUTPUT / "topic_registry.csv"
DELTA_DIR     = STAGE1_OUTPUT / "delta_reports"      # written by delta_analyzer.py
MARKDOWN_DIR  = Path(__file__).parent.parent / "Stage 1" / "data" / "pdfs"

# Query classification thresholds
CHROMA_N_CORPUS      = 10   # chunks returned from corpus collection
CHROMA_N_INTELLIGENCE = 6   # chunks returned from intelligence collection
CHROMA_N_DELTA        = 8   # chunks returned from delta collection
CHROMA_N_EVOLUTION    = 5   # evolution cards merged into the delta path
CHROMA_N_CROSS_CORPUS = 3   # cross-corpus relationship docs, ladder rung only

# ── Redis response cache (hackathon demo preset) ──────────────────────────────
# Overridable via env for deployed hosts using a managed Redis add-on
# (e.g. Render Key Value, Railway Redis) — defaults match local dev.
REDIS_URL        = _os.environ.get("REDIS_URL")  # if set, redis_cache.py uses this directly (takes priority)
REDIS_HOST       = _os.environ.get("REDIS_HOST", "127.0.0.1")
REDIS_PORT       = int(_os.environ.get("REDIS_PORT", "6379"))
REDIS_DB         = int(_os.environ.get("REDIS_DB", "0"))
REDIS_KEY_PREFIX = "hackcache:v1:"    # bump version suffix if answer format changes
REDIS_TTL_SECONDS = None              # None = no expiry (static demo preset, not live traffic)
CACHE_PRESET_PATH = STAGE1_OUTPUT / "hackathon_cache_preset.json"
CACHE_PRESET_DOC  = STAGE1_OUTPUT / "hackathon_cache_demo.docx"
CACHE_PRESET_MD   = STAGE1_OUTPUT / "hackathon_cache_demo.md"

# Confidence threshold — if top rerank score on HNSW falls below this, first
# try sibling sub-parents under the same routed parent(s) (router.py:
# Router.expand_siblings — free, reuses the same query's hits). Separately,
# an empty HNSW result still widens straight to corpus (retriever.py).
CONFIDENCE_THRESHOLD = 0.30

# Gates the whole confidence ladder (sibling expansion -> cross-parent
# bridge -> relationship-cluster lookup) in retriever.py:_retrieve_specific.
# Measured on the 19-query eval: zero retrieval-quality improvement on any
# query, but a real 4x latency cost (+139ms) whenever any tier fires. Off by
# default until a real query demonstrates the ladder resolves something the
# plain router+rerank path can't — flip to True to re-enable it.
ENABLE_CONFIDENCE_LADDER = False
