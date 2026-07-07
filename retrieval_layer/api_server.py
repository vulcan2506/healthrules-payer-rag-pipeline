"""
api_server.py
──────────────
Thin FastAPI layer over the existing retrieval_layer + Stage 1 backend, for
the Next.js frontend. Every handler is a direct call into existing modules
(redis_cache, retriever, config, llm_client) or a subprocess invocation of an
existing script (main.py, run_tail.py) — no retrieval, gating, reranking, OCR,
delta-analysis, or citation-selection logic lives in this file.

Run:
    cd retrieval_layer && uvicorn api_server:app --reload --port 8000

CORS is open to the local Next.js dev server only (http://localhost:3000).
"""

import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
import llm_client
import redis_cache
import retriever
import session as session_module

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="HealthRules Payer Knowledge API")

# ALLOWED_ORIGINS: comma-separated list, e.g. "https://your-app.vercel.app,http://localhost:3000".
# Defaults to local dev only — set this env var on the deployed backend host
# to the deployed Vercel frontend URL (and localhost, if you also test locally
# against the deployed backend).
_allowed_origins = [
    o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",") if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STAGE1_DIR = config.STAGE1_DIR
PDF_DIR    = STAGE1_DIR / "data" / "pdfs"
OUTPUT_DIR = config.STAGE1_OUTPUT
ENV_PATH   = STAGE1_DIR / ".env"

# ── Conversation sessions ────────────────────────────────────────────────────
# One ConversationSession per frontend chat thread (session.py — already used
# by cli.py's default REPL loop for every non-`--session` mode, including the
# gated/cached path this endpoint wraps). Follow-ups like "explain that in
# more detail" get rewritten into a standalone query BEFORE retrieval, using
# the trimmed conversation window — without this, a pronoun-only follow-up is
# retrieved on its own literal (near-contentless) text, pulls in unrelated
# chunks, and the LLM ends up answering ungrounded — reads as hallucination.
# Keyed by the frontend's per-thread session_id; a request with no session_id
# stays fully stateless (unchanged prior behavior).
_sessions: Dict[str, session_module.ConversationSession] = {}
_sessions_lock = threading.Lock()


def _get_session(session_id: str) -> session_module.ConversationSession:
    with _sessions_lock:
        sess = _sessions.get(session_id)
        if sess is None:
            sess = session_module.ConversationSession()
            _sessions[session_id] = sess
        return sess


@app.get("/api/health")
def health() -> Dict:
    return {"status": "ok"}


# ── /api/chat ────────────────────────────────────────────────────────────────
# Mirrors cli.py's real branching exactly (see cli.py's REPL loop):
#   - forcing `intent` only ever has effect on the raw/diagnostic path
#     (redis_cache.answer_query() never receives an intent — confirmed by
#     reading cli.py's gated-mode branch, which only forwards mode/best_of).
#     So selecting a non-"auto" intent here switches to raw mode automatically,
#     same as cli.py's `--intent` implicitly requiring `--raw` to have effect.
#   - raw + best_of together uses retrieve_best_of_n (cli.py's
#     `elif turn_raw and turn_best_of:` branch) — best-of still applies in
#     diagnostic mode, mode/best_of are just otherwise unused there.

class ChatRequest(BaseModel):
    query: str
    mode: str = "concise"           # "concise" | "detailed" — redis_cache.answer_query's two modes
    best_of: int = 3                # reformulation count — gated path (mode=concise) or raw path, both real
    intent: Optional[str] = None    # forcing this switches to the raw/diagnostic path (see above)
    version: Optional[str] = None   # only meaningful on the raw/diagnostic path
    raw: bool = False                # explicit opt-in to the raw/diagnostic path with no intent forced
    session_id: Optional[str] = None  # frontend chat-thread id — omit for stateless single-shot calls


@app.post("/api/chat")
def chat(req: ChatRequest) -> Dict:
    if not req.query.strip():
        raise HTTPException(400, "query is required")

    sess = _get_session(req.session_id) if req.session_id else None
    if sess is not None:
        standalone, was_rewritten = sess.prepare_turn(req.query)
    else:
        standalone, was_rewritten = req.query, False

    if req.raw or req.intent:
        t0 = time.time()
        if req.best_of and req.best_of > 1:
            result = retriever.retrieve_best_of_n(
                standalone, n=req.best_of, intent=req.intent, version=req.version
            )
        else:
            result = retriever.retrieve(standalone, intent=req.intent, version=req.version)
        if sess is not None:
            sess.record_turn(req.query, result)  # no answer generated on the raw path — nothing to add_assistant_turn
        return {
            "query":            req.query,
            "standalone_query": standalone,
            "was_rewritten":    was_rewritten,
            "method":      result.get("intent", "specific"),
            "mode":        "raw",
            "answer":      None,
            "confidence":  retriever._top_rerank_score(result),
            "chunks":      result.get("chunks", []),
            "from_cache":  False,
            "latency_s":   time.time() - t0,
        }

    if req.mode not in ("concise", "detailed"):
        raise HTTPException(400, f"mode must be 'concise' or 'detailed', got {req.mode!r}")

    try:
        result = redis_cache.answer_query(standalone, mode=req.mode, best_of=req.best_of)
    except Exception as e:
        log.exception("chat request failed")
        raise HTTPException(500, str(e))

    if sess is not None:
        sess.record_turn(req.query, result)
        if result.get("answer"):
            sess.add_assistant_turn(result["answer"])

    return {**result, "query": req.query, "standalone_query": standalone, "was_rewritten": was_rewritten}


# ── /api/settings ────────────────────────────────────────────────────────────

@app.get("/api/settings/status")
def settings_status() -> Dict:
    return {
        "backend":  config.LLM_BACKEND,
        "model":    config.ANTHROPIC_MODEL,
        "has_key":  bool(config.ANTHROPIC_API_KEY),
    }


class ApiKeyRequest(BaseModel):
    api_key: str


def _write_env_var(path: Path, key: str, value: str) -> None:
    """Rewrites one KEY=value line in a .env file in place, preserving every other line."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    out, found = [], False
    for line in lines:
        if line.startswith(f"{key}="):
            out.append(f"{key}={value}")
            found = True
        else:
            out.append(line)
    if not found:
        out.append(f"{key}={value}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)


@app.post("/api/settings/key")
def settings_set_key(req: ApiKeyRequest) -> Dict:
    key = req.api_key.strip()
    if not key:
        raise HTTPException(400, "api_key is required")

    _write_env_var(ENV_PATH, "ANTHROPIC_API_KEY", key)
    config.ANTHROPIC_API_KEY = key
    os.environ["ANTHROPIC_API_KEY"] = key
    llm_client._anthropic_client = None  # force reconstruction with the new key

    try:
        # Validate against Anthropic directly — bypass config.LLM_BACKEND and the
        # local-server fallback entirely, since this is specifically "does THIS
        # key work", not "does chat() succeed by any means".
        llm_client._chat_anthropic("Reply with exactly: PONG", None, 5, None, False, 10.0)
        return {"valid": True}
    except Exception as e:
        return {"valid": False, "error": _friendly_anthropic_error(e)}


def _friendly_anthropic_error(e: Exception) -> str:
    """Anthropic SDK exceptions stringify to a raw dict repr — translate the
    common cases into something worth showing a non-technical user."""
    name = type(e).__name__
    if name == "AuthenticationError":
        return "That key was rejected by Anthropic (invalid or revoked)."
    if name == "PermissionDeniedError":
        return "That key doesn't have permission to use this model."
    if name == "RateLimitError":
        return "Rate limited by Anthropic — the key is valid, but try again shortly."
    if name in ("APIConnectionError", "APITimeoutError"):
        return "Couldn't reach the Anthropic API — check your network connection."
    return f"{name}: {e}"


# ── /api/upload ──────────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> Dict:
    safe_name = os.path.basename(file.filename or "")
    if not safe_name or safe_name in (".", "..") or not safe_name.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF uploads are supported today.")

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    content = await file.read()
    (PDF_DIR / safe_name).write_bytes(content)
    return {"filename": safe_name, "size_bytes": len(content)}


# ── /api/process ─────────────────────────────────────────────────────────────
# Runs Stage 1's own main.py + run_tail.py --skip-eval as subprocesses — this
# reprocesses the ENTIRE data/pdfs/ corpus (main.py has no incremental/single-
# file mode), not just the file just uploaded. See Known Gaps in the plan.

_jobs: Dict[str, Dict] = {}
_jobs_lock = threading.Lock()
_STAGE_RE = re.compile(r"\[\d+/\d+\][^\n]*")


def _tail_log_into_job(proc: subprocess.Popen, log_path: Path, job_id: str, poll_interval: float = 1.0) -> None:
    while proc.poll() is None:
        time.sleep(poll_interval)
        _update_job_from_log(log_path, job_id)
    _update_job_from_log(log_path, job_id)


def _update_job_from_log(log_path: Path, job_id: str) -> None:
    if not log_path.exists():
        return
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    matches = _STAGE_RE.findall(text)
    with _jobs_lock:
        if job_id not in _jobs:
            return
        if matches:
            _jobs[job_id]["stage"] = matches[-1].strip()
        _jobs[job_id]["log_tail"] = text[-4000:]


def _run_process_job(job_id: str) -> None:
    log_path = OUTPUT_DIR / f"process_{job_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with _jobs_lock:
        _jobs[job_id] = {"status": "running", "stage": "Starting main.py (ingestion)…", "log_tail": ""}

    try:
        with open(log_path, "w", encoding="utf-8") as logf:
            proc = subprocess.Popen(
                [sys.executable, "main.py"], cwd=str(STAGE1_DIR),
                stdout=logf, stderr=subprocess.STDOUT,
            )
            _tail_log_into_job(proc, log_path, job_id)
            if proc.returncode != 0:
                raise RuntimeError(f"main.py exited with code {proc.returncode}")

        with _jobs_lock:
            _jobs[job_id]["stage"] = "Starting run_tail.py (taxonomy, delta, indexing)…"

        with open(log_path, "a", encoding="utf-8") as logf:
            proc = subprocess.Popen(
                [sys.executable, "run_tail.py", "--skip-eval"], cwd=str(STAGE1_DIR),
                stdout=logf, stderr=subprocess.STDOUT,
            )
            _tail_log_into_job(proc, log_path, job_id)
            if proc.returncode != 0:
                raise RuntimeError(f"run_tail.py exited with code {proc.returncode}")

        with _jobs_lock:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["stage"] = "Complete"
    except Exception as e:
        log.exception(f"process job {job_id} failed")
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["stage"] = str(e)


@app.post("/api/process")
def process() -> Dict:
    job_id = uuid.uuid4().hex[:12]
    threading.Thread(target=_run_process_job, args=(job_id,), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/process/{job_id}")
def process_status(job_id: str) -> Dict:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job_id")
    return {"job_id": job_id, **job}


# ── /api/knowledge ───────────────────────────────────────────────────────────
# Curated view of Stage 1/data/output/ — the real files discovered while
# planning this (not the illustrative names from the original spec).

_DISPLAY_NAMES = {
    "hierarchy_summaries":               "Knowledge Hierarchy",
    "topic_summaries":                   "Topic Insights",
    "delta_reports":                     "Delta Reports",
    "version_delta_report.md":           "Version Differences",
    "version_evolution_report.md":       "Version Evolution",
    "enterprise_nested_topics.json":     "Knowledge Structure",
    "parent_relationship_clusters.json": "Relationship Graph",
    "eval_report.md":                    "Evaluation Report",
}
_TOP_LEVEL_ALLOWLIST = set(_DISPLAY_NAMES)
_ALLOWED_EXTENSIONS  = {".md", ".json"}


def _display_name(name: str) -> str:
    return _DISPLAY_NAMES.get(name, name)


def _build_tree(path: Path, top_level: bool = True) -> List[Dict]:
    entries = []
    for child in sorted(path.iterdir()):
        if top_level and child.name not in _TOP_LEVEL_ALLOWLIST:
            continue
        if child.name.startswith(".") or ".bak" in child.name:
            continue
        rel = child.relative_to(OUTPUT_DIR)
        if child.is_dir():
            entries.append({
                "type": "directory", "name": child.name,
                "display_name": _display_name(child.name), "path": str(rel),
                "children": _build_tree(child, top_level=False),
            })
        elif child.suffix in _ALLOWED_EXTENSIONS:
            entries.append({
                "type": "file", "name": child.name,
                "display_name": _display_name(child.name), "path": str(rel),
                "extension": child.suffix, "size_bytes": child.stat().st_size,
            })
    return entries


_MAX_MARKDOWN_CHARS = 200_000  # some source docs are 2MB+ — cap so the browser doesn't choke


def _source_documents() -> List[Dict]:
    """
    Pre-converted markdown alternatives to Docling extraction — see
    ingest.py:_load_preconverted_md, which uses these instead of re-running
    OCR when present. Surfaced here read-only, under a "pdfs/" path prefix
    so /api/knowledge/file knows to resolve against PDF_DIR instead of
    OUTPUT_DIR.
    """
    entries = []
    if not PDF_DIR.exists():
        return entries
    for md_file in sorted(PDF_DIR.glob("*_Converted.md")):
        entries.append({
            "type": "file", "name": md_file.name,
            "display_name": md_file.stem.replace("_Converted", "").replace("_", " "),
            "path": f"pdfs/{md_file.name}",
            "extension": ".md", "size_bytes": md_file.stat().st_size,
        })
    return entries


@app.get("/api/knowledge/files")
def knowledge_files() -> Dict:
    tree = _build_tree(OUTPUT_DIR)
    source_docs = _source_documents()
    if source_docs:
        tree.insert(0, {
            "type": "directory", "name": "source_documents",
            "display_name": "Source Documents", "path": "pdfs",
            "children": source_docs,
        })
    return {"tree": tree}


@app.get("/api/knowledge/file")
def knowledge_file(path: str) -> Dict:
    if path.startswith("pdfs/"):
        target = (PDF_DIR / path[len("pdfs/"):]).resolve()
        root = PDF_DIR.resolve()
    else:
        target = (OUTPUT_DIR / path).resolve()
        root = OUTPUT_DIR.resolve()

    if not target.is_relative_to(root):
        raise HTTPException(400, "Invalid path")
    if not target.exists() or not target.is_file():
        raise HTTPException(404, "File not found")

    if target.suffix == ".json":
        return {"type": "json", "content": json.loads(target.read_text(encoding="utf-8"))}

    text = target.read_text(encoding="utf-8")
    if len(text) > _MAX_MARKDOWN_CHARS:
        text = text[:_MAX_MARKDOWN_CHARS] + "\n\n---\n*Truncated — file is larger than the preview limit.*"
    return {"type": "markdown", "content": text}


# ── /api/visualize ───────────────────────────────────────────────────────────
# Optional, post-answer-only feature: turns an already-grounded answer +
# its citations into a Mermaid.js diagram spec. Deliberately generated from
# the FINAL ANSWER + EVIDENCE, never the raw user query — this is what keeps
# it grounded instead of a second, independent (and potentially hallucinated)
# generation. Anthropic has no image-generation API, so this produces
# diagram-as-code (rendered client-side), not a raster image — Infographic
# and Comic Strip are intentionally NOT implemented here (see frontend).

_VIZ_TYPES = {
    "flowchart":    "a process flowchart — use Mermaid `flowchart TD` syntax showing sequential steps",
    "timeline":     "a timeline — use Mermaid `timeline` syntax showing chronological/version progression",
    "relationship": "a relationship diagram — use Mermaid `graph LR` syntax showing interconnected concepts, NOT a linear sequence",
    "mindmap":      "a mind map — use Mermaid `mindmap` syntax showing a central topic with branching sub-topics",
    "architecture": "an architecture diagram — use Mermaid `flowchart TD` syntax showing system components and how they interact",
}

_VIZ_SUGGEST_RULES = [
    (re.compile(r"chang(e|ed|es)?\b.*\bversion|\bversion.*\bchang|\bevolution\b|\bbetween v?\d", re.I), "timeline"),
    (re.compile(r"\brelat(e|ed|ionship)|\bconnect|\bassociat|\bversus\b|\bvs\b", re.I), "relationship"),
    (re.compile(r"\bhierarch|\bdomains?\b|\ball (the )?(topics|categories)|\blist all\b", re.I), "mindmap"),
    (re.compile(r"\barchitecture|\bsystem design\b|\bcomponents?\b interact|\bkubernetes\b", re.I), "architecture"),
    (re.compile(r"how does .* work|\bprocess\b|\bworkflow\b|\bexplain\b|\badjudicat", re.I), "flowchart"),
]


def _suggest_viz_type(question: str) -> str:
    for pattern, viz in _VIZ_SUGGEST_RULES:
        if pattern.search(question):
            return viz
    return "flowchart"


@app.get("/api/visualize/suggest")
def visualize_suggest(question: str) -> Dict:
    return {"suggested": _suggest_viz_type(question)}


class VisualizeRequest(BaseModel):
    question: str
    answer: str
    chunks: List[Dict] = []
    viz_type: str  # one of _VIZ_TYPES


_VISUALIZE_PROMPT = """You will produce a Mermaid.js diagram based STRICTLY on the grounded answer and evidence below. Do not introduce any fact, step, or relationship that is not explicitly present in the answer or evidence — if the answer doesn't specify enough detail for a rich diagram, keep the diagram simple rather than inventing detail.

Question: {question}

Answer: {answer}

Evidence:
{evidence}

Create {viz_hint}.

Requirements:
- Output ONLY a single Mermaid code block (```mermaid ... ```) — no other text before or after.
- Keep every label concise (a few words).
- Use only what's stated in the answer/evidence above.
- Clean and presentation-ready — enterprise style, no decorative elements.
"""


@app.post("/api/visualize")
def visualize(req: VisualizeRequest) -> Dict:
    if req.viz_type not in _VIZ_TYPES:
        raise HTTPException(400, f"Unsupported viz_type: {req.viz_type!r}")

    evidence = "\n".join(
        f"[{i + 1}] {c.get('section_header', '')}: {c.get('text', '')[:400]}"
        for i, c in enumerate(req.chunks[:6])
    ) or "(no additional evidence provided beyond the answer)"

    prompt = _VISUALIZE_PROMPT.format(
        question=req.question, answer=req.answer, evidence=evidence, viz_hint=_VIZ_TYPES[req.viz_type]
    )

    try:
        raw = llm_client.chat(prompt, max_tokens=1500)
    except Exception as e:
        raise HTTPException(500, str(e))

    match = re.search(r"```(?:mermaid)?\s*\n(.*?)```", raw, re.DOTALL)
    mermaid_code = match.group(1).strip() if match else raw.strip()
    return {"mermaid": mermaid_code, "viz_type": req.viz_type}
