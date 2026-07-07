"""
llm_client.py
─────────────
Single-tier LLM client backed by a local llama.cpp server.

All calls route to the same OpenAI-compatible endpoint at LLAMA_SERVER_URL.
The server runs Qwen3-14B-Q4_K_M locally — no external API, no rate limits.

Start the server before running the pipeline:
  bash start_server.sh

── Dynamic token budgeting ──────────────────────────────────────────────────
Each call site should pass stop sequences matching its expected output format.
This terminates generation the moment the output is complete, freeing the
parallel slot for the next prompt without waiting for unused token budget.

Use the budget() helper to right-size max_tokens from input length:
  max_tokens=llm_client.budget(prompt, ratio=4, ceil=200)

Stop sequences by output type (pass as stop=[...]):
  JSON object   → stop=STOP_JSON
  YES/NO flag   → stop=STOP_FLAG
  Free text     → stop=STOP_TEXT
  Short label   → stop=STOP_LABEL
"""

import gc
import time
import logging
import subprocess
import concurrent.futures
from pathlib import Path
from typing import List, Optional

import httpx
from openai import OpenAI

import config

log = logging.getLogger(__name__)

_anthropic_client = None  # lazy — only imported/constructed if LLM_BACKEND == "anthropic"
_local_server_starting = False  # guards against launching start_server.sh twice from parallel threads


def _is_local_server_up() -> bool:
    try:
        r = httpx.get(f"{config.LLAMA_SERVER_URL}/health", timeout=config.LOCAL_FALLBACK_HEALTH_TIMEOUT)
        return r.status_code == 200
    except Exception:
        return False


def _ensure_local_server() -> bool:
    """
    Starts the local llama.cpp server (start_server.sh) if it isn't already
    reachable, then polls /health until it responds or LOCAL_FALLBACK_STARTUP_TIMEOUT
    is hit. Returns True once the server is confirmed up, False otherwise.
    Safe to call from multiple threads — only the first caller launches the
    process; the rest just poll.
    """
    global _local_server_starting

    if _is_local_server_up():
        return True

    script = Path(__file__).parent / "start_server.sh"
    if not _local_server_starting:
        if not script.exists():
            log.error(f"Cannot start local fallback server — {script} not found")
            return False
        _local_server_starting = True
        log.warning("Claude unavailable — starting local llama.cpp server (start_server.sh) as fallback...")
        log_path = config.OUTPUT_DIR / "llama_server_fallback.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as logf:
            subprocess.Popen(
                ["bash", str(script)],
                cwd=str(script.parent),
                stdout=logf,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # detach — survives after this process/thread
            )

    waited = 0.0
    while waited < config.LOCAL_FALLBACK_STARTUP_TIMEOUT:
        time.sleep(config.LOCAL_FALLBACK_POLL_INTERVAL)
        waited += config.LOCAL_FALLBACK_POLL_INTERVAL
        if _is_local_server_up():
            log.info(f"Local llama.cpp server is up after {waited:.0f}s")
            return True

    log.error(f"Local llama.cpp server did not become ready within {config.LOCAL_FALLBACK_STARTUP_TIMEOUT}s")
    return False


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError(
                "LLM_BACKEND is 'anthropic' but ANTHROPIC_API_KEY is not set. "
                "BYOK: add your own key to Stage 1/.env — ANTHROPIC_API_KEY=sk-ant-... "
                "(never paste it into chat/logs). Set LLM_BACKEND=local in .env to use "
                "the local llama.cpp server instead."
            )
        import anthropic
        _anthropic_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        log.info(f"Anthropic client → model={config.ANTHROPIC_MODEL}")
    return _anthropic_client


def _progress(done: int, total: int, desc: str, t0: float) -> None:
    elapsed = time.time() - t0
    rate = done / elapsed if elapsed > 0 else 0
    rem = (total - done) / rate if rate > 0 else 0
    log.info(f"{desc}: {done}/{total} ({100*done//total}%) — ~{int(rem//60)}m{int(rem%60):02d}s remaining")

# ── Stop sequence constants ───────────────────────────────────────────────────
STOP_JSON  = ["```\n", "\n\n"]        # JSON block ends at closing brace line
STOP_FLAG  = ["\n", ".", ",", " "]   # YES/NO/score — stop after first token
STOP_TEXT  = ["\n\n\n"]              # Free-text paragraphs — stop at blank line
STOP_LABEL = ["\n\n", "```"]         # Short structured outputs

_client: Optional[OpenAI] = None
_openrouter_client: Optional[OpenAI] = None
_groq_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=config.LLAMA_SERVER_URL + "/v1",
            api_key="none",
        )
        log.info(f"llama.cpp client → {config.LLAMA_SERVER_URL}")
    return _client


def _get_openrouter_client() -> Optional[OpenAI]:
    global _openrouter_client
    if not config.OPENROUTER_API_KEY:
        return None
    if _openrouter_client is None:
        _openrouter_client = OpenAI(
            base_url=config.OPENROUTER_BASE_URL,
            api_key=config.OPENROUTER_API_KEY,
        )
        log.info(f"OpenRouter client → model={config.OPENROUTER_MODEL}")
    return _openrouter_client


def _get_groq_client() -> Optional[OpenAI]:
    global _groq_client
    if not config.GROQ_API_KEY:
        return None
    if _groq_client is None:
        _groq_client = OpenAI(
            base_url=config.GROQ_BASE_URL,
            api_key=config.GROQ_API_KEY,
        )
        log.info(f"Groq client → model={config.GROQ_MODEL}")
    return _groq_client


def budget(
    prompt: str,
    ratio: float = 4.0,
    floor: int = 50,
    ceil: int = 400,
) -> int:
    """
    Estimate output token budget from input prompt length.

    ratio: input_words / ratio = expected output tokens
           Lower ratio = more output relative to input (complex tasks)
           Higher ratio = less output relative to input (extraction/classification)
    floor: minimum tokens regardless of input size
    ceil:  hard upper limit — prevents runaway on edge cases
    """
    input_words = len(prompt.split())
    return max(floor, min(ceil, int(input_words / ratio)))


def _chat(
    prompt: str,
    max_tokens: int,
    temperature: float,
    system_prompt: Optional[str],
    stop: Optional[List[str]] = None,
    enable_thinking: bool = False,
) -> str:
    if config.LLM_BACKEND == "anthropic":
        try:
            return _chat_anthropic(prompt, max_tokens, system_prompt, stop, enable_thinking)
        except Exception as e_claude:
            log.warning(f"Claude call failed after retries ({type(e_claude).__name__}: {e_claude}) — trying OpenRouter fallback")

            or_client = _get_openrouter_client()
            if or_client is not None:
                try:
                    return _chat_openai_compatible(
                        or_client, config.OPENROUTER_MODEL, prompt, max_tokens, temperature,
                        system_prompt, stop, enable_thinking, label="OpenRouter",
                    )
                except Exception as e_or:
                    log.warning(f"OpenRouter call failed after retries ({type(e_or).__name__}: {e_or}) — trying Groq fallback")
            else:
                log.warning("OPENROUTER_API_KEY not set — skipping OpenRouter fallback, trying Groq")

            groq_client = _get_groq_client()
            if groq_client is not None:
                try:
                    return _chat_openai_compatible(
                        groq_client, config.GROQ_MODEL, prompt, max_tokens, temperature,
                        system_prompt, stop, enable_thinking, label="Groq",
                    )
                except Exception as e_groq:
                    log.warning(f"Groq call failed after retries ({type(e_groq).__name__}: {e_groq}) — falling back to local llama.cpp server")
            else:
                log.warning("GROQ_API_KEY not set — skipping Groq fallback, trying local llama.cpp server")

            if not _ensure_local_server():
                raise RuntimeError(
                    "Claude, OpenRouter, and Groq all failed, and the local llama.cpp "
                    "fallback server did not come up"
                ) from e_claude
            return _chat_local(prompt, max_tokens, temperature, system_prompt, stop, enable_thinking)
    return _chat_local(prompt, max_tokens, temperature, system_prompt, stop, enable_thinking)


def _chat_anthropic(
    prompt: str,
    max_tokens: int,
    system_prompt: Optional[str],
    stop: Optional[List[str]],
    enable_thinking: bool,
) -> str:
    client = _get_anthropic_client()
    kwargs = dict(
        model=config.ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    if system_prompt:
        kwargs["system"] = system_prompt
    if stop:
        kwargs["stop_sequences"] = stop
    if enable_thinking:
        kwargs["thinking"] = {"type": "adaptive"}
    # temperature is intentionally omitted — Opus 4.8 rejects sampling params
    # outright, and every call site here already runs at temperature=0.0.

    for attempt in range(3):
        try:
            resp = client.messages.create(**kwargs)
            return next((b.text for b in resp.content if b.type == "text"), "").strip()
        except Exception as e:
            if attempt < 2:
                log.warning(f"Anthropic API error (attempt {attempt+1}/3), retrying...")
            else:
                log.error(f"Anthropic API error after 3 attempts: {e}")
                raise


def _chat_openai_compatible(
    client: OpenAI,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    system_prompt: Optional[str],
    stop: Optional[List[str]],
    enable_thinking: bool,
    label: str,
) -> str:
    """Shared dispatch for any OpenAI-compatible fallback endpoint (OpenRouter, Groq).
    enable_thinking is accepted for signature parity with _chat_local but unused —
    neither provider's chat.completions endpoint supports llama.cpp's
    chat_template_kwargs extension."""
    messages = [{"role": "user", "content": prompt}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    kwargs = dict(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if stop:
        kwargs["stop"] = stop

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < 2:
                log.warning(f"{label} API error (attempt {attempt+1}/3), retrying...")
            else:
                log.error(f"{label} API error after 3 attempts: {e}")
                raise


def _chat_local(
    prompt: str,
    max_tokens: int,
    temperature: float,
    system_prompt: Optional[str],
    stop: Optional[List[str]] = None,
    enable_thinking: bool = False,
) -> str:
    client = _get_client()
    messages = [
        {"role": "user", "content": prompt},
    ]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    kwargs = dict(
        model=config.LLAMA_MODEL_NAME,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
    )
    if stop:
        kwargs["stop"] = stop

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < 2:
                log.warning(f"llama.cpp timeout (attempt {attempt+1}/3), retrying...")
            else:
                log.error(f"llama.cpp inference error after 3 attempts: {e}")
                raise


# ── Public API ────────────────────────────────────────────────────────────────

def generate(
    prompt: str,
    max_tokens: int = 500,
    temperature: float = 0.0,
    system_prompt: Optional[str] = None,
    stop: Optional[List[str]] = None,
    enable_thinking: bool = False,
) -> str:
    return _chat(prompt, max_tokens, temperature, system_prompt, stop, enable_thinking)


def generate_batch(
    prompts: List[str],
    max_tokens: int = 500,
    temperature: float = 0.0,
    system_prompt: Optional[str] = None,
    desc: str = "LLM Inference",
    stop: Optional[List[str]] = None,
    enable_thinking: bool = False,
) -> List[str]:
    if not prompts:
        return []

    results: List[Optional[str]] = [None] * len(prompts)
    total = len(prompts)
    milestone = max(1, total // 10)  # log every 10%

    def _call(idx_prompt):
        idx, prompt = idx_prompt
        return idx, _chat(prompt, max_tokens, temperature, system_prompt, stop, enable_thinking)

    slots = config.ANTHROPIC_PARALLEL_SLOTS if config.LLM_BACKEND == "anthropic" else config.LLAMA_PARALLEL_SLOTS
    t0 = time.time()
    log.info(f"{desc}: starting {total} items ({slots} parallel slots, backend={config.LLM_BACKEND})")
    done = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=slots) as pool:
        futures = {pool.submit(_call, (i, p)): i for i, p in enumerate(prompts)}
        for fut in concurrent.futures.as_completed(futures):
            try:
                idx, text = fut.result()
            except Exception as e:
                idx = futures[fut]
                log.warning(f"Slot {idx} failed ({type(e).__name__}) — returning empty")
                text = ""
            results[idx] = text
            done += 1
            if done % milestone == 0 or done == total:
                _progress(done, total, desc, t0)

    return results


def generate_local(
    prompt: str,
    max_tokens: int = 50,
    system_prompt: Optional[str] = None,
    stop: Optional[List[str]] = None,
    enable_thinking: bool = False,
) -> str:
    return _chat(prompt, max_tokens, 0.0, system_prompt, stop, enable_thinking)


def generate_local_batch(
    prompts: List[str],
    max_tokens: int = 50,
    batch_size: int = 16,  # noqa: ARG001 — kept for call-site compatibility
    system_prompt: Optional[str] = None,
    desc: str = "Local LLM Inference",
    stop: Optional[List[str]] = None,
    enable_thinking: bool = False,
) -> List[str]:
    return generate_batch(
        prompts,
        max_tokens=max_tokens,
        temperature=0.0,
        system_prompt=system_prompt,
        desc=desc,
        stop=stop,
        enable_thinking=enable_thinking,
    )


def get_anthropic_client():
    """Public accessor — used by ingest.py's Claude vision OCR path, which needs
    raw client.messages.create() for multimodal (image) content blocks that the
    text-only generate()/generate_batch() helpers above don't expose."""
    return _get_anthropic_client()


def unload():
    global _client, _anthropic_client
    _client = None
    _anthropic_client = None
    gc.collect()
    log.info("llm_client reset (llama-server keeps model loaded)")
