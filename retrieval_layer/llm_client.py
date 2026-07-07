"""
llm_client.py — retrieval_layer
────────────────────────────────
Single LLM client for retrieval-side calls: query reformulation
(retriever.py:_reformulate), query rewriting (graph.py:node_rewrite_query),
and answer generation (redis_cache.py:_generate_answer). Defaults to Claude
(Anthropic API, BYOK via Stage 1/.env's ANTHROPIC_API_KEY — see config.py).
Set LLM_BACKEND=local in Stage 1/.env to route back to the local llama.cpp
server instead, with zero call-site changes.
"""

import logging
import time
import subprocess
from typing import List, Optional

import httpx
import config

log = logging.getLogger(__name__)

_anthropic_client = None
_local_client = None
_openrouter_client = None
_groq_client = None
_local_server_starting = False  # guards against launching start_server.sh twice from parallel threads


def _is_local_server_up() -> bool:
    try:
        r = httpx.get(f"{config.LLAMA_SERVER_URL}/health", timeout=config.LOCAL_FALLBACK_HEALTH_TIMEOUT)
        return r.status_code == 200
    except Exception:
        return False


def _ensure_local_server() -> bool:
    """
    Starts Stage 1's llama.cpp server (start_server.sh) if it isn't already
    reachable, then polls /health until it responds or LOCAL_FALLBACK_STARTUP_TIMEOUT
    is hit. Returns True once confirmed up, False otherwise. Safe to call from
    multiple threads — only the first caller launches the process.
    """
    global _local_server_starting

    if _is_local_server_up():
        return True

    script = config.STAGE1_DIR / "start_server.sh"
    if not _local_server_starting:
        if not script.exists():
            log.error(f"Cannot start local fallback server — {script} not found")
            return False
        _local_server_starting = True
        log.warning("Claude unavailable — starting local llama.cpp server (start_server.sh) as fallback...")
        log_path = config.STAGE1_OUTPUT / "llama_server_fallback.log"
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


def _get_local_client():
    global _local_client
    if _local_client is None:
        from openai import OpenAI
        _local_client = OpenAI(base_url=config.LLAMA_SERVER_URL + "/v1", api_key="none")
    return _local_client


def _get_openrouter_client():
    global _openrouter_client
    if not config.OPENROUTER_API_KEY:
        return None
    if _openrouter_client is None:
        from openai import OpenAI
        _openrouter_client = OpenAI(base_url=config.OPENROUTER_BASE_URL, api_key=config.OPENROUTER_API_KEY)
        log.info(f"OpenRouter client → model={config.OPENROUTER_MODEL}")
    return _openrouter_client


def _get_groq_client():
    global _groq_client
    if not config.GROQ_API_KEY:
        return None
    if _groq_client is None:
        from openai import OpenAI
        _groq_client = OpenAI(base_url=config.GROQ_BASE_URL, api_key=config.GROQ_API_KEY)
        log.info(f"Groq client → model={config.GROQ_MODEL}")
    return _groq_client


def chat(
    prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 500,
    stop: Optional[List[str]] = None,
    temperature: float = 0.0,
    enable_thinking: bool = False,
    timeout: Optional[float] = None,
) -> str:
    """
    Single chat call, routed by config.LLM_BACKEND. Fallback chain on Claude
    failure: OpenRouter -> Groq -> local llama.cpp (auto-started). Raises on
    total failure — callers that need a soft-fail (e.g. _reformulate's
    fall-back-to-original-query behavior) should catch around this, same as
    before this abstraction existed.
    """
    if config.LLM_BACKEND == "local":
        return _chat_local(prompt, system_prompt, max_tokens, stop, temperature, enable_thinking, timeout)

    try:
        return _chat_anthropic(prompt, system_prompt, max_tokens, stop, enable_thinking, timeout)
    except Exception as e_claude:
        log.warning(f"Claude call failed ({type(e_claude).__name__}: {e_claude}) — trying OpenRouter fallback")

        or_client = _get_openrouter_client()
        if or_client is not None:
            try:
                return _chat_openai_compatible(
                    or_client, config.OPENROUTER_MODEL, prompt, system_prompt, max_tokens,
                    stop, temperature, enable_thinking, timeout, label="OpenRouter",
                )
            except Exception as e_or:
                log.warning(f"OpenRouter call failed ({type(e_or).__name__}: {e_or}) — trying Groq fallback")
        else:
            log.warning("OPENROUTER_API_KEY not set — skipping OpenRouter fallback, trying Groq")

        groq_client = _get_groq_client()
        if groq_client is not None:
            try:
                return _chat_openai_compatible(
                    groq_client, config.GROQ_MODEL, prompt, system_prompt, max_tokens,
                    stop, temperature, enable_thinking, timeout, label="Groq",
                )
            except Exception as e_groq:
                log.warning(f"Groq call failed ({type(e_groq).__name__}: {e_groq}) — falling back to local llama.cpp server")
        else:
            log.warning("GROQ_API_KEY not set — skipping Groq fallback, trying local llama.cpp server")

        if not _ensure_local_server():
            raise RuntimeError(
                "Claude, OpenRouter, and Groq all failed, and the local llama.cpp "
                "fallback server did not come up"
            ) from e_claude
        return _chat_local(prompt, system_prompt, max_tokens, stop, temperature, enable_thinking, timeout)


def _chat_anthropic(
    prompt: str,
    system_prompt: Optional[str],
    max_tokens: int,
    stop: Optional[List[str]],
    enable_thinking: bool,
    timeout: Optional[float],
) -> str:
    client = _get_anthropic_client()
    if timeout:
        client = client.with_options(timeout=timeout)

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
    # temperature intentionally omitted — Opus 4.8 rejects sampling params outright,
    # and every call site here already runs at temperature=0.0.

    resp = client.messages.create(**kwargs)
    return next((b.text for b in resp.content if b.type == "text"), "").strip()


def _chat_openai_compatible(
    client,
    model: str,
    prompt: str,
    system_prompt: Optional[str],
    max_tokens: int,
    stop: Optional[List[str]],
    temperature: float,
    enable_thinking: bool,  # unused — neither provider supports llama.cpp's chat_template_kwargs extension
    timeout: Optional[float],
    label: str,
) -> str:
    if timeout:
        client = client.with_options(timeout=timeout)

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

    resp = client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


def _chat_local(
    prompt: str,
    system_prompt: Optional[str],
    max_tokens: int,
    stop: Optional[List[str]],
    temperature: float,
    enable_thinking: bool,
    timeout: Optional[float],
) -> str:
    client = _get_local_client()
    if timeout:
        client = client.with_options(timeout=timeout)

    messages = [{"role": "user", "content": prompt}]
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

    resp = client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()
