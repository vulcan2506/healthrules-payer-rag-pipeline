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
import concurrent.futures
from typing import List, Optional

from openai import OpenAI

import config

log = logging.getLogger(__name__)


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


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=config.LLAMA_SERVER_URL + "/v1",
            api_key="none",
        )
        log.info(f"llama.cpp client → {config.LLAMA_SERVER_URL}")
    return _client


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

    t0 = time.time()
    log.info(f"{desc}: starting {total} items ({config.LLAMA_PARALLEL_SLOTS} parallel slots)")
    done = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=config.LLAMA_PARALLEL_SLOTS) as pool:
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


def unload():
    global _client
    _client = None
    gc.collect()
    log.info("llm_client reset (llama-server keeps model loaded)")
