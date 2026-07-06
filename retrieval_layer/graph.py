"""
graph.py
────────
LangGraph retrieval pipeline.

Decomposes the monolithic retriever.retrieve() into discrete, inspectable nodes:

  rewrite_query   — resolve follow-ups using conversation history (LLM call if ambiguous)
  classify        — intent routing: specific | intelligence | delta
  retrieve        — HNSW or ChromaDB retrieval based on intent
  ↓ conditional: low confidence (< CONFIDENCE_THRESHOLD) AND first attempt → widen
  widen           — fall back from HNSW to corpus for broader coverage
  detect_conflict — check if chunks span multiple document versions
  ↓ conditional: conflict detected → inject_delta
  inject_delta    — pull delta report as authoritative resolver
  format          — assemble context block + versioned/standard system prompt

Conversation memory is managed by LangGraph's MemorySaver — one thread per
conversation, state persists automatically between invocations.

Install:
    pip install langgraph langchain-core

Usage:
    from graph import get_app, add_assistant_turn

    app    = get_app()
    thread = {"configurable": {"thread_id": "user-123"}}

    state = app.invoke({"query": "How does claim adjudication work?"}, thread)
    answer = your_llm(state["system_prompt"], state["context"])
    add_assistant_turn(answer, thread_id="user-123")

    # Follow-up — history auto-loaded from MemorySaver
    state2 = app.invoke({"query": "What changed in that area?"}, thread)
    # state2["was_rewritten"] == True
    # state2["standalone_query"] == "What changed in claim adjudication between versions?"
"""

import logging
import re
from typing import Annotated, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from openai import OpenAI
from typing_extensions import TypedDict

import config as app_config  # aliased to avoid collision with LangGraph's 'config' arg name
import retriever as ret

log = logging.getLogger(__name__)


# ── State ──────────────────────────────────────────────────────────────────────

class ChatState(TypedDict):
    # ── Persisted by MemorySaver across all turns in a thread ─────────────────
    messages:       Annotated[List[BaseMessage], add_messages]  # full conversation history
    pinned_version: Optional[str]   # e.g. "26.1" — set by explicit mention in dialogue
    active_topic:   Optional[str]   # last routed sub-topic, for context

    # ── Per-turn (overwritten fresh each invocation) ───────────────────────────
    query:               str
    standalone_query:    str        # possibly rewritten version of query
    was_rewritten:       bool
    intent:              str        # specific | intelligence | delta | factual
    path:                Dict       # routing path (parents, subs, or collection name)
    chunks:              List[Dict] # reranked retrieved chunks
    confidence:          float      # max rerank score of top chunk
    retry_count:         int        # 0 = first attempt; 1 = after widening
    versioned:           bool       # True if multi-version conflict detected
    conflicting_sections: List[str] # section headers present in 2+ versions
    delta_findings:      List[Dict] # injected delta report entries
    delta_used:          bool
    context:             str        # formatted context block ready for LLM
    system_prompt:       str        # standard or versioned instruction set


# ── LLM client ─────────────────────────────────────────────────────────────────

_llm: Optional[OpenAI] = None


def _get_llm() -> OpenAI:
    global _llm
    if _llm is None:
        _llm = OpenAI(base_url=app_config.LLAMA_SERVER_URL + "/v1", api_key="none")
    return _llm


# ── Shared helpers ─────────────────────────────────────────────────────────────

_AMBIGUOUS_RE = re.compile(
    r"\b("
    r"that|this|those|these|it\b|its\b|they|them|their|"
    r"the same|the previous|mentioned|above|earlier|before|"
    r"also|too|as well|more about|tell me more|elaborate|expand|"
    r"what about|how about|same thing|same section|same rule"
    r")\b",
    re.IGNORECASE,
)

_VERSION_PIN_RE = re.compile(
    r"\b(?:i['’m]*\s*(?:am\s+)?(?:using|on|running|have)|"
    r"using|running|on\s+version|for\s+version|version)\s+(v?\d+\.\d+)\b",
    re.IGNORECASE,
)

_REWRITE_SYSTEM = (
    "You are a query rewriter for a technical document Q&A system. "
    "Given a conversation history and a new user message, rewrite it as a complete, "
    "self-contained search query understandable without the history. "
    "RULES: If already standalone, return UNCHANGED. Resolve all pronouns. "
    "Output ONLY the rewritten query — one sentence, no preamble."
)


def _is_ambiguous(query: str) -> bool:
    return len(query.split()) < 4 or bool(_AMBIGUOUS_RE.search(query))


def _messages_to_text(messages: List[BaseMessage]) -> str:
    parts = []
    for m in messages[-8:]:
        role    = "User" if isinstance(m, HumanMessage) else "Assistant"
        content = m.content[:350] if isinstance(m, AIMessage) else m.content
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _top_score(chunks: List[Dict]) -> float:
    scores = [c.get("rerank_score", c.get("score", 0.0)) for c in chunks]
    return max(scores) if scores else 0.0


# ── Node: rewrite_query ────────────────────────────────────────────────────────

def node_rewrite_query(state: ChatState) -> Dict:
    """
    Step 1 — make the query standalone.

    - Auto-detects version pins ("I'm using version X") and sets pinned_version.
    - If history exists and query is ambiguous, calls local LLM to rewrite.
    - Always appends user's original message to conversation history.
    - Resets per-turn counters (retry_count, was_rewritten).
    """
    query    = state["query"]
    messages = state.get("messages") or []

    # Version pin extraction (regex, no LLM)
    pinned = state.get("pinned_version")
    m = _VERSION_PIN_RE.search(query)
    if m:
        pinned = m.group(1).lstrip("vV")
        log.info(f"Version auto-pinned: {pinned}")

    # Query rewriting (LLM call only if ambiguous + history exists)
    standalone    = query
    was_rewritten = False
    if messages and _is_ambiguous(query):
        history_text = _messages_to_text(messages)
        prompt = (
            f"Conversation so far:\n{history_text}\n\n"
            f"New user message: {query}\n\nRewritten standalone query:"
        )
        try:
            resp = _get_llm().chat.completions.create(
                model=app_config.LLAMA_MODEL_NAME,
                messages=[
                    {"role": "system", "content": _REWRITE_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=80,
                temperature=0.0,
                stop=["\n"],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            rewritten = resp.choices[0].message.content.strip()
            if rewritten.lower().strip() != query.lower().strip():
                standalone    = rewritten
                was_rewritten = True
                log.info(f"Rewrite: {query!r} → {rewritten!r}")
        except Exception as e:
            log.warning(f"Query rewrite failed ({type(e).__name__}: {e}) — using original")

    return {
        "messages":          [HumanMessage(content=query)],  # add_messages appends
        "standalone_query":  standalone,
        "was_rewritten":     was_rewritten,
        "pinned_version":    pinned,
        "retry_count":       0,
        "delta_findings":    [],
        "conflicting_sections": [],
        "versioned":         False,
        "delta_used":        False,
    }


# ── Node: classify ─────────────────────────────────────────────────────────────

def node_classify(state: ChatState) -> Dict:
    """Step 2 — classify intent from standalone query."""
    intent = ret.classify(state["standalone_query"])
    log.info(f"Intent: {intent!r}")
    return {"intent": intent}


# ── Node: retrieve ─────────────────────────────────────────────────────────────

def node_retrieve(state: ChatState) -> Dict:
    """
    Step 3 — route to the appropriate retrieval backend.

    specific    → HNSW hierarchical (falls back to corpus if zero results)
    intelligence → ChromaDB intelligence collection
    delta       → ChromaDB delta collection
    factual     → ChromaDB corpus (also used after widening)
    """
    query   = state["standalone_query"]
    intent  = state["intent"]
    version = state.get("pinned_version")

    if intent == "specific":
        raw = ret._retrieve_specific(query, verbose=False)
        if not raw["chunks"]:                          # zero-result HNSW → immediate widen
            log.info("HNSW returned zero chunks — widening to corpus")
            raw = ret._retrieve_corpus(query, version, None)
    elif intent == "intelligence":
        raw = ret._retrieve_intelligence(query, version)
    elif intent == "delta":
        raw = ret._retrieve_delta(query, None, None)
    else:                                              # "factual" or post-widen
        raw = ret._retrieve_corpus(query, version, None)

    chunks = raw["chunks"]
    path   = raw.get("path", {})

    # Track active topic for session state
    subs         = path.get("subs", [])
    active_topic = subs[0] if subs else (path.get("parents", [None])[0])

    return {
        "path":         path,
        "chunks":       chunks,
        "confidence":   _top_score(chunks),
        "active_topic": active_topic or state.get("active_topic"),
    }


# ── Node: widen ────────────────────────────────────────────────────────────────

def node_widen(state: ChatState) -> Dict:
    """
    Fallback — broaden from HNSW to corpus when confidence is low.
    Only fires once (retry_count gate prevents infinite loops).
    """
    query   = state["standalone_query"]
    version = state.get("pinned_version")
    log.info(f"Low confidence ({state['confidence']:.3f}) — widening HNSW → corpus")
    raw    = ret._retrieve_corpus(query, version, None)
    chunks = raw["chunks"]
    return {
        "intent":      "factual",
        "path":        raw["path"],
        "chunks":      chunks,
        "confidence":  _top_score(chunks),
        "retry_count": state["retry_count"] + 1,
    }


# ── Node: detect_conflict ──────────────────────────────────────────────────────

def node_detect_conflict(state: ChatState) -> Dict:
    """Step 4 — find section headers that appear in chunks from 2+ versions."""
    sections = ret.detect_version_conflict(state["chunks"])
    if sections:
        log.info(f"Version conflict in {len(sections)} section(s): {sections[:3]}")
    return {
        "versioned":             bool(sections),
        "conflicting_sections":  sections,
    }


# ── Node: inject_delta ─────────────────────────────────────────────────────────

def node_inject_delta(state: ChatState) -> Dict:
    """
    Step 5 (conditional) — query delta collection for conflicting sections.
    Falls back to the original query if section-targeted lookup returns nothing.
    """
    findings = ret._inject_delta(
        state["conflicting_sections"],
        state["standalone_query"],
    )
    log.info(f"Delta injected: {len(findings)} finding(s)")
    return {
        "delta_findings": findings,
        "delta_used":     bool(findings),
    }


# ── Node: format ───────────────────────────────────────────────────────────────

def node_format(state: ChatState) -> Dict:
    """Step 6 — assemble context block and choose the right system prompt."""
    chunks   = state["chunks"]
    findings = state.get("delta_findings") or []

    if state.get("versioned"):
        context       = ret.format_context_versioned(chunks, findings or None)
        system_prompt = ret.SYSTEM_PROMPT_VERSIONED
    else:
        context       = ret.format_context(chunks)
        system_prompt = ret.SYSTEM_PROMPT_STANDARD

    return {
        "context":       context,
        "system_prompt": system_prompt,
        "delta_used":    bool(findings),
    }


# ── Conditional edges ──────────────────────────────────────────────────────────

def _edge_confidence(state: ChatState) -> str:
    """After retrieve: low confidence on first HNSW attempt → widen."""
    if (
        state["confidence"] < app_config.CONFIDENCE_THRESHOLD
        and state["retry_count"] == 0
        and state["intent"] == "specific"
    ):
        return "widen"
    return "detect_conflict"


def _edge_conflict(state: ChatState) -> str:
    """After detect_conflict: inject delta only if conflict was found."""
    return "inject_delta" if state.get("versioned") else "format"


# ── Graph assembly ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(ChatState)

    g.add_node("rewrite_query",   node_rewrite_query)
    g.add_node("classify",        node_classify)
    g.add_node("retrieve",        node_retrieve)
    g.add_node("widen",           node_widen)
    g.add_node("detect_conflict", node_detect_conflict)
    g.add_node("inject_delta",    node_inject_delta)
    g.add_node("format",          node_format)

    g.add_edge(START,             "rewrite_query")
    g.add_edge("rewrite_query",   "classify")
    g.add_edge("classify",        "retrieve")
    g.add_conditional_edges(
        "retrieve",
        _edge_confidence,
        {"widen": "widen", "detect_conflict": "detect_conflict"},
    )
    g.add_edge("widen",           "detect_conflict")
    g.add_conditional_edges(
        "detect_conflict",
        _edge_conflict,
        {"inject_delta": "inject_delta", "format": "format"},
    )
    g.add_edge("inject_delta",    "format")
    g.add_edge("format",          END)

    return g


# ── Singleton app ──────────────────────────────────────────────────────────────

_app = None


def get_app():
    """Return the compiled LangGraph app (singleton, lazy init)."""
    global _app
    if _app is None:
        checkpointer = MemorySaver()
        _app = build_graph().compile(checkpointer=checkpointer)
        log.info("LangGraph app compiled with MemorySaver")
    return _app


# ── Public convenience functions ───────────────────────────────────────────────

def chat(query: str, thread_id: str = "default") -> ChatState:
    """
    Send one user message and get the full retrieval state back.

    Same thread_id = shared conversation history (loaded from MemorySaver).
    Different thread_id = fresh conversation.

    After receiving the LLM answer, call add_assistant_turn() so future
    rewrites can reference it.
    """
    lc = {"configurable": {"thread_id": thread_id}}
    return get_app().invoke({"query": query}, lc)


def add_assistant_turn(answer: str, thread_id: str = "default") -> None:
    """
    Store the LLM's response in conversation history for this thread.
    Call this after generating the answer so follow-ups can reference it.
    """
    lc = {"configurable": {"thread_id": thread_id}}
    get_app().update_state(lc, {"messages": [AIMessage(content=answer)]})
