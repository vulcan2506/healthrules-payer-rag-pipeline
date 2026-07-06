"""
session.py
──────────
Stateful session wrapper — mode-agnostic, so it can wrap ANY retrieval
function (plain pipeline, gated/cached, best-of-N, compare, naive,
traditional), not just retriever.retrieve().

Adds four things a stateless retrieval call can't do:

  1. Short conversational memory — a small TRIMMED sliding window (last
                             max_history_turns, default 6) so follow-ups can
                             be understood without re-stating context. Kept
                             deliberately short because this is what feeds
                             the query-rewrite prompt on every ambiguous
                             follow-up — a bigger window means a bigger,
                             slower prompt on every turn.

  2. Full session summary  — a SEPARATE, never-trimmed full_log plus
                             summary(), covering every turn since the
                             session started (or was last cleared). Nothing
                             is lost just because the rewrite window is
                             short — summary() is the place to look for the
                             whole conversation, not self.history.

  3. Session state          — pins extracted automatically from dialogue:
                               • pinned_version  ("I'm on 26.1" → scopes retrieval)
                               • active_topic    (last routed topic, for context)

  4. Query rewriting        — if the new message is ambiguous (pronouns, short,
                             references to prior turns), the local LLM rewrites
                             it into a standalone query before retrieval.
                             Clear queries pass through with zero LLM overhead.

Two ways to use it:

  A) Convenience wrapper, plain pipeline only:

    session = ConversationSession()
    result = session.chat("How does claim adjudication work?")
    answer = llm.generate(result["system_prompt"], result["context"], result["query"])
    session.add_assistant_turn(answer)          # store so next rewrite can use it
    result2 = session.chat("What changed in that area?")
    # result2["standalone_query"] == "What changed in claim adjudication between versions?"
    # result2["was_rewritten"]    == True

  B) Building blocks, any retrieval mode (this is how cli.py wires session
     into --compare / --naive / --overlap / --best-of / the gated default):

    standalone, was_rewritten = session.prepare_turn(user_message)
    result = <any retrieval function>(standalone, ...)
    session.record_turn(user_message, result)
    # ... later, any time ...
    recap = session.summary()   # {"turn_count", "questions", "topics_touched",
                                 #  "versions_seen", "pinned_version", "narrative"}
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

import config
import retriever

log = logging.getLogger(__name__)


# ── Ambiguity detection ────────────────────────────────────────────────────────

_AMBIGUOUS_RE = re.compile(
    r"\b("
    r"that|this|those|these|"
    r"it\b|its\b|they|them|their|"
    r"the same|the previous|"
    r"mentioned|above|earlier|before|"
    r"also|too|as well|"
    r"more about|tell me more|elaborate|expand|go deeper|"
    r"what about|how about|and also|"
    r"same thing|same section|same rule"
    r")\b",
    re.IGNORECASE,
)

_SHORT_QUERY_WORDS = 4      # queries under this word count are likely follow-ups


# ── Version pin extraction ─────────────────────────────────────────────────────

_VERSION_PIN_RE = re.compile(
    r"\b(?:"
    r"i['’m]*\s*(?:am\s+)?(?:using|on|running|have)|"
    r"using|running|on\s+version|for\s+version|version"
    r")\s+(v?\d+\.\d+)\b",
    re.IGNORECASE,
)


# ── Rewrite prompt ─────────────────────────────────────────────────────────────

_REWRITE_SYSTEM = (
    "You are a query rewriter for a technical document Q&A system. "
    "Given a conversation history and a new user message, rewrite the user message "
    "as a complete, self-contained search query that can be understood without the history. "
    "RULES:\n"
    "- If the message is already fully standalone, return it UNCHANGED.\n"
    "- Resolve all pronouns and vague references using the conversation history.\n"
    "- Keep the rewritten query concise (one sentence, no preamble).\n"
    "- Output ONLY the rewritten query — nothing else."
)


def _history_to_text(history: List[Dict]) -> str:
    parts = []
    for turn in history:
        role = "User" if turn["role"] == "user" else "Assistant"
        # Truncate long assistant turns — we only need the gist
        content = turn["content"][:350] if turn["role"] == "assistant" else turn["content"]
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


# ── Thin LLM client (no Stage 1 dependency) ───────────────────────────────────

_llm: Optional[OpenAI] = None


def _get_llm() -> OpenAI:
    global _llm
    if _llm is None:
        _llm = OpenAI(base_url=config.LLAMA_SERVER_URL + "/v1", api_key="none")
    return _llm


_SUMMARY_SYSTEM = (
    "You are summarizing a technical Q&A conversation between a user and a document "
    "assistant for HealthRules Payer release documentation. Write ONE short paragraph "
    "(3-4 sentences) recapping what topics were discussed and what was established. "
    "Be factual and concise — do not add information that wasn't in the conversation. "
    "Output ONLY the paragraph, no preamble."
)


def _summarize_conversation(full_log: List[Dict]) -> str:
    """One-paragraph LLM recap of the ENTIRE conversation (full_log, not the
    trimmed rewrite window) — used by ConversationSession.summary(). Reference-
    free and best-effort: falls back to an empty string on any failure rather
    than raising, since this is a convenience view, not part of the retrieval
    path."""
    text = _history_to_text(full_log)
    try:
        resp = _get_llm().chat.completions.create(
            model=config.LLAMA_MODEL_NAME,
            messages=[
                {"role": "system", "content": _SUMMARY_SYSTEM},
                {"role": "user",   "content": f"Conversation:\n{text}\n\nSummary:"},
            ],
            max_tokens=200,
            temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        log.warning(f"Session summary generation failed ({type(e).__name__}: {e})")
        return ""


def _rewrite_query(query: str, history: List[Dict]) -> str:
    """Ask the local LLM to rephrase an ambiguous query as a standalone query."""
    history_text = _history_to_text(history)
    prompt = (
        f"Conversation so far:\n{history_text}\n\n"
        f"New user message: {query}\n\n"
        f"Rewritten standalone query:"
    )
    try:
        resp = _get_llm().chat.completions.create(
            model=config.LLAMA_MODEL_NAME,
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
        log.info(f"Rewrite: {query!r} → {rewritten!r}")
        return rewritten
    except Exception as e:
        log.warning(f"Query rewrite failed ({type(e).__name__}: {e}) — using original")
        return query


# ── ConversationSession ────────────────────────────────────────────────────────

class ConversationSession:
    """
    Stateful retrieval session.

    history  — list of {"role": "user"/"assistant", "content": str}
    state    — {"pinned_version": str|None, "active_topic": str|None}
    """

    def __init__(self, max_history_turns: int = 6):
        self.history: List[Dict] = []       # trimmed sliding window — feeds the rewrite prompt
        self.full_log: List[Dict] = []      # untrimmed — every turn, for summary() / full recap
        self.max_turns = max_history_turns
        self.topics_touched: List[str] = []      # distinct active_topic values, in order seen
        self.versions_seen: List[str] = []       # distinct pinned_version values, in order seen
        self.state: Dict = {
            "pinned_version": None,
            "active_topic":   None,
        }

    # ── Main entry point (plain-pipeline convenience wrapper) ─────────────────

    def chat(self, user_message: str, verbose: bool = False) -> Dict:
        """
        Process one user turn against the plain retriever.retrieve() path.

        Returns retriever.retrieve() dict extended with:
          standalone_query : str   — query used for retrieval (may differ from input)
          was_rewritten    : bool  — True if LLM rewrote the query
          session_state    : dict  — current pinned_version and active_topic
        """
        standalone, was_rewritten = self.prepare_turn(user_message)
        result = retriever.retrieve(
            standalone,
            version=self.state["pinned_version"],
            verbose=verbose,
        )
        self.record_turn(user_message, result)
        return {
            **result,
            "standalone_query": standalone,
            "was_rewritten":    was_rewritten,
            "session_state":    dict(self.state),
        }

    # ── Building blocks — usable by ANY retrieval mode, not just chat() ───────

    def prepare_turn(self, user_message: str) -> Tuple[str, bool]:
        """
        Step 1+2 of chat(), split out so any retrieval function (gated/cached,
        best-of-N, compare, naive, overlap, ...) can precede its own call with
        session-aware rewriting instead of only the plain pipeline path.

        Extracts version/topic pins (regex, no LLM), then rewrites the message
        into a standalone query if it looks like an ambiguous follow-up and
        history exists. Does NOT retrieve or store history yet — call
        record_turn() after retrieval with whatever result came back.
        """
        self._extract_pins(user_message)
        return self._maybe_rewrite(user_message)

    def record_turn(self, user_message: str, result: Dict) -> None:
        """
        Step 4+5 of chat(), split out so any retrieval mode can record its own
        result's routing path (for active_topic tracking) and store the turn
        in history — regardless of which retrieval function produced `result`.
        """
        self._update_topic(result)
        # Store original, not rewrite — keeps history readable.
        turn = {"role": "user", "content": user_message}
        self.history.append(turn)
        self.full_log.append(turn)
        self._trim()

    def add_assistant_turn(self, content: str) -> None:
        """
        Call after the LLM generates its response.
        Stores the answer in history so future rewrites can reference it.
        """
        turn = {"role": "assistant", "content": content}
        self.history.append(turn)
        self.full_log.append(turn)
        self._trim()

    def pin_version(self, version: str) -> None:
        """Manually pin a version for the rest of the session."""
        self.state["pinned_version"] = version
        if version not in self.versions_seen:
            self.versions_seen.append(version)
        log.info(f"Version manually pinned: {version}")

    def clear(self) -> None:
        """Reset the session (new conversation)."""
        self.history.clear()
        self.full_log.clear()
        self.topics_touched.clear()
        self.versions_seen.clear()
        self.state = {"pinned_version": None, "active_topic": None}
        log.info("Session cleared")

    @property
    def turn_count(self) -> int:
        return sum(1 for t in self.full_log if t["role"] == "user")

    # ── Full session summary ───────────────────────────────────────────────────

    def summary(self, use_llm: bool = True) -> Dict:
        """
        Recap the WHOLE conversation so far — not just the trimmed sliding
        window used for rewriting (self.history / max_turns, deliberately kept
        short so the rewrite prompt stays cheap). full_log is never trimmed,
        so this always covers every turn since the session started or was
        last cleared.

        Returns:
          turn_count      : int
          questions       : list of every user question asked, in order
          topics_touched  : distinct routed topics/parents seen, in order
          versions_seen   : distinct version pins encountered, in order
          pinned_version  : current pin (or None)
          narrative       : str — one-paragraph LLM recap (use_llm=True only;
                            falls back to "" if generation fails or there's
                            no history yet)
        """
        questions = [t["content"] for t in self.full_log if t["role"] == "user"]
        out = {
            "turn_count":     self.turn_count,
            "questions":      questions,
            "topics_touched": list(self.topics_touched),
            "versions_seen":  list(self.versions_seen),
            "pinned_version": self.state["pinned_version"],
            "narrative":      "",
        }
        if use_llm and questions:
            out["narrative"] = _summarize_conversation(self.full_log)
        return out

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _extract_pins(self, message: str) -> None:
        m = _VERSION_PIN_RE.search(message)
        if m:
            version = m.group(1).lstrip("vV")
            if self.state["pinned_version"] != version:
                self.state["pinned_version"] = version
                log.info(f"Version auto-pinned: {version}")
            if version not in self.versions_seen:
                self.versions_seen.append(version)

    def _is_ambiguous(self, query: str) -> bool:
        if len(query.split()) < _SHORT_QUERY_WORDS:
            return True
        return bool(_AMBIGUOUS_RE.search(query))

    def _maybe_rewrite(self, query: str) -> Tuple[str, bool]:
        if not self.history:
            return query, False
        if not self._is_ambiguous(query):
            return query, False
        # Deliberately uses the TRIMMED window (self.history[-max_turns:]), not
        # full_log — this is the "short conversational memory" that keeps the
        # rewrite prompt small; summary() below is what covers the full log.
        rewritten = _rewrite_query(query, self.history[-self.max_turns:])
        changed = rewritten.lower().strip() != query.lower().strip()
        return rewritten, changed

    def _update_topic(self, result: Dict) -> None:
        path = result.get("path", {})
        subs = path.get("subs", [])
        topic = subs[0] if subs else (path.get("parents") or [None])[0]
        if topic:
            self.state["active_topic"] = topic
            if topic not in self.topics_touched:
                self.topics_touched.append(topic)

    def _trim(self) -> None:
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]


# ── Singleton ──────────────────────────────────────────────────────────────────

_session: Optional[ConversationSession] = None


def get_session(max_history_turns: int = 6) -> ConversationSession:
    global _session
    if _session is None:
        _session = ConversationSession(max_history_turns)
    return _session
