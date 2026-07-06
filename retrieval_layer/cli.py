"""
cli.py
──────
Interactive CLI for testing the retrieval layer.

Default behavior (as of this version): every query goes through the gated/
cached path (redis_cache.answer_query, mode="concise") — best-of-3
reformulation + confidence-gated escalation + Redis caching — same as the
old explicit --cached flag. Use --raw to get the old plain-retrieve()
default back (no gating, no caching, no reformulation, shows routing/chunk
debug info instead of a generated answer).

Session tracking (short conversational memory, version pinning, ambiguous-
follow-up rewriting, and a full-conversation `summary` command) now applies
to EVERY mode below, not just --session — see session.py. --session is a
separate thing: it swaps the retrieval engine itself for graph.py's fully
inspectable LangGraph node-by-node pipeline (its own internal rewriting/
pinning, decomposed into named nodes) rather than adding memory on top of
whichever mode you picked.

Usage:
    python cli.py                  # interactive REPL — gated/cached, best-of-3, sessioned
    python cli.py --raw            # old plain-retrieve() default, no gating/caching
    python cli.py --verbose        # show routing debug info
    python cli.py --version 26.1   # scope to a specific version
    python cli.py --intent delta   # force an intent
    python cli.py --detailed       # same as --mode detailed: always merged_all, no confidence check
    python cli.py --concise        # same as --mode concise (the default)
    python cli.py --session        # swap engine to graph.py's LangGraph pipeline

REPL commands (persistent — full line, no "--"): 'quit' | 'intent <type>' |
'clear intent' | 'summary' (full conversation recap) | 'clear session' |
'pin <version>'

Inline flags (ONE-TURN overrides — ANY startup flag can also be typed as
part of the question itself, and combined freely):
    Chat > --detailed what changed in claim editing between versions?
    Chat > --raw --best-of 5 how does prior authorization work?
    Chat > --naive --verbose how are remittance advices generated?
Recognized inline: --raw --cached --concise --detailed --compare --naive
--overlap --verbose --session --mode <concise|detailed> --version <X>
--intent <type> --best-of <N>. Unrecognized "--" tokens are left in the
query text untouched (so a literal "--" in a real question isn't eaten).
These apply for that single turn only — the persistent REPL defaults
(set at startup, or via the commands above) are unaffected.
"""

import argparse
import logging
import shlex
import sys
import textwrap

import retriever
import graph as graph_module
import redis_cache
import session

logging.basicConfig(level=logging.WARNING, format="%(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

# ── Display helpers ────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
RED    = "\033[91m"
DIM    = "\033[2m"


def _header(text: str, color: str = CYAN):
    width = 60
    print(f"\n{color}{BOLD}{'─' * width}{RESET}")
    print(f"{color}{BOLD}  {text}{RESET}")
    print(f"{color}{BOLD}{'─' * width}{RESET}")


def _print_result(result: dict, verbose: bool):
    intent    = result["intent"]
    versioned = result["versioned"]
    delta     = result["delta_used"]
    path      = result["path"]
    chunks    = result["chunks"]

    # ── Routing summary ────────────────────────────────────────────────────────
    _header("ROUTING", CYAN)
    print(f"  Intent   : {BOLD}{intent.upper()}{RESET}")

    if "parents" in path:
        print(f"  Parents  : {', '.join(path['parents'])}")
    if "subs" in path:
        print(f"  Subs     : {', '.join(path['subs'])}")
    if "collection" in path:
        print(f"  Collection: {path['collection']}")

    if versioned:
        print(f"\n  {YELLOW}{BOLD}⚠  Multi-version conflict detected{RESET}")
    if delta:
        print(f"  {GREEN}{BOLD}✓  Delta report injected{RESET}")

    # ── Retrieved chunks ───────────────────────────────────────────────────────
    _header(f"RETRIEVED CHUNKS  ({len(chunks)} total)", CYAN)
    for i, c in enumerate(chunks, 1):
        score   = c.get("rerank_score", c.get("score", 0))
        src     = c.get("source_doc", "")
        section = c.get("section_header", "")
        preview = c.get("text", "")[:160].replace("\n", " ")
        print(f"\n  {BOLD}[{i}]{RESET} {DIM}{src} — {section}{RESET}")
        print(f"       Score : {score:.4f}")
        print(f"       {textwrap.fill(preview + '…', width=72, subsequent_indent='       ')}")

    # ── Context block ──────────────────────────────────────────────────────────
    _header("CONTEXT SENT TO LLM", CYAN)
    ctx_lines = result["context"].split("\n")
    for line in ctx_lines[:60]:                   # cap display at 60 lines
        print(f"  {line}")
    if len(ctx_lines) > 60:
        print(f"  {DIM}… ({len(ctx_lines) - 60} more lines){RESET}")

    # ── System prompt ──────────────────────────────────────────────────────────
    _header("SYSTEM PROMPT", CYAN)
    label = f"{YELLOW}VERSIONED{RESET}" if versioned else f"{GREEN}STANDARD{RESET}"
    print(f"  Mode: {label}")
    print(f"\n  {DIM}{result['system_prompt'][:200]}…{RESET}")

    # ── Debug ──────────────────────────────────────────────────────────────────
    if verbose and result.get("debug"):
        _header("DEBUG", DIM)
        dbg = result["debug"]
        print(f"  Source             : {dbg.get('source')}")
        print(f"  Conflicting sections: {dbg.get('conflicting_sections', [])}")
        print(f"  Delta findings     : {dbg.get('delta_findings_count', 0)}")
        if dbg.get("hnsw_candidates"):
            print(f"\n  HNSW top candidates:")
            for c in dbg["hnsw_candidates"][:5]:
                print(f"    [{c['type']}] {c['label'][:60]}  score={c['score']:.3f}")

    print()


# ── Gated / cached answer display ──────────────────────────────────────────────

def _print_cached_result(result: dict):
    hit = result["from_cache"]
    _header("GATED ANSWER", GREEN if hit else YELLOW)
    src = f"{GREEN}{BOLD}CACHE HIT{RESET}" if hit else f"{YELLOW}{BOLD}LIVE{RESET}"
    print(f"  Source  : {src}")
    print(f"  Method  : {result['method']}")
    print(f"  Latency : {result['latency_s']:.3f}s")
    print()
    print(textwrap.fill(result["answer"], width=76, initial_indent="  ", subsequent_indent="  "))
    print()


# ── Session metadata display ───────────────────────────────────────────────────

def _print_session_meta(result: dict):
    """Print session-specific info (rewrite, pins) before the standard result."""
    state    = result.get("session_state", {})
    rewritten = result.get("was_rewritten", False)
    standalone = result.get("standalone_query", "")

    if rewritten:
        print(f"\n  {YELLOW}↺  Query rewritten:{RESET} {standalone}")
    version = state.get("pinned_version")
    topic   = state.get("active_topic")
    if version or topic:
        pin_parts = []
        if version:
            pin_parts.append(f"version={version}")
        if topic:
            pin_parts.append(f"topic={topic[:40]}")
        print(f"  {DIM}Session: {' | '.join(pin_parts)}{RESET}")


def _print_summary(recap: dict):
    """Full-conversation recap — see session.ConversationSession.summary()."""
    _header("SESSION SUMMARY", CYAN)
    print(f"  Turns           : {recap['turn_count']}")
    if recap["pinned_version"]:
        print(f"  Currently pinned: {recap['pinned_version']}")
    if recap["versions_seen"]:
        print(f"  Versions seen   : {', '.join(recap['versions_seen'])}")
    if recap["topics_touched"]:
        print(f"  Topics touched  : {', '.join(t[:40] for t in recap['topics_touched'])}")
    if recap["questions"]:
        print(f"\n  {BOLD}Questions asked:{RESET}")
        for i, q in enumerate(recap["questions"], 1):
            print(f"    {i}. {q}")
    if recap["narrative"]:
        print(f"\n  {BOLD}Recap:{RESET}")
        print(textwrap.fill(recap["narrative"], width=76, initial_indent="  ", subsequent_indent="  "))
    if not recap["questions"]:
        print(f"  {DIM}Nothing asked yet this session.{RESET}")
    print()


# ── Inline flag parsing ─────────────────────────────────────────────────────────

_INLINE_BOOL  = {"raw", "cached", "concise", "detailed", "compare", "naive", "overlap", "verbose", "session"}
_INLINE_VALUE = {"mode", "version", "intent", "best-of"}


def _extract_inline_flags(text: str):
    """
    Pull recognized --flag / --flag value tokens out of a free-text query
    line so any startup flag can also be typed as part of the question
    itself, combined freely — e.g.:

        "--detailed --best-of 5 what changed in claim editing?"
        -> ("what changed in claim editing?", {"detailed": True, "best_of": 5})

    Unrecognized tokens (including any "--xyz" not in the tables above) are
    left in the returned query text untouched, so a literal "--" in a real
    question doesn't silently vanish. Returns (clean_query, overrides) —
    overrides is empty if nothing matched.
    """
    try:
        tokens = shlex.split(text)
    except ValueError:
        return text, {}   # unbalanced quotes etc. — treat the whole line as query text

    overrides: dict = {}
    remaining = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--"):
            name = tok[2:].lower()
            if name in _INLINE_VALUE and i + 1 < len(tokens):
                key = name.replace("-", "_")
                val = tokens[i + 1]
                if key == "best_of":
                    try:
                        val = int(val)
                    except ValueError:
                        remaining.append(tok)     # "--best-of" not followed by a number — keep literal
                        i += 1
                        continue
                overrides[key] = val
                i += 2
                continue
            if name in _INLINE_BOOL:
                overrides[name] = True
                i += 1
                continue
        remaining.append(tok)
        i += 1
    return " ".join(remaining), overrides


def _resolve_turn_mode(overrides: dict, persistent_mode: str) -> str:
    """--detailed / --concise (booleans) win over --mode <value> if both are
    somehow given in the same turn; falls back to the persistent default."""
    if overrides.get("detailed"):
        return "detailed"
    if overrides.get("concise"):
        return "concise"
    return overrides.get("mode", persistent_mode)


# ── REPL ───────────────────────────────────────────────────────────────────────

def repl(args):
    use_session = args.session               # opt into the LangGraph *engine* specifically
    thread_id   = "cli-session"
    conv        = session.get_session()       # lightweight session tracking — ALWAYS active,
                                               # regardless of which retrieval mode is selected

    if use_session:
        mode_label = f"{YELLOW}LANGGRAPH ENGINE{RESET}"
    elif args.raw:
        mode_label = f"{GREEN}RAW (no gating/caching){RESET}"
    else:
        mode_label = f"{GREEN}GATED + CACHED  (mode={args.mode}, best-of-3){RESET}"
    print(f"\n{BOLD}{CYAN}HealthRules Payer — Retrieval Layer CLI{RESET}  [{mode_label}]")
    print(f"{DIM}Commands: 'quit' | 'intent <type>' | 'clear intent' | 'summary' | "
          f"'clear session' | 'pin <version>'{RESET}\n")
    print(f"{DIM}Intents: specific | factual | intelligence | delta{RESET}\n")

    forced_intent  = args.intent
    forced_version = args.version

    while True:
        prompt_label = f"{BOLD}Chat >{RESET} "
        try:
            query = input(prompt_label).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not query:
            continue

        # Inline "--flag" tokens are stripped FIRST, before any command or
        # session logic sees the text — so 'pin 25.2' etc. below always
        # match against the actual command/question, never a raw "--..." token.
        query, inline = _extract_inline_flags(query)
        if not query and inline:
            print(f"  {GREEN}Flags noted ({', '.join(inline)}) — no question text, nothing to run{RESET}\n")
            continue

        if query.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break

        if query.lower().startswith("intent "):
            forced_intent = query.split(" ", 1)[1].strip()
            print(f"  {GREEN}Intent forced to: {forced_intent}{RESET}\n")
            continue

        if query.lower() == "clear intent":
            forced_intent = None
            print(f"  {GREEN}Intent cleared — auto-classify active{RESET}\n")
            continue

        if query.lower() == "summary":
            _print_summary(conv.summary())
            continue

        if query.lower() == "clear session":
            conv.clear()
            if use_session:
                thread_id = f"cli-session-{id(query)}"   # new thread = fresh LangGraph memory too
            print(f"  {GREEN}Session cleared — history, pins, and summary reset{RESET}\n")
            continue

        if query.lower().startswith("pin "):
            version = query.split(" ", 1)[1].strip()
            conv.pin_version(version)
            print(f"  {GREEN}Version pinned: {version}{RESET}\n")
            continue

        # Per-turn effective config: persistent REPL defaults, with any
        # inline flags from THIS line layered on top. Nothing here mutates
        # the persistent args/forced_intent/forced_version — next turn
        # reverts unless the same inline flags are given again.
        turn_session  = inline.get("session", use_session)
        turn_compare  = inline.get("compare", args.compare)
        turn_naive    = inline.get("naive", args.naive)
        turn_overlap  = inline.get("overlap", args.overlap)
        turn_raw      = inline.get("raw", args.raw)
        turn_verbose  = inline.get("verbose", args.verbose)
        turn_intent   = inline.get("intent", forced_intent)
        turn_version  = inline.get("version", forced_version)
        turn_mode     = _resolve_turn_mode(inline, args.mode)
        turn_best_of  = inline.get("best_of", args.best_of)   # None = "not specified anywhere"

        if inline:
            print(f"  {DIM}Inline flags this turn: "
                  f"{', '.join(f'{k}={v}' for k, v in inline.items())}{RESET}")

        try:
            if turn_session:
                # graph.py's own engine does its own internal rewriting/pinning via
                # MemorySaver — don't double-rewrite with conv on top of it. Still
                # feed conv.record_turn() afterward so 'summary' has something to
                # show regardless of which engine answered.
                result = graph_module.chat(query, thread_id=thread_id)
                _print_session_meta(result)
                _print_result(result, verbose=turn_verbose)
                conv.record_turn(query, result)
                continue

            # Every other mode shares the same lightweight session prep: extract
            # version pins, rewrite ambiguous follow-ups using the SHORT trimmed
            # window (session.py), then run whichever mode was asked for on the
            # standalone query.
            standalone, was_rewritten = conv.prepare_turn(query)
            version_to_use = turn_version or conv.state["pinned_version"]
            _print_session_meta({
                "was_rewritten":    was_rewritten,
                "standalone_query": standalone,
                "session_state":    dict(conv.state),
            })

            if turn_compare:
                p = retriever.retrieve(standalone, intent=turn_intent,
                                       version=version_to_use, verbose=turn_verbose)
                n = retriever.retrieve_naive(standalone)
                o = retriever.retrieve_overlap(standalone)
                _header("PIPELINE", CYAN)
                _print_result(p, verbose=turn_verbose)
                _header("NAIVE RAG (filtered chunks, flat)", YELLOW)
                _print_result(n, verbose=False)
                _header("TRADITIONAL RAG (raw markdown, overlap chunks)", GREEN)
                _print_result(o, verbose=False)
                def _top(r): return max((c.get("rerank_score", c.get("score", 0)) for c in r["chunks"]), default=0)
                tp, tn, to = _top(p), _top(n), _top(o)
                winner = "Pipeline" if tp >= tn and tp >= to else ("Naive" if tn >= to else "Traditional")
                print(f"\n  {BOLD}Top scores — Pipeline: {tp:.3f} | Naive: {tn:.3f} | Traditional: {to:.3f}{RESET}")
                print(f"  Winner: {winner}\n")
                conv.record_turn(query, p)

            elif turn_naive:
                result = retriever.retrieve_naive(standalone)
                _print_result(result, verbose=turn_verbose)
                conv.record_turn(query, result)

            elif turn_overlap:
                result = retriever.retrieve_overlap(standalone)
                _print_result(result, verbose=turn_verbose)
                conv.record_turn(query, result)

            elif turn_raw and turn_best_of:
                # --raw + --best-of combined: standalone diagnostic view —
                # every reformulation + its score, no caching/generation.
                # Independent of the gated default's own internal best-of-3.
                result = retriever.retrieve_best_of_n(
                    standalone, n=turn_best_of,
                    intent=turn_intent, version=version_to_use, verbose=turn_verbose,
                )
                scores = result.get("best_of_scores", [])
                refs   = result.get("reformulations", [])
                print(f"\n  {DIM}Best-of-{turn_best_of} reformulations:{RESET}")
                for j, (q, s) in enumerate(zip(refs, scores), 1):
                    marker = f"{GREEN}★{RESET}" if s == max(scores) else " "
                    print(f"  {marker} [{j}] {DIM}score={s:.3f}{RESET}  {q}")
                _print_result(result, verbose=turn_verbose)
                conv.record_turn(query, result)

            elif turn_raw:
                # Old default: plain retrieve(), no gating/caching/reformulation,
                # shows routing/chunk debug info instead of a generated answer.
                result = retriever.retrieve(
                    standalone, intent=turn_intent, version=version_to_use, verbose=turn_verbose,
                )
                _print_result(result, verbose=turn_verbose)
                conv.record_turn(query, result)

            else:
                # Gated + Redis-cached default (also what --cached explicitly
                # requests — the two are equivalent now). --best-of this turn
                # (if given) tunes the gate's own internal reformulation count
                # instead of triggering the standalone diagnostic view above —
                # this is what lets "--detailed --best-of 5" combine into ONE
                # gated call rather than best-of silently taking over. Ignored
                # by the library itself when mode="detailed" (see redis_cache.
                # answer_query's docstring — detailed never reformulates).
                kwargs = {"mode": turn_mode}
                if turn_best_of is not None:
                    kwargs["best_of"] = turn_best_of
                if turn_mode == "detailed" and turn_best_of:
                    print(f"  {DIM}Note: --best-of is ignored in detailed mode "
                          f"(it never reformulates — see redis_cache.py){RESET}")
                result = redis_cache.answer_query(standalone, **kwargs)
                _print_cached_result(result)
                conv.record_turn(query, result)
                conv.add_assistant_turn(result["answer"])

        except Exception as e:
            print(f"\n  {RED}Error: {e}{RESET}\n")
            if turn_verbose:
                import traceback
                traceback.print_exc()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Retrieval layer CLI")
    parser.add_argument("--verbose",  action="store_true", help="Show debug info")
    parser.add_argument("--session",  action="store_true",
                        help="Enable session mode: history, version pinning, query rewriting")
    parser.add_argument("--naive",    action="store_true",
                        help="Use traditional flat RAG (baseline, no routing)")
    parser.add_argument("--cached",   action="store_true",
                        help="Gated + Redis-cached (now the DEFAULT — this flag is kept only "
                             "for backward compatibility, behaves identically to no flag at all)")
    parser.add_argument("--raw",      action="store_true",
                        help="Opt OUT of the gated/cached default: plain retrieve(), no gating, "
                             "no caching, no best-of-N reformulation — shows routing/chunk debug "
                             "info instead of a generated answer (this was the old default)")
    parser.add_argument("--mode",     type=str, default="concise", choices=["concise", "detailed"],
                        help="Gated/cached mode: 'concise' (default, best-of-3 + confidence gate, "
                             "doesn't drift) or 'detailed' (always fullest pool, no best-of-N)")
    parser.add_argument("--concise",  action="store_true", help="Shorthand for --mode concise")
    parser.add_argument("--detailed", action="store_true", help="Shorthand for --mode detailed")
    parser.add_argument("--overlap",   action="store_true",
                        help="Use traditional RAG with sliding-window chunks from raw markdown")
    parser.add_argument("--best-of",  type=int, default=None, metavar="N",
                        help="With the gated default: overrides its internal reformulation count "
                             "(default 3). Combined with --raw instead: standalone diagnostic view "
                             "— reformulate N times via plain retrieve(), show every reformulation's "
                             "score, return the best.")
    parser.add_argument("--compare",  action="store_true",
                        help="Run both pipeline and naive RAG, show side-by-side")
    parser.add_argument("--version",  type=str, default=None,
                        help="Scope retrieval to a version e.g. 25.2 or 26.1")
    parser.add_argument("--intent",   type=str, default=None,
                        choices=["specific", "factual", "intelligence", "delta"],
                        help="Force a retrieval intent")
    parser.add_argument("query", nargs="?", default=None,
                        help="Single query (non-interactive mode). Inline --flags typed INSIDE "
                             "this quoted string also work, same as in the REPL — e.g. "
                             "python cli.py \"--detailed what changed in claim editing?\"")
    args = parser.parse_args()

    if args.detailed:
        args.mode = "detailed"
    if args.concise:               # --concise wins if both are somehow given
        args.mode = "concise"

    if args.query:
        # Single-shot mode — no persisted history across separate process
        # invocations, so session tracking (pins/rewriting/summary) has
        # nothing to build on here; that's the REPL's job. Same inline-flag
        # support as the REPL, and the same gated-by-default / --raw
        # opt-out choice.
        query_text, inline = _extract_inline_flags(args.query)
        turn_raw     = inline.get("raw", args.raw)
        turn_mode    = _resolve_turn_mode(inline, args.mode)
        turn_best_of = inline.get("best_of", args.best_of)
        turn_verbose = inline.get("verbose", args.verbose)

        if turn_raw and turn_best_of:
            result = retriever.retrieve_best_of_n(
                query_text, n=turn_best_of,
                intent=inline.get("intent", args.intent),
                version=inline.get("version", args.version),
                verbose=turn_verbose,
            )
            _print_result(result, verbose=turn_verbose)
        elif turn_raw:
            result = retriever.retrieve(
                query_text,
                intent=inline.get("intent", args.intent),
                version=inline.get("version", args.version),
                verbose=turn_verbose,
            )
            _print_result(result, verbose=turn_verbose)
        else:
            kwargs = {"mode": turn_mode}
            if turn_best_of is not None:
                kwargs["best_of"] = turn_best_of
            result = redis_cache.answer_query(query_text, **kwargs)
            _print_cached_result(result)
    else:
        repl(args)


if __name__ == "__main__":
    main()
