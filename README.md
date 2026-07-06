# HealthRules Payer Documentation Intelligence Pipeline

Turns a folder of multi-version enterprise release-note PDFs into a retrieval-augmented
chatbot that is measurably better than a plain vector-search baseline — and proves it,
on every change, against three other retrieval methods.

Full architecture, design history, and every decision behind this system are written up
in **[`KT_Session_Document.docx`](./KT_Session_Document.docx)** — read that for the deep
dive. This README is just enough to get it running.

## What's in here

```
Stage 1/           offline ingestion pipeline — PDFs -> chunked, labeled, taxonomy-aware
                   knowledge base (run once per corpus update)
retrieval_layer/   the serving layer — routing, reranking, confidence-gated retrieval,
                   evaluation harness, Redis-cached chatbot CLI
```

Two phases, one venv. `Stage 1/` reads raw PDFs and writes structured JSON/CSV artifacts.
`retrieval_layer/` reads those artifacts, builds a vector index, and answers questions.

## Quick start

```bash
cd "Stage 1"
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

You'll also need two things running locally (not pip-installable):

```bash
bash start_server.sh                # local Qwen3.5-9B LLM via llama.cpp, port 8080
sudo systemctl start redis-server   # response cache for the demo
```

### Run the pipeline (only needed once, or when the source PDFs change)

```bash
cd "Stage 1"
python main.py         # ingest -> label -> group -> build taxonomy
python run_tail.py     # cross-version analysis -> index -> ChromaDB -> eval -> cache seed
```

Drop your PDFs in `Stage 1/data/pdfs/` first — that folder isn't included in this repo
(it's the actual licensed source documentation, and it's large).

### Ask it questions

```bash
cd retrieval_layer
python cli.py                              # interactive chat — gated, cached, sessioned
python cli.py "What changed in claim editing between versions?"
python cli.py --raw --compare "..."        # see pipeline vs naive vs traditional RAG
```

Any flag can also be typed inline as part of the question itself, e.g.
`--detailed what changed in claim editing?` — see the KT document for the full list.

## What makes this more than a basic RAG demo

- **Self-comparing.** Every retrieval path (pipeline, naive, traditional RAG, and two pooled
  ensembles) is measured head-to-head on the same 53-query set, on both retrieval quality
  and generated-answer quality (faithfulness, correctness, completeness).
- **Dynamic confidence gating, not a fixed router.** The production path runs the cheap
  method first, measures its own confidence live, and only escalates to a more expensive
  pooled method when a specific query actually needs it.
- **Version-aware.** Detects when an answer would blend facts from two different document
  versions and injects the actual delta report instead of silently merging them.
- **Redis-backed caching** for instant, zero-token demo answers, side by side with the real
  live path for comparison.

## Where to look next

- `KT_Session_Document.docx` — full architecture, every design decision and why, eval
  results, known limitations, and a step-by-step run guide.
- `Stage 1/requirements.txt` — pinned dependencies for the whole project (both folders
  share one venv).
