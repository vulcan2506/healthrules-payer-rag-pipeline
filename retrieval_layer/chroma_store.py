"""
chroma_store.py
───────────────
Manages 4 ChromaDB collections, each with a distinct retrieval purpose:

  corpus         — filtered chunk texts (factual: what things ARE, how they work)
  intelligence   — topic + document summaries (overview: what domains/topics exist)
  delta          — delta analysis reports (change: what shifted between versions)
  corpus_overlap — traditional sliding-window chunks from raw markdown files
                   (baseline for ablation: no taxonomy, raw text with overlap)

Ingestion:
  build_all()  — populates all 4 collections from Stage 1 outputs

Querying:
  query_corpus(query, n, filters)
  query_intelligence(query, n, filters)
  query_delta(query, n, filters)
  query_overlap(query, n)
"""

import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
import pandas as pd
from sentence_transformers import SentenceTransformer

import config

log = logging.getLogger(__name__)


# ── Custom embedding function for ChromaDB ─────────────────────────────────────

class _MiniLMEmbedder(EmbeddingFunction):
    """Wraps our pipeline SentenceTransformer so ChromaDB uses the same model."""

    def __init__(self):
        self._model: Optional[SentenceTransformer] = None
        self._lock  = threading.Lock()

    def _get_model(self) -> SentenceTransformer:
        # Double-checked locking — see router.py's get_router() docstring for
        # why: retrieve_best_of_n's parallel reformulations (now the default
        # cheap-first-pass, not just an opt-in debug flag) can call this from
        # multiple threads on the very first query of a fresh process.
        if self._model is None:
            with self._lock:
                if self._model is None:
                    log.info(f"Loading embedder for ChromaDB: {config.EMBED_MODEL}")
                    self._model = SentenceTransformer(config.EMBED_MODEL, device="cpu")
        return self._model

    def __call__(self, input: Documents) -> Embeddings:
        model = self._get_model()
        return model.encode(list(input), normalize_embeddings=True).tolist()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _version_from_doc(source_doc: str) -> str:
    """Extract version string from filename e.g. '25.2_HR_Payer...' → '25.2'."""
    m = re.match(r"(\d+\.\d+)", source_doc)
    return m.group(1) if m else "unknown"


def _safe_str(val: Any) -> str:
    return "" if val is None or (isinstance(val, float) and str(val) == "nan") else str(val)


def _sliding_chunks(text: str, size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks. Tries to break on newline boundaries
    rather than mid-word. This is the standard approach for traditional RAG.
    """
    chunks = []
    start  = 0
    length = len(text)
    while start < length:
        end = min(start + size, length)
        # Prefer to end at a natural line break
        if end < length:
            nl = text.rfind("\n", start, end)
            if nl > start:
                end = nl + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start
    return chunks


# ── ChromaStore ────────────────────────────────────────────────────────────────

class ChromaStore:

    def __init__(self):
        config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self._client      = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        self._embedder    = _MiniLMEmbedder()
        self._corpus      = None
        self._intelligence = None
        self._delta       = None
        self._overlap     = None

    def _col(self, name: str):
        return self._client.get_or_create_collection(
            name=name,
            embedding_function=self._embedder,
            metadata={"hnsw:space": "cosine"},
        )

    def warm(self) -> None:
        """
        Force-load the shared embedder model now. retrieve_best_of_n()
        (retriever.py) calls this on the MAIN thread before spinning up its
        ThreadPoolExecutor — see reranker.warm()'s docstring for why.
        """
        self._embedder._get_model()

    @property
    def corpus(self):
        if self._corpus is None:
            self._corpus = self._col(config.CORPUS_COLLECTION)
        return self._corpus

    @property
    def intelligence(self):
        if self._intelligence is None:
            self._intelligence = self._col(config.INTELLIGENCE_COLLECTION)
        return self._intelligence

    @property
    def delta(self):
        if self._delta is None:
            self._delta = self._col(config.DELTA_COLLECTION)
        return self._delta

    @property
    def overlap(self):
        if self._overlap is None:
            self._overlap = self._col(config.OVERLAP_COLLECTION)
        return self._overlap

    # ── Ingestion ──────────────────────────────────────────────────────────────

    def ingest_corpus(self, path: Path = config.FILTERED_CHUNKS, batch: int = 256):
        """
        Load filtered_chunks.json → corpus collection.
        One document per chunk. Metadata tags: source_doc, version, section_header,
        master_label (if present).
        """
        log.info(f"Ingesting corpus from {path.name}...")
        with open(path, encoding="utf-8") as f:
            chunks = json.load(f)

        ids, docs, metas = [], [], []
        for c in chunks:
            text = _safe_str(c.get("text", "")).strip()
            if not text:
                continue
            cid = str(c.get("chunk_id", ""))
            ids.append(f"chunk_{cid}")
            docs.append(text)
            metas.append({
                "chunk_id":       cid,
                "source_doc":     _safe_str(c.get("source_doc", "")),
                "version":        _version_from_doc(_safe_str(c.get("source_doc", ""))),
                "section_header": _safe_str(c.get("section_header", "")),
                "master_label":   _safe_str(c.get("master_label", "")),
                "type":           "chunk",
            })

        self._upsert_batched(self.corpus, ids, docs, metas, batch, "corpus")

    def ingest_intelligence(self, registry_path: Path = config.REGISTRY_CSV, batch: int = 128):
        """
        Load topic_registry.csv → intelligence collection.
        One document per master_label using summarized_description.
        """
        log.info(f"Ingesting intelligence from {registry_path.name}...")
        df = pd.read_csv(registry_path)

        ids, docs, metas = [], [], []
        for _, row in df.iterrows():
            label   = _safe_str(row.get("master_label", ""))
            summary = _safe_str(row.get("summarized_description", ""))
            raw_desc = _safe_str(row.get("description", ""))
            text = summary if summary.strip() else raw_desc
            if not text.strip() or not label:
                continue

            src_docs = _safe_str(row.get("source_docs", ""))
            versions = sorted({_version_from_doc(s.strip()) for s in src_docs.split("|") if s.strip()})

            # qna is stored as JSON dict {"source_doc": [{q,a},...]} — keep as string
            qna_raw = _safe_str(row.get("qna", "{}"))

            ids.append(f"topic_{re.sub(r'[^a-z0-9]', '_', label.lower())[:60]}")
            docs.append(f"{label}. {text}")
            metas.append({
                "master_label": label,
                "source_docs":  src_docs,
                "versions":     ",".join(versions),
                "keywords":     _safe_str(row.get("keywords", "")),
                "n_chunks":     int(row.get("n_chunks", 0)),
                "qna":          qna_raw,   # JSON dict keyed by source_doc
                "type":         "topic_summary",
            })

        self._upsert_batched(self.intelligence, ids, docs, metas, batch, "intelligence")

    def ingest_hierarchy_summaries(self, nested_path: Path = config.NESTED_JSON, batch: int = 64):
        """
        Load parent/sub-category group profiles (hierarchy_summarizer.py
        output) into the SAME intelligence collection as topic summaries —
        these are what actually answer breadth queries like "list all the
        domains covered" that no single topic summary can.

        Embeds `*_overview` (short prose) but STORES `*_detail` (the full
        delta-analyzer-style structured profile: named features + notable
        details). These need to differ: the bi-encoder used for embedding
        (max_seq_length=128 tokens) was measured to do worse the denser the
        text gets — a raw name list front-loaded into the overview dropped
        Q9's score from -8.03 to -11.32 — so the embedding stays short and
        coherent. The cross-encoder reranker (max_seq_length=512) and
        answer-generation both read whatever's stored, independent of what
        was embedded, so that's where the granularity belongs. Passing
        precomputed `embeddings` to upsert() is what makes embed-text and
        stored-text able to differ at all.
        """
        log.info(f"Ingesting hierarchy summaries from {nested_path.name}...")
        if not nested_path.exists():
            log.warning(f"{nested_path.name} not found — skipping hierarchy ingestion")
            return

        with open(nested_path, encoding="utf-8") as f:
            taxonomy = json.load(f)["taxonomy"]

        ids, docs, metas, embed_texts = [], [], [], []
        for p_idx, parent in enumerate(taxonomy):
            pname     = _safe_str(parent.get("parent_category_name", ""))
            poverview = _safe_str(parent.get("parent_category_overview", ""))
            pdetail   = _safe_str(parent.get("parent_category_detail", "")) or poverview
            if pname and poverview.strip():
                ids.append(f"parent_{p_idx}_{re.sub(r'[^a-z0-9]', '_', pname.lower())[:50]}")
                embed_texts.append(f"{pname}. {poverview}")
                docs.append(f"{pname}. {pdetail}")
                metas.append({"parent_category_name": pname, "type": "parent_summary"})

            for s_idx, sub in enumerate(parent.get("sub_categories", [])):
                sname     = _safe_str(sub.get("sub_category_name", ""))
                soverview = _safe_str(sub.get("sub_category_overview", ""))
                sdetail   = _safe_str(sub.get("sub_category_detail", "")) or soverview
                if sname and soverview.strip():
                    ids.append(f"sub_{p_idx}_{s_idx}_{re.sub(r'[^a-z0-9]', '_', sname.lower())[:50]}")
                    embed_texts.append(f"{sname}. {soverview}")
                    docs.append(f"{sname}. {sdetail}")
                    metas.append({
                        "parent_category_name": pname,
                        "sub_category_name":     sname,
                        "type":                  "sub_summary",
                    })

        if not ids:
            log.warning("intelligence (hierarchy): nothing to ingest")
            return

        embedder = SentenceTransformer(config.EMBED_MODEL, device="cpu")
        embeddings = embedder.encode(embed_texts, normalize_embeddings=True).tolist()

        self._upsert_batched(self.intelligence, ids, docs, metas, batch,
                              "intelligence (hierarchy)", embeddings=embeddings)

    def ingest_enumeration_doc(self, nested_path: Path = config.NESTED_JSON):
        """
        Adds ONE document to the intelligence collection that IS a flat list
        of every parent category name. Every other row in this collection
        describes a single category; queries like "list all the domains" or
        "what broadly does this system handle" are themselves list-shaped —
        no single-category description can match that shape no matter how
        many candidates get reranked. This is the one row built to match it.
        """
        log.info(f"Ingesting enumeration doc from {nested_path.name}...")
        if not nested_path.exists():
            log.warning(f"{nested_path.name} not found — skipping enumeration doc")
            return

        with open(nested_path, encoding="utf-8") as f:
            taxonomy = json.load(f)["taxonomy"]

        names = [_safe_str(p.get("parent_category_name", "")) for p in taxonomy]
        names = [n for n in names if n]
        if not names:
            log.warning("enumeration doc: no parent names found — skipping")
            return

        text = "This documentation covers the following domains:\n\n" + "\n".join(
            f"{i + 1}. {n}" for i, n in enumerate(names)
        )

        self._upsert_batched(
            self.intelligence,
            ids=["enumeration_all_domains"],
            docs=[text],
            metas=[{"type": "enumeration", "n_domains": len(names)}],
            batch_size=1,
            name="intelligence (enumeration)",
        )


    def ingest_parent_relationships(self, path: Path = config.PARENT_RELATIONSHIPS, batch: int = 64):
        """
        Grandparent layer (Tier B of the cross-family expansion design —
        parent_relationship.py). A pure additive wrapper: each row is
        {parent_category_name, parent_category_description, member_parents:
        [full original parent objects, untouched]} — the grouping comes from
        Louvain community detection over sub-parent-grounded similarity, not
        a top-K nearest-neighbor heuristic (see parent_relationship.py). This
        ingests only the grandparent's own name+description as a new
        intelligence row; the member parents' own content is already
        ingested individually by ingest_hierarchy_summaries — this doesn't
        re-ingest them.
        """
        log.info(f"Ingesting grandparent groups from {path.name}...")
        if not path.exists():
            log.warning(f"{path.name} not found — skipping relationship ingestion "
                        f"(run parent_relationship.py first)")
            return

        with open(path, encoding="utf-8") as f:
            clusters = json.load(f)

        # Clear any prior relationship rows before re-ingesting — otherwise
        # stale rows with the same type would sit alongside the new ones and
        # still surface via query_relationships().
        existing = self.intelligence.get(where={"type": "parent_relationship"})
        if existing["ids"]:
            self.intelligence.delete(ids=existing["ids"])
            log.info(f"Cleared {len(existing['ids'])} prior relationship row(s)")

        ids, docs, metas, embed_texts = [], [], [], []
        for i, c in enumerate(clusters):
            gp_name = _safe_str(c.get("parent_category_name", ""))
            gp_desc = _safe_str(c.get("parent_category_description", ""))
            members = [
                _safe_str(m.get("parent_category_name", ""))
                for m in c.get("member_parents", []) if m.get("parent_category_name")
            ]
            if not (gp_name and gp_desc.strip() and members):
                continue

            uid = f"relationship_cluster_{i}"
            ids.append(uid)
            embed_texts.append(f"{gp_name}. {gp_desc}")
            docs.append(f"{gp_name} ({' × '.join(members)}). {gp_desc}")
            metas.append({
                "parent_category_name": gp_name,
                "members": " | ".join(members),
                "n_members": len(members),
                "type": "parent_relationship",
            })

        if not ids:
            log.warning("intelligence (relationships): nothing to ingest "
                        "(no genuine clusters found, or file empty)")
            return

        embedder = SentenceTransformer(config.EMBED_MODEL, device="cpu")
        embeddings = embedder.encode(embed_texts, normalize_embeddings=True).tolist()

        self._upsert_batched(self.intelligence, ids, docs, metas, batch,
                              "intelligence (relationships)", embeddings=embeddings)

    def ingest_cross_corpus_relationships(self, path: Path = config.CROSS_CORPUS_RELATIONSHIPS, batch: int = 64):
        """
        Cross-corpus relationship layer (cross_corpus_relationship.py) — links
        parent categories across TWO SEPARATE document corpora (e.g. payer and
        a future connector/security taxonomy), unlike ingest_parent_relationships
        which links parents WITHIN one taxonomy. Same embed/store split pattern.
        No-ops quietly if the file doesn't exist or is empty — expected today,
        since only one real corpus ("payer") exists (see cross_corpus_relationship.py
        module docstring for validation status).
        """
        log.info(f"Ingesting cross-corpus relationships from {path.name}...")
        if not path.exists():
            log.warning(f"{path.name} not found — skipping cross-corpus ingestion "
                        f"(no second corpus yet, or run cross_corpus_relationship.py first)")
            return

        with open(path, encoding="utf-8") as f:
            clusters = json.load(f)

        existing = self.intelligence.get(where={"type": "cross_corpus_relationship"})
        if existing["ids"]:
            self.intelligence.delete(ids=existing["ids"])
            log.info(f"Cleared {len(existing['ids'])} prior cross-corpus relationship row(s)")

        ids, docs, metas, embed_texts = [], [], [], []
        for i, c in enumerate(clusters):
            gp_name = _safe_str(c.get("parent_category_name", ""))
            gp_desc = _safe_str(c.get("parent_category_description", ""))
            source_tags = [_safe_str(s) for s in c.get("source_type_keys", [])]
            members = [
                _safe_str(m.get("parent_category_name", ""))
                for m in c.get("member_parents", []) if m.get("parent_category_name")
            ]
            if not (gp_name and gp_desc.strip() and members):
                continue

            uid = f"cross_corpus_cluster_{i}"
            ids.append(uid)
            embed_texts.append(f"{gp_name}. {gp_desc}")
            docs.append(f"{gp_name} ({' × '.join(members)}). {gp_desc}")
            metas.append({
                "parent_category_name": gp_name,
                "members": " | ".join(members),
                "n_members": len(members),
                "source_type_keys": " | ".join(source_tags),
                "type": "cross_corpus_relationship",
            })

        if not ids:
            log.warning("intelligence (cross-corpus): nothing to ingest "
                        "(no genuine clusters found, or file empty)")
            return

        embedder = SentenceTransformer(config.EMBED_MODEL, device="cpu")
        embeddings = embedder.encode(embed_texts, normalize_embeddings=True).tolist()

        self._upsert_batched(self.intelligence, ids, docs, metas, batch,
                              "intelligence (cross-corpus)", embeddings=embeddings)

    def ingest_evolution_cards(self, path: Path = config.EVOLUTION_CARDS, batch: int = 64):
        """
        Value-add / evolution cards (evolution_analyzer.py) — the constructive
        counterpart to the delta collection. Where `delta` findings are neutral
        about whether a change is good, bad, or a contradiction, these cards are
        pre-filtered to only constructive change types and are framed as
        "what foundation did the older version lay, what value did the newer
        version add on top of it." Same embed/store split as
        ingest_parent_relationships: embed the short narrative, store the full
        card (foundation + value_added bullets + narrative) so the reranker's
        512-token cross-encoder sees the concrete details the 128-token
        bi-encoder would dilute if asked to embed all of it.
        """
        log.info(f"Ingesting evolution cards from {path.name}...")
        if not path.exists():
            log.warning(f"{path.name} not found — skipping evolution ingestion "
                        f"(run evolution_analyzer.py first)")
            return

        with open(path, encoding="utf-8") as f:
            entries = json.load(f)

        existing = self.intelligence.get(where={"type": "evolution_card"})
        if existing["ids"]:
            self.intelligence.delete(ids=existing["ids"])
            log.info(f"Cleared {len(existing['ids'])} prior evolution card row(s)")

        ids, docs, metas, embed_texts = [], [], [], []
        for i, entry in enumerate(entries):
            card = entry.get("card", {})
            feature_name = _safe_str(card.get("feature_name", entry.get("topic", "")))
            narrative    = _safe_str(card.get("narrative", ""))
            foundation   = _safe_str(card.get("foundation", ""))
            value_added  = [_safe_str(v) for v in card.get("value_added", [])]
            if not (feature_name and narrative):
                continue

            vA, vB = _safe_str(entry.get("vA", "")), _safe_str(entry.get("vB", ""))
            stored = (
                f"{feature_name} ({vA} → {vB}). Foundation: {foundation} "
                f"Value added: {'; '.join(value_added)} {narrative}"
            )

            uid = f"evolution_{i}"
            ids.append(uid)
            embed_texts.append(f"{feature_name}. {narrative}")
            docs.append(stored)
            metas.append({
                "feature_name": feature_name,
                "vA": vA,
                "vB": vB,
                "change_type": _safe_str(card.get("change_type", "")),
                "parent": _safe_str(entry.get("parent", "")),
                "sub": _safe_str(entry.get("sub", "")),
                "type": "evolution_card",
            })

        if not ids:
            log.warning("intelligence (evolution): nothing to ingest")
            return

        embedder = SentenceTransformer(config.EMBED_MODEL, device="cpu")
        embeddings = embedder.encode(embed_texts, normalize_embeddings=True).tolist()

        self._upsert_batched(self.intelligence, ids, docs, metas, batch,
                              "intelligence (evolution)", embeddings=embeddings)

    def ingest_delta(self, delta_dir: Path = config.DELTA_DIR, batch: int = 64):
        """
        Load delta analysis report files → delta collection.

        Expects JSON files in delta_dir, each being a list of findings:
        [{"section": str, "change_type": str, "summary": str,
          "detail": str, "version_from": str, "version_to": str}, ...]

        Also accepts a single dict with a "findings" key.

        NOTE: a "No Change Detected" filter was tried here (2026-07-05) —
        it measurably fixed Q12-style "what's new" queries by removing
        hollow boilerplate findings, but was reverted at the user's request
        to keep the aggregate eval mean score at its prior 1.985 baseline.
        See [[evolution-analyzer]] memory for the full tradeoff (quality vs.
        aggregate-score) before re-adding this.
        """
        if not delta_dir.exists():
            log.warning(f"Delta dir not found: {delta_dir} — skipping delta ingestion")
            return

        report_files = list(delta_dir.glob("*.json"))
        if not report_files:
            log.warning(f"No delta report JSON files in {delta_dir}")
            return

        # Clear all prior rows before re-ingesting — deterministic IDs mean
        # upsert() only touches rows still generated this run; a finding
        # that gets filtered out (or a source file that's removed) would
        # otherwise leave its stale row behind forever. Found exactly this
        # bug immediately after adding the "No Change Detected" filter above
        # — the filter worked, but 11 stale rows from the prior ingestion
        # kept answering queries anyway until this clear was added.
        existing = self.delta.get()
        if existing["ids"]:
            self.delta.delete(ids=existing["ids"])
            log.info(f"Cleared {len(existing['ids'])} prior delta row(s)")

        log.info(f"Ingesting delta from {len(report_files)} report file(s)...")
        ids, docs, metas = [], [], []

        for fpath in report_files:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)

            findings = data if isinstance(data, list) else data.get("findings", [])

            for i, finding in enumerate(findings):
                section      = _safe_str(finding.get("section", ""))
                change_type  = _safe_str(finding.get("change_type", ""))
                summary      = _safe_str(finding.get("summary", ""))
                detail       = _safe_str(finding.get("detail", ""))
                v_from       = _safe_str(finding.get("version_from", ""))
                v_to         = _safe_str(finding.get("version_to", ""))

                text = f"{section}. {summary} {detail}".strip()
                if not text:
                    continue

                uid = f"delta_{fpath.stem}_{i}"
                ids.append(uid)
                docs.append(text)
                metas.append({
                    "section":      section,
                    "change_type":  change_type,
                    "version_from": v_from,
                    "version_to":   v_to,
                    "source_file":  fpath.name,
                    "type":         "delta_finding",
                })

        self._upsert_batched(self.delta, ids, docs, metas, batch, "delta")

    def ingest_overlap(
        self,
        md_dir: Path = config.MARKDOWN_DIR,
        chunk_size: int = 1000,
        overlap: int = 200,
        batch: int = 256,
    ):
        """
        Load raw markdown files → corpus_overlap collection.

        Splits each file into overlapping sliding-window chunks (no taxonomy,
        no pre-filtering) so it mirrors how a traditional RAG pipeline works.
        Source doc label is derived dynamically from the filename — no version
        names are hardcoded here.
        """
        md_files = sorted(md_dir.glob("*.md")) if md_dir.exists() else []
        if not md_files:
            log.warning(f"No markdown files found in {md_dir} — skipping overlap ingestion")
            return

        ids, docs, metas = [], [], []
        for md_path in md_files:
            # Derive a human-readable source label from filename, no hardcoding
            stem = re.sub(r"_Converted$", "", md_path.stem, flags=re.IGNORECASE)
            source_doc = stem.replace("_", " ")

            text = md_path.read_text(encoding="utf-8")
            chunks = _sliding_chunks(text, chunk_size, overlap)
            log.info(f"overlap: {md_path.name} → {len(chunks)} chunks "
                     f"(size={chunk_size}, overlap={overlap})")

            for i, chunk_text in enumerate(chunks):
                ids.append(f"overlap_{md_path.stem}_{i:04d}")
                docs.append(chunk_text)
                metas.append({
                    "source_doc":  source_doc,
                    "chunk_idx":   i,
                    "source_file": md_path.name,
                    "type":        "overlap_chunk",
                })

        self._upsert_batched(self.overlap, ids, docs, metas, batch, "overlap")

    def _upsert_batched(self, collection, ids, docs, metas, batch_size, name, embeddings=None):
        """
        embeddings, if given, are used instead of auto-embedding `docs` via
        the collection's embedding_function — lets the stored/returned text
        differ from what gets embedded (see ingest_hierarchy_summaries).
        """
        total = len(ids)
        if total == 0:
            log.warning(f"{name}: nothing to ingest")
            return
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            kwargs = dict(ids=ids[start:end], documents=docs[start:end], metadatas=metas[start:end])
            if embeddings is not None:
                kwargs["embeddings"] = embeddings[start:end]
            collection.upsert(**kwargs)
        log.info(f"{name}: {total} documents ingested into '{collection.name}'")

    # ── Querying ───────────────────────────────────────────────────────────────

    def query_corpus(
        self,
        query: str,
        n: int = config.CHROMA_N_CORPUS,
        version: Optional[str] = None,
        source_doc: Optional[str] = None,
    ) -> List[Dict]:
        """Factual retrieval from raw chunks."""
        where = self._build_where(version=version, source_doc=source_doc)
        return self._query(self.corpus, query, n, where)

    def query_intelligence(
        self,
        query: str,
        n: int = config.CHROMA_N_INTELLIGENCE,
        version: Optional[str] = None,
    ) -> List[Dict]:
        """
        Overview/summary retrieval from topic descriptions.

        Excludes type=parent_relationship, type=evolution_card, and
        type=cross_corpus_relationship — those rows share this collection
        but are only meant to answer their own dedicated query methods
        (query_relationships / query_evolution / query_cross_corpus). Left
        unfiltered here, they'd compete as extra candidates against every
        intelligence-intent query; parent_relationship was measured to
        regress one query this way (Q10, -3.817 -> -4.220) despite being
        purely additive rows — a shared collection can still get
        contaminated by irrelevant competition, not just by editing existing
        rows.
        """
        conditions = [{"$and": [
            {"type": {"$ne": "parent_relationship"}},
            {"type": {"$ne": "evolution_card"}},
            {"type": {"$ne": "cross_corpus_relationship"}},
        ]}]
        base_where = self._build_where(version=version)
        if base_where:
            conditions.append(base_where)
        where = conditions[0] if len(conditions) == 1 else {"$and": conditions}
        return self._query(self.intelligence, query, n, where)

    def query_relationships(self, query: str, n: int = 3) -> List[Dict]:
        """
        Parent×Parent relationship docs only (Tier B) — used as the last
        rung of the specific-path confidence ladder, after same-family
        (expand_siblings) and different-family raw-chunk (expand_cross_parent)
        both left confidence low. Filtered to type=parent_relationship so a
        genuinely cross-domain query can surface the synthesized connection
        doc instead of two disconnected single-parent summaries.
        """
        where = {"type": {"$eq": "parent_relationship"}}
        return self._query(self.intelligence, query, n, where)

    def query_evolution(self, query: str, n: int = config.CHROMA_N_EVOLUTION) -> List[Dict]:
        """
        Value-add / evolution cards only (evolution_analyzer.py). Filtered to
        type=evolution_card so "how has X evolved" / "what value did X add"
        style questions can surface the constructive foundation->value-added
        narrative instead of (or alongside) the neutral delta collection's
        raw change classification.
        """
        where = {"type": {"$eq": "evolution_card"}}
        return self._query(self.intelligence, query, n, where)

    def query_cross_corpus(self, query: str, n: int = config.CHROMA_N_CROSS_CORPUS) -> List[Dict]:
        """
        Cross-corpus relationship docs only (cross_corpus_relationship.py) —
        parent categories linked ACROSS two separate document corpora, e.g.
        payer <-> connector. Used as a confidence-ladder rung, same position
        as query_relationships (Tier B) but for the cross-corpus case. Empty
        today — no second corpus exists yet — but wired so it activates the
        moment cross_corpus_relationship_clusters.json has real content.
        """
        where = {"type": {"$eq": "cross_corpus_relationship"}}
        return self._query(self.intelligence, query, n, where)

    def query_delta(
        self,
        query: str,
        n: int = config.CHROMA_N_DELTA,
        change_type: Optional[str] = None,
        version_from: Optional[str] = None,
        version_to: Optional[str] = None,
    ) -> List[Dict]:
        """Change-analysis retrieval from delta reports."""
        conditions = {}
        if change_type:
            conditions["change_type"] = {"$eq": change_type}
        if version_from:
            conditions["version_from"] = {"$eq": version_from}
        if version_to:
            conditions["version_to"] = {"$eq": version_to}
        where = {"$and": list(conditions.values())} if len(conditions) > 1 else (
            list(conditions.values())[0] if conditions else None
        )
        return self._query(self.delta, query, n, where)

    def query_overlap(
        self,
        query: str,
        n: int = config.CHROMA_N_CORPUS,
    ) -> List[Dict]:
        """Traditional sliding-window RAG retrieval from raw markdown chunks."""
        return self._query(self.overlap, query, n, where=None)

    def _build_where(self, version: Optional[str] = None,
                     source_doc: Optional[str] = None) -> Optional[Dict]:
        conditions = []
        if version:
            conditions.append({"version": {"$eq": version}})
        if source_doc:
            conditions.append({"source_doc": {"$eq": source_doc}})
        if len(conditions) == 0:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def _query(self, collection, query: str, n: int, where: Optional[Dict]) -> List[Dict]:
        kwargs = {"query_texts": [query], "n_results": n,
                  "include": ["documents", "metadatas", "distances"]}
        if where:
            kwargs["where"] = where
        try:
            res = collection.query(**kwargs)
        except Exception as e:
            log.warning(f"ChromaDB query failed on '{collection.name}': {e}")
            return []

        out = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            out.append({
                "text":     doc,
                "score":    round(1 - dist, 4),   # cosine distance → similarity
                **meta,
            })
        return out

    # ── Stats ──────────────────────────────────────────────────────────────────

    def stats(self) -> Dict:
        return {
            "corpus":         self.corpus.count(),
            "intelligence":   self.intelligence.count(),
            "delta":          self.delta.count(),
            "corpus_overlap": self.overlap.count(),
        }


# ── Build helper ───────────────────────────────────────────────────────────────

def build_all():
    """Ingest all available Stage 1 outputs into the 3 collections."""
    store = ChromaStore()

    if config.FILTERED_CHUNKS.exists():
        store.ingest_corpus()
    else:
        log.warning(f"filtered_chunks.json not found — skipping corpus ingestion")

    if config.REGISTRY_CSV.exists():
        store.ingest_intelligence()
    else:
        log.warning(f"topic_registry.csv not found — skipping intelligence ingestion")

    store.ingest_hierarchy_summaries()
    store.ingest_enumeration_doc()
    store.ingest_parent_relationships()
    store.ingest_evolution_cards()  # warns internally if cache missing
    store.ingest_cross_corpus_relationships()  # warns internally if cache missing (no 2nd corpus yet)

    store.ingest_delta()  # warns internally if dir missing

    if config.MARKDOWN_DIR.exists():
        store.ingest_overlap()
    else:
        log.warning(f"Markdown dir not found at {config.MARKDOWN_DIR} — skipping overlap ingestion")

    log.info(f"ChromaDB stats: {store.stats()}")
    return store


# ── Singleton ──────────────────────────────────────────────────────────────────

_store: Optional[ChromaStore] = None
_store_lock = threading.Lock()

def get_store() -> ChromaStore:
    # Double-checked locking — same reason as router.get_router() and
    # reranker._get_model(): retrieve_best_of_n's parallel reformulations are
    # now the default cheap-first-pass, so multiple threads can race here on
    # the first query of a fresh process.
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = ChromaStore()
    return _store


def warm() -> None:
    """Module-level convenience — see ChromaStore.warm()'s docstring."""
    get_store().warm()


def reset_store() -> None:
    """
    Drops the cached ChromaStore so the next get_store() call re-opens
    chroma_db/ from disk. Needed after a corpus reset + reprocess — without
    this, a long-lived api_server process keeps serving the OLD in-memory
    store even after the underlying files are replaced.
    """
    global _store
    with _store_lock:
        _store = None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    build_all()
