"""
cross_corpus_relationship.py
─────────────────────────────
Generalizes parent_relationship.py's Louvain-clustering "grandparent" layer
to link parent categories ACROSS two or more separate document corpora
(e.g. a "payer" taxonomy and a future "connector"/"security" taxonomy),
instead of within one taxonomy.

KEY DIFFERENCE FROM parent_relationship.py
───────────────────────────────────────────
parent_relationship.py groups parents WITHIN one enterprise_nested_topics.json
— that's already a solved, validated problem (see PARENT_RELATIONSHIPS /
[[confidence-ladder]]). This file exists for the genuinely new case: linking
parents that live in ENTIRELY SEPARATE taxonomy files (each built by its own
Stage 1 run against its own document set — see nested.py's CSV_PATH/
OUTPUT_JSON_PATH, now read from config.py so a second corpus can point
OUTPUT_DIR elsewhere and get its own file without colliding with payer's).

Same embedding + Louvain + outlier-audit + base-name-dedup machinery as
parent_relationship.py (matching nested.py's own MACRO_THRESHOLD mechanism),
but graph edges are restricted to CROSS-SOURCE pairs only — same-source pairs
are skipped entirely, since in-corpus grouping is parent_relationship.py's
job, not this file's. This is what makes the output "cross-corpus linking"
rather than a duplicate of the existing grandparent layer.

VALIDATION STATUS (as of 2026-07-05)
─────────────────────────────────────
Only ONE real document type ("payer") exists today — there is no second
corpus to link against yet. This file is built generic (no doc-type
hardcoded) and mechanism-tested via a synthetic 2-way split of the existing
payer taxonomy (see the throwaway test script used for that — NOT part of
this file), which only proves the cross-source-only edge restriction and
composite (source_tag, idx) member lookup behave correctly. It does NOT
prove anything about real cross-corpus relationship QUALITY — that can only
be validated once a second real corpus exists, at which point the
similarity threshold below should be recalibrated (cross-corpus vocabulary
overlap is likely lower than within one corpus, so 0.60 may be too strict).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import community.community_louvain as community_louvain
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import config
import llm_client
from nested import parse_llm_list
from topic_summarizer import _get_domain_profile

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

OUT_PATH = config.OUTPUT_DIR / "cross_corpus_relationship_clusters.json"

EMBED_MODEL      = "all-MiniLM-L6-v2"   # matches nested.py / parent_relationship.py
EDGE_CASE_NAME   = "Edge Cases & Standalone Topics"
CROSS_CORPUS_THRESHOLD = 0.60   # same starting point as nested.py's MACRO_THRESHOLD —
                                 # recalibrate once a real second corpus exists (see module docstring)

# Sources to cross-link. Each entry is (source_tag, path to that corpus's own
# enterprise_nested_topics.json-shaped file). Only one real corpus exists
# today — add a second line here (e.g. ("connector", Path(...))) once a
# second document type has been through its own Stage 1 run.
SOURCES: List[Tuple[str, Path]] = [
    ("payer", config.NESTED_OUTPUT_PATH),
]


def _load_all_parents(sources: List[Tuple[str, Path]]) -> Tuple[List[Dict], Dict[str, Dict]]:
    """
    Loads parents from every source, tagging each with its source_tag and a
    GLOBAL sequential idx (0..N-1 across all sources combined) — this is the
    one real coupling point parent_relationship.py has that doesn't carry
    over: its member_parents lookup indexes a single flat
    taxonomy_data["taxonomy"][m] array. Here that lookup needs the
    (source_tag, local_idx) pair, tracked via each parent dict's own
    "source_tag"/"local_idx" fields plus the returned taxonomy_by_tag dict.
    """
    parents: List[Dict] = []
    taxonomy_by_tag: Dict[str, Dict] = {}
    global_idx = 0

    for source_tag, path in sources:
        if not path.exists():
            log.warning(f"Source '{source_tag}': {path} not found — skipping")
            continue
        with open(path, encoding="utf-8") as f:
            taxonomy_data = json.load(f)
        taxonomy_by_tag[source_tag] = taxonomy_data

        for local_idx, p in enumerate(taxonomy_data["taxonomy"]):
            name = p.get("parent_category_name", "")
            subs = p.get("sub_categories", [])
            sub_lines = [
                f"{s.get('sub_category_name','')}: {s.get('sub_category_description','')}"
                for s in subs if s.get("sub_category_name") and s.get("sub_category_name") != EDGE_CASE_NAME
            ]
            parents.append({
                "idx": global_idx,
                "source_tag": source_tag,
                "local_idx": local_idx,
                "name": name,
                "combined_text": f"{name}. " + " ".join(sub_lines),
            })
            global_idx += 1

    return parents, taxonomy_by_tag


def _embed_parents(parents: List[Dict]):
    embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
    return embedder.encode([p["combined_text"] for p in parents], show_progress_bar=False)


def _build_cross_source_graph(parents: List[Dict], embs) -> nx.Graph:
    """
    Same similarity-threshold graph as parent_relationship.py's
    _build_parent_graph, EXCEPT edges are only added between parents from
    DIFFERENT source_tags. Same-source pairs never get a direct edge here —
    in-corpus grouping is parent_relationship.py's job, not this file's.
    """
    sim = cosine_similarity(embs)
    G = nx.Graph()
    G.add_nodes_from(range(len(embs)))
    for i in range(len(sim)):
        for j in range(i + 1, len(sim)):
            if parents[i]["source_tag"] == parents[j]["source_tag"]:
                continue
            if sim[i][j] > CROSS_CORPUS_THRESHOLD:
                G.add_edge(i, j, weight=float(sim[i][j]))
    return G


def _outlier_prompt(names: List[str], analyst_role: str) -> str:
    """Identical prompt shape to parent_relationship.py's _outlier_prompt —
    kept in sync deliberately rather than imported, since the two files are
    meant to stay independently readable/removable."""
    return (
        f"You are {analyst_role}, acting as an expert systems auditor. "
        "I will give you a list of top-level feature-area categories, drawn "
        "from DIFFERENT document sets, that an algorithm grouped together as "
        "related.\n"
        "Your job is to find the 'Odd One Out' (Outliers). An outlier is a "
        "category that does NOT share a genuine functional or technical "
        "connection with the rest.\n\n"
        "### Examples:\n"
        "Group:\n1. Claims Processing Workflow\n2. Remittance Advice Management\n"
        "3. Network Security Protocols\n"
        "Outliers: [\"Network Security Protocols\"]\n\n"
        "Group:\n1. Patient Registration\n2. Appointment Scheduling\n"
        "3. Clinical Documentation\n"
        "Outliers: [\"NONE\"]\n\n"
        "### Now evaluate this Group:\n"
        + "\n".join(f"- {n}" for n in names) + "\n\n"
        "Reply EXACTLY with a Python list of the exact category names that are "
        "outliers. If all fit perfectly, reply [\"NONE\"]. Do not add any "
        "conversational text."
    )


def _audit_outliers(clusters: Dict[int, List[int]], parent_names: Dict[int, str], analyst_role: str) -> Dict[int, List[int]]:
    multi = {cid: members for cid, members in clusters.items() if len(members) > 1}
    if not multi:
        return clusters

    cluster_ids = list(multi.keys())
    prompts = [_outlier_prompt([parent_names[m] for m in multi[cid]], analyst_role) for cid in cluster_ids]

    log.info(f"Auditing {len(prompts)} multi-member cross-corpus group(s) for outliers...")
    responses = llm_client.generate_batch(prompts, max_tokens=80, desc="Cross-corpus outlier audit")

    cleaned: Dict[int, List[int]] = dict(clusters)
    next_singleton_id = max(clusters.keys(), default=0) + 1000
    for cid, raw in zip(cluster_ids, responses):
        outlier_names = set(parse_llm_list(raw))
        if not outlier_names or outlier_names == {"NONE"}:
            continue
        kept, removed = [], []
        for m in multi[cid]:
            (removed if parent_names[m] in outlier_names else kept).append(m)
        if removed:
            log.info(f"Group {cid}: removed outlier(s) {[parent_names[m] for m in removed]}")
            cleaned[cid] = kept
            for m in removed:
                cleaned[next_singleton_id] = [m]
                next_singleton_id += 1
    return cleaned


_GRANDPARENT_NAME_PROMPT = """\
You are organizing a {organizer_role} enterprise knowledge base.
Analyze the following group of top-level categories, drawn from different \
document sets, and their descriptions to find the common ground:

{context}

Provide a broad, overarching Cross-Domain Category Name (max 4 words) that \
accurately encompasses ALL of these categories.
IMPORTANT RULES:
- Use descriptive, functional terms (e.g., 'Financial Operations', 'Infrastructure & Security', 'Claims Processing').
- DO NOT use specific company, vendor, or brand names.
- Reply ONLY with the name, no quotes."""

_GRANDPARENT_DESC_PROMPT = """\
You are organizing a {organizer_role} enterprise knowledge base.
Analyze the following group of top-level categories, drawn from different \
document sets, and their descriptions:

{context}

Write a comprehensive 2-sentence description that summarizes how these \
categories connect across document sets. Focus on the shared operational, \
technical, or business goals.
Reply ONLY with the description text, no preamble."""


def build_cross_corpus_relationships(sources: List[Tuple[str, Path]] = SOURCES) -> List[Dict]:
    parents, taxonomy_by_tag = _load_all_parents(sources)
    n_sources = len({p["source_tag"] for p in parents})
    if n_sources < 2:
        log.warning(
            f"Only {n_sources} distinct source(s) loaded — "
            "cross-corpus linking needs at least 2. Add a second (source_tag, path) "
            "entry to SOURCES once a second corpus's taxonomy exists. Nothing to do."
        )
        return []

    parent_names = {p["idx"]: p["name"] for p in parents}
    log.info(f"Loaded {len(parents)} parents across {len({p['source_tag'] for p in parents})} source(s)")

    embs = _embed_parents(parents)
    graph = _build_cross_source_graph(parents, embs)
    log.info(f"Cross-source graph: {graph.number_of_edges()} edges (threshold={CROSS_CORPUS_THRESHOLD})")

    partition = community_louvain.best_partition(graph) if graph.number_of_edges() > 0 \
        else {i: i for i in parent_names}
    clusters: Dict[int, List[int]] = {}
    for node_id, comm_id in partition.items():
        clusters.setdefault(comm_id, []).append(node_id)

    multi_before = {c: m for c, m in clusters.items() if len(m) > 1}
    log.info(f"Louvain: {len(clusters)} group(s), {len(multi_before)} multi-parent")

    # Only groups with members from >1 source_tag are genuinely "cross-corpus"
    # — a multi-member group that Louvain formed entirely within one source
    # (possible if same-source parents both bridge through a different-source
    # neighbor) isn't what this file is for.
    cross_only = {
        cid: members for cid, members in multi_before.items()
        if len({parents[m]["source_tag"] for m in members}) > 1
    }
    log.info(f"{len(cross_only)}/{len(multi_before)} multi-parent group(s) are genuinely cross-source")

    analyst_role, _ = _get_domain_profile(next(iter(taxonomy_by_tag.values())))
    cross_only = _audit_outliers(cross_only, parent_names, analyst_role)
    # Deliberately NOT running parent_relationship.py's base-name-dedup filter
    # here. That filter drops clusters where every member reduces to the same
    # base name after stripping nested.py's "(N)" disambiguator — correct
    # in-corpus (it's nested.py's own numbering artifact there), but WRONG
    # cross-corpus: two parents in SEPARATE taxonomies sharing the exact same
    # name is potentially the single strongest legitimate link (same concept
    # independently named the same way in both doc types), not a collision.
    # The outlier audit above already judges genuine thematic connection from
    # the actual descriptions — trust that instead of a blind string match.
    cross_only = {cid: members for cid, members in cross_only.items() if len(members) > 1}

    jobs = []
    for cid, members in cross_only.items():
        context = "\n".join(
            f"- {parent_names[m]} [{parents[m]['source_tag']}]: "
            f"{taxonomy_by_tag[parents[m]['source_tag']]['taxonomy'][parents[m]['local_idx']].get('parent_category_description','')}"
            for m in members
        )
        jobs.append({"cid": cid, "members": members, "context": context})

    log.info(f"Generating {len(jobs)} cross-corpus group name(s) + description(s)...")
    name_prompts = [_GRANDPARENT_NAME_PROMPT.format(organizer_role=analyst_role, context=j["context"]) for j in jobs]
    desc_prompts = [_GRANDPARENT_DESC_PROMPT.format(organizer_role=analyst_role, context=j["context"]) for j in jobs]
    name_responses = llm_client.generate_batch(name_prompts, max_tokens=15, desc="Cross-corpus naming") if jobs else []
    desc_responses = llm_client.generate_batch(desc_prompts, max_tokens=60, desc="Cross-corpus describing") if jobs else []

    results = []
    seen_names: Dict[str, int] = {}
    for job, n_resp, d_resp in zip(jobs, name_responses, desc_responses):
        gp_name = n_resp.replace('"', '').replace('\n', '').strip()
        if len(gp_name) > 50 or not gp_name:
            gp_name = "Related Cross-Domain Systems"
        if gp_name in seen_names:
            seen_names[gp_name] += 1
            gp_name = f"{gp_name} ({seen_names[gp_name]})"
        else:
            seen_names[gp_name] = 1

        gp_desc = d_resp.replace('"', '').replace('\n', ' ').strip()
        member_parents = [
            taxonomy_by_tag[parents[m]["source_tag"]]["taxonomy"][parents[m]["local_idx"]]
            for m in job["members"]
        ]
        source_tags = sorted({parents[m]["source_tag"] for m in job["members"]})

        results.append({
            "parent_category_name":        gp_name,
            "parent_category_description": gp_desc,
            "source_type_keys":            source_tags,
            "member_parents":              member_parents,
        })

    return results


def main():
    results = build_cross_corpus_relationships()
    if not results:
        return

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote {len(results)} cross-corpus group(s) → {OUT_PATH}")
    for r in results:
        member_names = [m["parent_category_name"] for m in r["member_parents"]]
        print(f"\n--- {r['parent_category_name']} ({' × '.join(r['source_type_keys'])}) ---")
        print(r["parent_category_description"])
        print(f"Members: {member_names}")


if __name__ == "__main__":
    main()
