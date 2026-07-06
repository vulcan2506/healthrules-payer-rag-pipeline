import pandas as pd
import json
import networkx as nx
import community.community_louvain as community_louvain
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import ast
import numpy as np

import context_profiler
import llm_client
import config

# ==========================================
# CONFIGURATION
# ==========================================
# Read from config.py (not hardcoded) so a future separate pipeline run
# scoped to a different document type — its own PDF folder, its own
# config.OUTPUT_DIR — gets its own topic_registry.csv/enterprise_nested_
# topics.json without touching this file. Cross-corpus linking wants
# separate taxonomies to bridge between (see cross_corpus_relationship.py),
# not one merged clustering pass over multiple doc types — so this file
# intentionally still clusters everything it's given as ONE corpus.
CSV_PATH = config.REGISTRY_PATH
OUTPUT_JSON_PATH = config.NESTED_OUTPUT_PATH

MACRO_THRESHOLD = 0.50
MICRO_THRESHOLD = 0.60

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def parse_llm_list(text):
    try:
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if match:
            return ast.literal_eval(match.group(0))
    except:
        pass
    return []


# ==========================================
# MAIN PIPELINE
# ==========================================
def run_enterprise_pipeline():
    print("1. Loading CSV Data...")
    df = pd.read_csv(CSV_PATH)
    df['description'] = df['description'].fillna("")
    df["summarized_description"] = df["summarized_description"].fillna("")
    df['keywords'] = df['keywords'].fillna("")
    df['chunk_ids'] = df['chunk_ids'].fillna("")
    df['source_docs'] = df['source_docs'].fillna("")
    df['qna'] = df['qna'].fillna("[]") if 'qna' in df.columns else "[]"
    df['grounded_summary'] = df['grounded_summary'].fillna("") if 'grounded_summary' in df.columns else ""

    df['combined_text'] = df['master_label'] + ". " + df['summarized_description'] + " " + df['keywords']

    # Domain-aware roles from context_profiler (see context_profiler.py) —
    # previously this whole file hardcoded "healthcare IT" phrasing
    # regardless of what document type actually ran through the pipeline.
    # Falls back to the same generic phrasing only if no profile is loaded
    # (e.g. running this script standalone, outside main.py's process).
    _ref_doc = next(
        (d.strip() for docs in df['source_docs'] for d in str(docs).split('|') if d.strip()),
        "",
    )
    _profile = context_profiler.get_profile(_ref_doc) if _ref_doc else None
    AUDITOR_ROLE   = context_profiler.get_role(_ref_doc, "analyst") if _ref_doc else "an expert healthcare IT and technology systems auditor"
    ORGANIZER_ROLE = _profile.get("domain", "healthcare IT and technology") if _profile else "healthcare IT and technology"
    WRITER_ROLE    = context_profiler.get_role(_ref_doc, "specialist") if _ref_doc else "a healthcare IT documentation specialist"

    print("2. Generating Semantic Embeddings (For Initial Graph Clustering)...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    embeddings = embedder.encode(df['combined_text'].tolist(), show_progress_bar=True)
    sim_matrix = cosine_similarity(embeddings)

    print("3. Pass 1: Building Macro-Graph...")
    G_macro = nx.Graph()
    for idx, row in df.iterrows():
        c_ids = [c.strip() for c in str(row['chunk_ids']).split('|') if c.strip()]
        s_docs = [s.strip() for s in str(row['source_docs']).split('|') if s.strip()]
        
        G_macro.add_node(
            idx,
            label=row['master_label'],
            desc=row['summarized_description'],
            desc_=row['description'],
            keywords=row['keywords'],
            chunk_ids=c_ids,
            source_docs=s_docs,
            qna=row['qna'],
            grounded_summary=row['grounded_summary'],
        )
        
    for i in range(len(sim_matrix)):
        for j in range(i + 1, len(sim_matrix)):
            if sim_matrix[i][j] > MACRO_THRESHOLD:
                G_macro.add_edge(i, j, weight=sim_matrix[i][j])

    macro_partition = community_louvain.best_partition(G_macro)
    
    macro_clusters = {}
    for node_id, comm_id in macro_partition.items():
        if comm_id not in macro_clusters: macro_clusters[comm_id] = []
        macro_clusters[comm_id].append(node_id)


    print("\n4. LLM client ready (OpenRouter)...")

    print("\n5. Pass 2: Structuring Micro-Clusters (Graph Analysis)...")
    
    work_items = []  
    macro_orphans = {m_id: [] for m_id in macro_clusters.keys()}

    for macro_id, node_indices in macro_clusters.items():
        G_micro = nx.Graph()
        for idx in node_indices:
            G_micro.add_node(idx, **G_macro.nodes[idx])
            
        for i in range(len(node_indices)):
            for j in range(i + 1, len(node_indices)):
                u, v = node_indices[i], node_indices[j]
                if sim_matrix[u][v] > MICRO_THRESHOLD:
                    G_micro.add_edge(u, v, weight=sim_matrix[u][v])
                    
        micro_partition = community_louvain.best_partition(G_micro)
        
        micro_clusters_dict = {}
        for n_id, c_id in micro_partition.items():
            if c_id not in micro_clusters_dict: micro_clusters_dict[c_id] = []
            micro_clusters_dict[c_id].append(G_micro.nodes[n_id])

        for micro_id, topics in micro_clusters_dict.items():
            if len(topics) <= 1:
                for t in topics:
                    t['confidence_score'] = 0.000 # Orphans score 0 for cohesion
                macro_orphans[macro_id].extend(topics)
            else:
                work_items.append({
                    "macro_id": macro_id,
                    "micro_id": micro_id,
                    "topics": topics,
                    "valid_topics": [],
                    "sub_category_name": "",
                    "sub_category_description": ""
                })

    # ==========================================
    # GLOBAL BATCH 1: OUTLIER DETECTION
    # ==========================================
    print(f"\n--- Batch Executing {len(work_items)} Audit Prompts ---")
    audit_prompts = []
    for item in work_items:
        topic_labels = [t['label'] for t in item['topics']]
        judge_prompt = (
            f"You are {AUDITOR_ROLE}, acting as an expert systems auditor. "
            "I will give you a list of topics that an algorithm grouped together.\n"
            "Your job is to find the 'Odd One Out' (Outliers). An outlier is a topic "
            "that does NOT share the core functional or technical theme of the rest.\n\n"
            "### Examples:\n"
            "Group:\n1. Claims Processing Workflow\n2. Remittance Advice Management\n"
            "3. Network Security Protocols\n"
            "Outliers: [\"Network Security Protocols\"]\n\n"
            "Group:\n1. Patient Registration\n2. Appointment Scheduling\n"
            "3. Clinical Documentation\n"
            "Outliers: [\"NONE\"]\n\n"
            "### Now evaluate this Group:\n"
            + "\n".join([f"- {t}" for t in topic_labels]) + "\n\n"
            "Reply EXACTLY with a Python list of the exact topic names that are outliers. "
            "If all fit perfectly, reply [\"NONE\"]. Do not add any conversational text."
        )
        audit_prompts.append(judge_prompt)

    audit_responses = llm_client.generate_batch(audit_prompts, max_tokens=50, desc="Auditing")

    for item, response in zip(work_items, audit_responses):
        outlier_names = parse_llm_list(response)
        
        for t in item['topics']:
            if t['label'] in outlier_names:
                t['confidence_score'] = 0.000 # Rejected by LLM -> zero cohesion
                macro_orphans[item['macro_id']].append(t)
            else:
                item['valid_topics'].append(t)

    work_items = [item for item in work_items if len(item['valid_topics']) > 0]


    # ==========================================
    # GLOBAL BATCH 2 & 3: NAMING AND DESCRIPTIONS
    # ==========================================
    print(f"\n--- Batch Executing {len(work_items)} Naming & Description Prompts ---")
    name_prompts = []
    desc_prompts = []

    for item in work_items:
        titles_str = ", ".join([t['label'] for t in item['valid_topics']])
        
        name_prompt = (
            f"You are organizing a {ORGANIZER_ROLE} knowledge base.\n"
            f"Group these topics under a single, professional Sub-Category Name (max 4 words) "
            f"that describes their shared functional domain:\n{titles_str}\n"
            f"Reply ONLY with the name, no quotes."
        )
        name_prompts.append(name_prompt)

        desc_prompt = (
            f"You are {WRITER_ROLE}.\n"
            f"Write a single, comprehensive 2-sentence summary describing the core "
            f"functional or technical theme that connects all these topics together:\n"
            f"{titles_str}\n"
            f"Reply ONLY with the summary text."
        )
        desc_prompts.append(desc_prompt)

    name_responses = llm_client.generate_batch(name_prompts, max_tokens=15, desc="Naming")
    desc_responses = llm_client.generate_batch(desc_prompts, max_tokens=60, desc="Describing")

    for item, n_resp, d_resp in zip(work_items, name_responses, desc_responses):
        sub_name = n_resp.replace('"', '').replace('\n', '').strip()
        sub_desc = d_resp.replace('"', '').replace('\n', ' ').strip()
        
        item['sub_category_name'] = sub_name if len(sub_name) <= 50 else "Healthcare System Components"
        item['sub_category_description'] = sub_desc


    # ==========================================
    # GLOBAL BATCH 4: LLM-BASED CONFIDENCE SCORING
    # ==========================================
    print("\n--- Batch Executing LLM Confidence Scoring ---")
    
    score_prompts = []
    score_tracking = []

    for item in work_items:
        sub_name = item['sub_category_name']
        sub_desc = item['sub_category_description']
        
        if len(item['valid_topics']) <= 1:
            for t in item['valid_topics']:
                t['confidence_score'] = 0.000
            continue
            
        for t in item['valid_topics']:
            topic_label = t['label']
            topic_desc = t['desc']
            
            score_prompt = (
                f"You are a strict data auditor. Evaluate how well a specific Topic fits into a Parent Category.\n\n"
                f"Parent Category: {sub_name}\n"
                f"Parent Description: {sub_desc}\n\n"
                f"Topic: {topic_label}\n"
                f"Topic Description: {topic_desc}\n\n"
                f"Rate the conceptual fit from 0.00 (terrible fit) to 1.00 (perfect fit).\n"
                f"- 0.90 to 1.00: Perfect conceptual match.\n"
                f"- 0.70 to 0.89: Good fit, closely related.\n"
                f"- 0.50 to 0.69: Loose fit, tangentially related.\n"
                f"- Below 0.50: Does not belong.\n\n"
                f"Reply ONLY with the decimal number (e.g., 0.92). Do not include any other text."
            )
            score_prompts.append(score_prompt)
            score_tracking.append(t)

    if score_prompts:
        score_responses = llm_client.generate_batch(score_prompts, max_tokens=5, desc="Scoring")

        for t, response in zip(score_tracking, score_responses):
            try:
                match = re.search(r"0\.\d+|1\.00?", response)
                if match:
                    score = float(match.group(0))
                else:
                    score = 0.500 
            except Exception:
                score = 0.500
                
            t['confidence_score'] = score


    # ==========================================
    # GLOBAL BATCH 5 & 6: MACRO CATEGORY NAMING & DESCRIPTION
    # ==========================================
    print(f"\n--- Batch Executing {len(macro_clusters)} Macro Naming & Description Prompts ---")
    
    macro_name_prompts = []
    macro_desc_prompts = []
    macro_keys = list(macro_clusters.keys())
    
    for macro_id in macro_keys:
        # Build a deep context block using both the Sub-Category Names AND their Descriptions
        context_parts = []
        for item in work_items:
            if item['macro_id'] == macro_id:
                context_parts.append(f"- {item['sub_category_name']}: {item['sub_category_description']}")
                
        if not context_parts:
            macro_context = "- General Uncategorized Topics"
        else:
            macro_context = "\n".join(context_parts)
            
        # 1. Macro Naming Prompt
        name_prompt = (
            f"You are organizing a {ORGANIZER_ROLE} enterprise knowledge base.\n"
            f"Analyze the following group of Sub-Categories and their descriptions to find the common ground:\n\n"
            f"{macro_context}\n\n"
            f"Provide a broad, overarching Parent Category Name (max 4 words) that accurately encompasses ALL of these Sub-Categories.\n"
            f"IMPORTANT RULES:\n"
            f"- Use descriptive, functional terms (e.g., 'Financial Operations', 'Infrastructure & Security', 'Claims Processing').\n"
            f"- DO NOT use specific company, vendor, or brand names.\n"
            f"- Reply ONLY with the name, no quotes."
        )
        macro_name_prompts.append(name_prompt)

        # 2. Macro Description Prompt
        desc_prompt = (
            f"You are organizing a {ORGANIZER_ROLE} enterprise knowledge base.\n"
            f"Analyze the following group of Sub-Categories and their descriptions:\n\n"
            f"{macro_context}\n\n"
            f"Write a comprehensive 2-sentence Parent Category Description that summarizes the overarching theme connecting all of these Sub-Categories. "
            f"Focus on the shared operational, technical, or business goals.\n"
            f"Reply ONLY with the description text, no preamble."
        )
        macro_desc_prompts.append(desc_prompt)

    macro_name_responses = llm_client.generate_batch(macro_name_prompts, max_tokens=15, desc="Macro Naming")
    macro_desc_responses = llm_client.generate_batch(macro_desc_prompts, max_tokens=60, desc="Macro Describing")


    # ==========================================
    # JSON RECONSTRUCTION
    # ==========================================
    print("\n--- Assembling Final JSON ---")
    final_taxonomy = []
    seen_macro_names = {}

    for macro_id, n_resp, d_resp in zip(macro_keys, macro_name_responses, macro_desc_responses):
        # Process Parent Category Name
        macro_name = n_resp.replace('"', '').replace('\n', '').strip()
        if len(macro_name) > 50 or not macro_name: 
            macro_name = "Healthcare Technology Systems"
            
        if macro_name in seen_macro_names:
            seen_macro_names[macro_name] += 1
            macro_name = f"{macro_name} ({seen_macro_names[macro_name]})"
        else:
            seen_macro_names[macro_name] = 1
            
        # Process Parent Category Description
        macro_desc = d_resp.replace('"', '').replace('\n', ' ').strip()
        
        clean_sub_categories = []
        
        for item in work_items:
            if item['macro_id'] == macro_id:
                clean_sub_categories.append({
                    "sub_category_name": item['sub_category_name'],
                    "sub_category_description": item['sub_category_description'],
                    "topics": [{
                        "master_label": t['label'],
                        "summarized_description": t['desc'],
                        "description": t['desc_'],
                        "keywords": t['keywords'],
                        "source_docs": t['source_docs'],
                        "chunk_ids": t['chunk_ids'],
                        "qna": json.loads(t.get('qna', '[]') or '[]'),
                        "grounded_summary": t.get('grounded_summary', ''),
                        "relevance_confidence_score": t['confidence_score'],
                    } for t in item['valid_topics']]
                })

        orphans = macro_orphans[macro_id]
        if orphans:
            clean_sub_categories.append({
                "sub_category_name": "Edge Cases & Standalone Topics",
                "sub_category_description": "A collection of standalone topics that did not strongly align with other clusters.",
                "topics": [{
                    "master_label": t['label'],
                    "description": t['desc'],
                    "keywords": t['keywords'],
                    "source_docs": t['source_docs'],
                    "chunk_ids": t['chunk_ids'],
                    "qna": json.loads(t.get('qna', '[]') or '[]'),
                    "grounded_summary": t.get('grounded_summary', ''),
                    "relevance_confidence_score": t['confidence_score'],
                    "is_standalone_outlier": True,
                } for t in orphans]
            })
            
        final_taxonomy.append({
            "parent_category_name": macro_name,
            "parent_category_description": macro_desc, # <--- NEW ADDITION
            "sub_categories": clean_sub_categories
        })


    # ==========================================
    # EXPORT
    # ==========================================
    output_data = {"taxonomy": final_taxonomy}
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)
        
    print(f"\n✅ Enterprise JSON with Parent Descriptions & Confidence Scoring saved to {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    run_enterprise_pipeline()