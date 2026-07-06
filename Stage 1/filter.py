"""
filter_cross_version.py
───────────────────────
Filters the enterprise_nested_topics.json file to ONLY keep topics 
that appear in two or more source documents (PDFs).

This creates a lightweight "Delta" file perfect for feeding into an LLM 
to analyze version-to-version contradictions and evolutions.
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

# Use your actual paths here
INPUT_JSON_PATH = "/home/vulcan/Documents/Projects/Research Maker/Stage 1/data/output/enterprise_nested_topics.json"
OUTPUT_JSON_PATH = "/home/vulcan/Documents/Projects/Research Maker/Stage 1/data/output/cross_version_topics_only.json"

def has_multiple_sources(topic: dict) -> bool:
    """Safely checks if a topic has 2 or more source docs."""
    docs = topic.get("source_docs", [])
    
    # Handle if source_docs is a list: ["doc1.pdf", "doc2.pdf"]
    if isinstance(docs, list):
        return len(docs) >= 2
        
    # Handle if source_docs is a pipe-separated string: "doc1.pdf | doc2.pdf"
    elif isinstance(docs, str):
        parsed_docs = [d.strip() for d in docs.split("|") if d.strip()]
        return len(parsed_docs) >= 2
        
    return False

def filter_json():    
    with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    original_topic_count = 0
    kept_topic_count = 0

    new_taxonomy = []

    for parent in data.get("taxonomy", []):
        new_sub_categories = []
        
        for sub in parent.get("sub_categories", []):
            original_topics = sub.get("topics", [])
            original_topic_count += len(original_topics)
            
            # Keep only topics with >= 2 source documents
            filtered_topics = [t for t in original_topics if has_multiple_sources(t)]
            
            # If the sub-category still has topics left, keep it
            if filtered_topics:
                sub["topics"] = filtered_topics
                new_sub_categories.append(sub)
                kept_topic_count += len(filtered_topics)

        # If the parent category still has sub-categories left, keep it
        if new_sub_categories:
            parent["sub_categories"] = new_sub_categories
            new_taxonomy.append(parent)

    # Overwrite taxonomy with the filtered data
    data["taxonomy"] = new_taxonomy

    # Save to a NEW file so we don't destroy the original enterprise data
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    filter_json()