"""
quality_filter.py
─────────────────
Drops chunks that have a text_quality_score below a strict threshold (e.g. 70%).
Ensures downstream taxonomy isn't polluted with garbage data.
"""

import logging
from typing import List, Dict

log = logging.getLogger(__name__)

def run_filter(chunks: List[Dict], threshold: float = 0.70) -> List[Dict]:
    initial_count = len(chunks)
    valid_chunks = []
    
    for c in chunks:
        # Default to 1.0 if the score key is completely missing so we don't accidentally drop good chunks
        score = c.get("text_quality_score", 1.0) 
        
        if score >= threshold:
            valid_chunks.append(c)

    dropped_count = initial_count - len(valid_chunks)
    
    log.info(f"[Quality Filter] Dropped {dropped_count} chunks with quality score < {threshold*100}%.")
    log.info(f"[Quality Filter] Remaining high-quality chunks: {len(valid_chunks)}")
    
    return valid_chunks