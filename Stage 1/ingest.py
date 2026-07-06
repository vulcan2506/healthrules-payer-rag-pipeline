"""
ingest.py
─────────
Extracts a single PDF using Docling TableFormer, formats tables to JSONL, 
applies structural chunking, and runs LLM cohesion checks.
"""

import re
import os
import math
import json
import logging
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Optional
from pypdf import PdfReader, PdfWriter
from docling.datamodel.pipeline_options import AcceleratorOptions, PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption

from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

import config
import llm_client

log = logging.getLogger(__name__)

# ── Singletons ─────────────────────────────────────────────────────────────────
_embedder:      Optional[SentenceTransformer] = None
_keybert_model: Optional[KeyBERT]             = None

def unload():
    """Free all GPU models held by this module."""
    global _embedder, _keybert_model
    del _embedder, _keybert_model
    _embedder = None
    _keybert_model = None
    _free_vram()


def _free_vram():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        log.info(f"Loading embedder: {config.EMBEDDING_MODEL}")
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")
    return _embedder

def _get_keybert() -> KeyBERT:
    global _keybert_model
    if _keybert_model is None:
        _keybert_model = KeyBERT(model=_get_embedder())
    return _keybert_model

# ── STEP 1: Docling Multiprocessing Extraction (Replaces pymupdf4llm) ──────────

def _docling_worker(chunk_idx: int, start_page: int, end_page: int, pdf_path: str) -> tuple[int, str]:
    """Isolated worker for Docling extraction."""
    try:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        
        reader = PdfReader(pdf_path)
        writer = PdfWriter()
        for page_num in range(start_page, end_page):
            writer.add_page(reader.pages[page_num])
            
        temp_path = Path(f"/tmp/docling_chunk_{os.getpid()}_{chunk_idx}.pdf")
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_path, "wb") as f:
            writer.write(f)
            
        accelerator = AcceleratorOptions(num_threads="16" ,device="cpu")
        pipeline_opts = PdfPipelineOptions(accelerator_options=accelerator)
        pipeline_opts.do_ocr = False 
        pipeline_opts.do_table_structure = True 
        pipeline_opts.table_structure_options.mode = TableFormerMode.ACCURATE
        pipeline_opts.generate_page_images = False 

        converter = DocumentConverter(
            format_options={"pdf": PdfFormatOption(pipeline_options=pipeline_opts)}
        )
        
        result = converter.convert(temp_path)
        markdown_text = result.document.export_to_markdown()
        
        if temp_path.exists():
            temp_path.unlink()
            
        return chunk_idx, markdown_text
        
    except Exception as e:
        log.error(f"Docling worker failed on chunk {chunk_idx}: {e}")
        return chunk_idx, ""

def _extract_pdf_with_docling(pdf_path: Path) -> List[Dict]:
    """
    Orchestrates memory-safe multiprocessing.
    Returns a list of dicts matching the old pymupdf4llm format so existing code works.
    """
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    pages_per_chunk = getattr(config, 'PAGE_BATCH_SIZE', 4)
    workers = getattr(config, 'N_WORKERS', 5)
    total_chunks = math.ceil(total_pages / pages_per_chunk)
    
    tasks = []
    for i in range(total_chunks):
        start_page = i * pages_per_chunk
        end_page = min(start_page + pages_per_chunk, total_pages)
        tasks.append((i, start_page, end_page, str(pdf_path)))
        
    results_dict = {}
    with mp.Pool(processes=workers, maxtasksperchild=1) as pool:
        for idx, text in pool.starmap(_docling_worker, tasks):
            results_dict[idx] = text
            
    # Bridge to old pymupdf4llm format
    md_pages = []
    for i in range(total_chunks):
        if i in results_dict and results_dict[i].strip():
            md_pages.append({"text": results_dict[i]})
            
    return md_pages


# ── Cleaners & Converters (RESTORED EXACTLY AS PROVIDED) ───────────────────────

def _is_toc_page(md_text: str) -> bool:
    lines = md_text.splitlines()
    if not lines:
        return False
    toc_lines = sum(1 for ln in lines if re.search(r"\]\(#page=\d+\)|\.\.\.\.\s*\d+$", ln))
    return (toc_lines / len(lines)) > 0.35

def _is_alphabetical_index(md_text: str) -> bool:
    return bool(re.search(r"^#{1,3}\s+Index\b", md_text.strip(), re.IGNORECASE))

def _clean_text(text: str) -> str:
    text = text.replace("<br>", "\n")
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    return text.strip()

def _md_table_to_jsonl(table_lines: List[str]) -> str:
    if len(table_lines) < 3:
        return "\n".join(table_lines)

    all_jsonl = []
    sep_indices = [i for i, line in enumerate(table_lines) if re.match(r"^\|[\s\-:|]+\|$", line.strip())]
    if not sep_indices:
        sep_indices = [1]
        
    for idx_count, sep_idx in enumerate(sep_indices):
        header_idx = sep_idx - 1
        if header_idx < 0: continue
            
        end_idx = sep_indices[idx_count+1] - 1 if idx_count + 1 < len(sep_indices) else len(table_lines)
        raw_headers = [h.strip() for h in table_lines[header_idx].strip().strip("|").split("|")]
        
        headers, counts = [], {}
        for h in raw_headers:
            h = _clean_text(h)
            if h in counts:
                counts[h] += 1
                headers.append(f"{h}_{counts[h]}")
            else:
                counts[h] = 0
                headers.append(h)
                
        for line in table_lines[sep_idx + 1:end_idx]:
            if not line.strip(): continue
            cols = [c.strip() for c in line.strip().strip("|").split("|")]
            while len(cols) > len(headers):
                headers.append(f"Column_{len(headers)+1}")
            row = {headers[i]: _clean_text(cols[i]) if i < len(cols) else "" for i in range(len(headers))}
            if any(row.values()):
                all_jsonl.append(json.dumps(row))
                
    return "\n".join(all_jsonl)

def _detect_section_break(line: str) -> Optional[str]:
    s = line.strip()
    if 5 < len(s) < 200:
        if s.isupper() and not re.search(r"\d{2,}", s):
            return s
        if (s.startswith("**") and s.endswith("**")) or (s.startswith("__") and s.endswith("__")):
            return s.strip("*_").strip()
    return None

def _chunk_by_structure(md_pages: List[Dict]) -> List[Dict]:
    chunks, current_header = [], "Introduction / Front Matter"
    current_text, current_table = [], []
    in_table = False
    current_pages: set = set()

    def flush_text():
        body = _clean_text("\n".join(current_text))
        if body and not re.match(r"^#+\s*$", body):
            chunks.append({"section_header": current_header, "text": body, "pages": sorted(current_pages)})
        current_text.clear(); current_pages.clear()

    def flush_table():
        if current_table:
            body = _md_table_to_jsonl(current_table)
            raw_md = "\n".join(current_table) 
            if body.strip():
                chunks.append({
                    "section_header": current_header + " (Table)",
                    "text": body, 
                    "_temp_raw_md": raw_md,
                    "pages": sorted(current_pages)
                })
        current_table.clear(); current_pages.clear()

    for page_num, page in enumerate(md_pages, start=1):
        for line in page.get("text", "").splitlines():
            is_table = line.strip().startswith("|") and line.strip().endswith("|")
            hm       = re.match(r"^(#{1,6})\s+(.*)", line.strip())
            sb       = _detect_section_break(line)

            if is_table:
                if not in_table:
                    flush_text(); in_table = True
                current_table.append(line); current_pages.add(page_num)
            else:
                if in_table:
                    flush_table(); in_table = False
                if hm:
                    flush_text(); current_header = hm.group(2).strip()
                    current_pages.add(page_num)
                elif sb:
                    flush_text(); current_header = sb
                    current_pages.add(page_num)
                else:
                    current_text.append(line); current_pages.add(page_num)

    flush_text(); flush_table()
    return chunks

def _extract_keywords_batch(texts: List[str], batch_size: int = 32) -> List[List[str]]:
    kw = _get_keybert()
    all_kw: List[List[str]] = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            results = kw.extract_keywords(
                batch,
                keyphrase_ngram_range=(config.KEYBERT_NGRAM_MIN, config.KEYBERT_NGRAM_MAX),
                stop_words="english", use_mmr=True, diversity=config.KEYBERT_DIVERSITY, top_n=config.KEYBERT_TOP_N,
            )
            if batch and not isinstance(results[0], list):
                results = [results]
            all_kw.extend([[k for k, _ in r] for r in results])
        except Exception:
            all_kw.extend([[] for _ in batch])
    return all_kw

# ── Cohesion Rectifier (RESTORED EXACTLY AS PROVIDED) ──────────────────────────

_RECTIFY_PROMPT = (
    "Does this text abruptly switch to a completely unrelated topic halfway through?\n"
    "If NO — reply exactly: COHESIVE\n"
    "If YES — reply with the first 5-8 words of the sentence where the new topic starts.\n\n"
    "Text: {text}\n\nAnswer:"
)

def _is_jsonl_table(text: str) -> bool:
    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
    return bool(lines) and all(ln.startswith("{") and ln.endswith("}") for ln in lines)

def _rectify_batch(batch: List[Dict]) -> List[Dict]:
    text_idxs = [i for i, c in enumerate(batch) if not _is_jsonl_table(c["text"])]
    result = list(batch)

    if not text_idxs:
        return result

    prompts = [_RECTIFY_PROMPT.format(text=batch[i]["text"][:1500]) for i in text_idxs]
    outputs = llm_client.generate_local_batch(prompts, max_tokens=20, batch_size=16, desc="Cohesion Check", stop=llm_client.STOP_FLAG)

    splits = []
    for idx, response in zip(text_idxs, outputs):
        chunk = batch[idx]

        if "COHESIVE" not in response.upper() and len(response) > 10:
            target = response.replace('"', "").replace("'", "").strip()
            split_pos = chunk["text"].find(target)
            if split_pos > 50:
                p1 = chunk["text"][:split_pos].strip()
                p2 = chunk["text"][split_pos:].strip()
                if p1: result[idx] = {**chunk, "text": p1, "_rectified": True}
                if p2: splits.append({**chunk, "text": p2, "section_header": chunk["section_header"] + " (Cont.)", "_rectified": True})
                continue
        result[idx] = {**chunk, "_rectified": True}

    return result + splits


# ── Main API Function ──────────────────────────────────────────────────────────

def _load_preconverted_md(pdf_path: Path) -> Optional[List[Dict]]:
    """
    Check if a pre-converted markdown file exists for this PDF.
    Naming convention: <stem>_Converted.md  in the same directory.
    Returns md_pages list (same format as Docling output) or None.
    """
    md_path = pdf_path.with_name(pdf_path.stem + "_Converted.md")
    if not md_path.exists():
        return None

    log.info(f"Found pre-converted markdown: {md_path.name} — skipping Docling extraction")
    text = md_path.read_text(encoding="utf-8")

    # Split on markdown H2+ headers to approximate page-level chunks,
    # matching the structure Docling produces.
    sections = re.split(r'(?=^## )', text, flags=re.MULTILINE)
    md_pages = []
    for section in sections:
        section = section.strip()
        if section:
            md_pages.append({"text": section})

    log.info(f"Loaded {len(md_pages)} sections from {md_path.name}")
    return md_pages


def process_single_pdf(pdf_path: Path, global_id: dict) -> List[Dict]:
    """
    Extracts a single PDF, runs keyword extraction, and applies text cohesion.
    Returns a list of chunks in memory.
    """
    log.info(f"Extracting: {pdf_path.name}...")

    temp_md_dir = config.OUTPUT_DIR / "temp_md"
    temp_md_dir.mkdir(parents=True, exist_ok=True)

    md_pages = _load_preconverted_md(pdf_path)
    if md_pages is None:
        md_pages = _extract_pdf_with_docling(pdf_path)

    clean_pages = []
    toc_texts = []
    for page in md_pages:
        if _is_toc_page(page["text"]):
            toc_texts.append(page["text"])
            continue
        if _is_alphabetical_index(page["text"]): break
        clean_pages.append(page)

    if toc_texts:
        import context_profiler
        context_profiler.store_toc(pdf_path.name, "\n\n".join(toc_texts))

    raw_chunks = _chunk_by_structure(clean_pages)
    
    # Ban TOC headers
    valid = []
    for c in raw_chunks:
        hdr = c["section_header"].lower()
        if "contents" in hdr or "table of contents" in hdr or "index" in hdr:
            continue
        if len(c["text"].split()) > 10:
            valid.append(c)

    texts = [c["text"] for c in valid]
    kw_lists = _extract_keywords_batch(texts, batch_size=config.KEYBERT_BATCH_SIZE)

    pdf_chunks = []
    for chunk, kws in zip(valid, kw_lists):
        cid = global_id["id"]
        pages = chunk.get("pages", [])
        
        # In Docling batching, 'pages' represents the batch index. This keeps it stable.
        page_str = (f"{pages[0]}-{pages[-1]}" if len(pages) > 1 else str(pages[0]) if pages else "?")

        md_file_path = None
        # PRESERVES YOUR TEMP MD SAVING FOR RE-RECTIFIER
        if "_temp_raw_md" in chunk:
            md_path = temp_md_dir / f"table_{cid}.md"
            md_path.write_text(chunk["_temp_raw_md"], encoding="utf-8")
            md_file_path = str(md_path)

        pdf_chunks.append({
            "chunk_id": cid,
            "section_header": chunk["section_header"],
            "text": chunk["text"],
            "md_file_path": md_file_path,
            "keywords": kws,
            "source_doc": pdf_path.name,
            "page_range": page_str,
            "master_label": None,
            "description": None,
            "_rectified": False,
        })
        global_id["id"] += 1

    # --- Cohesion Rectify (In-Memory) ---
    if config.USE_LLM_NOISE_FILTER:
        batch_sz = config.RECTIFY_BATCH_SIZE * 4
        
        final_pdf_chunks = []
        for i in range(0, len(pdf_chunks), batch_sz):
            batch = pdf_chunks[i : i + batch_sz]
            final_pdf_chunks.extend(_rectify_batch(batch))
        return final_pdf_chunks
        
    return pdf_chunks