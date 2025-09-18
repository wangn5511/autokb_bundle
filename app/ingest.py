# -*- coding: utf-8 -*-
import os, re, json, hashlib, pickle, glob
from typing import List, Dict, Any
from dataclasses import dataclass
from loguru import logger
from pathlib import Path
from tqdm import tqdm

# 尝试导入 PyMuPDF；失败则回退到 pdfminer.six（页码将不可用）
HAVE_FITZ = False
try:
    import fitz  # PyMuPDF
    HAVE_FITZ = True
except Exception:
    HAVE_FITZ = False

def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_md(path: str) -> str:
    return read_txt(path)

def read_pdf(path: str) -> List[Dict[str, Any]]:
    """返回每页文本。
    - 使用 PyMuPDF 时按页返回（含页码）
    - 回退 pdfminer 时整份文本放在单页，page=None
    - 扫描件需 OCR（未在此实现）
    """
    if HAVE_FITZ:
        doc = fitz.open(path)
        pages = []
        for i in range(len(doc)):
            text = doc[i].get_text("text")
            pages.append({"page": i + 1, "text": text})
        return pages
    else:
        from pdfminer.high_level import extract_text
        text = extract_text(path) or ""
        return [{"page": None, "text": text}]

def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def chunk_text(text: str, chunk_size=800, overlap=150) -> List[str]:
    text = normalize_ws(text)
    if not text: return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text): break
        start = max(0, end - overlap)
    return chunks

@dataclass
class Chunk:
    text: str
    meta: Dict[str, Any]

def hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def iter_files(root: str) -> List[str]:
    exts = ["*.pdf", "*.md", "*.txt"]
    paths = []
    for ext in exts:
        paths += glob.glob(os.path.join(root, "**", ext), recursive=True)
    return sorted(list(set(paths)))

def build_chunks_for_file(path: str, chunk_size=800, overlap=150) -> List[Chunk]:
    chunks: List[Chunk] = []
    name = os.path.basename(path)
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        pages = read_pdf(path)
        for p in pages:
            ctexts = chunk_text(p["text"], chunk_size, overlap)
            for i, c in enumerate(ctexts):
                chunks.append(Chunk(text=c, meta={"source": name, "page": p["page"], "chunk_idx": i}))
    elif ext in [".md", ".txt"]:
        text = read_txt(path)
        ctexts = chunk_text(text, chunk_size, overlap)
        for i, c in enumerate(ctexts):
            chunks.append(Chunk(text=c, meta={"source": name, "page": None, "chunk_idx": i}))
    return chunks

def load_or_init_state(index_dir: str):
    os.makedirs(index_dir, exist_ok=True)
    meta_path = os.path.join(index_dir, "docmeta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"files": {}}

def save_state(index_dir: str, state: Dict):
    with open(os.path.join(index_dir, "docmeta.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def ingest(root: str, index_dir: str, chunk_size=800, overlap=150, update=False):
    state = load_or_init_state(index_dir)
    changed_files, new_chunks = [], []
    all_paths = iter_files(root)
    logger.info(f"发现 {len(all_paths)} 个文件")
    for p in tqdm(all_paths):
        sha = hash_file(p)
        record = state["files"].get(p)
        if (not record) or (record.get("sha256") != sha) or update:
            cks = build_chunks_for_file(p, chunk_size, overlap)
            new_chunks.extend(cks)
            state["files"][p] = {"sha256": sha, "chunks": len(cks)}
            changed_files.append(p)

    ds_pkl = os.path.join(index_dir, "docstore.pkl")
    with open(ds_pkl, "wb") as f:
        pickle.dump(new_chunks, f)

    save_state(index_dir, state)
    logger.info(f"完成，变更文件数：{len(changed_files)}，chunk 总数：{len(new_chunks)}")
    return changed_files, len(new_chunks)
