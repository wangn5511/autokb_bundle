# -*- coding: utf-8 -*-
import os
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from app.ingest import ingest
from app.retriever import AutoKBRetriever
from app.llm import llm_query_rewrite, simple_query_rewrite, generate_answer

load_dotenv()
app = FastAPI(title="Auto-KB API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

R: Optional[AutoKBRetriever] = None

class IngestReq(BaseModel):
    path: str
    update: bool = False
    chunk_size: int = 800
    overlap: int = 150

class QueryReq(BaseModel):
    query: str
    top_k: int = int(os.getenv("TOP_K", 8))
    use_rerank: bool = True
    rewrite: str = "rules"  # "rules" | "llm" | "off"

@app.post("/ingest")
def api_ingest(req: IngestReq):
    changed, n = ingest(req.path, "indexes", req.chunk_size, req.overlap, update=req.update)
    global R
    R = AutoKBRetriever(
        index_dir="indexes",
        embedding_model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
        reranker_model_name=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-large"),
        fusion_dense=float(os.getenv("FUSION_WEIGHT_DENSE", 0.65)),
        fusion_sparse=float(os.getenv("FUSION_WEIGHT_SPARSE", 0.35)),
    )
    return {"changed_files": changed, "chunks": n}

@app.post("/query")
def api_query(req: QueryReq):
    global R
    if R is None:
        R = AutoKBRetriever(
            index_dir="indexes",
            embedding_model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
            reranker_model_name=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-large"),
            fusion_dense=float(os.getenv("FUSION_WEIGHT_DENSE", 0.65)),
            fusion_sparse=float(os.getenv("FUSION_WEIGHT_SPARSE", 0.35)),
        )
    rewrites = [req.query]
    if req.rewrite == "rules":
        rewrites = simple_query_rewrite(req.query)
    elif req.rewrite == "llm":
        rewrites = llm_query_rewrite(req.query)

    pool = []
    for q in rewrites:
        rs = R.search(q, top_k=req.top_k, use_rerank=req.use_rerank)
        pool.extend(rs)
    seen = {}
    for r in pool:
        key = (r.meta.get("source"), r.meta.get("page"), r.meta.get("chunk_idx"))
        if (key not in seen) or (seen[key].score < r.score):
            seen[key] = r
    merged = sorted(seen.values(), key=lambda x: x.score, reverse=True)[:req.top_k]
    contexts = [{"text": m.text, "meta": m.meta, "score": m.score} for m in merged]
    out = generate_answer(contexts, req.query)

    min_conf = float(os.getenv("MIN_CONFIDENCE", 0.35))
    if out["confidence"] < min_conf:
        out["answer"] = f"我不确定能准确回答。建议提供更具体的问题或上传相关文档。当前置信度={out['confidence']:.2f}，阈值={min_conf:.2f}。"
    return out

@app.get("/")
def root():
    return {"ok": True, "msg": "Auto-KB running"}
