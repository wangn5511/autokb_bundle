# -*- coding: utf-8 -*-
import os, pickle, math
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from loguru import logger
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

@dataclass
class ScoredChunk:
    text: str
    meta: Dict[str, Any]
    score: float
    score_sparse: float = 0.0
    score_dense: float = 0.0

class AutoKBRetriever:
    def __init__(self, index_dir="indexes",
                 embedding_model_name="BAAI/bge-m3",
                 reranker_model_name="BAAI/bge-reranker-large",
                 fusion_dense=0.65, fusion_sparse=0.35, device=None):
        self.index_dir = index_dir
        self.embedder = SentenceTransformer(embedding_model_name, device=device)
        self.reranker_name = reranker_model_name
        self.reranker = None  # 懒加载
        self.fusion_dense = fusion_dense
        self.fusion_sparse = fusion_sparse
        self.docstore = self._load_docstore()
        self.bm25 = None
        self.faiss_index = None
        self.embeddings = None
        self._build_sparse()
        self._build_dense()

    def _load_docstore(self):
        p = os.path.join(self.index_dir, "docstore.pkl")
        if not os.path.exists(p):
            raise FileNotFoundError("docstore.pkl 不存在，请先运行摄取构建。")
        with open(p, "rb") as f:
            return pickle.load(f)

    def _tokenize(self, text: str) -> List[str]:
        import re
        tokens = re.findall(r"[A-Za-z]+|\S", text.lower())
        return tokens

    def _build_sparse(self):
        corpus = [c.text for c in self.docstore]
        tokenized = [self._tokenize(t) for t in corpus]
        self.bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 就绪，文档数={len(corpus)}")

    def _build_dense(self):
        corpus = [c.text for c in self.docstore]
        self.embeddings = self.embedder.encode(corpus, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
        d = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(d)
        self.faiss_index.add(self.embeddings.astype(np.float32))
        logger.info(f"FAISS 向量就绪，维度={d}, 条数={self.embeddings.shape[0]}")

    def _ensure_reranker(self):
        if self.reranker is None and self.reranker_name:
            self.reranker = CrossEncoder(self.reranker_name)

    def search(self, query: str, top_k=8, use_rerank=True) -> List[ScoredChunk]:
        qtok = self._tokenize(query)
        scores_sparse = self.bm25.get_scores(qtok)
        qv = self.embedder.encode([query], normalize_embeddings=True)[0].astype(np.float32)
        D, I = self.faiss_index.search(np.array([qv]), k=min(top_k*3, len(self.docstore)))
        scores_dense = D[0]

        idx_sparse = np.argsort(scores_sparse)[::-1][:top_k*3]
        idx_dense = I[0]
        candidate = list(set(idx_sparse.tolist() + idx_dense.tolist()))

        def norm(v):
            if len(v)==0: return v
            v = np.array(v, dtype=float)
            if np.max(v)==np.min(v): 
                return np.zeros_like(v)
            return (v - np.min(v)) / (np.max(v) - np.min(v))

        ss = scores_sparse[candidate]
        sd = np.array([scores_dense[np.where(idx_dense==i)[0][0]] if i in idx_dense else 0.0 for i in candidate])
        ns, nd = norm(ss), norm(sd)

        fused = self.fusion_dense*nd + self.fusion_sparse*ns
        order = np.argsort(fused)[::-1][:top_k]
        results = []
        for j in order:
            idx = candidate[j]
            ck = self.docstore[idx]
            results.append(ScoredChunk(
                text=ck.text, meta=ck.meta,
                score=float(fused[j]),
                score_sparse=float(ns[j]), score_dense=float(nd[j])
            ))

        if use_rerank and len(results) > 1 and self.reranker_name:
            self._ensure_reranker()
            pairs = [[query, r.text] for r in results]
            rerank_scores = self.reranker.predict(pairs)
            rns = norm(rerank_scores)
            alpha = 0.6
            final_scores = alpha*np.array([r.score for r in results]) + (1-alpha)*rns
            order2 = np.argsort(final_scores)[::-1]
            results = [ScoredChunk(
                text=results[i].text, meta=results[i].meta,
                score=float(final_scores[i]),
                score_sparse=results[i].score_sparse,
                score_dense=results[i].score_dense
            ) for i in order2]

        return results
