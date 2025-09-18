# -*- coding: utf-8 -*-
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name="BAAI/bge-reranker-large"):
        self.model = CrossEncoder(model_name)

    def score(self, query: str, docs):
        pairs = [[query, d] for d in docs]
        return self.model.predict(pairs).tolist()
