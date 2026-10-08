"""Hybrid retrieval: BM25 + vectors fused with Reciprocal Rank Fusion, then rerank."""
import json
import re
import time

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from .config import (CHUNKS_PATH, EMB_PATH, EMBED_MODEL, RERANK_MODEL,
                     TOP_K_FIRST, TOP_K_FINAL)


def tok(s):
    return re.findall(r"\w+", s.lower())


class Retriever:
    def __init__(self, use_reranker=True):
        self.chunks = [json.loads(l) for l in open(CHUNKS_PATH, encoding="utf-8")]
        self.emb = np.load(EMB_PATH)
        self.bm25 = BM25Okapi([tok(c["text"]) for c in self.chunks])
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.reranker = CrossEncoder(RERANK_MODEL) if use_reranker else None

    def _allowed(self, filters):
        if not filters:
            return None
        ok = set()
        for i, c in enumerate(self.chunks):
            good = True
            for k, v in filters.items():
                if k == "year_from":
                    good &= bool(c.get("date")) and str(c["date"])[:4] >= str(v)
                elif k == "year_to":
                    good &= bool(c.get("date")) and str(c["date"])[:4] <= str(v)
                else:
                    good &= str(c.get(k, "")).lower() == str(v).lower()
            if good:
                ok.add(i)
        return ok

    def _rank(self, scores, allowed, k):
        order = np.argsort(-scores)
        if allowed is not None:
            order = [i for i in order if i in allowed]
        return list(order[:k])

    def search(self, query, mode="hybrid", k=TOP_K_FIRST, filters=None):
        """mode: bm25 | vector | hybrid. Returns list of chunk indices."""
        allowed = self._allowed(filters)
        bm = self._rank(self.bm25.get_scores(tok(query)), allowed, k) if mode != "vector" else []
        if mode != "bm25":
            q = self.embedder.encode([query], normalize_embeddings=True)[0]
            vec = self._rank(self.emb @ q, allowed, k)
        else:
            vec = []
        if mode == "bm25":
            return bm
        if mode == "vector":
            return vec
        fused = {}
        for ranking in (bm, vec):  # RRF, constant 60
            for r, i in enumerate(ranking):
                fused[i] = fused.get(i, 0) + 1 / (60 + r + 1)
        return [i for i, _ in sorted(fused.items(), key=lambda x: -x[1])][:k]

    def retrieve(self, query, mode="hybrid", rerank=True, k_first=TOP_K_FIRST,
                 k_final=TOP_K_FINAL, filters=None):
        t0 = time.perf_counter()
        idx = self.search(query, mode, k_first, filters)
        scores = [None] * len(idx)
        if rerank and self.reranker and idx:
            s = self.reranker.predict([(query, self.chunks[i]["text"]) for i in idx])
            order = np.argsort(-s)
            idx = [idx[j] for j in order]
            scores = [float(s[j]) for j in order]
        idx, scores = idx[:k_final], scores[:k_final]
        return {
            "chunks": [self.chunks[i] for i in idx],
            "scores": scores,
            "latency_s": time.perf_counter() - t0,
        }
