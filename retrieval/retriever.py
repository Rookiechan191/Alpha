"""
Two-stage retrieval:
  1. FAISS dense retrieval (embedding cosine similarity) -> top TOP_K_RETRIEVE
  2. Cross-encoder reranking -> top TOP_K_RERANK

WHY a cross-encoder reranker:
- The embedding model scores query and document independently (bi-encoder),
  so similarity is a coarse approximation. A cross-encoder feeds
  (query, document) as a single input and produces a relevance score that's
  typically much more accurate -- at the cost of being too slow to run over
  the whole corpus, hence the two-stage funnel (cheap recall -> expensive
  precision on a small candidate set).

cross-encoder/ms-marco-MiniLM-L-6-v2 is a small, fast, well-established
reranker trained on MS MARCO passage ranking.
"""
import json
import os
from pathlib import Path

import faiss
from sentence_transformers import CrossEncoder, SentenceTransformer

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
TOP_K_RETRIEVE = int(os.environ.get("TOP_K_RETRIEVE", 20))
TOP_K_RERANK = int(os.environ.get("TOP_K_RERANK", 5))


class Retriever:
    def __init__(self, index_dir="data/index"):
        index_dir = Path(index_dir)
        self.index = faiss.read_index(str(index_dir / "index.faiss"))
        with open(index_dir / "chunks.json", encoding="utf-8") as f:
            self.chunks = json.load(f)

        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.reranker = CrossEncoder(RERANKER_MODEL)

    def retrieve(self, query: str, top_k_retrieve=None, top_k_rerank=None):
        top_k_retrieve = top_k_retrieve or TOP_K_RETRIEVE
        top_k_rerank = top_k_rerank or TOP_K_RERANK

        # bge models recommend a query instruction prefix for retrieval tasks
        query_emb = self.embedder.encode(
            [f"Represent this sentence for searching relevant passages: {query}"],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        scores, ids = self.index.search(query_emb, top_k_retrieve)
        candidates = []
        for idx, sim in zip(ids[0], scores[0]):
            if idx == -1:
                continue
            chunk = dict(self.chunks[idx])
            chunk["retrieval_score"] = float(sim)
            candidates.append(chunk)

        if not candidates:
            return []

        # Rerank with cross-encoder
        pairs = [[query, c["text"]] for c in candidates]
        rerank_scores = self.reranker.predict(pairs)
        for c, s in zip(candidates, rerank_scores):
            c["rerank_score"] = float(s)

        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
        return candidates[:top_k_rerank]
