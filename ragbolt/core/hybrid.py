from __future__ import annotations

from ragbolt.core.bm25 import BM25Runner
from ragbolt.core.policy import Chunk, Corpus


class HybridRunner:
    """
    BM25 + FAISS dense retrieval with Reciprocal Rank Fusion.
    Requires: pip install faiss-cpu sentence-transformers
    Falls back to BM25-only if FAISS unavailable.
    """

    def __init__(self, corpus: Corpus, config: dict):
        self._config = dict(config)
        self._top_k = int(config["top_k"])
        self._top_k_max = int(config["top_k_max"])
        self._embedding_model = str(
            config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        )
        self._rrf_k = int(config.get("rrf_k", 60))
        self._bm25 = BM25Runner(corpus, config)
        self._chunks = corpus.chunks
        self._index = None
        self._embedder = None
        self._index_build_attempted = False

    def _build_index(self):
        if self._index_build_attempted:
            return
        self._index_build_attempted = True
        try:
            import faiss
            import numpy as np
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self._embedding_model)
            embeddings = self._embedder.encode([c.text for c in self._chunks])
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings = embeddings / norms
            self._index = faiss.IndexFlatIP(embeddings.shape[1])
            self._index.add(embeddings.astype("float32"))
        except Exception:
            self._index = None
            self._embedder = None

    def retrieve(self, query: str) -> tuple[list[Chunk], float]:
        if self._index is None:
            self._build_index()

        bm25_chunks, top_score = self._bm25.retrieve(query)
        bm25_ranks = {chunk.chunk_id: rank for rank, chunk in enumerate(bm25_chunks)}

        if self._index is None or self._embedder is None:
            return bm25_chunks, top_score

        import numpy as np

        top_k = min(self._top_k, self._top_k_max)
        query_vec = self._embedder.encode([query])
        query_norm = np.linalg.norm(query_vec, axis=1, keepdims=True)
        query_norm[query_norm == 0] = 1.0
        query_vec = query_vec / query_norm
        _, indices = self._index.search(query_vec.astype("float32"), top_k)
        dense_ranks = {
            self._chunks[i].chunk_id: rank
            for rank, i in enumerate(indices[0])
            if i >= 0 and i < len(self._chunks)
        }

        all_ids = set(bm25_ranks) | set(dense_ranks)
        rrf_score: dict[str, float] = {}
        fallback_rank = top_k * 2
        for cid in all_ids:
            bm25_r = bm25_ranks.get(cid, fallback_rank)
            dense_r = dense_ranks.get(cid, fallback_rank)
            rrf_score[cid] = (1.0 / (self._rrf_k + bm25_r)) + (
                1.0 / (self._rrf_k + dense_r)
            )

        chunk_by_id = {chunk.chunk_id: chunk for chunk in self._chunks}
        ranked_ids = sorted(rrf_score, key=rrf_score.get, reverse=True)[:top_k]
        fused_chunks = [chunk_by_id[cid] for cid in ranked_ids if cid in chunk_by_id]
        return fused_chunks, top_score
