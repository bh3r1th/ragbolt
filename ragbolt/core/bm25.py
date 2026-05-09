from __future__ import annotations

from rank_bm25 import BM25Okapi

from ragbolt.core.policy import Chunk, Corpus


class BM25Runner:
    def __init__(self, corpus: Corpus, config: dict):
        self._chunks = corpus.chunks
        self._top_k = int(config["top_k"])
        self._top_k_max = int(config["top_k_max"])
        self._bm25_min_score = float(config["bm25_min_score"])
        tokenized_corpus = [chunk.text.lower().split() for chunk in self._chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str) -> tuple[list[Chunk], float]:
        query_tokens = query.lower().split()
        scores = self._bm25.get_scores(query_tokens).tolist()
        ranked = sorted(
            ((index, float(score)) for index, score in enumerate(scores)),
            key=lambda item: item[1],
            reverse=True,
        )
        if not ranked:
            return [], 0.0
        top_score = ranked[0][1]
        k = min(self._top_k, self._top_k_max)
        selected: list[Chunk] = []
        for index, score in ranked[:k]:
            if score >= self._bm25_min_score:
                selected.append(self._chunks[index])
        if not selected:
            return [], top_score
        return selected, top_score
