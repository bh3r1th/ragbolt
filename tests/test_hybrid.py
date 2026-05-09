from ragbolt.core.hybrid import HybridRunner
from ragbolt.core.policy import Chunk, Corpus


def _hybrid_test_corpus() -> Corpus:
    return Corpus(
        chunks=[
            Chunk(chunk_id="c1", text="retrieval ranking bm25", source="a.txt"),
            Chunk(chunk_id="c2", text="language model generation", source="b.txt"),
            Chunk(chunk_id="c3", text="grounding evidence verification", source="c.txt"),
        ],
        corpus_id="test",
    )


def _hybrid_test_config() -> dict:
    return {
        "bm25_min_score": 0.0,
        "top_k": 5,
        "top_k_max": 10,
        "rrf_k": 60,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    }


def test_hybrid_bm25_fallback() -> None:
    corpus = _hybrid_test_corpus()
    config = _hybrid_test_config()
    runner = HybridRunner(corpus, config)
    chunks, score = runner.retrieve("retrieval ranking")
    assert len(chunks) >= 1
    assert score > 0.0


def test_hybrid_returns_chunks_in_order() -> None:
    corpus = _hybrid_test_corpus()
    config = _hybrid_test_config()
    runner = HybridRunner(corpus, config)
    chunks, _ = runner.retrieve("retrieval ranking")
    assert chunks[0].chunk_id == "c1"
