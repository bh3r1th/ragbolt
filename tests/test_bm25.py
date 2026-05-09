import pytest

from ragbolt.core.bm25 import BM25Runner
from ragbolt.core.policy import Chunk, Corpus


@pytest.fixture
def small_corpus() -> Corpus:
    return Corpus(
        corpus_id="small",
        chunks=[
            Chunk(chunk_id="c1", text="alpha beta gamma", source="a.txt"),
            Chunk(chunk_id="c2", text="alpha delta", source="b.txt"),
            Chunk(chunk_id="c3", text="omega theta", source="c.txt"),
        ],
    )


def test_retrieve_returns_results(small_corpus: Corpus) -> None:
    runner = BM25Runner(
        small_corpus,
        {"top_k": 5, "top_k_max": 10, "bm25_min_score": 0.0},
    )
    chunks, top_score = runner.retrieve("gamma")
    assert len(chunks) >= 1
    assert top_score > 0.0


def test_retrieve_below_threshold(small_corpus: Corpus) -> None:
    runner = BM25Runner(
        small_corpus,
        {"top_k": 5, "top_k_max": 10, "bm25_min_score": 0.30},
    )
    result = runner.retrieve("qzxjv nnnn")
    assert result == ([], 0.0)


def test_retrieve_ordering(small_corpus: Corpus) -> None:
    runner = BM25Runner(
        small_corpus,
        {"top_k": 3, "top_k_max": 10, "bm25_min_score": -1.0},
    )
    chunks, _ = runner.retrieve("alpha beta gamma")
    assert chunks[0].chunk_id == "c1"


def test_top_k_respected() -> None:
    corpus = Corpus(
        corpus_id="ten",
        chunks=[
            Chunk(chunk_id=f"c{i}", text=f"token {i} common", source="bulk.txt")
            for i in range(10)
        ],
    )
    runner = BM25Runner(
        corpus,
        {"top_k": 3, "top_k_max": 10, "bm25_min_score": -1.0},
    )
    chunks, _ = runner.retrieve("common")
    assert len(chunks) <= 3
