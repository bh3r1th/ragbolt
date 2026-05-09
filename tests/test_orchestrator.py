import pytest

from ragbolt.core.generator import StubGenerationProvider
from ragbolt.core.orchestrator import RepairOrchestrator
from ragbolt.core.policy import Chunk, Corpus, DecisionOutcome, FailureClass, RepairPolicy
from ragbolt.verify.stub import StubEGAVerifier


@pytest.fixture
def stub_corpus() -> Corpus:
    return Corpus(
        corpus_id="stub",
        chunks=[
            Chunk(chunk_id="c1", text="rag repair retrieval grounding evidence", source="a.txt"),
            Chunk(chunk_id="c2", text="retrieval confidence and rerank behavior", source="b.txt"),
            Chunk(chunk_id="c3", text="grounding checks evidence overlap tokens", source="c.txt"),
            Chunk(chunk_id="c4", text="generation output quality for query response", source="d.txt"),
            Chunk(chunk_id="c5", text="repair policy ordered attempts bounded", source="e.txt"),
        ],
    )


@pytest.fixture
def stub_config() -> dict:
    return {
        "bm25_min_score": 0.30,
        "overlap_min_jaccard": 0.15,
        "unsupported_ratio_threshold": 0.25,
        "top_k": 5,
        "top_k_max": 10,
        "context_reduction_mode": "chunk",
    }


def test_accepted(stub_corpus: Corpus, stub_config: dict) -> None:
    orchestrator = RepairOrchestrator(
        stub_corpus,
        stub_config,
        StubGenerationProvider(),
        StubEGAVerifier(stub_config),
    )
    result = orchestrator.run("retrieval grounding evidence")
    assert result.outcome == DecisionOutcome.ACCEPTED
    assert result.response is not None


def test_retrieval_failure(stub_corpus: Corpus, stub_config: dict) -> None:
    orchestrator = RepairOrchestrator(
        stub_corpus,
        stub_config,
        StubGenerationProvider(),
        StubEGAVerifier(stub_config),
    )
    result = orchestrator.run("qzxjv nnnn zzzqqq")
    assert result.outcome == DecisionOutcome.ABSTAINED
    assert result.response is None


def test_grounding_repair(
    stub_corpus: Corpus,
    stub_config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unrelated_generate(self, query: str, context: str) -> str:
        return "unrelated content xyz"

    monkeypatch.setattr(StubGenerationProvider, "generate", unrelated_generate)
    orchestrator = RepairOrchestrator(
        stub_corpus,
        stub_config,
        StubGenerationProvider(),
        StubEGAVerifier(stub_config),
    )
    result = orchestrator.run("retrieval grounding evidence")
    assert result.outcome == DecisionOutcome.FAILED
    assert FailureClass.GROUNDING_FAILED in result.failure_classes
    assert result.repair_attempts == 1


def test_repair_attempts_bounded(
    stub_corpus: Corpus,
    stub_config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = {"n": 0}
    real_chunks = stub_corpus.chunks[:2]

    def staged_retrieve(self, query: str):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return [], 0.0
        return real_chunks, 1.0

    def always_fail(self, response: str, chunks: list) -> tuple[bool, float]:
        return False, 0.9

    monkeypatch.setattr("ragbolt.core.orchestrator.BM25Runner.retrieve", staged_retrieve)
    monkeypatch.setattr(StubEGAVerifier, "verify", always_fail)
    orchestrator = RepairOrchestrator(
        stub_corpus,
        stub_config,
        StubGenerationProvider(),
        StubEGAVerifier(stub_config),
    )
    result = orchestrator.run("retrieval grounding evidence")
    assert result.repair_attempts == RepairPolicy.MAX_ATTEMPTS


def test_retrieval_repair_expands_top_k() -> None:
    config = {
        "bm25_min_score": 99.0,
        "overlap_min_jaccard": 0.15,
        "unsupported_ratio_threshold": 0.25,
        "top_k": 5,
        "top_k_max": 10,
        "context_reduction_mode": "chunk",
    }
    corpus = Corpus(
        corpus_id="generic",
        chunks=[
            Chunk(chunk_id=f"c{i}", text=f"generic text content number {i}", source="x.txt")
            for i in range(5)
        ],
    )
    orchestrator = RepairOrchestrator(
        corpus,
        config,
        StubGenerationProvider(),
        StubEGAVerifier(config),
    )
    result = orchestrator.run("text content")
    assert result.repair_attempts == 1
    assert result.outcome in (DecisionOutcome.ABSTAINED, DecisionOutcome.REPAIRED_ACCEPTED)


def test_failure_classes_no_duplicates() -> None:
    config = {
        "bm25_min_score": 99.0,
        "overlap_min_jaccard": 0.15,
        "unsupported_ratio_threshold": 0.25,
        "top_k": 5,
        "top_k_max": 10,
        "context_reduction_mode": "chunk",
    }
    corpus = Corpus(
        corpus_id="generic",
        chunks=[
            Chunk(chunk_id=f"c{i}", text=f"generic text content number {i}", source="x.txt")
            for i in range(5)
        ],
    )
    orchestrator = RepairOrchestrator(
        corpus,
        config,
        StubGenerationProvider(),
        StubEGAVerifier(config),
    )
    result = orchestrator.run("text content")
    assert result.failure_classes.count(FailureClass.RETRIEVAL_LOW_CONFIDENCE) == 1
