import json

import pytest

from ragbolt.core.batch import load_queries, run_batch
from ragbolt.core.generator import StubGenerationProvider
from ragbolt.core.orchestrator import RepairOrchestrator
from ragbolt.core.policy import Chunk, Corpus
from ragbolt.trace.emitter import TraceEmitter
from ragbolt.verify.stub import StubEGAVerifier


@pytest.fixture
def stub_corpus() -> Corpus:
    return Corpus(
        corpus_id="b",
        chunks=[
            Chunk(chunk_id="c1", text="rag retrieval grounding evidence", source="a.txt"),
            Chunk(chunk_id="c2", text="retrieval augmented generation language", source="b.txt"),
            Chunk(chunk_id="c3", text="grounding evidence overlap supported", source="c.txt"),
            Chunk(chunk_id="c4", text="repair attempts policy bounded run", source="d.txt"),
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


def test_batch_all_accepted(tmp_path, stub_corpus, stub_config) -> None:
    orch = RepairOrchestrator(
        stub_corpus,
        stub_config,
        StubGenerationProvider(),
        StubEGAVerifier(stub_config),
    )
    emitter = TraceEmitter(tmp_path / "batch_trace.json")
    queries = [
        "retrieval grounding evidence",
        "retrieval augmented generation",
        "grounding evidence overlap",
    ]
    result = run_batch(queries, orch, emitter, stub_corpus.corpus_id)
    assert result.total == 3
    assert result.outcome_counts["ACCEPTED"] >= 1


def test_batch_never_raises_on_bad_query(tmp_path, stub_corpus, stub_config) -> None:
    orch = RepairOrchestrator(
        stub_corpus,
        stub_config,
        StubGenerationProvider(),
        StubEGAVerifier(stub_config),
    )
    emitter = TraceEmitter(tmp_path / "batch_trace.json")
    queries = ["", "retrieval grounding evidence"]
    result = run_batch(queries, orch, emitter, stub_corpus.corpus_id)
    assert result.total == 2


def test_load_queries_txt(tmp_path) -> None:
    path = tmp_path / "queries.txt"
    path.write_text(
        "first query\n"
        "\n"
        "# this is a comment\n"
        "second query\n"
        "third query\n",
        encoding="utf-8",
    )
    queries = load_queries(path)
    assert len(queries) == 3


def test_load_queries_jsonl(tmp_path) -> None:
    path = tmp_path / "queries.jsonl"
    path.write_text(
        json.dumps({"query": "first query"}) + "\n"
        + json.dumps({"query": "second query"}) + "\n",
        encoding="utf-8",
    )
    queries = load_queries(path)
    assert len(queries) == 2
