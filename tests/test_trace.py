import json
import re

from ragbolt.core.policy import DecisionOutcome, FailureClass
from ragbolt.trace.emitter import TraceEmitter


def test_record_returns_uuid4() -> None:
    emitter = TraceEmitter()
    run_id = emitter.record(
        corpus_id="corpus_a",
        query="what is alpha",
        failure_classes=[FailureClass.RETRIEVAL_LOW_CONFIDENCE],
        repair_attempts=1,
        outcome=DecisionOutcome.ABSTAINED,
        top_score=0.0,
        chunks_retrieved=0,
    )
    assert re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
        run_id,
    )


def test_flush_writes_json(tmp_path) -> None:
    out = tmp_path / "trace.json"
    emitter = TraceEmitter(out)
    emitter.record(
        corpus_id="corpus_b",
        query="beta",
        failure_classes=[],
        repair_attempts=0,
        outcome=DecisionOutcome.ACCEPTED,
        top_score=1.23,
        chunks_retrieved=2,
    )
    emitter.flush()

    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert len(payload) == 1
    expected_keys = {
        "run_id",
        "corpus_id",
        "query",
        "timestamp_utc",
        "failure_classes",
        "repair_attempts",
        "outcome",
        "top_score",
        "chunks_retrieved",
    }
    assert expected_keys <= set(payload[0].keys())


def test_multiple_records(tmp_path) -> None:
    out = tmp_path / "trace_multi.json"
    emitter = TraceEmitter(out)
    emitter.record(
        corpus_id="corpus_c",
        query="q1",
        failure_classes=[],
        repair_attempts=0,
        outcome=DecisionOutcome.ACCEPTED,
        top_score=1.0,
        chunks_retrieved=1,
    )
    emitter.record(
        corpus_id="corpus_c",
        query="q2",
        failure_classes=[FailureClass.GROUNDING_FAILED],
        repair_attempts=1,
        outcome=DecisionOutcome.FAILED,
        top_score=0.4,
        chunks_retrieved=1,
    )
    emitter.flush()

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert len(payload) == 2


def test_flush_appends_existing(tmp_path) -> None:
    out = tmp_path / "trace_append.json"
    seed_event = {
        "run_id": "00000000-0000-4000-8000-000000000000",
        "corpus_id": "seed",
        "query": "seed",
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "failure_classes": [],
        "repair_attempts": 0,
        "outcome": "ACCEPTED",
        "top_score": 1.0,
        "chunks_retrieved": 1,
        "unsupported_ratio": 0.0,
        "chunk_ids": ["c1"],
        "raw_top_score": 1.0,
    }
    out.write_text(json.dumps([seed_event]), encoding="utf-8")

    emitter = TraceEmitter(out)
    emitter.record(
        corpus_id="new",
        query="new",
        failure_classes=[],
        repair_attempts=0,
        outcome=DecisionOutcome.ACCEPTED,
        top_score=2.0,
        chunks_retrieved=1,
    )
    emitter.flush()

    events = json.loads(out.read_text(encoding="utf-8"))
    assert len(events) == 2


def test_trace_includes_new_fields(tmp_path) -> None:
    out = tmp_path / "trace_new_fields.json"
    emitter = TraceEmitter(out)
    emitter.record(
        corpus_id="x",
        query="q",
        failure_classes=[],
        repair_attempts=0,
        outcome=DecisionOutcome.ACCEPTED,
        top_score=1.0,
        chunks_retrieved=2,
        unsupported_ratio=0.3,
        chunk_ids=["c1", "c2"],
        raw_top_score=0.8,
    )
    emitter.flush()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload[0]["unsupported_ratio"] == 0.3
    assert payload[0]["chunk_ids"] == ["c1", "c2"]
    assert payload[0]["raw_top_score"] == 0.8
