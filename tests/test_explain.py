from ragbolt.trace.explain import explain_event


def test_explain_event_accepted() -> None:
    event = {
        "run_id": "11111111-1111-4111-8111-111111111111",
        "corpus_id": "corpus_a",
        "query": "alpha beta",
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "failure_classes": [],
        "repair_attempts": 0,
        "outcome": "ACCEPTED",
        "top_score": 1.2,
        "chunks_retrieved": 2,
        "chunk_ids": ["c1", "c2"],
        "unsupported_ratio": 0.0,
        "raw_top_score": 1.2,
    }
    lines = explain_event(event)
    assert any("ACCEPTED" in line for line in lines)
    assert any("c1" in line for line in lines)


def test_explain_event_grounding_failed() -> None:
    event = {
        "run_id": "22222222-2222-4222-8222-222222222222",
        "corpus_id": "corpus_b",
        "query": "gamma",
        "timestamp_utc": "2026-01-02T00:00:00+00:00",
        "failure_classes": ["GROUNDING_FAILED"],
        "repair_attempts": 1,
        "outcome": "FAILED",
        "top_score": 0.5,
        "chunks_retrieved": 1,
        "chunk_ids": ["c1"],
        "unsupported_ratio": 0.6,
        "raw_top_score": 0.5,
    }
    lines = explain_event(event)
    assert any("60%" in line for line in lines)
    assert any("Repair" in line for line in lines)


def test_explain_handles_missing_fields() -> None:
    event = {
        "run_id": "x",
        "corpus_id": "c",
        "query": "q",
        "timestamp_utc": "t",
        "outcome": "ACCEPTED",
    }
    lines = explain_event(event)
    assert len(lines) > 0
