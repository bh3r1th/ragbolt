import json

import pytest

from ragbolt.core.calibrate import calibrate_from_trace


def _event(outcome: str, top_score: float = 0.0, unsupported_ratio: float = 0.0,
           failure_classes: list | None = None) -> dict:
    return {
        "run_id": "11111111-1111-4111-8111-111111111111",
        "corpus_id": "c",
        "query": "q",
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "failure_classes": failure_classes or [],
        "repair_attempts": 0,
        "outcome": outcome,
        "top_score": top_score,
        "chunks_retrieved": 0,
        "unsupported_ratio": unsupported_ratio,
        "chunk_ids": [],
        "raw_top_score": top_score,
    }


def test_calibrate_insufficient_data(tmp_path) -> None:
    path = tmp_path / "trace.json"
    events = [_event("ACCEPTED") for _ in range(5)]
    path.write_text(json.dumps(events), encoding="utf-8")
    with pytest.raises(ValueError):
        calibrate_from_trace(path)


def test_calibrate_high_abstain_rate(tmp_path) -> None:
    path = tmp_path / "trace.json"
    events: list = []
    for _ in range(10):
        events.append(_event("ABSTAINED", top_score=0.0))
    accepted_scores = [0.10, 0.15, 0.20, 0.25, 0.30]
    for s in accepted_scores:
        events.append(_event("ACCEPTED", top_score=s))
    path.write_text(json.dumps(events), encoding="utf-8")
    result = calibrate_from_trace(path, target_abstain_rate=0.05)
    assert result.recommended_bm25_min_score < 0.30
    assert result.abstain_rate > 0.05


def test_calibrate_result_clamped(tmp_path) -> None:
    path = tmp_path / "trace.json"
    events: list = []
    for _ in range(10):
        events.append(_event("ABSTAINED", top_score=0.0))
    for s in [0.10, 0.15, 0.20, 0.25, 0.30]:
        events.append(_event("ACCEPTED", top_score=s))
    path.write_text(json.dumps(events), encoding="utf-8")
    result = calibrate_from_trace(path)
    assert 0.05 <= result.recommended_bm25_min_score <= 0.80
    assert 0.10 <= result.recommended_unsupported_ratio_threshold <= 0.40
