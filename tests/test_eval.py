import json

from ragbolt.core.policy import DecisionOutcome
from ragbolt.eval.report import build_report, write_report


def _event(outcome: str) -> dict:
    return {
        "run_id": "11111111-1111-4111-8111-111111111111",
        "corpus_id": "c",
        "query": "q",
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "failure_classes": [],
        "repair_attempts": 0,
        "outcome": outcome,
        "top_score": 1.0,
        "chunks_retrieved": 1,
    }


def test_build_report_counts() -> None:
    events = [
        _event(DecisionOutcome.ACCEPTED.value),
        _event(DecisionOutcome.ACCEPTED.value),
        _event(DecisionOutcome.FAILED.value),
    ]
    report = build_report(events)
    assert report.total_cases == 3
    assert report.outcome_distribution[DecisionOutcome.ACCEPTED.value] == 2
    assert report.outcome_distribution[DecisionOutcome.FAILED.value] == 1


def test_all_outcomes_present() -> None:
    events = [_event(DecisionOutcome.ACCEPTED.value)]
    report = build_report(events)
    for outcome in DecisionOutcome:
        assert outcome.value in report.outcome_distribution


def test_write_and_read(tmp_path) -> None:
    events = [
        _event(DecisionOutcome.ACCEPTED.value),
        _event(DecisionOutcome.ABSTAINED.value),
    ]
    report = build_report(events)
    out = tmp_path / "eval_report.json"
    write_report(report, out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["total_cases"] == 2
