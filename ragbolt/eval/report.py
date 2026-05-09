import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from ragbolt.core.policy import DecisionOutcome
from ragbolt.trace.emitter import TraceEvent


@dataclass
class EvalReport:
    generated_at: str
    total_cases: int
    outcome_distribution: dict[str, int]
    per_case: list[TraceEvent]


def build_report(events: list[TraceEvent]) -> EvalReport:
    distribution = {outcome.value: 0 for outcome in DecisionOutcome}
    for event in events:
        outcome = event.get("outcome")
        if outcome in distribution:
            distribution[outcome] += 1
    return EvalReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        total_cases=len(events),
        outcome_distribution=distribution,
        per_case=events,
    )


def write_report(report: EvalReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)


def load_and_build_report(trace_path: Path, report_path: Path) -> EvalReport:
    if not trace_path.exists():
        raise FileNotFoundError(trace_path)
    events = json.loads(trace_path.read_text(encoding="utf-8"))
    if not events:
        raise ValueError("Empty trace file")
    report = build_report(events)
    write_report(report, report_path)
    return report


def summary_lines(report: EvalReport) -> list[str]:
    return [
        "ragbolt eval report",
        "─────────────────────────────",
        f"Total cases : {report.total_cases}",
        f"Generated at: {report.generated_at}",
        "",
        "Outcome distribution:",
        f"  ACCEPTED          : {report.outcome_distribution.get(DecisionOutcome.ACCEPTED.value, 0)}",
        f"  REPAIRED_ACCEPTED : {report.outcome_distribution.get(DecisionOutcome.REPAIRED_ACCEPTED.value, 0)}",
        f"  ABSTAINED         : {report.outcome_distribution.get(DecisionOutcome.ABSTAINED.value, 0)}",
        f"  FAILED            : {report.outcome_distribution.get(DecisionOutcome.FAILED.value, 0)}",
    ]
