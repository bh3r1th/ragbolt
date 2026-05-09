from pathlib import Path
import json

from ragbolt.trace.emitter import TraceEvent


def load_trace(trace_path: Path) -> list[TraceEvent]:
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")
    events = json.loads(trace_path.read_text(encoding="utf-8"))
    if not isinstance(events, list) or len(events) == 0:
        raise ValueError("Trace file is empty or malformed")
    return events


def explain_event(event: TraceEvent) -> list[str]:
    """
    Return human-readable narrative lines for a single trace event.
    No jargon. Plain English. Useful for debugging a specific run.
    """
    lines = []
    lines.append(f"Run ID  : {event.get('run_id', 'UNKNOWN')}")
    lines.append(f"Corpus  : {event.get('corpus_id', 'UNKNOWN')}")
    lines.append(f"Query   : {event.get('query', '')}")
    lines.append(f"Time    : {event.get('timestamp_utc', 'UNKNOWN')}")
    lines.append("")

    score = event.get("raw_top_score", event.get("top_score", 0.0))
    n = event.get("chunks_retrieved", 0)
    ids = event.get("chunk_ids", [])
    lines.append(f"Retrieval: {n} chunk(s) retrieved (top BM25 score: {score:.4f})")
    if ids:
        lines.append(f"  Chunks : {', '.join(ids)}")

    failures = event.get("failure_classes", [])
    repairs = event.get("repair_attempts", 0)
    if not failures:
        lines.append("Failures : none")
    else:
        for f in failures:
            if f == "RETRIEVAL_LOW_CONFIDENCE":
                lines.append("  ✗ Retrieval score too low — repair attempted (top_k expanded)")
            elif f == "GROUNDING_FAILED":
                ratio = event.get("unsupported_ratio", 0.0)
                lines.append(
                    f"  ✗ Grounding failed — {ratio:.0%} of response unsupported by evidence"
                )
                lines.append("    Repair attempted (context reduced to top chunk)")
            elif f == "GENERATION_MALFORMED":
                lines.append("  ✗ Generation returned empty or errored — no repair in v0.2.0")
    lines.append(f"Repairs  : {repairs} attempt(s)")

    outcome = event.get("outcome", "UNKNOWN")
    outcome_text = {
        "ACCEPTED": "✓ Response accepted — fully grounded, no repairs needed.",
        "REPAIRED_ACCEPTED": "✓ Response accepted after repair — grounding confirmed.",
        "ABSTAINED": "✗ Abstained — retrieval could not find confident evidence.",
        "FAILED": "✗ Failed — could not produce a grounded response.",
    }.get(outcome, f"Unknown outcome: {outcome}")
    lines.append("")
    lines.append(f"Outcome  : {outcome} — {outcome_text}")

    return lines


def explain_trace(events: list[TraceEvent]) -> list[str]:
    """Explain all events in a trace file, separated by dividers."""
    divider = "─" * 52
    lines = []
    for i, event in enumerate(events):
        if i > 0:
            lines.append("")
            lines.append(divider)
            lines.append("")
        lines.extend(explain_event(event))
    return lines
