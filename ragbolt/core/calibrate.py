from dataclasses import dataclass
from pathlib import Path
import json
import statistics

from ragbolt.core.policy import DecisionOutcome, FailureClass
from ragbolt.trace.emitter import TraceEvent


@dataclass
class CalibrationResult:
    recommended_bm25_min_score: float
    recommended_overlap_min_jaccard: float
    recommended_unsupported_ratio_threshold: float
    sample_size: int
    abstain_rate: float
    fail_rate: float
    notes: list[str]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _percentile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile of empty data")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def calibrate_from_trace(
    trace_path: Path,
    target_abstain_rate: float = 0.05,
    target_fail_rate: float = 0.10,
) -> CalibrationResult:
    """
    Analyze a trace file and recommend threshold adjustments.

    Algorithm:
    1. Load trace events from trace_path
    2. Separate events by outcome
    3. BM25 threshold recommendation:
       - Collect top_score values for ABSTAINED events
       - Collect top_score values for ACCEPTED + REPAIRED_ACCEPTED events
       - If abstain_rate > target_abstain_rate:
           recommend lower bm25_min_score (5th percentile of accepted scores)
       - If abstain_rate < target_abstain_rate / 2:
           recommend higher bm25_min_score (median of abstained scores * 0.9)
       - Else: keep current (use median of all top_scores * 0.8 as safe default)
    4. Grounding threshold recommendation:
       - Collect unsupported_ratio values for FAILED events (GROUNDING_FAILED)
       - Collect unsupported_ratio values for ACCEPTED events
       - Recommend threshold = midpoint between max(accepted ratios) and
         min(failed ratios), clamped to [0.10, 0.40]
       - If insufficient data: keep default 0.25
    5. Jaccard threshold: keep default 0.15 (insufficient signal from trace alone)
    6. Add notes explaining each recommendation with supporting evidence counts

    Returns CalibrationResult with all recommended values.
    Raises FileNotFoundError if trace_path missing.
    Raises ValueError if trace has fewer than 10 events (insufficient data).
    """
    if not trace_path.exists():
        raise FileNotFoundError(trace_path)
    events = json.loads(trace_path.read_text(encoding="utf-8"))
    if not isinstance(events, list):
        raise ValueError("Trace file must contain a JSON array")
    if len(events) < 10:
        raise ValueError("Insufficient data: need at least 10 trace events")

    sample_size = len(events)
    abstained = [e for e in events if e.get("outcome") == DecisionOutcome.ABSTAINED.value]
    failed = [e for e in events if e.get("outcome") == DecisionOutcome.FAILED.value]
    accepted = [e for e in events if e.get("outcome") == DecisionOutcome.ACCEPTED.value]
    repaired = [e for e in events if e.get("outcome") == DecisionOutcome.REPAIRED_ACCEPTED.value]

    abstain_rate = len(abstained) / sample_size
    fail_rate = len(failed) / sample_size

    abstained_scores = [float(e.get("top_score", 0.0)) for e in abstained]
    accepted_scores = [float(e.get("top_score", 0.0)) for e in accepted + repaired]
    all_scores = [float(e.get("top_score", 0.0)) for e in events]

    notes: list[str] = []
    notes.append(
        f"Targets used: abstain<={target_abstain_rate:.1%}, fail<={target_fail_rate:.1%}"
    )

    if abstain_rate > target_abstain_rate and accepted_scores:
        bm25 = _percentile(accepted_scores, 0.05)
        notes.append(
            f"Abstain rate is high ({abstain_rate:.1%}); lowered bm25_min_score "
            f"using 5th percentile of {len(accepted_scores)} accepted/repaired scores."
        )
    elif abstain_rate < (target_abstain_rate / 2.0) and abstained_scores:
        bm25 = statistics.median(abstained_scores) * 0.9
        notes.append(
            f"Abstain rate is very low ({abstain_rate:.1%}); raised bm25_min_score "
            f"using median of {len(abstained_scores)} abstained scores * 0.9."
        )
    elif all_scores:
        bm25 = statistics.median(all_scores) * 0.8
        notes.append(
            f"Abstain rate near target; used median of all {len(all_scores)} "
            "top scores * 0.8 as safe default."
        )
    else:
        bm25 = 0.30
        notes.append("No top_score values found; used default bm25_min_score=0.30.")

    bm25 = _clamp(float(bm25), 0.05, 0.80)

    failed_grounding = [
        float(e.get("unsupported_ratio", 0.0))
        for e in failed
        if FailureClass.GROUNDING_FAILED.value in (e.get("failure_classes") or [])
    ]
    accepted_ratios = [float(e.get("unsupported_ratio", 0.0)) for e in accepted]

    if accepted_ratios and failed_grounding:
        midpoint = (max(accepted_ratios) + min(failed_grounding)) / 2.0
        unsupported = _clamp(float(midpoint), 0.10, 0.40)
        notes.append(
            f"Set unsupported_ratio_threshold from accepted({len(accepted_ratios)}) "
            f"and grounding-failed({len(failed_grounding)}) separation midpoint."
        )
    else:
        unsupported = 0.25
        notes.append(
            "Insufficient grounding separation data; kept "
            "unsupported_ratio_threshold at default 0.25."
        )

    overlap = 0.15
    notes.append("Kept overlap_min_jaccard at 0.15 (insufficient direct signal in trace).")

    return CalibrationResult(
        recommended_bm25_min_score=bm25,
        recommended_overlap_min_jaccard=overlap,
        recommended_unsupported_ratio_threshold=unsupported,
        sample_size=sample_size,
        abstain_rate=abstain_rate,
        fail_rate=fail_rate,
        notes=notes,
    )


def format_calibration(result: CalibrationResult) -> list[str]:
    """
    Return human-readable lines:
    "ragbolt threshold calibration"
    "────────────────────────────────────────"
    "Sample size : {n} events"
    "Abstain rate: {rate:.1%}  (target: ≤{target:.1%})"
    "Fail rate   : {rate:.1%}  (target: ≤{target:.1%})"
    ""
    "Recommended config.yaml changes:"
    "  bm25_min_score              : {old} → {new}"
    "  unsupported_ratio_threshold : {old} → {new}"
    "  overlap_min_jaccard         : {old}  (unchanged)"
    ""
    "Notes:"
    "  - {note}" for each note
    """
    target_abstain = 0.05
    target_fail = 0.10
    for note in result.notes:
        if note.startswith("Targets used:"):
            try:
                raw = note.replace("Targets used:", "").strip()
                parts = [p.strip() for p in raw.split(",")]
                for part in parts:
                    if part.startswith("abstain<="):
                        target_abstain = float(part.split("<=")[1].rstrip("%")) / 100.0
                    elif part.startswith("fail<="):
                        target_fail = float(part.split("<=")[1].rstrip("%")) / 100.0
            except Exception:
                pass

    lines = [
        "ragbolt threshold calibration",
        "────────────────────────────────────────",
        f"Sample size : {result.sample_size} events",
        f"Abstain rate: {result.abstain_rate:.1%}  (target: ≤{target_abstain:.1%})",
        f"Fail rate   : {result.fail_rate:.1%}  (target: ≤{target_fail:.1%})",
        "",
        "Recommended config.yaml changes:",
        f"  bm25_min_score              : {0.30:.2f} → {result.recommended_bm25_min_score:.2f}",
        "  unsupported_ratio_threshold : "
        f"{0.25:.2f} → {result.recommended_unsupported_ratio_threshold:.2f}",
        f"  overlap_min_jaccard         : {0.15:.2f}  (unchanged)",
        "",
        "Notes:",
    ]
    for note in result.notes:
        lines.append(f"  - {note}")
    return lines
