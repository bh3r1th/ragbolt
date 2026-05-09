import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

from ragbolt.core.policy import DecisionOutcome, FailureClass


class TraceEvent(TypedDict):
    run_id: str
    corpus_id: str
    query: str
    timestamp_utc: str
    failure_classes: list[str]
    repair_attempts: int
    outcome: str
    top_score: float
    chunks_retrieved: int
    unsupported_ratio: float
    chunk_ids: list[str]
    raw_top_score: float


class TraceEmitter:
    def __init__(self, output_path: Path = Path("rag_trace.json")):
        self.output_path = output_path
        self._events: list[TraceEvent] = []

    def record(
        self,
        *,
        corpus_id: str,
        query: str,
        failure_classes: list[FailureClass],
        repair_attempts: int,
        outcome: DecisionOutcome,
        top_score: float,
        chunks_retrieved: int,
        unsupported_ratio: float = 0.0,
        chunk_ids: list[str] | None = None,
        raw_top_score: float = 0.0,
    ) -> str:
        run_id = str(uuid.uuid4())
        timestamp_utc = datetime.now(timezone.utc).isoformat()
        event: TraceEvent = {
            "run_id": run_id,
            "corpus_id": corpus_id,
            "query": query,
            "timestamp_utc": timestamp_utc,
            "failure_classes": [failure.value for failure in failure_classes],
            "repair_attempts": repair_attempts,
            "outcome": outcome.value,
            "top_score": top_score,
            "chunks_retrieved": chunks_retrieved,
            "unsupported_ratio": unsupported_ratio,
            "chunk_ids": list(chunk_ids) if chunk_ids is not None else [],
            "raw_top_score": raw_top_score,
        }
        self._events.append(event)
        return run_id

    def flush(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        existing: list = []
        if self.output_path.exists():
            try:
                existing = json.loads(self.output_path.read_text(encoding="utf-8"))
                if not isinstance(existing, list):
                    existing = []
            except (json.JSONDecodeError, OSError):
                existing = []
        combined = existing + self._events
        self.output_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
