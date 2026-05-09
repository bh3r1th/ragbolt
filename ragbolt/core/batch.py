from dataclasses import dataclass
from pathlib import Path
import json

from ragbolt.core.orchestrator import OrchestratorResult, RepairOrchestrator
from ragbolt.core.policy import Corpus, DecisionOutcome
from ragbolt.trace.emitter import TraceEmitter


@dataclass
class BatchResult:
    total: int
    results: list[tuple[str, OrchestratorResult]]
    outcome_counts: dict[str, int]


def run_batch(
    queries: list[str],
    orchestrator: RepairOrchestrator,
    emitter: TraceEmitter,
    corpus_id: str,
) -> BatchResult:
    """
    Run orchestrator.run(query) for each query in queries.
    Record each result via emitter.record().
    Return BatchResult with outcome_counts for all four outcomes.
    Never raise — if a single query fails, record outcome=FAILED and continue.
    """
    outcome_counts = {o.value: 0 for o in DecisionOutcome}
    results: list[tuple[str, OrchestratorResult]] = []

    for query in queries:
        try:
            result = orchestrator.run(query)
        except Exception:
            result = OrchestratorResult(
                outcome=DecisionOutcome.FAILED,
                response=None,
                failure_classes=[],
                repair_attempts=0,
                top_score=0.0,
                chunks_retrieved=0,
                unsupported_ratio=0.0,
                chunk_ids=[],
                raw_top_score=0.0,
            )

        results.append((query, result))
        outcome_counts[result.outcome.value] += 1

        try:
            emitter.record(
                corpus_id=corpus_id,
                query=query,
                failure_classes=result.failure_classes,
                repair_attempts=result.repair_attempts,
                outcome=result.outcome,
                top_score=result.top_score,
                chunks_retrieved=result.chunks_retrieved,
                unsupported_ratio=result.unsupported_ratio,
                chunk_ids=result.chunk_ids,
                raw_top_score=result.raw_top_score,
            )
        except Exception:
            outcome_counts[result.outcome.value] -= 1
            outcome_counts[DecisionOutcome.FAILED.value] += 1
            failed_result = OrchestratorResult(
                outcome=DecisionOutcome.FAILED,
                response=None,
                failure_classes=[],
                repair_attempts=0,
                top_score=0.0,
                chunks_retrieved=0,
                unsupported_ratio=0.0,
                chunk_ids=[],
                raw_top_score=0.0,
            )
            results[-1] = (query, failed_result)

    return BatchResult(total=len(queries), results=results, outcome_counts=outcome_counts)


def load_queries(path: Path) -> list[str]:
    """
    Load queries from file. Supports two formats:
    - .jsonl: one JSON object per line with "query" key
    - .txt: one query per line, skip blank lines and lines starting with #
    Raises FileNotFoundError if path missing.
    Raises ValueError if format unrecognized or no valid queries found.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    ext = path.suffix.lower()
    queries: list[str] = []

    if ext == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL line: {e}") from e
            if not isinstance(obj, dict) or "query" not in obj:
                raise ValueError("JSONL lines must be objects with 'query' key")
            query = str(obj["query"]).strip()
            if query:
                queries.append(query)
    elif ext == ".txt":
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            queries.append(text)
    else:
        raise ValueError("Unsupported query file format; use .jsonl or .txt")

    if not queries:
        raise ValueError("No valid queries found")
    return queries
