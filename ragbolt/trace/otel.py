from pathlib import Path

from ragbolt.trace.emitter import TraceEvent


def _event_ids(event: TraceEvent) -> tuple[int, int]:
    import hashlib
    import string

    raw = str(event.get("run_id", "")).replace("-", "").lower()
    filtered = "".join(ch for ch in raw if ch in string.hexdigits.lower())
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    material = (filtered + digest) if filtered else digest
    trace_hex = material[:32]
    span_hex = material[-16:]
    return int(trace_hex, 16), int(span_hex, 16)


def _start_time_nano(event: TraceEvent) -> int:
    from datetime import datetime, timezone

    ts = str(event.get("timestamp_utc", "")).strip()
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    except Exception:
        dt = datetime.now(timezone.utc)
    return int(dt.timestamp() * 1_000_000_000)


def export_to_otel(
    events: list[TraceEvent],
    service_name: str = "ragbolt",
    endpoint: str = "http://localhost:4318/v1/traces",
) -> tuple[int, list[str]]:
    """
    Export trace events to an OpenTelemetry collector via OTLP/HTTP.
    Returns (success_count, errors).

    Requires: pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http

    For each TraceEvent, create an OTLP span:
      trace_id: derived from run_id UUID (strip hyphens, take first 32 hex chars)
      span_id:  derived from run_id UUID (take last 16 hex chars)
      name: f"ragbolt.run.{event['outcome'].lower()}"
      start_time_unix_nano: parse timestamp_utc → unix nanoseconds
      end_time_unix_nano: start + 1_000_000 (1ms placeholder)
      attributes:
        ragbolt.corpus_id: str
        ragbolt.query: str
        ragbolt.outcome: str
        ragbolt.repair_attempts: int
        ragbolt.top_score: float
        ragbolt.chunks_retrieved: int
        ragbolt.unsupported_ratio: float (if present)
        ragbolt.failure_classes: comma-joined str

    Use OTLPSpanExporter with endpoint.
    If opentelemetry packages not installed: return (0, ["opentelemetry-sdk not installed"])
    If export fails: return (0, [str(error)])
    On success: return (len(events), [])
    """
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.id_generator import IdGenerator
    except ImportError:
        return (0, ["opentelemetry-sdk not installed"])

    try:
        trace_ids: list[int] = []
        span_ids: list[int] = []
        for event in events:
            tid, sid = _event_ids(event)
            trace_ids.append(tid)
            span_ids.append(sid)

        class _QueuedIdGenerator(IdGenerator):
            def __init__(self, trace_values: list[int], span_values: list[int]):
                self._trace_values = trace_values
                self._span_values = span_values
                self._trace_idx = 0
                self._span_idx = 0

            def generate_span_id(self) -> int:
                if self._span_idx < len(self._span_values):
                    value = self._span_values[self._span_idx]
                    self._span_idx += 1
                    return value
                import random

                return random.getrandbits(64)

            def generate_trace_id(self) -> int:
                if self._trace_idx < len(self._trace_values):
                    value = self._trace_values[self._trace_idx]
                    self._trace_idx += 1
                    return value
                import random

                return random.getrandbits(128)

        id_generator = _QueuedIdGenerator(trace_ids, span_ids)
        provider = TracerProvider(
            resource=Resource.create({"service.name": service_name}),
            id_generator=id_generator,
        )
        exporter = OTLPSpanExporter(endpoint=endpoint)
        processor = SimpleSpanProcessor(exporter)
        provider.add_span_processor(processor)
        tracer = provider.get_tracer("ragbolt.trace.otel")

        for event in events:
            outcome = str(event.get("outcome", "unknown"))
            span_name = f"ragbolt.run.{outcome.lower()}"
            start_nano = _start_time_nano(event)
            attributes = {
                "ragbolt.corpus_id": str(event.get("corpus_id", "")),
                "ragbolt.query": str(event.get("query", "")),
                "ragbolt.outcome": outcome,
                "ragbolt.repair_attempts": int(event.get("repair_attempts", 0)),
                "ragbolt.top_score": float(event.get("top_score", 0.0)),
                "ragbolt.chunks_retrieved": int(event.get("chunks_retrieved", 0)),
                "ragbolt.failure_classes": ",".join(event.get("failure_classes", [])),
            }
            if "unsupported_ratio" in event:
                attributes["ragbolt.unsupported_ratio"] = float(
                    event.get("unsupported_ratio", 0.0)
                )
            span = tracer.start_span(
                span_name,
                start_time=start_nano,
                attributes=attributes,
            )
            span.end(end_time=start_nano + 1_000_000)

        provider.force_flush()
        provider.shutdown()
        return (len(events), [])
    except Exception as error:
        return (0, [str(error)])


def load_and_export(
    trace_path: Path,
    endpoint: str = "http://localhost:4318/v1/traces",
    service_name: str = "ragbolt",
) -> tuple[int, list[str]]:
    """Load trace file and export all events to OTEL collector."""
    import json

    if not trace_path.exists():
        return (0, [f"Trace not found: {trace_path}"])
    events = json.loads(trace_path.read_text(encoding="utf-8"))
    return export_to_otel(events, service_name, endpoint)
