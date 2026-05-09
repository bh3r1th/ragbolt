## Trace Format

Each entry in `rag_trace.json` is a `TraceEvent` object.

| Field | Type | Description |
| --- | --- | --- |
| `run_id` | `str` | UUID for one run execution. |
| `corpus_id` | `str` | Corpus id derived from input filename stem. |
| `query` | `str` | Query string sent to the orchestrator. |
| `timestamp_utc` | `str` | UTC timestamp for the run event (`ISO 8601`). |
| `failure_classes` | `list[str]` | Ordered list of detected failures. |
| `repair_attempts` | `int` | Number of executed repair attempts. |
| `outcome` | `str` | Final decision outcome (`ACCEPTED`, `REPAIRED_ACCEPTED`, `ABSTAINED`, `FAILED`). |
| `top_score` | `float` | Threshold score used for decisioning (BM25 top score). |
| `chunks_retrieved` | `int` | Number of chunks used for generation/verification. |
| `unsupported_ratio` | `float` | Fraction of unsupported response sentences from verifier. |
| `chunk_ids` | `list[str]` | Chunk ids used in the final context. |
| `raw_top_score` | `float` | Raw top retrieval score before any post-processing. |

Example `rag_trace.json`:

```json
[
  {
    "run_id": "8c3fc4b8-0c73-4b6b-8c9f-0b6b2bb6d4b7",
    "corpus_id": "corpus",
    "query": "Where is the Eiffel Tower?",
    "timestamp_utc": "2026-05-09T18:30:45.123456+00:00",
    "failure_classes": [],
    "repair_attempts": 0,
    "outcome": "ACCEPTED",
    "top_score": 1.2345,
    "chunks_retrieved": 3,
    "unsupported_ratio": 0.0,
    "chunk_ids": ["c1", "c2", "c3"],
    "raw_top_score": 1.2345
  }
]
```

The trace file is append-only: multiple runs accumulate in one file.
