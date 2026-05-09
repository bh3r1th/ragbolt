# ragbolt
Failure-aware RAG repair layer for Python.

## What it does
ragbolt runs a bounded failure-handling loop around retrieval, generation, and grounding checks. It classifies retrieval low-confidence, malformed generation output, and grounding failures. Applies at most one repair per failure class per run (max 2 total). RETRIEVAL_LOW_CONFIDENCE and GROUNDING_FAILED are repaired; GENERATION_MALFORMED fails fast in v0.1.0. Grounding decisions are checked with EGA (Evidence-Gated Generation) using a verifier interface and a lexical stub implementation. Each run emits an auditable `rag_trace.json` event stream with failure classes, attempt counts, and outcome.

## What it is not
- Not a RAG framework
- Not an eval dashboard
- Not an agent system
- Not a grounding verifier

## Install
```bash
pip install ragbolt          # core (BM25, stub EGA, CLI)
```
Optional extras (FAISS, sentence-transformers) planned for future releases.

## Quickstart
1. Minimal corpus JSON:

```json
[
  {
    "chunk_id": "c1",
    "text": "The Eiffel Tower is in Paris.",
    "source": "facts.txt"
  },
  {
    "chunk_id": "c2",
    "text": "Paris is the capital of France.",
    "source": "facts.txt"
  },
  {
    "chunk_id": "c3",
    "text": "The Seine runs through Paris.",
    "source": "facts.txt"
  }
]
```

2. Run CLI:

```bash
ragbolt run corpus.json "your query" --output trace.json
```

Expected output:

```text
Outcome: ACCEPTED  run_id: <uuid>  trace: trace.json
```

## Eval
```bash
ragbolt eval report.json
```
Prints outcome distribution from an existing eval_report.json.

To generate a report from a trace file:
```python
from pathlib import Path
from ragbolt.eval.report import load_and_build_report
load_and_build_report(Path("rag_trace.json"), Path("eval_report.json"))
```

## Configuration
```yaml
bm25_min_score: 0.30
overlap_min_jaccard: 0.15
unsupported_ratio_threshold: 0.25
top_k: 5
top_k_max: 10
context_reduction_mode: chunk
```
Copy to `config.yaml` and pass via `--config`.

## Failure classes and outcomes
| Class | Trigger |
| --- | --- |
| RETRIEVAL_LOW_CONFIDENCE | BM25 top score < bm25_min_score |
| GENERATION_MALFORMED | Empty or error from provider — fails fast, no repair in v0.1.0 |
| GROUNDING_FAILED | EGA unsupported_ratio >= threshold |

| Outcome | Meaning |
| --- | --- |
| ACCEPTED | No failures detected |
| REPAIRED_ACCEPTED | Repair applied, final EGA passed |
| ABSTAINED | Retrieval could not be repaired |
| FAILED | Generation or grounding could not be repaired |

## Trace output
```json
[
  {
    "run_id": "8c3fc4b8-0c73-4b6b-8c9f-0b6b2bb6d4b7",
    "corpus_id": "corpus",
    "query": "your query",
    "timestamp_utc": "2026-05-09T18:30:45.123456+00:00",
    "failure_classes": [],
    "repair_attempts": 0,
    "outcome": "ACCEPTED",
    "top_score": 1.2345,
    "chunks_retrieved": 3
  }
]
```

## Project layout
```text
ragbolt/
  __init__.py
  cli/
  core/
    generator.py
    orchestrator.py
  eval/
    report.py
  trace/
  verify/
    stub.py
tests/
  __init__.py
  test_bm25.py
  test_eval.py
  test_orchestrator.py
  test_policy.py
  test_trace.py
```

## License
MIT
