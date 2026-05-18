# ragbolt
Failure-aware RAG repair layer for Python.

## What it does
ragbolt runs a bounded failure-handling loop around retrieval, generation, and grounding checks. It classifies retrieval low-confidence, malformed generation output, and grounding failures. Applies at most one repair per failure class per run (max 2 total). RETRIEVAL_LOW_CONFIDENCE and GROUNDING_FAILED are repaired; GENERATION_MALFORMED fails fast in v0.2.0. Grounding decisions are checked with EGA (Evidence-Gated Generation) using a verifier interface and both stub and production verifier implementations. Each run emits an auditable `rag_trace.json` event stream with failure classes, attempt counts, and outcome.

## What it is not
- Not a RAG framework
- Not an eval dashboard
- Not an agent system
- Not a grounding verifier

## Install
```bash
pip install ragbolt              # BM25, stub EGA, CLI
pip install ragbolt[full]        # + FAISS hybrid, NLI verifier
```

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

Phase 3 options:

```bash
# Use Anthropic provider with production EGA verifier
ragbolt run corpus.json "query" --provider anthropic --verifier production

# Use hybrid retrieval (requires ragbolt[full])
ragbolt run corpus.json "query" --retriever hybrid
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

# Generation providers
anthropic_model: claude-sonnet-4-20250514
openai_model: gpt-4o-mini
max_tokens: 1024

# Hybrid retrieval
embedding_model: sentence-transformers/all-MiniLM-L6-v2
rrf_k: 60

# Production EGA verifier
nli_model: cross-encoder/nli-deberta-v3-small
nli_batch_size: 8
```
Copy to `config.yaml` and pass via `--config`.

## Failure classes and outcomes
| Class | Trigger |
| --- | --- |
| RETRIEVAL_LOW_CONFIDENCE | BM25 top score < bm25_min_score |
| GENERATION_MALFORMED | Empty or error from provider — fails fast, no repair in v0.2.0 |
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

## Post-retrieval sanity checks\r\nragbolt 0.6.0 adds deterministic sanity checks on retrieved chunks before generation.\r\n\r\nfrom ragbolt.sanity import sanity_check, repair\r\n\r\nreport = sanity_check(chunks)          # pure, stateless\r\nresult = repair(chunks, report)        # safe bounded repairs only\r\n\r\nThree detectors ship in v1:\r\n\r\n| Detector | Severity | Default action |\r\n| --- | --- | --- |\r\n| unicode_normalization_corruption | low | auto-normalize to NFC |\r\n| ocr_hard_corruption | medium | annotate risk only |\r\n| semantic_orphan_reference | medium | annotate; expand if neighbors provided |\r\n\r\nChunks accept plain strings, dicts with a text field, or RagboltChunk models.\r\nNeighbor expansion requires explicit opt-in — ragbolt never fetches context silently.\r\n\r\n## Post-retrieval sanity checks
ragbolt 0.6.0 adds deterministic sanity checks on retrieved chunks before generation.

from ragbolt.sanity import sanity_check, repair

report = sanity_check(chunks)          # pure, stateless
result = repair(chunks, report)        # safe bounded repairs only

Three detectors ship in v1:

| Detector | Severity | Default action |
| --- | --- | --- |
| unicode_normalization_corruption | low | auto-normalize to NFC |
| ocr_hard_corruption | medium | annotate risk only |
| semantic_orphan_reference | medium | annotate; expand if neighbors provided |

Chunks accept plain strings, dicts with a text field, or RagboltChunk models.
Neighbor expansion requires explicit opt-in — ragbolt never fetches context silently.

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
  sanity/
    __init__.py
    check.py
    repair.py
    models.py
    _adapter.py
    detectors/
      unicode_normalization.py
      ocr_hard_corruption.py
      semantic_orphan.py
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
