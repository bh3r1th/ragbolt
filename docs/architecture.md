## Architecture

`core/` contains the pipeline primitives: corpus validation (`policy.py`), sparse retrieval (`bm25.py`), optional hybrid retrieval (`hybrid.py`), generation providers (`generator.py`, `providers.py`), and the repair loop orchestrator (`orchestrator.py`).

`verify/` defines the grounding contract (`protocol.py`) and verifier implementations: lexical overlap stub (`stub.py`) plus production NLI verifier with fallback behavior (`production.py`).

`trace/` records run-level telemetry (`emitter.py`) and renders run narratives for debugging (`explain.py`).

`eval/` builds aggregate reports from trace events (`report.py`) and provides summary output consumed by the CLI.

`cli/` exposes user-facing commands (`run`, `eval`, `explain`) and wires config loading, provider/retriever/verifier selection, orchestration, and trace/report output.

Repair loop:

```text
Query
  └─ BM25Runner.retrieve()
       ├─ score < threshold → RETRIEVAL repair → re-retrieve
       │    └─ still fails → ABSTAINED
       └─ score ok
            └─ GenerationProvider.generate()
                 ├─ error/empty → GENERATION_MALFORMED → FAILED (no repair)
                 └─ response
                      └─ EGAVerifier.verify()
                           ├─ not grounded → GROUNDING repair → re-verify
                           │    └─ still fails → FAILED
                           └─ grounded → ACCEPTED / REPAIRED_ACCEPTED
```

| Outcome | Meaning |
| --- | --- |
| ACCEPTED | No failures detected |
| REPAIRED_ACCEPTED | Repair applied, final EGA passed |
| ABSTAINED | Retrieval could not be repaired |
| FAILED | Generation or grounding could not be repaired |

| Class | Trigger |
| --- | --- |
| RETRIEVAL_LOW_CONFIDENCE | BM25 top score < bm25_min_score |
| GENERATION_MALFORMED | Empty or error from provider — fails fast, no repair in v0.2.0 |
| GROUNDING_FAILED | EGA unsupported_ratio >= threshold |
