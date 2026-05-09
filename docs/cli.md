## Commands

### `run`
Run ragbolt on a single query against a corpus.

Arguments:

| Name | Type | Required | Description |
| --- | --- | --- | --- |
| `corpus` | `PATH` | Yes | Path to corpus JSON file. |
| `query` | `TEXT` | Yes | Query string. |

Options:

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `--config` | `PATH` | `config.yaml` | YAML config path. |
| `--output` | `PATH` | `rag_trace.json` | Trace output path. |
| `--provider` | `TEXT` | `stub` | Generation provider: `stub`, `anthropic`, or `openai`. |
| `--retriever` | `TEXT` | `bm25` | Retriever: `bm25` or `hybrid`. |
| `--verifier` | `TEXT` | `stub` | Verifier: `stub` or `production`. |

Example:

```bash
ragbolt run corpus.json "Where is the Eiffel Tower?" --output rag_trace.json
```

Example output:

```text
Outcome: ACCEPTED  run_id: <uuid>  trace: rag_trace.json
```

### `eval`
Generate and print an eval report from a trace file.

Arguments:

| Name | Type | Required | Description |
| --- | --- | --- | --- |
| `trace` | `PATH` | Yes | Path to `rag_trace.json`. |

Options:

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `--report` | `PATH` | `eval_report.json` | Path to write eval report JSON. |

Example:

```bash
ragbolt eval rag_trace.json --report eval_report.json
```

Example output:

```text
ragbolt eval report
─────────────────────────────
Total cases : 1
Generated at: 2026-05-09T18:31:00.000000+00:00

Outcome distribution:
  ACCEPTED          : 1
  REPAIRED_ACCEPTED : 0
  ABSTAINED         : 0
  FAILED            : 0

Report written to: eval_report.json
```

### `explain`
Print a human-readable explanation of a trace file.

Arguments:

| Name | Type | Required | Description |
| --- | --- | --- | --- |
| `trace` | `PATH` | Yes | Path to `rag_trace.json`. |

Options:

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `--run-id` | `TEXT` | `None` | Explain only a specific run id from the trace. |

Example:

```bash
ragbolt explain rag_trace.json --run-id 8c3fc4b8-0c73-4b6b-8c9f-0b6b2bb6d4b7
```

Example output:

```text
Run ID  : 8c3fc4b8-0c73-4b6b-8c9f-0b6b2bb6d4b7
Corpus  : corpus
Query   : Where is the Eiffel Tower?
Time    : 2026-05-09T18:30:45.123456+00:00

Retrieval: 3 chunk(s) retrieved (top BM25 score: 1.2345)
  Chunks : c1, c2, c3
Failures : none
Repairs  : 0 attempt(s)

Outcome  : ✓ Response accepted — fully grounded, no repairs needed.
```

### `ragbolt --help`
```text
                                                                               
 Usage: ragbolt [OPTIONS] COMMAND [ARGS]...                                    
                                                                               
 Failure-aware RAG repair layer.                                               
                                                                               
+- Options -------------------------------------------------------------------+
| --install-completion          Install completion for the current shell.     |
| --show-completion             Show completion for the current shell, to     |
|                               copy it or customize the installation.        |
| --help                        Show this message and exit.                   |
+-----------------------------------------------------------------------------+
+- Commands ------------------------------------------------------------------+
| run       Run ragbolt on a single query against a corpus.                   |
| eval      Generate and print eval report from a trace file.                 |
| explain   Print human-readable explanation of a trace file.                 |
+-----------------------------------------------------------------------------+
```
