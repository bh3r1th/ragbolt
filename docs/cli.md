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

### `ingest`
Ingest text files into a ragbolt corpus JSON.

Arguments:

| Name | Type | Required | Description |
| --- | --- | --- | --- |
| `source` | `PATH` | Yes | File or directory to ingest. Supports `.txt`, `.md`, `.json`. |

Options:

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `--output` | `PATH` | `corpus.json` | Output corpus JSON path. |
| `--chunk-size` | `INT` | `512` | Max words per chunk. |
| `--recursive`, `-r` | `FLAG` | off | Recurse into subdirectories. |

Example:

```bash
ragbolt ingest docs/ --output corpus.json --recursive
```

### `batch`
Run ragbolt on multiple queries from a file.

Arguments:

| Name | Type | Required | Description |
| --- | --- | --- | --- |
| `corpus` | `PATH` | Yes | Path to corpus JSON. |
| `queries` | `PATH` | Yes | Path to `queries.txt` (one per line) or `queries.jsonl` (`{"query": "..."}` per line). |

Options:

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `--config` | `PATH` | `config.yaml` | YAML config path. |
| `--output` | `PATH` | `rag_trace.json` | Trace output path (appended). |
| `--provider` | `TEXT` | `stub` | Generation provider. |
| `--retriever` | `TEXT` | `bm25` | Retriever. |
| `--verifier` | `TEXT` | `stub` | Verifier. |

Example:

```bash
ragbolt batch corpus.json queries.txt --output rag_trace.json
```

Example output:

```text
Batch complete: 25 queries
  ACCEPTED            : 18
  REPAIRED_ACCEPTED   : 4
  ABSTAINED           : 2
  FAILED              : 1
Trace written to: rag_trace.json
```

### `serve`
Start ragbolt as a REST API server. Requires `pip install ragbolt[serve]`.

Arguments:

| Name | Type | Required | Description |
| --- | --- | --- | --- |
| `corpus` | `PATH` | Yes | Path to corpus JSON. |

Options:

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `--config` | `PATH` | `config.yaml` | YAML config path. |
| `--provider` | `TEXT` | `stub` | Generation provider. |
| `--retriever` | `TEXT` | `bm25` | Retriever. |
| `--verifier` | `TEXT` | `stub` | Verifier. |
| `--host` | `TEXT` | `127.0.0.1` | Bind host. |
| `--port` | `INT` | `8000` | Bind port. |

Endpoints: `POST /query`, `GET /health`, `GET /trace`.

Example:

```bash
ragbolt serve corpus.json --host 0.0.0.0 --port 8000
```

### `calibrate`
Recommend threshold adjustments based on trace history.

Arguments:

| Name | Type | Required | Description |
| --- | --- | --- | --- |
| `trace` | `PATH` | Yes | Path to `rag_trace.json` (≥10 events). |

Options:

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `--config` | `PATH` | `config.yaml` | Current config path. |
| `--target-abstain-rate` | `FLOAT` | `0.05` | Target abstain rate. |
| `--target-fail-rate` | `FLOAT` | `0.10` | Target fail rate. |
| `--apply` | `FLAG` | off | Write recommended values to `config.yaml`. |

Example:

```bash
ragbolt calibrate rag_trace.json --apply
```

### `export`
Export trace to OpenTelemetry collector. Requires `pip install ragbolt[otel]`.

Arguments:

| Name | Type | Required | Description |
| --- | --- | --- | --- |
| `trace` | `PATH` | Yes | Path to `rag_trace.json`. |

Options:

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `--endpoint` | `TEXT` | `http://localhost:4318/v1/traces` | OTLP HTTP endpoint. |
| `--service-name` | `TEXT` | `ragbolt` | OTEL service name. |

Example:

```bash
ragbolt export rag_trace.json --endpoint http://otel-collector:4318/v1/traces
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
