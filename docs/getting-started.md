## Quickstart
Create a small corpus:

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

Run one query:

```bash
ragbolt run corpus.json "Where is the Eiffel Tower?" --output rag_trace.json
```

Expected output:

```text
Outcome: ACCEPTED  run_id: <uuid>  trace: rag_trace.json
```

Explain the trace:

```bash
ragbolt explain rag_trace.json
```

Expected output:

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

Generate an eval report from trace:

```bash
ragbolt eval rag_trace.json --report eval_report.json
```

Expected output:

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

## Ingesting documents
ragbolt can build a corpus directly from `.txt`, `.md`, or `.json` files. Pass a single file or a directory:

```bash
ragbolt ingest docs/ --output corpus.json --recursive
```

Behavior:

- `.txt` / `.md`: paragraphs split on double newline. Long paragraphs split further on sentences. Each chunk gets `chunk_id = "{file_stem}_{NNNN}"` and `source = "{file.name}"`.
- `.json`: must be an array of objects with at least `text` and `chunk_id`. Validated via the `Chunk` model.
- Directories: all supported files merged, duplicates auto-disambiguated, `corpus_id = directory.stem`.

## Batch queries
Run many queries against the same corpus in a single CLI call. Two formats supported:

`queries.txt` — one per line, blanks and `#`-comment lines skipped:

```text
Where is the Eiffel Tower?
What is the capital of France?
# this line is ignored
Which river runs through Paris?
```

`queries.jsonl` — one JSON object per line with a `query` key:

```jsonl
{"query": "Where is the Eiffel Tower?"}
{"query": "What is the capital of France?"}
```

Then run:

```bash
ragbolt batch corpus.json queries.txt --output rag_trace.json
```

Each query becomes one event in the trace file (appended, not overwritten). `run_batch` never raises on a single bad query — failed queries are recorded as `FAILED` and the loop continues.

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
stream_generation: false

# Hybrid retrieval
embedding_model: sentence-transformers/all-MiniLM-L6-v2
rrf_k: 60

# Production EGA verifier
nli_model: cross-encoder/nli-deberta-v3-small
nli_batch_size: 8
```

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `bm25_min_score` | `float` | `0.30` | Minimum BM25 score required for confident retrieval. |
| `overlap_min_jaccard` | `float` | `0.15` | Minimum Jaccard overlap for stub EGA sentence support. |
| `unsupported_ratio_threshold` | `float` | `0.25` | Maximum unsupported sentence ratio before grounding fails. |
| `top_k` | `int` | `5` | Number of retrieved chunks requested per query. |
| `top_k_max` | `int` | `10` | Upper bound for retrieval expansion repair. |
| `context_reduction_mode` | `str` | `chunk` | Context reduction strategy used during grounding repair. |
| `anthropic_model` | `str` | `claude-sonnet-4-20250514` | Anthropic model id for generation provider. |
| `openai_model` | `str` | `gpt-4o-mini` | OpenAI model id for generation provider. |
| `max_tokens` | `int` | `1024` | Maximum generated tokens for API providers. |
| `stream_generation` | `bool` | `false` | Enables streaming generation for API providers. |
| `embedding_model` | `str` | `sentence-transformers/all-MiniLM-L6-v2` | Dense embedding model for hybrid retrieval. |
| `rrf_k` | `int` | `60` | Reciprocal Rank Fusion constant for BM25 + dense ranks. |
| `nli_model` | `str` | `cross-encoder/nli-deberta-v3-small` | Cross-encoder NLI model for production EGA verifier. |
| `nli_batch_size` | `int` | `8` | Batch size for sentence/chunk NLI scoring. |
