# ragbolt
Failure-aware RAG repair layer for Python.

ragbolt runs a bounded failure-handling loop around retrieval, generation, and grounding checks. It classifies retrieval low-confidence, malformed generation output, and grounding failures. It applies one bounded repair at a time with a maximum of two repairs in a run. Grounding is re-checked with EGA (Evidence-Gated Generation). Each run emits an auditable `rag_trace.json` trace.

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

Continue with [Getting Started](getting-started.md).
