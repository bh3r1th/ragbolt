from pathlib import Path


def create_app(
    corpus_path: Path,
    config: dict,
    provider_name: str,
    retriever_name: str,
    verifier_name: str,
):
    """
    Create and return a FastAPI app instance.
    Lazy-import fastapi inside this function.
    """
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError:
        raise ImportError(
            "ragbolt serve requires fastapi and uvicorn. "
            "Install with: pip install fastapi uvicorn[standard]"
        )

    from ragbolt.core.policy import load_corpus
    from ragbolt.core.orchestrator import RepairOrchestrator
    from ragbolt.core.generator import StubGenerationProvider
    from ragbolt.verify.stub import StubEGAVerifier
    from ragbolt.trace.emitter import TraceEmitter

    corpus = load_corpus(corpus_path)

    if provider_name == "stub":
        provider = StubGenerationProvider()
    elif provider_name == "anthropic":
        from ragbolt.core.providers import AnthropicGenerationProvider

        provider = AnthropicGenerationProvider(config)
    elif provider_name == "openai":
        from ragbolt.core.providers import OpenAIGenerationProvider

        provider = OpenAIGenerationProvider(config)
    else:
        raise ValueError("unknown provider")

    if verifier_name == "stub":
        verifier = StubEGAVerifier(config)
    elif verifier_name == "production":
        from ragbolt.verify.production import ProductionEGAVerifier

        verifier = ProductionEGAVerifier(config)
    else:
        raise ValueError("unknown verifier")

    if retriever_name == "bm25":
        retriever = None
    elif retriever_name == "hybrid":
        from ragbolt.core.hybrid import HybridRunner

        retriever = HybridRunner(corpus, config)
    else:
        raise ValueError("unknown retriever")

    app = FastAPI(
        title="ragbolt",
        description="Failure-aware RAG repair layer",
        version="0.3.0",
    )

    class QueryRequest(BaseModel):
        query: str
        trace_output: str = "rag_trace.json"

    class QueryResponse(BaseModel):
        run_id: str
        outcome: str
        response: str | None
        failure_classes: list[str]
        repair_attempts: int
        top_score: float
        chunks_retrieved: int
        unsupported_ratio: float
        chunk_ids: list[str]

    @app.get("/health")
    def health():
        return {"status": "ok", "corpus_id": corpus.corpus_id, "chunks": len(corpus.chunks)}

    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest):
        orchestrator = RepairOrchestrator(
            corpus,
            config,
            provider,
            verifier,
            retriever=retriever,
        )
        emitter = TraceEmitter(Path(req.trace_output))
        result = orchestrator.run(req.query)
        run_id = emitter.record(
            corpus_id=corpus.corpus_id,
            query=req.query,
            failure_classes=result.failure_classes,
            repair_attempts=result.repair_attempts,
            outcome=result.outcome,
            top_score=result.top_score,
            chunks_retrieved=result.chunks_retrieved,
            unsupported_ratio=result.unsupported_ratio,
            chunk_ids=result.chunk_ids,
            raw_top_score=result.raw_top_score,
        )
        emitter.flush()
        return QueryResponse(
            run_id=run_id,
            outcome=result.outcome.value,
            response=result.response,
            failure_classes=[f.value for f in result.failure_classes],
            repair_attempts=result.repair_attempts,
            top_score=result.top_score,
            chunks_retrieved=result.chunks_retrieved,
            unsupported_ratio=result.unsupported_ratio,
            chunk_ids=result.chunk_ids,
        )

    @app.get("/trace")
    def trace(output: str = "rag_trace.json"):
        import json

        p = Path(output)
        if not p.exists():
            raise HTTPException(status_code=404, detail="Trace file not found")
        return json.loads(p.read_text(encoding="utf-8"))

    return app
