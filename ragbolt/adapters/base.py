from typing import Protocol, runtime_checkable, Any
from pathlib import Path
from dataclasses import dataclass
from ragbolt.core.policy import Chunk, Corpus, load_corpus, DecisionOutcome
from ragbolt.core.orchestrator import RepairOrchestrator, OrchestratorResult
from ragbolt.core.generator import StubGenerationProvider
from ragbolt.verify.stub import StubEGAVerifier


@dataclass
class RagboltResponse:
    """
    Unified response object returned by all ragbolt adapters.
    Framework-agnostic. Adapters map this to framework-native types.
    """

    query: str
    response: str | None
    outcome: str
    run_id: str
    corpus_id: str
    chunks: list[Chunk]
    chunk_ids: list[str]
    failure_classes: list[str]
    repair_attempts: int
    top_score: float
    unsupported_ratio: float
    metadata: dict[str, Any]


@runtime_checkable
class RagboltAdapter(Protocol):
    def query(self, query: str, **kwargs) -> RagboltResponse: ...
    def retrieve(self, query: str, **kwargs) -> list[Chunk]: ...


class BaseRagboltAdapter:
    """
    Shared implementation inherited by all framework adapters.
    Handles corpus loading, orchestrator setup, and result packaging.
    """

    def __init__(
        self,
        corpus_path: Path | str,
        config: dict | None = None,
        provider_name: str = "stub",
        verifier_name: str = "stub",
        retriever_name: str = "bm25",
    ):
        self.corpus_path = Path(corpus_path)
        self.config = config or {
            "bm25_min_score": 0.30,
            "overlap_min_jaccard": 0.15,
            "unsupported_ratio_threshold": 0.25,
            "top_k": 5,
            "top_k_max": 10,
            "context_reduction_mode": "chunk",
            "max_tokens": 1024,
        }
        self.corpus: Corpus = load_corpus(self.corpus_path)
        self._provider = self._build_provider(provider_name)
        self._verifier = self._build_verifier(verifier_name)
        self._retriever = self._build_retriever(retriever_name)

    def _build_provider(self, name: str):
        if name == "stub":
            return StubGenerationProvider()
        elif name == "anthropic":
            from ragbolt.core.providers import AnthropicGenerationProvider

            return AnthropicGenerationProvider(self.config)
        elif name == "openai":
            from ragbolt.core.providers import OpenAIGenerationProvider

            return OpenAIGenerationProvider(self.config)
        else:
            raise ValueError(f"Unknown provider: {name}")

    def _build_verifier(self, name: str):
        if name == "stub":
            return StubEGAVerifier(self.config)
        elif name == "production":
            from ragbolt.verify.production import ProductionEGAVerifier

            return ProductionEGAVerifier(self.config)
        else:
            raise ValueError(f"Unknown verifier: {name}")

    def _build_retriever(self, name: str):
        if name == "bm25":
            return None
        elif name == "hybrid":
            from ragbolt.core.hybrid import HybridRunner

            return HybridRunner(self.corpus, self.config)
        else:
            raise ValueError(f"Unknown retriever: {name}")

    def _run(self, query: str) -> tuple[OrchestratorResult, str]:
        from ragbolt.trace.emitter import TraceEmitter

        orchestrator = RepairOrchestrator(
            self.corpus, self.config, self._provider, self._verifier, self._retriever
        )
        result = orchestrator.run(query)
        emitter = TraceEmitter()
        run_id = emitter.record(
            corpus_id=self.corpus.corpus_id,
            query=query,
            failure_classes=result.failure_classes,
            repair_attempts=result.repair_attempts,
            outcome=result.outcome,
            top_score=result.top_score,
            chunks_retrieved=result.chunks_retrieved,
            unsupported_ratio=result.unsupported_ratio,
            chunk_ids=result.chunk_ids,
            raw_top_score=result.raw_top_score,
        )
        return result, run_id

    def query(self, query: str, **kwargs) -> RagboltResponse:
        result, run_id = self._run(query)
        return RagboltResponse(
            query=query,
            response=result.response,
            outcome=result.outcome.value,
            run_id=run_id,
            corpus_id=self.corpus.corpus_id,
            chunks=result.chunks if hasattr(result, "chunks") else [],
            chunk_ids=result.chunk_ids,
            failure_classes=[f.value for f in result.failure_classes],
            repair_attempts=result.repair_attempts,
            top_score=result.top_score,
            unsupported_ratio=result.unsupported_ratio,
            metadata=kwargs,
        )

    def retrieve(self, query: str, **kwargs) -> list[Chunk]:
        from ragbolt.core.bm25 import BM25Runner

        chunks, _ = BM25Runner(self.corpus, self.config).retrieve(query)
        return chunks
