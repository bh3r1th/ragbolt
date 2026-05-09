from dataclasses import dataclass, field

from ragbolt.core.bm25 import BM25Runner
from ragbolt.core.generator import GenerationError, GenerationProvider
from ragbolt.core.policy import Chunk, Corpus, DecisionOutcome, FailureClass, RepairPolicy
from ragbolt.verify.protocol import EGAVerifier


@dataclass
class OrchestratorResult:
    outcome: DecisionOutcome
    response: str | None
    failure_classes: list[FailureClass]
    repair_attempts: int
    top_score: float
    chunks_retrieved: int
    unsupported_ratio: float = 0.0
    chunk_ids: list[str] = field(default_factory=list)
    raw_top_score: float = 0.0


class RepairOrchestrator:
    def __init__(
        self,
        corpus: Corpus,
        config: dict,
        generator: GenerationProvider,
        verifier: EGAVerifier,
        retriever=None,
    ):
        self._corpus = corpus
        self._config = dict(config)
        self._generator = generator
        self._verifier = verifier
        if retriever is None:
            self._retriever = BM25Runner(corpus, config)
            self._retriever_type = BM25Runner
        else:
            self._retriever = retriever
            self._retriever_type = retriever.__class__

    def _append_failure(self, failures: list, fc: FailureClass) -> None:
        if fc not in failures:
            failures.append(fc)

    def run(self, query: str) -> OrchestratorResult:
        failure_classes: list[FailureClass] = []
        repair_attempts = 0
        top_k = int(self._config["top_k"])
        top_k_max = int(self._config["top_k_max"])
        bm25_min_score = float(self._config["bm25_min_score"])

        retrieval_config = dict(self._config)
        retrieval_config["top_k"] = top_k
        retriever = self._retriever
        chunks, top_score = retriever.retrieve(query)
        raw_top_score = top_score
        chunk_ids = [c.chunk_id for c in chunks]

        if top_score < bm25_min_score:
            self._append_failure(failure_classes, FailureClass.RETRIEVAL_LOW_CONFIDENCE)
            expanded_config = dict(self._config)
            expanded_config["top_k"] = min(int(self._config["top_k"]) + 5, top_k_max)
            repair_attempts += 1
            retriever = self._retriever_type(self._corpus, expanded_config)
            chunks, top_score = retriever.retrieve(query)
            raw_top_score = top_score
            chunk_ids = [c.chunk_id for c in chunks]
            if top_score < bm25_min_score:
                self._append_failure(failure_classes, FailureClass.RETRIEVAL_LOW_CONFIDENCE)
                return OrchestratorResult(
                    outcome=DecisionOutcome.ABSTAINED,
                    response=None,
                    failure_classes=failure_classes,
                    repair_attempts=repair_attempts,
                    top_score=top_score,
                    chunks_retrieved=len(chunks),
                    unsupported_ratio=0.0,
                    chunk_ids=chunk_ids,
                    raw_top_score=raw_top_score,
                )

        active_chunks: list[Chunk] = chunks
        chunk_ids = [c.chunk_id for c in active_chunks]
        context = "\n\n".join(chunk.text for chunk in active_chunks)
        # No repair for GENERATION_MALFORMED in v0.1.0 — fails immediately
        try:
            response = self._generator.generate(query, context)
        except GenerationError:
            self._append_failure(failure_classes, FailureClass.GENERATION_MALFORMED)
            return OrchestratorResult(
                outcome=DecisionOutcome.FAILED,
                response=None,
                failure_classes=failure_classes,
                repair_attempts=repair_attempts,
                top_score=top_score,
                chunks_retrieved=len(active_chunks),
                unsupported_ratio=0.0,
                chunk_ids=chunk_ids,
                raw_top_score=raw_top_score,
            )
        if response.strip() == "":
            self._append_failure(failure_classes, FailureClass.GENERATION_MALFORMED)
            return OrchestratorResult(
                outcome=DecisionOutcome.FAILED,
                response=None,
                failure_classes=failure_classes,
                repair_attempts=repair_attempts,
                top_score=top_score,
                chunks_retrieved=len(active_chunks),
                unsupported_ratio=0.0,
                chunk_ids=chunk_ids,
                raw_top_score=raw_top_score,
            )

        is_grounded, unsupported_ratio = self._verifier.verify(response, active_chunks)
        if not is_grounded:
            self._append_failure(failure_classes, FailureClass.GROUNDING_FAILED)
            if repair_attempts >= RepairPolicy.MAX_ATTEMPTS:
                return OrchestratorResult(
                    outcome=DecisionOutcome.FAILED,
                    response=None,
                    failure_classes=failure_classes,
                    repair_attempts=repair_attempts,
                    top_score=top_score,
                    chunks_retrieved=len(active_chunks),
                    unsupported_ratio=unsupported_ratio,
                    chunk_ids=chunk_ids,
                    raw_top_score=raw_top_score,
                )
            repair_attempts += 1
            active_chunks = active_chunks[:1]
            chunk_ids = [c.chunk_id for c in active_chunks]
            context = "\n\n".join(chunk.text for chunk in active_chunks)
            try:
                repaired_response = self._generator.generate(query, context)
            except GenerationError:
                self._append_failure(failure_classes, FailureClass.GENERATION_MALFORMED)
                return OrchestratorResult(
                    outcome=DecisionOutcome.FAILED,
                    response=None,
                    failure_classes=failure_classes,
                    repair_attempts=repair_attempts,
                    top_score=top_score,
                    chunks_retrieved=len(active_chunks),
                    unsupported_ratio=unsupported_ratio,
                    chunk_ids=chunk_ids,
                    raw_top_score=raw_top_score,
                )
            if repaired_response.strip() == "":
                self._append_failure(failure_classes, FailureClass.GENERATION_MALFORMED)
                return OrchestratorResult(
                    outcome=DecisionOutcome.FAILED,
                    response=None,
                    failure_classes=failure_classes,
                    repair_attempts=repair_attempts,
                    top_score=top_score,
                    chunks_retrieved=len(active_chunks),
                    unsupported_ratio=unsupported_ratio,
                    chunk_ids=chunk_ids,
                    raw_top_score=raw_top_score,
                )
            repaired_grounded, repaired_ratio = self._verifier.verify(repaired_response, active_chunks)
            unsupported_ratio = repaired_ratio
            if not repaired_grounded:
                self._append_failure(failure_classes, FailureClass.GROUNDING_FAILED)
                return OrchestratorResult(
                    outcome=DecisionOutcome.FAILED,
                    response=None,
                    failure_classes=failure_classes,
                    repair_attempts=repair_attempts,
                    top_score=top_score,
                    chunks_retrieved=len(active_chunks),
                    unsupported_ratio=unsupported_ratio,
                    chunk_ids=chunk_ids,
                    raw_top_score=raw_top_score,
                )
            return OrchestratorResult(
                outcome=DecisionOutcome.REPAIRED_ACCEPTED,
                response=repaired_response,
                failure_classes=failure_classes,
                repair_attempts=repair_attempts,
                top_score=top_score,
                chunks_retrieved=len(active_chunks),
                unsupported_ratio=unsupported_ratio,
                chunk_ids=chunk_ids,
                raw_top_score=raw_top_score,
            )

        outcome = (
            DecisionOutcome.REPAIRED_ACCEPTED
            if repair_attempts > 0
            else DecisionOutcome.ACCEPTED
        )
        return OrchestratorResult(
            outcome=outcome,
            response=response,
            failure_classes=failure_classes,
            repair_attempts=repair_attempts,
            top_score=top_score,
            chunks_retrieved=len(active_chunks),
            unsupported_ratio=unsupported_ratio,
            chunk_ids=chunk_ids,
            raw_top_score=raw_top_score,
        )
