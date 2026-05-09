from ragbolt.core.policy import Chunk
from ragbolt.verify.protocol import EGAVerifier
from ragbolt.verify.stub import StubEGAVerifier


class ProductionEGAVerifier:
    """
    NLI-based grounding verifier using cross-encoder/nli-deberta-v3-small.
    Requires: pip install sentence-transformers
    Falls back to StubEGAVerifier behavior if model unavailable.
    """

    def __init__(self, config: dict):
        self._config = dict(config)
        self._unsupported_ratio_threshold = float(
            config.get("unsupported_ratio_threshold", 0.25)
        )
        self._nli_model = str(config.get("nli_model", "cross-encoder/nli-deberta-v3-small"))
        self._nli_batch_size = int(config.get("nli_batch_size", 8))
        self._model = None
        self._model_load_attempted = False

    def _load_model(self):
        if self._model_load_attempted:
            return
        self._model_load_attempted = True
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._nli_model)
        except Exception:
            self._model = None

    def verify(self, response: str, chunks: list[Chunk]) -> tuple[bool, float]:
        if self._model is None:
            self._load_model()

        sentences = [s.strip() for s in response.split(". ") if s.strip()]
        if not sentences:
            return True, 0.0

        context = "\n".join(c.text for c in chunks)
        if self._model is None:
            fallback_config = dict(self._config)
            fallback_config.setdefault("overlap_min_jaccard", 0.15)
            return StubEGAVerifier(fallback_config).verify(response, chunks)

        unsupported = 0
        for sentence in sentences:
            pairs = [(sentence, chunk.text) for chunk in chunks]
            entailment_score = 0.0
            for start in range(0, len(pairs), self._nli_batch_size):
                batch = pairs[start : start + self._nli_batch_size]
                scores = self._model.predict(batch)
                for score in scores:
                    entailment_score = max(entailment_score, float(score[2]))
            if entailment_score < 0.5:
                unsupported += 1

        unsupported_ratio = unsupported / len(sentences)
        is_grounded = unsupported_ratio < self._unsupported_ratio_threshold
        return is_grounded, unsupported_ratio
