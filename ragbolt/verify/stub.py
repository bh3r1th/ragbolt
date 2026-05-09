from ragbolt.core.policy import Chunk


class StubEGAVerifier:
    """
    Lexical overlap verifier. No ML dependencies required.
    Uses Jaccard similarity between response tokens and chunk tokens.
    """

    def __init__(self, config: dict):
        self._overlap_min_jaccard = float(config["overlap_min_jaccard"])
        self._unsupported_ratio_threshold = float(config["unsupported_ratio_threshold"])

    def verify(self, response: str, chunks: list[Chunk]) -> tuple[bool, float]:
        chunk_token_sets = [set(chunk.text.lower().split()) for chunk in chunks]
        sentences = [s.strip() for s in response.split(". ") if s.strip()]
        if not sentences:
            return False, 1.0
        unsupported_sentences = 0
        for sentence in sentences:
            sentence_tokens = set(sentence.lower().split())
            max_jaccard = 0.0
            for chunk_tokens in chunk_token_sets:
                overlap = self._jaccard(sentence_tokens, chunk_tokens)
                if overlap > max_jaccard:
                    max_jaccard = overlap
            if max_jaccard < self._overlap_min_jaccard:
                unsupported_sentences += 1
        unsupported_ratio = unsupported_sentences / len(sentences)
        is_grounded = unsupported_ratio < self._unsupported_ratio_threshold
        return is_grounded, unsupported_ratio

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        union = a | b
        if not union:
            return 0.0
        return len(a & b) / len(union)
