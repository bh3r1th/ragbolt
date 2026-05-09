from typing import Protocol, runtime_checkable


@runtime_checkable
class GenerationProvider(Protocol):
    def generate(self, query: str, context: str) -> str:
        """
        Generate a response string given a query and retrieved context.
        Must return a non-empty string or raise GenerationError.
        Never returns None.
        """


class GenerationError(Exception):
    """Raised when generation fails or returns malformed output."""


class StubGenerationProvider:
    """
    Deterministic stub for testing. Returns a canned response that
    includes tokens from the query so EGA lexical overlap works.
    No API calls. No randomness.
    """

    def generate(self, query: str, context: str) -> str:
        return f"Based on the context: {context[:200]}. Query was: {query}"
