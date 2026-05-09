from typing import Protocol, runtime_checkable

from ragbolt.core.policy import Chunk


@runtime_checkable
class EGAVerifier(Protocol):
    def verify(self, response: str, chunks: list[Chunk]) -> tuple[bool, float]:
        """
        Returns (is_grounded, unsupported_ratio).
        is_grounded = True if unsupported_ratio < threshold.
        unsupported_ratio in [0.0, 1.0].
        """
