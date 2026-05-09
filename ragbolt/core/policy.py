from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    text: str = Field(max_length=8000)
    source: str
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("chunk_id", "text", "source")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be a non-empty string")
        return value


class Corpus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunks: list[Chunk]
    corpus_id: str

    @model_validator(mode="after")
    def validate_unique_chunk_ids(self) -> Corpus:
        seen: set[str] = set()
        for chunk in self.chunks:
            if chunk.chunk_id in seen:
                raise ValueError(f"duplicate chunk_id: {chunk.chunk_id}")
            seen.add(chunk.chunk_id)
        return self


def load_corpus(path: Path) -> Corpus:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    try:
        return Corpus.model_validate({"chunks": data, "corpus_id": path.stem})
    except ValidationError as e:
        raise ValueError(f"Corpus validation failed: {e}") from e


class FailureClass(str, Enum):
    RETRIEVAL_LOW_CONFIDENCE = "RETRIEVAL_LOW_CONFIDENCE"
    GENERATION_MALFORMED = "GENERATION_MALFORMED"
    GROUNDING_FAILED = "GROUNDING_FAILED"


class DecisionOutcome(str, Enum):
    ACCEPTED = "ACCEPTED"
    REPAIRED_ACCEPTED = "REPAIRED_ACCEPTED"
    ABSTAINED = "ABSTAINED"
    FAILED = "FAILED"


class RepairPolicy:
    MAX_ATTEMPTS: int = 2
    # GENERATION_MALFORMED: no repair in v0.1.0 — fails fast
    ORDERED_REPAIRS: list[FailureClass] = [
        FailureClass.RETRIEVAL_LOW_CONFIDENCE,
        FailureClass.GROUNDING_FAILED,
    ]
