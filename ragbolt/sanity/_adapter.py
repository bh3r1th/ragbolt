from .models import RagboltChunk, ChunkInput
from typing import List

def to_chunk(raw: ChunkInput, index: int) -> RagboltChunk:
    if isinstance(raw, RagboltChunk):
        return raw
    if isinstance(raw, str):
        return RagboltChunk(id=f"chunk_{index}", text=raw)
    if isinstance(raw, dict):
        text = raw.get("text")
        if not text:
            raise ValueError(f"dict chunk at index {index} missing 'text' field")
        return RagboltChunk(
            id=raw.get("id", f"chunk_{index}"),
            text=text,
            metadata=raw.get("metadata", {}),
        )
    raise TypeError(f"Unsupported chunk type: {type(raw)}")

def normalize_chunks(raws: List[ChunkInput]) -> List[RagboltChunk]:
    return [to_chunk(r, i) for i, r in enumerate(raws)]
