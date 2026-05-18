import unicodedata
from ..models import RagboltChunk, ChunkIssue

def detect(chunk: RagboltChunk) -> ChunkIssue | None:
    normalized = unicodedata.normalize("NFC", chunk.text)
    if normalized == chunk.text:
        return None
    return ChunkIssue(
        chunk_id=chunk.id or "unknown",
        detector="unicode_normalization_corruption",
        severity="low",
        action="normalized",
        evidence="text was not NFC normalized",
        verification="nfc_normalized",
    )

def repair(chunk: RagboltChunk) -> RagboltChunk:
    return chunk.model_copy(update={"text": unicodedata.normalize("NFC", chunk.text)})
