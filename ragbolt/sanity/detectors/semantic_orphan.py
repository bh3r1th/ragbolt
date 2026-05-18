from __future__ import annotations
import re
from typing import List, Optional
from ..models import RagboltChunk, ChunkIssue

_PHRASES = [
    "as shown above", "as shown below",
    "see above", "see below",
    "see table", "see figure",
    "shown in the table", "shown in the figure",
    "as discussed earlier", "as mentioned above",
]
_PATTERN = re.compile(
    "|".join(re.escape(p) for p in _PHRASES), re.IGNORECASE
)

def detect(chunk: RagboltChunk) -> ChunkIssue | None:
    if _PATTERN.search(chunk.text):
        return ChunkIssue(
            chunk_id=chunk.id or "unknown",
            detector="semantic_orphan_reference",
            severity="medium",
            action="risk_annotated",
            evidence=f"structural reference phrase found in text",
            verification="risk_only",
        )
    return None

def repair(
    chunk: RagboltChunk,
    issue: ChunkIssue,
    neighbors: Optional[List[RagboltChunk]] = None,
) -> tuple[RagboltChunk, ChunkIssue]:
    if not neighbors:
        return chunk, issue
    neighbor_text = " ".join(n.text for n in neighbors[:2])
    expanded = f"{neighbor_text} {chunk.text}"
    updated_chunk = chunk.model_copy(update={"text": expanded})
    updated_issue = issue.model_copy(update={
        "action": "repair_applied",
        "verification": "bounded_neighbor_context_added",
    })
    return updated_chunk, updated_issue
