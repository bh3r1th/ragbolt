from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field

class RagboltChunk(BaseModel):
    id: Optional[str] = None
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

ChunkInput = Union[str, Dict[str, Any], RagboltChunk]

class ChunkIssue(BaseModel):
    chunk_id: str
    detector: str
    severity: Literal["low", "medium", "high"]
    action: Literal["normalized", "risk_annotated", "repair_applied", "no_op"]
    evidence: str
    verification: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SanityReport(BaseModel):
    issues: List[ChunkIssue] = Field(default_factory=list)

class RepairResult(BaseModel):
    chunks: List[RagboltChunk]
    repairs: List[ChunkIssue] = Field(default_factory=list)
