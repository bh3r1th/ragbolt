from typing import List, Optional
from .models import ChunkInput, RepairResult, SanityReport
from ._adapter import normalize_chunks
from .detectors import unicode_normalization as ud
from .detectors import semantic_orphan as so

def repair(
    chunks: List[ChunkInput],
    sanity_report: SanityReport,
    neighbors: Optional[dict] = None,
) -> RepairResult:
    normalized = normalize_chunks(chunks)
    repaired = []
    repairs = []
    flagged_by_id: dict = {}
    for issue in sanity_report.issues:
        flagged_by_id.setdefault(issue.chunk_id, []).append(issue)

    for chunk in normalized:
        issues = flagged_by_id.get(chunk.id or "unknown", [])
        for issue in issues:
            if issue.detector == "unicode_normalization_corruption":
                chunk = ud.repair(chunk)
                repairs.append(issue)
            elif issue.detector == "semantic_orphan_reference" and neighbors:
                nbrs = neighbors.get(chunk.id, [])
                chunk, updated_issue = so.repair(chunk, issue, nbrs)
                repairs.append(updated_issue)
        repaired.append(chunk)
    return RepairResult(chunks=repaired, repairs=repairs)

def repair_with_neighbors(
    chunks: List[ChunkInput],
    sanity_report: SanityReport,
    neighbors: Optional[dict] = None,
) -> RepairResult:
    return repair(chunks, sanity_report, neighbors)
