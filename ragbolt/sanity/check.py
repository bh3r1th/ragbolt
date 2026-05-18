from typing import List
from .models import ChunkInput, SanityReport
from ._adapter import normalize_chunks
from .detectors.unicode_normalization import detect as detect_unicode
from .detectors.ocr_hard_corruption import detect as detect_ocr
from .detectors.semantic_orphan import detect as detect_orphan

def sanity_check(chunks: List[ChunkInput]) -> SanityReport:
    normalized = normalize_chunks(chunks)
    issues = []
    for chunk in normalized:
        for detector in [detect_unicode, detect_ocr, detect_orphan]:
            issue = detector(chunk)
            if issue:
                issues.append(issue)
    return SanityReport(issues=issues)
