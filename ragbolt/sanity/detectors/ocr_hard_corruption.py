import re
from ..models import RagboltChunk, ChunkIssue

_MOJIBAKE = ["Ã", "Â", "â€™", "â€œ", "â€", "â€\"", "â€\""]
_CONTROL_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')

def detect(chunk: RagboltChunk) -> ChunkIssue | None:
    text = chunk.text
    evidence_parts = []
    if "\ufffd" in text:
        evidence_parts.append("U+FFFD replacement character present")
    if "\x00" in text:
        evidence_parts.append("null bytes present")
    ctrl_matches = _CONTROL_RE.findall(text)
    if len(ctrl_matches) > 3:
        evidence_parts.append(f"{len(ctrl_matches)} non-printable control chars")
    for seq in _MOJIBAKE:
        if seq in text:
            evidence_parts.append(f"mojibake sequence '{seq}' found")
    if not evidence_parts:
        return None
    return ChunkIssue(
        chunk_id=chunk.id or "unknown",
        detector="ocr_hard_corruption",
        severity="medium",
        action="risk_annotated",
        evidence="; ".join(evidence_parts),
        verification="evidence_captured",
    )
