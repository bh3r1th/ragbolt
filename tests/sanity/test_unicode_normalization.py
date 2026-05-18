import unicodedata
import pytest
from ragbolt.sanity.models import RagboltChunk, SanityReport
from ragbolt.sanity.detectors.unicode_normalization import detect, repair

NFC = "café"
NFD = unicodedata.normalize("NFD", NFC)

def test_positive_detects_nfd():
    chunk = RagboltChunk(id="c1", text=NFD)
    issue = detect(chunk)
    assert issue is not None
    assert issue.detector == "unicode_normalization_corruption"
    assert issue.severity == "low"
    assert issue.action == "normalized"
    assert issue.verification == "nfc_normalized"

def test_negative_nfc_text():
    chunk = RagboltChunk(id="c2", text=NFC)
    assert detect(chunk) is None

def test_false_positive_guard_ascii():
    chunk = RagboltChunk(id="c3", text="plain ascii text")
    assert detect(chunk) is None

def test_repair_idempotent():
    chunk = RagboltChunk(id="c4", text=NFD)
    once = repair(chunk)
    twice = repair(once)
    assert once.text == twice.text
    assert unicodedata.is_normalized("NFC", once.text)

def test_repair_does_not_alter_metadata():
    chunk = RagboltChunk(id="c5", text=NFD, metadata={"source": "doc.pdf", "page": 3})
    repaired = repair(chunk)
    assert repaired.metadata == {"source": "doc.pdf", "page": 3}

def test_trace_schema():
    chunk = RagboltChunk(id="c6", text=NFD)
    issue = detect(chunk)
    assert issue.chunk_id == "c6"
    assert issue.evidence
    assert issue.metadata == {}
