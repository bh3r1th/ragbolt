import pytest
from ragbolt.sanity.models import RagboltChunk
from ragbolt.sanity.detectors.ocr_hard_corruption import detect

def test_positive_replacement_char():
    chunk = RagboltChunk(id="c1", text="bad \ufffd text")
    issue = detect(chunk)
    assert issue is not None
    assert issue.detector == "ocr_hard_corruption"
    assert issue.action == "risk_annotated"
    assert issue.verification == "evidence_captured"

def test_positive_mojibake():
    chunk = RagboltChunk(id="c2", text="some â€™ text")
    assert detect(chunk) is not None

def test_negative_clean_text():
    chunk = RagboltChunk(id="c3", text="This is a normal sentence.")
    assert detect(chunk) is None

def test_false_positive_guard_multilingual():
    chunk = RagboltChunk(id="c4", text="Das ist ein Satz. Это предложение. 这是一个句子。")
    assert detect(chunk) is None

def test_trace_schema():
    chunk = RagboltChunk(id="c5", text="null\x00byte")
    issue = detect(chunk)
    assert issue.chunk_id == "c5"
    assert issue.severity == "medium"
    assert issue.evidence
