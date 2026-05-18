import pytest
from ragbolt.sanity.models import RagboltChunk
from ragbolt.sanity.detectors.semantic_orphan import detect, repair

def test_positive_detects_phrase():
    chunk = RagboltChunk(id="c1", text="As shown above, the result is clear.")
    issue = detect(chunk)
    assert issue is not None
    assert issue.detector == "semantic_orphan_reference"
    assert issue.verification == "risk_only"

def test_negative_no_phrase():
    chunk = RagboltChunk(id="c2", text="The results are conclusive.")
    assert detect(chunk) is None

def test_false_positive_guard_partial_word():
    chunk = RagboltChunk(id="c3", text="The table was set for dinner.")
    assert detect(chunk) is None

def test_no_expansion_without_neighbors():
    chunk = RagboltChunk(id="c4", text="See figure for details.")
    issue = detect(chunk)
    repaired_chunk, repaired_issue = repair(chunk, issue, neighbors=None)
    assert repaired_chunk.text == chunk.text
    assert repaired_issue.verification == "risk_only"

def test_expansion_with_neighbors():
    chunk = RagboltChunk(id="c4", text="See figure for details.")
    issue = detect(chunk)
    prev = RagboltChunk(id="c3", text="Figure 1 shows the distribution.")
    repaired_chunk, repaired_issue = repair(chunk, issue, neighbors=[prev])
    assert "Figure 1" in repaired_chunk.text
    assert repaired_issue.verification == "bounded_neighbor_context_added"
    assert repaired_issue.action == "repair_applied"

def test_trace_schema():
    chunk = RagboltChunk(id="c5", text="As mentioned above, this is key.")
    issue = detect(chunk)
    assert issue.chunk_id == "c5"
    assert issue.severity == "medium"
