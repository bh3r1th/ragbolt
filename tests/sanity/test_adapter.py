import pytest
from ragbolt.sanity._adapter import normalize_chunks
from ragbolt.sanity.models import RagboltChunk

def test_string_input():
    chunks = normalize_chunks(["hello world"])
    assert chunks[0].text == "hello world"
    assert chunks[0].id == "chunk_0"

def test_dict_input():
    chunks = normalize_chunks([{"id": "x1", "text": "some text", "metadata": {"page": 1}}])
    assert chunks[0].id == "x1"
    assert chunks[0].metadata == {"page": 1}

def test_ragboltchunk_input():
    c = RagboltChunk(id="c1", text="direct")
    chunks = normalize_chunks([c])
    assert chunks[0] is c

def test_missing_id_fallback():
    chunks = normalize_chunks([{"text": "no id here"}])
    assert chunks[0].id == "chunk_0"

def test_metadata_preserved():
    chunks = normalize_chunks([{"text": "t", "metadata": {"source": "x.pdf"}}])
    assert chunks[0].metadata["source"] == "x.pdf"
