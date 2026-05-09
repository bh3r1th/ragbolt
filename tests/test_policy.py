import json

import pytest
from pydantic import ValidationError

from ragbolt.core.policy import Chunk, load_corpus


def test_load_corpus_valid(tmp_path) -> None:
    path = tmp_path / "demo.json"
    payload = [
        {"chunk_id": "c1", "text": "alpha text", "source": "a.txt"},
        {"chunk_id": "c2", "text": "beta text", "source": "b.txt", "metadata": {"k": "v"}},
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")

    corpus = load_corpus(path)
    assert corpus.corpus_id == "demo"
    assert len(corpus.chunks) == 2


def test_load_corpus_duplicate_ids(tmp_path) -> None:
    path = tmp_path / "dupe.json"
    payload = [
        {"chunk_id": "same", "text": "alpha", "source": "a.txt"},
        {"chunk_id": "same", "text": "beta", "source": "b.txt"},
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        load_corpus(path)


def test_load_corpus_missing_field(tmp_path) -> None:
    path = tmp_path / "missing.json"
    payload = [
        {"chunk_id": "c1", "text": "alpha"},
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        load_corpus(path)


def test_load_corpus_file_not_found(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "nope.json")


def test_chunk_text_max_length() -> None:
    with pytest.raises(ValidationError):
        Chunk(chunk_id="c1", text="x" * 8001, source="a.txt")
