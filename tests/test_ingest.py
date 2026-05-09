import json

import pytest

from ragbolt.ingest import ingest_directory, ingest_file


def test_ingest_txt_file(tmp_path) -> None:
    path = tmp_path / "doc.txt"
    path.write_text(
        "First paragraph with enough content to qualify.\n\n"
        "Second paragraph also has plenty of words inside.\n\n"
        "Third paragraph closes the file with more content.",
        encoding="utf-8",
    )
    chunks = ingest_file(path)
    assert len(chunks) >= 3
    assert all(c.source == path.name for c in chunks)
    assert len(set(c.chunk_id for c in chunks)) == len(chunks)


def test_ingest_json_corpus(tmp_path) -> None:
    path = tmp_path / "corpus.json"
    payload = [
        {"chunk_id": "c1", "text": "First chunk text content here", "source": "x.txt"},
        {"chunk_id": "c2", "text": "Second chunk text content here", "source": "x.txt"},
        {"chunk_id": "c3", "text": "Third chunk text content here", "source": "x.txt"},
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")
    chunks = ingest_file(path)
    assert len(chunks) == 3


def test_ingest_directory(tmp_path) -> None:
    (tmp_path / "a.txt").write_text(
        "Document A first paragraph content here.", encoding="utf-8"
    )
    (tmp_path / "b.txt").write_text(
        "Document B first paragraph content here.", encoding="utf-8"
    )
    corpus = ingest_directory(tmp_path)
    assert len(corpus.chunks) >= 2
    assert corpus.corpus_id == tmp_path.stem


def test_ingest_empty_dir_raises(tmp_path) -> None:
    with pytest.raises(ValueError):
        ingest_directory(tmp_path)


def test_ingest_skips_short_chunks(tmp_path) -> None:
    path = tmp_path / "mixed.txt"
    path.write_text(
        "This is a valid paragraph with enough content to qualify.\n\n"
        "a b c",
        encoding="utf-8",
    )
    chunks = ingest_file(path)
    assert all(len(c.text) >= 10 for c in chunks)
