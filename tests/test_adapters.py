import json
from pathlib import Path

import pytest

from ragbolt.adapters.base import BaseRagboltAdapter, RagboltResponse


def _write_corpus(tmp_path: Path) -> Path:
    corpus_path = tmp_path / "corpus.json"
    corpus = [{"chunk_id": "c1", "text": "BM25 retrieval ranking", "source": "a.txt"}]
    corpus_path.write_text(json.dumps(corpus), encoding="utf-8")
    return corpus_path


def test_base_adapter_init(tmp_path: Path):
    corpus_path = _write_corpus(tmp_path)
    adapter = BaseRagboltAdapter(str(corpus_path))
    assert adapter.corpus.corpus_id == "corpus"
    assert len(adapter.corpus.chunks) == 1


def test_base_adapter_query_returns_response(tmp_path: Path):
    corpus_path = _write_corpus(tmp_path)
    adapter = BaseRagboltAdapter(str(corpus_path))
    response = adapter.query("retrieval ranking")
    assert isinstance(response, RagboltResponse)
    assert response.outcome in ("ACCEPTED", "REPAIRED_ACCEPTED", "ABSTAINED", "FAILED")
    assert response.run_id is not None
    assert response.corpus_id == "corpus"


def test_base_adapter_retrieve_returns_chunks(tmp_path: Path):
    corpus_path = _write_corpus(tmp_path)
    adapter = BaseRagboltAdapter(str(corpus_path))
    chunks = adapter.retrieve("retrieval ranking")
    assert isinstance(chunks, list)


def test_base_adapter_unknown_provider_raises(tmp_path: Path):
    corpus_path = _write_corpus(tmp_path)
    with pytest.raises(ValueError):
        BaseRagboltAdapter(str(corpus_path), provider_name="bad")


def test_base_adapter_unknown_verifier_raises(tmp_path: Path):
    corpus_path = _write_corpus(tmp_path)
    with pytest.raises(ValueError):
        BaseRagboltAdapter(str(corpus_path), verifier_name="bad")


def test_ragbolt_response_fields(tmp_path: Path):
    corpus_path = _write_corpus(tmp_path)
    adapter = BaseRagboltAdapter(str(corpus_path))
    response = adapter.query("retrieval")
    assert hasattr(response, "query")
    assert hasattr(response, "outcome")
    assert hasattr(response, "chunk_ids")
    assert hasattr(response, "failure_classes")
    assert hasattr(response, "repair_attempts")
    assert hasattr(response, "unsupported_ratio")
    assert isinstance(response.failure_classes, list)
    assert isinstance(response.chunk_ids, list)


def test_langchain_adapter_import_error_is_helpful(tmp_path: Path):
    corpus_path = _write_corpus(tmp_path)
    from ragbolt.adapters.langchain_adapter import RagboltRetriever

    retriever = RagboltRetriever(str(corpus_path))
    try:
        docs = retriever.get_relevant_documents("retrieval")
        assert isinstance(docs, list)
    except ImportError as e:
        assert "langchain-core" in str(e)


def test_llamaindex_adapter_import_error_is_helpful(tmp_path: Path):
    corpus_path = _write_corpus(tmp_path)
    from ragbolt.adapters.llamaindex_adapter import RagboltLlamaIndexRetriever

    retriever = RagboltLlamaIndexRetriever(str(corpus_path))
    try:
        nodes = retriever.retrieve("retrieval")
        assert isinstance(nodes, list)
    except ImportError as e:
        assert "llama-index-core" in str(e)


def test_adapters_init_importable():
    from ragbolt.adapters import BaseRagboltAdapter, RagboltResponse

    assert BaseRagboltAdapter is not None
    assert RagboltResponse is not None


def test_ragbolt_qa_call_interface(tmp_path: Path):
    corpus_path = _write_corpus(tmp_path)
    from ragbolt.adapters.langchain_adapter import RagboltLangChainQA

    qa = RagboltLangChainQA(str(corpus_path))
    result = qa("retrieval ranking")
    assert "answer" in result
    assert "outcome" in result
    assert "sources" in result
    assert "run_id" in result
