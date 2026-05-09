# Framework Adapters

ragbolt works standalone through its CLI and API, and it can also be used as a drop-in component inside LangChain and LlamaIndex pipelines through adapter classes that preserve ragbolt's retrieval, verification, and repair behavior.

## Installation
```bash
pip install ragbolt[langchain]    # LangChain adapter
pip install ragbolt[llamaindex]   # LlamaIndex adapter
pip install ragbolt[all]          # everything
```

## LangChain

### As a retriever
```python
from ragbolt.adapters.langchain_adapter import RagboltRetriever

retriever = RagboltRetriever("corpus.json")
docs = retriever.get_relevant_documents("What is BM25?")

for d in docs:
    print(d.page_content[:80])
    print(d.metadata["chunk_id"], d.metadata["source"])
```

### As a QA component
```python
from ragbolt.adapters.langchain_adapter import RagboltLangChainQA

qa = RagboltLangChainQA("corpus.json", provider_name="stub")
result = qa("What is BM25?")

print(result["answer"])
print(result["outcome"])
print(result["sources"])
```

### With a LangChain chain
```python
from ragbolt.adapters.langchain_adapter import RagboltRetriever
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

retriever = RagboltRetriever("corpus.json")
chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    retriever=retriever.as_langchain_retriever(),
)

response = chain({"query": "What is BM25?"})
print(response)
```

Note: this pattern requires `langchain` and an LLM provider package separately.

## LlamaIndex

### As a retriever
```python
from ragbolt.adapters.llamaindex_adapter import RagboltLlamaIndexRetriever

retriever = RagboltLlamaIndexRetriever("corpus.json")
llama_retriever = retriever.as_llama_retriever()
nodes = llama_retriever.retrieve("What is BM25?")

for n in nodes:
    print(n.node.text[:80], n.score)
```

### As a query engine
```python
from ragbolt.adapters.llamaindex_adapter import RagboltQueryEngine

engine = RagboltQueryEngine("corpus.json", provider_name="stub")
response = engine.query("What is BM25?")

print(response.response)
print(response.metadata["outcome"])
```

### With a RetrieverQueryEngine
```python
from ragbolt.adapters.llamaindex_adapter import RagboltLlamaIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

retriever = RagboltLlamaIndexRetriever("corpus.json")
engine = RetrieverQueryEngine.from_args(retriever=retriever.as_llama_retriever())

response = engine.query("What is BM25?")
print(response)
```

## RagboltResponse fields

| Field | Type | Description |
| --- | --- | --- |
| `query` | `str` | Original user query passed into adapter `query()`. |
| `response` | `str \| None` | Final generated answer text, or `None` if abstained/failed. |
| `outcome` | `str` | Final decision outcome (`ACCEPTED`, `REPAIRED_ACCEPTED`, `ABSTAINED`, `FAILED`). |
| `run_id` | `str` | Trace run identifier generated during adapter execution. |
| `corpus_id` | `str` | Corpus identifier loaded from the corpus file stem. |
| `chunks` | `list[Chunk]` | Retrieved chunks attached by the adapter response object. |
| `chunk_ids` | `list[str]` | Retrieved chunk identifiers used in the run. |
| `failure_classes` | `list[str]` | Failure classes observed in order during execution. |
| `repair_attempts` | `int` | Number of repair attempts executed. |
| `top_score` | `float` | Retrieval score used for threshold decisions. |
| `unsupported_ratio` | `float` | Verifier-estimated unsupported response ratio. |
| `metadata` | `dict[str, Any]` | Framework-specific extras passed through adapter calls. |

## Provider and verifier options

| Option | Value | Description |
| --- | --- | --- |
| `provider_name` | `stub` | Deterministic local generator for testing and offline runs. |
| `provider_name` | `anthropic` | Calls Anthropic Messages API with ragbolt prompt constraints. |
| `provider_name` | `openai` | Calls OpenAI Chat Completions API with ragbolt prompt constraints. |
| `verifier_name` | `stub` | Lexical overlap verifier with lightweight heuristic grounding checks. |
| `verifier_name` | `production` | NLI-based verifier with lexical fallback when model is unavailable. |
