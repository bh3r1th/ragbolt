from pathlib import Path
from typing import Any

from ragbolt.adapters.base import BaseRagboltAdapter, RagboltResponse
from ragbolt.core.policy import Chunk


class RagboltRetriever(BaseRagboltAdapter):
    """
    ragbolt as a LangChain BaseRetriever drop-in.

    Usage:
        from ragbolt.adapters.langchain_adapter import RagboltRetriever
        retriever = RagboltRetriever("corpus.json")
        docs = retriever.get_relevant_documents("my query")

        # Or as part of a chain:
        from langchain.chains import RetrievalQA
        from langchain_openai import ChatOpenAI
        chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(),
            retriever=retriever,
        )
    """

    def __init__(
        self,
        corpus_path,
        config=None,
        provider_name="stub",
        verifier_name="stub",
        retriever_name="bm25",
    ):
        super().__init__(corpus_path, config, provider_name, verifier_name, retriever_name)
        self._lc_retriever = None

    def _chunk_to_document(self, chunk: Chunk) -> Any:
        """Convert ragbolt Chunk to LangChain Document."""
        try:
            from langchain_core.documents import Document
        except ImportError:
            raise ImportError(
                "LangChain adapter requires langchain-core. "
                "Install with: pip install langchain-core"
            )
        return Document(
            page_content=chunk.text,
            metadata={
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                **chunk.metadata,
            },
        )

    def get_relevant_documents(self, query: str) -> list[Any]:
        """
        LangChain BaseRetriever interface.
        Returns list[Document] — ragbolt retrieves and verifies,
        returns only grounded chunks.
        """
        chunks = self.retrieve(query)
        return [self._chunk_to_document(c) for c in chunks]

    async def aget_relevant_documents(self, query: str) -> list[Any]:
        """Async variant — runs sync version in thread pool."""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_relevant_documents, query)

    def as_langchain_retriever(self) -> Any:
        """
        Return a proper LangChain BaseRetriever instance that wraps this adapter.
        Use this when a chain requires an exact BaseRetriever subclass.
        """
        try:
            from langchain_core.retrievers import BaseRetriever
            from langchain_core.documents import Document
            from langchain_core.callbacks import CallbackManagerForRetrieverRun
        except ImportError:
            raise ImportError(
                "LangChain adapter requires langchain-core. "
                "Install with: pip install langchain-core"
            )

        adapter = self

        class _WrappedRetriever(BaseRetriever):
            def _get_relevant_documents(
                self,
                query: str,
                *,
                run_manager: CallbackManagerForRetrieverRun,
            ) -> list[Document]:
                return adapter.get_relevant_documents(query)

            async def _aget_relevant_documents(
                self,
                query: str,
                *,
                run_manager: CallbackManagerForRetrieverRun,
            ) -> list[Document]:
                return await adapter.aget_relevant_documents(query)

        return _WrappedRetriever()


class RagboltLangChainQA(BaseRagboltAdapter):
    """
    ragbolt as a complete LangChain-compatible QA component.
    Runs the full ragbolt pipeline (retrieve + generate + verify + repair).
    Returns RagboltResponse but also exposes __call__ for chain compatibility.

    Usage:
        qa = RagboltLangChainQA("corpus.json", provider_name="anthropic")
        result = qa("what is BM25?")
        print(result["answer"])     # response text
        print(result["outcome"])    # ACCEPTED / REPAIRED_ACCEPTED / etc
        print(result["sources"])    # list of chunk_ids
    """

    def __call__(self, query: str | dict, **kwargs) -> dict:
        """
        LangChain chain-compatible __call__.
        Accepts str or dict with "query" key (LangChain passes dicts).
        Returns dict with keys: answer, outcome, sources, run_id, repair_attempts
        """
        if isinstance(query, dict):
            q = query.get("query") or query.get("question") or ""
        else:
            q = query
        response: RagboltResponse = self.query(q, **kwargs)
        return {
            "query": q,
            "answer": response.response or "",
            "outcome": response.outcome,
            "sources": response.chunk_ids,
            "run_id": response.run_id,
            "repair_attempts": response.repair_attempts,
            "failure_classes": response.failure_classes,
        }
