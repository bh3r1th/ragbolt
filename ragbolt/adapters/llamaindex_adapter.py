from pathlib import Path
from typing import Any

from ragbolt.adapters.base import BaseRagboltAdapter, RagboltResponse
from ragbolt.core.policy import Chunk


class RagboltLlamaIndexRetriever(BaseRagboltAdapter):
    """
    ragbolt as a LlamaIndex BaseRetriever drop-in.

    Usage:
        from ragbolt.adapters.llamaindex_adapter import RagboltLlamaIndexRetriever
        retriever = RagboltLlamaIndexRetriever("corpus.json")
        nodes = retriever.retrieve("my query")

        # As part of a RetrieverQueryEngine:
        from llama_index.core.query_engine import RetrieverQueryEngine
        engine = RetrieverQueryEngine.from_args(retriever=retriever.as_llama_retriever())
    """

    def _chunk_to_node(self, chunk: Chunk, score: float = 1.0) -> Any:
        """Convert ragbolt Chunk to LlamaIndex NodeWithScore."""
        try:
            from llama_index.core.schema import TextNode, NodeWithScore
        except ImportError:
            raise ImportError(
                "LlamaIndex adapter requires llama-index-core. "
                "Install with: pip install llama-index-core"
            )
        node = TextNode(
            text=chunk.text,
            id_=chunk.chunk_id,
            metadata={
                "source": chunk.source,
                "chunk_id": chunk.chunk_id,
                **chunk.metadata,
            },
        )
        return NodeWithScore(node=node, score=score)

    def retrieve(self, query: str, **kwargs) -> list[Any]:
        """
        LlamaIndex retriever interface.
        Returns list[NodeWithScore].
        """
        from ragbolt.core.bm25 import BM25Runner

        chunks, top_score = BM25Runner(self.corpus, self.config).retrieve(query)
        return [
            self._chunk_to_node(
                chunk,
                score=min(1.0, top_score) if top_score > 0 else 0.5,
            )
            for chunk in chunks
        ]

    def as_llama_retriever(self) -> Any:
        """
        Return a proper LlamaIndex BaseRetriever subclass instance.
        Use this when a query engine requires an exact BaseRetriever subclass.
        """
        try:
            from llama_index.core.retrievers import BaseRetriever
            from llama_index.core.schema import NodeWithScore, QueryBundle
        except ImportError:
            raise ImportError(
                "LlamaIndex adapter requires llama-index-core. "
                "Install with: pip install llama-index-core"
            )

        adapter = self

        class _WrappedRetriever(BaseRetriever):
            def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
                return adapter.retrieve(query_bundle.query_str)

        return _WrappedRetriever()


class RagboltQueryEngine(BaseRagboltAdapter):
    """
    ragbolt as a complete LlamaIndex-compatible query engine.
    Runs the full ragbolt pipeline (retrieve + generate + verify + repair).

    Usage:
        from ragbolt.adapters.llamaindex_adapter import RagboltQueryEngine
        engine = RagboltQueryEngine("corpus.json", provider_name="anthropic")
        response = engine.query("what is BM25?")
        print(response.response)       # answer text
        print(response.metadata)       # outcome, run_id, repair_attempts
    """

    def query(self, query: str, **kwargs) -> Any:
        """
        LlamaIndex query engine interface.
        Returns a LlamaIndex Response object if llama-index-core installed,
        otherwise returns RagboltResponse.
        """
        rb_response: RagboltResponse = super().query(query, **kwargs)
        try:
            from llama_index.core.base.response.schema import Response
            from llama_index.core.schema import NodeWithScore, TextNode

            source_nodes = [
                NodeWithScore(
                    node=TextNode(text=cid, id_=cid),
                    score=rb_response.top_score,
                )
                for cid in rb_response.chunk_ids
            ]
            return Response(
                response=rb_response.response or "",
                source_nodes=source_nodes,
                metadata={
                    "outcome": rb_response.outcome,
                    "run_id": rb_response.run_id,
                    "repair_attempts": rb_response.repair_attempts,
                    "failure_classes": rb_response.failure_classes,
                    "corpus_id": rb_response.corpus_id,
                    "unsupported_ratio": rb_response.unsupported_ratio,
                },
            )
        except ImportError:
            return rb_response

    async def aquery(self, query: str, **kwargs) -> Any:
        """Async variant — runs sync query in thread pool."""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.query, query)
