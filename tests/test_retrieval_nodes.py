from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.retrieval.vector_retriever import HybridCandidateRetriever
from tests.rag_test_support import (
    StaticRetriever,
)


def test_hybrid_retriever_preserves_channel_union_and_fusion_trace() -> None:
    dense = StaticRetriever(
        documents=[
            Document(page_content="dense", metadata={"dsid": "a", "chunk_id": "a::0"})
        ]
    )
    bm25 = StaticRetriever(
        documents=[
            Document(page_content="bm25", metadata={"dsid": "b", "chunk_id": "b::0"})
        ]
    )
    retriever = HybridCandidateRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
        candidate_k=10,
        mode="rrf",
    )

    documents = retriever.invoke("query")
    trace = documents[0].metadata["_retrieval_trace"]

    assert [row["dsid"] for row in trace["stages"]["dense"]] == ["a"]
    assert [row["dsid"] for row in trace["stages"]["bm25"]] == ["b"]
    assert {row["dsid"] for row in trace["stages"]["union"]} == {"a", "b"}
    assert {row["dsid"] for row in trace["stages"]["channel_fusion"]} == {"a", "b"}


def test_dense_only_does_not_call_bm25() -> None:
    class FailingRetriever(BaseRetriever):
        def _get_relevant_documents(self, query: str, *, run_manager):
            raise AssertionError("disabled channel was invoked")

    dense = StaticRetriever(
        documents=[
            Document(page_content="dense", metadata={"dsid": "a", "chunk_id": "a::0"})
        ]
    )
    retriever = HybridCandidateRetriever(
        dense_retriever=dense,
        bm25_retriever=FailingRetriever(),
        mode="dense",
    )

    assert retriever.invoke("query")[0].metadata["dsid"] == "a"
