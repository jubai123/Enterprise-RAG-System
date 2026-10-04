"""两个受支持图（minimal / chunkce）的端到端集成：检索 → CE → 上下文 → 生成。"""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.retrieval.vector_retriever import (
    SimilarityCandidateRetriever,
    build_parent_document_store,
)
from tests.rag_test_support import build_test_graph, graph_config


class _FakeVectorStore:
    def __init__(self, document: Document) -> None:
        self.document = document

    def similarity_search(self, query: str, **kwargs):
        return [self.document]


def _retriever(document: Document) -> SimilarityCandidateRetriever:
    return SimilarityCandidateRetriever.model_construct(
        vector_store=_FakeVectorStore(document),
        candidate_k=10,
    )


def test_minimal_graph_runs_ce_then_parent_expansion_then_generate() -> None:
    document = Document(
        page_content="The limit is 10 MiB.",
        metadata={
            "dsid": "a",
            "chunk_id": "a::description::0",
            "title": "Upload limit",
        },
    )
    retriever = _retriever(document)
    llm = FakeListChatModel(
        responses=[
            '{"strategy":"single","requirements":["limit"]}',
            "The limit is 10 MiB.",
        ]
    )
    graph = build_test_graph(
        retriever,
        llm,
        build_parent_document_store(retriever.invoke("seed")),
        graph_config(
            retrieval={
                "max_queries": 5,
                "max_documents": 10,
                "candidate_documents": 10,
            }
        ),
    )

    state = graph.invoke({"question": "What is the upload limit?"})

    assert state["answer"] == "The limit is 10 MiB."
    assert state["document_ids"] == ["a"]
    assert state["executed_queries"] == ["What is the upload limit?"]
    assert state["answer_docs"]  # 父展开后的生成上下文非空
    assert state["evidence_sufficient"] is True


def test_chunkce_graph_scores_raw_chunks_without_rrf_pooling() -> None:
    document = Document(
        page_content="The exact limit is 10 MiB.",
        metadata={
            "dsid": "a",
            "chunk_id": "a::description::0",
            "title": "Upload limit",
        },
    )
    retriever = _retriever(document)
    llm = FakeListChatModel(
        responses=[
            '{"strategy":"single","requirements":["exact limit"]}',
            "The exact limit is 10 MiB.",
        ]
    )
    graph = build_test_graph(
        retriever,
        llm,
        {},
        graph_config(
            retrieval={"max_queries": 5, "max_documents": 10, "candidate_documents": 10},
            graph={"mode": "chunkce"},
        ),
    )

    state = graph.invoke({"question": "What is the exact limit?"})

    assert state["answer"] == "The exact limit is 10 MiB."
    assert state["document_ids"] == ["a"]
    assert state["rerank_history"][0]["mode"] == "chunk_ce"
    assert [chunk.page_content for chunk in state["answer_docs"]] == [
        "The exact limit is 10 MiB."
    ]
