from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.retrieval.lexical_retriever import LocalBM25Retriever, tokenize_technical_text
from src.retrieval.vector_retriever import (
    HybridCandidateRetriever,
    SimilarityCandidateRetriever,
    reciprocal_rank_fuse,
)


def test_similarity_candidate_retriever_uses_configured_k() -> None:
    class FakeVectorStore:
        def __init__(self) -> None:
            self.call: dict = {}

        def similarity_search(self, query: str, **kwargs):
            self.call = {"query": query, **kwargs}
            return [Document(page_content="a", metadata={"dsid": "a"})]

    store = FakeVectorStore()
    retriever = SimilarityCandidateRetriever.model_construct(
        vector_store=store,
        candidate_k=30,
    )
    assert retriever.invoke("query")[0].page_content == "a"
    assert store.call == {"query": "query", "k": 30}


def test_similarity_candidate_retriever_passes_source_type_filter() -> None:
    class FakeVectorStore:
        def __init__(self) -> None:
            self.call: dict = {}

        def similarity_search(self, query: str, **kwargs):
            self.call = {"query": query, **kwargs}
            return [Document(page_content="a", metadata={"dsid": "a"})]

    store = FakeVectorStore()
    retriever = SimilarityCandidateRetriever.model_construct(
        vector_store=store,
        candidate_k=30,
    )

    retriever.invoke(
        "query",
        config={"metadata": {"source_types": ["github", "jira"]}},
    )

    assert store.call["query"] == "query"
    assert store.call["k"] == 30
    condition = store.call["filter"].must[0]
    assert condition.key == "metadata.source_type"
    assert condition.match.any == ["github", "jira"]


def test_similarity_candidate_retriever_omits_filter_without_source_types() -> None:
    class FakeVectorStore:
        def __init__(self) -> None:
            self.call: dict = {}

        def similarity_search(self, query: str, **kwargs):
            self.call = {"query": query, **kwargs}
            return [Document(page_content="a", metadata={"dsid": "a"})]

    store = FakeVectorStore()
    retriever = SimilarityCandidateRetriever.model_construct(
        vector_store=store,
        candidate_k=30,
    )

    retriever.invoke("query")

    assert store.call == {"query": "query", "k": 30}


def test_bm25_restricts_results_to_declared_source_types() -> None:
    retriever = LocalBM25Retriever(
        documents=[
            Document(
                page_content="workspace_id isolation controls tenant isolation",
                metadata={"dsid": "conf", "source_type": "confluence"},
            ),
            Document(
                page_content="workspace_id",
                metadata={"dsid": "gh1", "source_type": "github"},
            ),
            Document(
                page_content="workspace",
                metadata={"dsid": "gh2", "source_type": "github"},
            ),
        ],
        k=2,
    )

    filtered = retriever.invoke(
        "workspace_id isolation",
        config={"metadata": {"source_types": ["github"]}},
    )
    unfiltered = retriever.invoke("workspace_id isolation")

    # 过滤后：跳过得分最高的 confluence 文档，只返回 github 来源的 top-2。
    assert [document.metadata["dsid"] for document in filtered] == ["gh1", "gh2"]
    assert [document.metadata["dsid"] for document in unfiltered][0] == "conf"


def test_hybrid_forwards_source_types_to_both_channels() -> None:
    captured: dict[str, list] = {"dense": [], "bm25": []}

    class CapturingRetriever(BaseRetriever):
        channel: str

        def _get_relevant_documents(self, query: str, *, run_manager):
            captured[self.channel].append(
                (run_manager.metadata or {}).get("source_types")
            )
            return [Document(page_content="x", metadata={"dsid": "a", "chunk_id": "a::0"})]

    retriever = HybridCandidateRetriever(
        dense_retriever=CapturingRetriever(channel="dense"),
        bm25_retriever=CapturingRetriever(channel="bm25"),
        candidate_k=3,
        rrf_k=60,
    )

    retriever.invoke(
        "query",
        config={"metadata": {"source_types": ["github"]}},
    )
    assert captured["dense"] == [["github"]]
    assert captured["bm25"] == [["github"]]

    retriever.invoke("query")
    assert captured["dense"] == [["github"], None]
    assert captured["bm25"] == [["github"], None]


def test_technical_tokenizer_keeps_identifier_and_parts() -> None:
    assert tokenize_technical_text("workspace_id v2.1") == [
        "workspace_id",
        "workspace",
        "id",
        "v2.1",
        "v2",
        "1",
    ]


def test_local_bm25_retrieves_exact_technical_identifier() -> None:
    retriever = LocalBM25Retriever(
        documents=[
            Document(
                page_content="The workspace_id controls tenant isolation.",
                metadata={"dsid": "exact", "chunk_id": "exact::0"},
            ),
            Document(
                page_content="Workspace setup and user permissions.",
                metadata={"dsid": "broad", "chunk_id": "broad::0"},
            ),
        ],
        k=2,
    )

    results = retriever.invoke("workspace_id")

    assert results[0].metadata["dsid"] == "exact"
    assert results[0].metadata["retrieval_channels"] == ["bm25"]


def test_hybrid_retriever_fuses_dense_and_bm25_channels() -> None:
    class StaticRetriever(BaseRetriever):
        documents: list[Document]

        def _get_relevant_documents(self, query: str, *, run_manager):
            return self.documents

    shared = Document(page_content="shared", metadata={"dsid": "a", "chunk_id": "a::0"})
    dense_only = Document(
        page_content="dense", metadata={"dsid": "b", "chunk_id": "b::0"}
    )
    bm25_only = Document(
        page_content="bm25", metadata={"dsid": "c", "chunk_id": "c::0"}
    )
    retriever = HybridCandidateRetriever(
        dense_retriever=StaticRetriever(documents=[dense_only, shared]),
        bm25_retriever=StaticRetriever(documents=[bm25_only, shared]),
        candidate_k=3,
        rrf_k=60,
    )

    results = retriever.invoke("query")

    assert results[0].metadata["dsid"] == "a"
    assert results[0].metadata["retrieval_channels"] == ["dense", "bm25"]
    assert results[0].metadata["retrieval_channel_ranks"] == {
        "dense": 2,
        "bm25": 2,
    }


def test_rrf_rewards_documents_retrieved_by_multiple_queries() -> None:
    a1 = Document(page_content="a1", metadata={"dsid": "a", "chunk_id": "a1"})
    a2 = Document(page_content="a2", metadata={"dsid": "a", "chunk_id": "a2"})
    b = Document(page_content="b", metadata={"dsid": "b", "chunk_id": "b1"})
    c = Document(page_content="c", metadata={"dsid": "c", "chunk_id": "c1"})

    fused = reciprocal_rank_fuse(
        [[b, a1], [c, a2]],
        max_documents=3,
        chunks_per_document=2,
        rrf_k=60,
    )

    assert list(fused)[0] == "a"
    assert [document.page_content for document in fused["a"]] == ["a1", "a2"]
    assert fused["a"][0].metadata["document_rrf_rank"] == 1
    assert fused["a"][0].metadata["query_hit_count"] == 2


def test_rrf_ranks_by_aggregate_score_then_document() -> None:
    # 融合池只按 RRF 聚合分排序（单查询链只喂一个 query_results；实体链接扩展
    # 与配额保留参数随 full 图下线删除）。
    first = [
        Document(page_content=f"a{i}", metadata={"dsid": f"a{i}", "chunk_id": f"a{i}"})
        for i in range(3)
    ]
    second = [
        Document(page_content=f"b{i}", metadata={"dsid": f"b{i}", "chunk_id": f"b{i}"})
        for i in range(3)
    ]

    fused = reciprocal_rank_fuse(
        [first, second],
        max_documents=4,
        chunks_per_document=1,
    )

    assert list(fused) == ["a0", "b0", "a1", "b1"]


def test_bm25_does_not_double_downweight_text_sections() -> None:
    retriever = LocalBM25Retriever(
        documents=[
            Document(
                page_content="origin verification release index",
                metadata={"dsid": "direct", "section": "description"},
            ),
            Document(
                page_content="origin verification release index",
                metadata={"dsid": "notes", "section": "text"},
            ),
        ],
        k=2,
        text_section_weight=0.8,
    )

    results = retriever.invoke("origin verification release index")

    assert [document.metadata["dsid"] for document in results] == [
        "direct",
        "notes",
    ]
    # BM25 不再应用 section 先验（避免与融合层重复降权 0.8×0.8）；
    # 两篇内容相同、section 不同的文档得分应一致，顺序仅由索引 tie-break 决定。
    assert results[0].metadata["bm25_score"] == results[1].metadata["bm25_score"]

