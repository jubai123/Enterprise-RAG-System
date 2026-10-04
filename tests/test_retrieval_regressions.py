from __future__ import annotations

from langchain_core.documents import Document

from src.graphs.retrieval_nodes import expand_selected_documents_query_aware


def test_query_aware_parent_expansion_selects_deep_relevant_chunk() -> None:
    selected = Document(
        page_content="partial evidence",
        metadata={"dsid": "gold", "chunk_id": "gold::0"},
    )
    parent_chunks = [
        Document(
            page_content="Overview of unrelated batching migration.",
            metadata={"dsid": "gold", "chunk_id": "gold::1"},
        ),
        Document(
            page_content="The exact version 3.2.1 is enabled by --fast.",
            metadata={"dsid": "gold", "chunk_id": "gold::2"},
        ),
    ]

    expanded = expand_selected_documents_query_aware(
        ["gold"],
        {"gold": [selected]},
        {"gold": [*parent_chunks, selected]},
        question="Which flag enables version 3.2.1?",
        requirements=["flag"],
        max_parent_chunks=2,
    )

    # 旧逻辑按存储顺序取 doc::0 + doc::1；新逻辑必须 doc::0 + doc::2
    assert [document.metadata["chunk_id"] for document in expanded] == [
        "gold::0",
        "gold::2",
    ]


def test_parent_expansion_without_stored_parents_keeps_candidates() -> None:
    selected = Document(
        page_content="partial evidence",
        metadata={"dsid": "gold", "chunk_id": "gold::0"},
    )

    expanded = expand_selected_documents_query_aware(
        ["gold"],
        {"gold": [selected]},
        {},
        question="What is the limit?",
        requirements=["limit"],
        max_parent_chunks=8,
    )

    assert [document.metadata["chunk_id"] for document in expanded] == ["gold::0"]
