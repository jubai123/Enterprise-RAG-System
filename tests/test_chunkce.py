from __future__ import annotations

from langchain_core.documents import Document

from src.graphs.chunkce import flatten_chunks, fuse_chunks_ce_node


class _FixedChunkScorer:
    """按 chunk_id 返回确定得分的 fake chunk-CE。"""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[list[str]] = []

    def score_chunks(self, question, chunks):
        self.calls.append([str(chunk.metadata.get("chunk_id")) for chunk in chunks])
        return [self.scores.get(str(chunk.metadata.get("chunk_id")), 0.0) for chunk in chunks]


def _chunk(dsid: str, index: int, content: str | None = None) -> Document:
    return Document(
        page_content=content or f"{dsid} body {index}",
        metadata={"dsid": dsid, "chunk_id": f"{dsid}::{index}"},
    )


def _chunk_ids(documents: list[Document]) -> list[str]:
    return [str(doc.metadata["chunk_id"]) for doc in documents]


def test_flatten_chunks_dedupes_across_queries() -> None:
    c1 = _chunk("d1", 0)
    c2 = _chunk("d1", 1)
    c3 = _chunk("d2", 0)
    pooled = flatten_chunks([[c1, c2], [c2, c3]])

    assert _chunk_ids(pooled) == ["d1::0", "d1::1", "d2::0"]


def test_node_empty_retrieval_does_not_crash() -> None:
    node = fuse_chunks_ce_node(scorer=_FixedChunkScorer({}), pool_cap=None)

    state = node({"question": "question", "query_results": []})

    assert state["answer_docs"] == []
    assert state["selected_document_ids"] == []
    assert state["rerank_history"][0]["pool_size"] == 0


def test_node_pool_cap_bounds_scored_set() -> None:
    docs = {
        "d1": [_chunk("d1", 0, "a"), _chunk("d1", 1, "b")],
        "d2": [_chunk("d2", 0, "c"), _chunk("d2", 1, "d")],
        "d3": [_chunk("d3", 0, "e"), _chunk("d3", 1, "f")],
    }
    all_chunks = [c for chunks in docs.values() for c in chunks]
    scorer = _FixedChunkScorer(
        {str(c.metadata["chunk_id"]): float(100 - i) for i, c in enumerate(all_chunks)}
    )
    node = fuse_chunks_ce_node(scorer=scorer, pool_cap=3)

    state = node({"question": "question", "query_results": [all_chunks]})

    assert state["rerank_history"][0]["pool_size"] == 3
    assert len(scorer.calls[0]) == 3  # 只对封顶后的池打分


def _parent_block(dsid: str, block_id: str, child_chunk_ids, text: str) -> Document:
    return Document(
        page_content=text,
        metadata={
            "dsid": dsid,
            "chunk_id": block_id,
            "parent_block_id": block_id,
            "child_chunk_ids": list(child_chunk_ids),
            "kind": "parent_block",
        },
    )


class _BlockIndexStub:
    """chunk_id → 父块 Document 的映射 stub（模拟 ParentBlockIndex.expand_chunks）。"""

    def __init__(self, child_to_block: dict[str, Document]) -> None:
        self.child_to_block = child_to_block

    def expand_chunks(self, chunks):
        expanded: list[Document] = []
        seen: set[str] = set()
        for chunk in chunks:
            block = self.child_to_block.get(str(chunk.metadata["chunk_id"]))
            if block is None:
                continue
            if block.metadata["parent_block_id"] in seen:
                continue
            seen.add(block.metadata["parent_block_id"])
            expanded.append(block)
        return expanded


def test_node_per_query_budget_is_not_global() -> None:
    # expand_after_ce_top_k 是**每子查询**预算：两个子查询各命中不同父块 → 两块都进。
    # 若 K 是全局预算，K=1 时只取全局 top-1 叶，第二个子查询的块会被丢掉。
    pb_a = _parent_block("d1", "d1::desc::block_0", ["d1::0", "d1::1"], "block a body")
    pb_b = _parent_block("d2", "d2::s::block_0", ["d2::0"], "block b body")
    index = _BlockIndexStub({"d1::0": pb_a, "d1::1": pb_a, "d2::0": pb_b})
    a0, a1 = _chunk("d1", 0, "a0"), _chunk("d1", 1, "a1")
    b0 = _chunk("d2", 0, "b0")
    scorer = _FixedChunkScorer({"d1::0": 0.9, "d1::1": 0.5, "d2::0": 0.2})
    node = fuse_chunks_ce_node(
        scorer=scorer,
        pool_cap=None,
        parent_blocks=index,
        expand_after_ce_top_k=1,
    )

    state = node(
        {
            "question": "question",
            # 子查询1 → 块A；子查询2 → 块B
            "query_results": [[a0, a1], [b0]],
        }
    )

    # 每子查询各取 top-1 叶 → 块A(0.9) + 块B(0.2)，按全局最佳叶 CE 序输出。
    assert [d.metadata["parent_block_id"] for d in state["answer_docs"]] == [
        pb_a.metadata["parent_block_id"],
        pb_b.metadata["parent_block_id"],
    ]
    assert state["selected_document_ids"] == ["d1", "d2"]
    record = state["rerank_history"][0]
    assert record["pool_size"] == 3  # 打分单位是合并叶池（CE 只打一次分）
    assert record["per_query_leaf_top_k"] == 1
    # chunk_rankings = 全局候选块序（块 rank 派生自块内最佳叶 CE rank），供归因。
    assert [r["parent_block_id"] for r in record["chunk_rankings"]] == [
        pb_a.metadata["parent_block_id"],
        pb_b.metadata["parent_block_id"],
    ]
    assert [r["best_leaf_rank"] for r in record["chunk_rankings"]] == [1, 3]
    assert all(row["kind"] == "parent_block" for row in record["chunk_rankings"])


def test_node_per_query_dedupes_shared_block_across_queries() -> None:
    # 同一父块被两个子查询经不同叶子命中 → 跨子查询合并后只出现一次。
    pb_a = _parent_block("d1", "d1::desc::block_0", ["d1::0", "d1::1"], "block a body")
    index = _BlockIndexStub({"d1::0": pb_a, "d1::1": pb_a})
    a0, a1 = _chunk("d1", 0, "a0"), _chunk("d1", 1, "a1")
    scorer = _FixedChunkScorer({"d1::0": 0.9, "d1::1": 0.5})
    node = fuse_chunks_ce_node(
        scorer=scorer,
        pool_cap=None,
        parent_blocks=index,
        expand_after_ce_top_k=1,
    )
    state = node({"question": "question", "query_results": [[a0], [a1]]})

    assert [d.metadata["parent_block_id"] for d in state["answer_docs"]] == [
        pb_a.metadata["parent_block_id"]
    ]
    assert state["selected_document_ids"] == ["d1"]


def test_node_no_blocks_falls_back_to_top_leaves() -> None:
    # 无父块索引 → expand_chunks 退化为 identity，每叶即自己的「块」。
    # 每子查询各取 top-2 叶，合并后按全局最佳叶 CE 序输出。
    pool_a = [_chunk("d1", 0, "a0"), _chunk("d1", 1, "a1")]
    pool_b = [_chunk("d2", 0, "b0")]
    scorer = _FixedChunkScorer({"d1::0": 0.9, "d2::0": 0.7, "d1::1": 0.1})
    node = fuse_chunks_ce_node(
        scorer=scorer,
        pool_cap=None,
        parent_blocks=None,
        expand_after_ce_top_k=2,
    )

    state = node({"question": "question", "query_results": [pool_a, pool_b]})

    assert _chunk_ids(state["answer_docs"]) == ["d1::0", "d2::0", "d1::1"]
