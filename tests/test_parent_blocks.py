from __future__ import annotations

from langchain_core.documents import Document

from src.ingestion.parent_blocks import (
    ParentBlockIndex,
    build_parent_blocks,
    load_parent_blocks,
    strip_chunk_header,
    write_parent_blocks,
)


def _leaf(
    dsid: str,
    section: str,
    index: int,
    body: str,
    *,
    context_header: str | None = None,
    char_body: str | None = None,
) -> Document:
    header = context_header if context_header is not None else f"title: t\nsection: {section}"
    prefix = f"{header}\n\nSection: {section}\nChunk: {index}\n\n"
    return Document(
        page_content=prefix + body,
        metadata={
            "dsid": dsid,
            "section": section,
            "chunk_id": f"{dsid}::{section}::{index}",
            "chunk_field_index": index,
            "context_header": header,
            "source_type": "github",
            "relative_path": "a/b.json",
            "title": "t",
        },
    )


def test_strip_chunk_header_removes_only_prefix() -> None:
    doc = _leaf("d1", "changes", 3, "real body\n\nSection: changes\nChunk: 9")
    body = strip_chunk_header(
        doc.page_content, "changes", 3, doc.metadata.get("context_header")
    )
    # 只剥最前面的确定性前缀，正文里再出现的 Section:/Chunk: 不动。
    assert body == "real body\n\nSection: changes\nChunk: 9"


def test_strip_chunk_header_no_context_header_returns_unchanged() -> None:
    text = "raw body without prefix"
    assert strip_chunk_header(text, "text", 0, None) == text


def test_build_parent_blocks_joins_sibling_leaves_in_order() -> None:
    docs = [
        _leaf("d1", "description", 1, "body one"),
        _leaf("d1", "description", 0, "body zero"),
        _leaf("d1", "description", 2, "body two"),
        _leaf("d2", "text", 0, "other doc"),
    ]
    blocks = build_parent_blocks(docs, max_chars=1000)

    assert len(blocks) == 2  # d1::description 并成 1 块 + d2::text 单块
    d1 = next(b for b in blocks if b.dsid == "d1")
    assert d1.section == "description"
    assert d1.child_chunk_ids == (
        "d1::description::0",
        "d1::description::1",
        "d1::description::2",
    )
    assert d1.text == "body zero\n\nbody one\n\nbody two"


def test_build_parent_blocks_caps_block_length_and_flushes() -> None:
    body = "x" * 30
    docs = [_leaf("d1", "s", i, body) for i in range(4)]
    # 单块上限约 60: 每加一片会超就封块 → 4 片应分成若干块。
    blocks = build_parent_blocks(docs, max_chars=65)

    assert len(blocks) >= 2
    child_ids = [cid for b in blocks for cid in b.child_chunk_ids]
    assert child_ids == ["d1::s::0", "d1::s::1", "d1::s::2", "d1::s::3"]
    for b in blocks:
        # 多叶块必须 ≤ cap; 单叶块允许 > cap(超长单叶唯一例外)。
        if len(b.child_chunk_ids) > 1:
            assert b.chars <= 65
    # 分块覆盖且互斥。
    flattened = [cid for b in blocks for cid in b.child_chunk_ids]
    assert len(flattened) == len(set(flattened)) == 4


def test_index_maps_chunk_to_block_and_expands_deduped() -> None:
    docs = [
        _leaf("d1", "s", 0, "b0"),
        _leaf("d1", "s", 1, "b1"),
        _leaf("d2", "t", 0, "other"),
    ]
    index = ParentBlockIndex(build_parent_blocks(docs, max_chars=1000))

    assert index.block_count == 2
    assert index.parent_id_of("d1::s::0") == index.parent_id_of("d1::s::1")

    # 顺序:同一块的兄弟切片应去重为 1 个父块,未命中切片原样回退。
    d2_leaf = docs[2]
    expanded = index.expand_chunks(
        [
            docs[0],
            docs[0],  # 同一片重复
            docs[1],  # 兄弟片 → 并入同父块
            d2_leaf,  # 命中单块
            Document(
                page_content="orphan",
                metadata={"chunk_id": "missing::x::0", "dsid": "missing"},
            ),  # 未命中 → 原样
        ]
    )
    assert len(expanded) == 3
    first = expanded[0]
    assert first.metadata["kind"] == "parent_block"
    assert first.metadata["parent_block_id"].startswith("d1::s::block_")
    assert first.metadata["child_chunk_ids"] == ["d1::s::0", "d1::s::1"]
    assert first.metadata["source_type"] == "github"  # 溯源键从叶子抄
    assert expanded[1].metadata["dsid"] == "d2"
    assert expanded[2].metadata["chunk_id"] == "missing::x::0"  # 原样回退


def test_serialization_roundtrip(tmp_path) -> None:
    docs = [_leaf("d1", "s", 0, "b0"), _leaf("d1", "s", 1, "b1")]
    blocks = build_parent_blocks(docs, max_chars=1000)
    path = tmp_path / "parent_blocks.jsonl"
    write_parent_blocks(path, blocks)

    reloaded = load_parent_blocks(path)

    assert [b.to_record() for b in reloaded] == [b.to_record() for b in blocks]
