"""父块索引：把 (dsid, section) 的多个子切片重建成长度受控的「父块」。

动机（2026-09-09 chunkce 深挖）：决定性答案常埋在某个 (dsid, section) 里，而检索只命中
同一 section 的其他子切片——chunk 级漏，不是 CE 判错。把命中切片展开到其父块（含未命中的
兄弟切片），让父块成为 CE/去噪/生成的单位即可捞回。

设计约束：
- 纯离线变换：输入 = 现成 manifest 叶子（chunk_id = "{dsid}::{section}::{n}"，n 为该
  section 内序号），不复跑 parse_documents、不动 Qdrant/embedding。
- 父块长度受控：同一 section 的叶子按 chunk_field_index 序拼成一块，拼到将超 parent_max_chars
  就封块另起；单叶本就超限的极端独立成块（唯一允许的超限父块）。
- 运行时 chunk→父块 一对一映射由父块记录的 child_chunk_ids 反推，不另存映射表。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document

_SOURCE_META_KEYS = ("source_type", "relative_path", "title", "context_header")
_JOIN_SEP = "\n\n"


def strip_chunk_header(
    text: str,
    section: str,
    chunk_field_index: int,
    context_header: str | None,
) -> str:
    """剥掉 split_documents 加的确定性前缀，取回纯正文。

    前缀只在 context_header 存在时被写入（parse_documents.py:554-561）。剥离按字面前缀精确
    匹配——前缀不存在（无 header 文档）时原样返回；正文中再次出现的 Section:/Chunk: 不剥。
    """
    header = context_header or ""
    prefix = f"{header}\n\nSection: {section}\nChunk: {chunk_field_index}\n\n"
    if not header or not text.startswith(prefix):
        return text
    return text[len(prefix) :]


@dataclass(frozen=True)
class ParentBlock:
    parent_id: str
    dsid: str
    section: str
    text: str
    child_chunk_ids: tuple[str, ...]

    @property
    def chars(self) -> int:
        return len(self.text)

    def to_record(self) -> dict:
        return {
            "parent_id": self.parent_id,
            "dsid": self.dsid,
            "section": self.section,
            "text": self.text,
            "child_chunk_ids": list(self.child_chunk_ids),
            "chars": self.chars,
        }

    @classmethod
    def from_record(cls, record: dict) -> ParentBlock:
        return cls(
            parent_id=record["parent_id"],
            dsid=record["dsid"],
            section=record["section"],
            text=record["text"],
            child_chunk_ids=tuple(record["child_chunk_ids"]),
        )

    def to_document(self, seed_meta: dict | None = None) -> Document:
        meta: dict = {}
        for key in _SOURCE_META_KEYS:
            if seed_meta and seed_meta.get(key):
                meta[key] = seed_meta[key]
        meta.update(
            {
                "dsid": self.dsid,
                "section": self.section,
                "chunk_id": self.parent_id,
                "parent_block_id": self.parent_id,
                "child_chunk_ids": list(self.child_chunk_ids),
                "kind": "parent_block",
            }
        )
        return Document(page_content=self.text, metadata=meta)


def _child_index(chunk: Document) -> tuple[str, str, int]:
    md = chunk.metadata
    dsid = str(md.get("dsid", ""))
    section = str(md.get("section", "text"))
    index = md.get("chunk_field_index")
    if index is None:
        cid = str(md.get("chunk_id", ""))
        tail = cid.rsplit("::", 1)[-1]
        index = int(tail) if tail.isdigit() else 0
    return dsid, section, int(index)


def build_parent_blocks(
    documents: Iterable[Document],
    max_chars: int,
) -> list[ParentBlock]:
    """把 manifest 叶子按 (dsid, section) 收拢、按 chunk_field_index 排序、贪心封块。

    max_chars 是正文拼接上限（含块内分隔符）；单叶正文本身就 > max_chars 时该叶独立成块。
    """
    groups: dict[tuple[str, str], list[tuple[int, str, str]]] = {}
    for chunk in documents:
        md = chunk.metadata
        cid = str(md.get("chunk_id", ""))
        if not cid:
            continue
        dsid, section, index = _child_index(chunk)
        body = strip_chunk_header(
            chunk.page_content,
            section,
            index,
            md.get("context_header"),
        )
        groups.setdefault((dsid, section), []).append((index, cid, body))

    blocks: list[ParentBlock] = []
    for (dsid, section) in sorted(groups):
        entries = sorted(groups[(dsid, section)])
        cur_ids: list[str] = []
        cur_parts: list[str] = []
        cur_chars = 0
        part_count = 0
        flushed = 0

        def flush() -> None:
            nonlocal cur_ids, cur_parts, cur_chars, part_count, flushed
            if not cur_ids:
                return
            blocks.append(
                ParentBlock(
                    parent_id=f"{dsid}::{section}::block_{flushed}",
                    dsid=dsid,
                    section=section,
                    text=_JOIN_SEP.join(cur_parts),
                    child_chunk_ids=tuple(cur_ids),
                )
            )
            flushed += 1
            cur_ids, cur_parts = [], []
            cur_chars = 0
            part_count = 0

        for _index, cid, body in entries:
            added = cur_chars + len(body) + (2 if part_count else 0)
            if cur_ids and added > max_chars:
                flush()
            cur_ids.append(cid)
            cur_parts.append(body)
            cur_chars += len(body) + (2 if part_count else 0)
            part_count += 1
        flush()
    return blocks


def write_parent_blocks(path: str | Path, blocks: Iterable[ParentBlock]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for block in blocks:
            file.write(json.dumps(block.to_record(), ensure_ascii=False) + "\n")
    return path


def load_parent_blocks(path: str | Path) -> list[ParentBlock]:
    blocks: list[ParentBlock] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                blocks.append(ParentBlock.from_record(json.loads(line)))
    return blocks


def summarize(blocks: Iterable[ParentBlock], max_chars: int) -> dict:
    """构建脚本用统计：块数/字符分布/单叶超限（唯一允许超限的父块）数。"""
    block_count = 0
    chars_sum = 0
    chars_max = 0
    oversized_single = 0
    multi_leaf = 0
    leaf_count = 0
    for block in blocks:
        block_count += 1
        chars_sum += block.chars
        chars_max = max(chars_max, block.chars)
        leaf_count += len(block.child_chunk_ids)
        child_count = len(block.child_chunk_ids)
        if child_count > 1:
            multi_leaf += 1
        if child_count == 1 and block.chars > max_chars:
            oversized_single += 1
    return {
        "blocks": block_count,
        "leaves": leaf_count,
        "mean_chars": round(chars_sum / block_count, 1) if block_count else 0,
        "max_chars": chars_max,
        "multi_leaf_blocks": multi_leaf,
        "oversized_single_blocks": oversized_single,
    }


class ParentBlockIndex:
    """运行时查询：chunk_id → 父块 Document。

    不命中（语料与 artifact 漂移）时原样回退该叶子，保证展开永不丢内容。
    """

    def __init__(self, blocks: Iterable[ParentBlock]) -> None:
        self._by_id: dict[str, ParentBlock] = {}
        self._block_of: dict[str, str] = {}
        for block in blocks:
            self._by_id[block.parent_id] = block
            for child in block.child_chunk_ids:
                self._block_of[child] = block.parent_id

    @classmethod
    def from_path(cls, path: str | Path) -> ParentBlockIndex:
        return cls(load_parent_blocks(path))

    @property
    def block_count(self) -> int:
        return len(self._by_id)

    def parent_id_of(self, chunk_id: str) -> str | None:
        return self._block_of.get(chunk_id)

    def expand_chunks(self, chunks: Iterable[Document]) -> list[Document]:
        """命中切片 → 父块 Document，按父块去重、保首见序；未命中切片原样保留。"""
        expanded: list[Document] = []
        seen: set[str] = set()
        for chunk in chunks:
            cid = str(chunk.metadata.get("chunk_id", ""))
            block = self._by_id.get(self._block_of.get(cid, ""))
            if block is None:
                if cid and cid not in seen:
                    seen.add(cid)
                    expanded.append(chunk)
                continue
            if block.parent_id in seen:
                continue
            seen.add(block.parent_id)
            expanded.append(block.to_document(seed_meta=dict(chunk.metadata)))
        return expanded
