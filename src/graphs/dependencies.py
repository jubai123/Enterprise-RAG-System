"""图构建共用的依赖容器。

图骨架本身在各自的构建器里（均接受 `(config, dependencies)`）：
- `src/graphs/minimal.py`  — 3 阶段最小管线，`graph.mode: minimal`（默认）
- `src/graphs/chunkce.py`  — 叶级 CE 干净路径，`graph.mode: chunkce`
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever

from src.retrieval.cross_encoder import CrossEncoderScorer


@dataclass(frozen=True)
class RagDependencies:
    llm: BaseChatModel
    retriever: BaseRetriever
    parent_documents: dict[str, list[Document]]
    cross_encoder: CrossEncoderScorer
    # 仅 chunkce 路径消费：索引层预建的父块 artifact。None = 退化为 identity 展开（无兄弟上下文）。
    parent_blocks: Any | None = None
