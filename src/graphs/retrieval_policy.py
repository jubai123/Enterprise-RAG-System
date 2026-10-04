from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Strategy = Literal[
    "single",
    "semantic",
    "multi_document",
    "conflicting",
    "completeness",
]


@dataclass(frozen=True)
class RetrievalPolicy:
    """按策略给出的检索预算；planner 用它把 LLM 给的计划收进硬边界。"""

    max_queries: int
    document_budget: int
    minimum_documents: int
    candidate_documents: int = 24


DEFAULT_RETRIEVAL_POLICIES: dict[Strategy, RetrievalPolicy] = {
    "single": RetrievalPolicy(1, 1, 1),
    "semantic": RetrievalPolicy(3, 4, 1),
    "multi_document": RetrievalPolicy(8, 8, 3),
    "conflicting": RetrievalPolicy(4, 4, 2),
    "completeness": RetrievalPolicy(10, 10, 6, candidate_documents=48),
}
