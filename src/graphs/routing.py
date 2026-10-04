"""Per-intent routing table for soft-routing the retrieval chain.

Each query intent (simple/colloquial/multi_perspective/composite) maps to a
concrete plan: which retrieval strategy to use, how many queries and documents
to budget, and whether the queries get rewritten.

planner 从 `IntentRoute` 取 strategy / source_scope / max_queries /
document_budget / minimum_documents / rewrite 六项；其中 rewrite 经
`plan["routing"]["rewrite"]` 在 normalize_plan 里被读取（口语化问题改写查询）。
"""
from __future__ import annotations

from dataclasses import dataclass

Intent = str

ROUTING_TOGGLE_KEYS = ("rewrite",)


@dataclass(frozen=True)
class IntentRoute:
    strategy: str
    source_scope: str
    max_queries: int
    document_budget: int
    minimum_documents: int
    # 查询改写：口语化问题先恢复精确措辞再检索。
    rewrite: bool = False

    @property
    def routing(self) -> dict[str, bool]:
        return {"rewrite": self.rewrite}


INTENT_ROUTING: dict[str, IntentRoute] = {
    # A 简单事实单跳：单 gold 文档，budget=1。
    "simple": IntentRoute(
        strategy="single",
        source_scope="single_source",
        max_queries=1,
        document_budget=1,
        minimum_documents=1,
    ),
    # B 口语化/模糊：改写查询恢复精确措辞后单文档检索；budget 2 给改写丢保真度留余量。
    "colloquial": IntentRoute(
        strategy="single",
        source_scope="single_source",
        max_queries=2,
        document_budget=2,
        minimum_documents=1,
        rewrite=True,
    ),
    # C 多视角/歧义：多查询、多预算，semantic 类问题的重灾区。
    "multi_perspective": IntentRoute(
        strategy="semantic",
        source_scope="single_source",
        max_queries=4,
        document_budget=4,
        minimum_documents=2,
    ),
    # D 复合多跳：跨源多文档，查询与预算都放宽到最大档。
    "composite": IntentRoute(
        strategy="multi_document",
        source_scope="multiple_sources",
        max_queries=8,
        document_budget=8,
        minimum_documents=3,
    ),
}
