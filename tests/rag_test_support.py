from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.config import AppConfig, load_config
from src.graphs.chunkce import build_chunkce_graph
from src.graphs.dependencies import RagDependencies
from src.graphs.minimal import build_minimal_graph


def graph_config(
    *,
    retrieval: dict[str, object] | None = None,
    features: dict[str, object] | None = None,
    graph: dict[str, object] | None = None,
) -> AppConfig:
    raw = load_config().model_dump()
    raw["retrieval"].update(retrieval or {})
    merged_features = dict(features or {})
    # 测试默认走纯 feature 开关（无 per-intent 路由）；需要路由时显式传入
    # intent_routing=True（如 test_planning_regressions 直接调 normalize_plan）。
    merged_features.setdefault("intent_routing", False)
    raw["features"].update(merged_features)
    # graph 覆盖用于选图模式（默认 minimal；chunkce 走叶级 CE 路径）。
    raw["graph"].update(graph or {})
    return AppConfig.model_validate(raw)


class GraphCrossEncoder:
    """确定性假 cross-encoder：全部候选同分，表达「按融合序输出」。

    minimal 路径按 doc 打分（score_documents），chunkce 路径按原始 chunk 打分
    （score_chunks）。
    """

    status = "success"
    last_error = None
    last_latency_ms = 1.0
    last_scored_count = 0

    def score_documents(self, question, documents_by_dsid):
        self.last_scored_count = len(documents_by_dsid)
        return {dsid: 1.0 for dsid in documents_by_dsid}

    def score_chunks(self, question, chunks):
        self.last_scored_count = len(chunks)
        return [1.0 for _ in chunks]


class StaticRetriever(BaseRetriever):
    documents: list[Document]

    def _get_relevant_documents(self, query: str, *, run_manager):
        return self.documents


def judge_state() -> dict:
    document = Document(
        page_content="The exact limit is 10 MiB.",
        metadata={"dsid": "gold", "chunk_id": "gold::description::0"},
    )
    return {
        "question": "What is the exact limit?",
        "plan": {
            "requirements": ["exact limit"],
            "retrieval_tasks": [],
        },
        "retrieved_docs": [document],
        "executed_queries": ["What is the exact limit?"],
        "retrieval_round": 1,
        "model_calls": [],
    }


def build_test_graph(
    retriever,
    llm,
    parent_documents,
    config,
    cross_encoder=None,
    parent_blocks=None,
):
    """按 config.graph.mode 构建被测图：minimal（默认）或 chunkce。"""
    dependencies = RagDependencies(
        llm=llm,
        retriever=retriever,
        parent_documents=parent_documents,
        cross_encoder=cross_encoder or GraphCrossEncoder(),
        parent_blocks=parent_blocks,
    )
    if config.graph.mode == "chunkce":
        return build_chunkce_graph(config, dependencies)
    return build_minimal_graph(config, dependencies)
