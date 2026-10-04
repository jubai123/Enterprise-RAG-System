from __future__ import annotations

from typing import Any, Literal, TypedDict

from langchain_core.documents import Document


class RetrievalTask(TypedDict):
    task_id: str
    requirement: str
    slot: str
    query: str


class Plan(TypedDict, total=False):
    strategy: str
    source_scope: str
    intent: str
    routing: dict[str, bool]
    queries: list[str]
    retrieval_tasks: list[RetrievalTask]
    document_budget: int
    minimum_documents: int
    requires_multiple_distinct_documents: bool
    requirements: list[str]


class RerankTrace(TypedDict, total=False):
    round: int
    retrieval_mode: str
    candidate_document_ids: list[str]
    selected_document_ids: list[str]
    cross_encoder_status: str
    cross_encoder_error: str | None
    cross_encoder_latency_ms: float | None
    cross_encoder_scored_count: int
    verification: dict[str, Any]


class ModelCall(TypedDict, total=False):
    node: str
    started_at_utc: str
    ended_at_utc: str
    duration_ms: float
    model: str
    prompt_sha256: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    retry: int
    status: str


class NodeMetric(TypedDict, total=False):
    node: str
    started_at_utc: str
    ended_at_utc: str
    duration_ms: float
    candidate_documents: int
    selected_documents: int
    status: Literal["success", "error"]
    error: str


class RagState(TypedDict, total=False):
    """图在 LangGraph 节点之间传递的状态（minimal / chunkce 共用）。"""

    question: str
    # 问题声明的语料来源范围（github/confluence/jira），用于检索层按 source_type 收窄候选。
    # None = 不过滤（全量检索）。
    source_types: list[str] | None
    plan: Plan
    pending_queries: list[str]
    pending_tasks: list[RetrievalTask]
    executed_queries: list[str]
    executed_tasks: list[RetrievalTask]
    query_results: list[list[Document]]
    base_query_results: list[list[Document]]
    result_tasks: list[RetrievalTask]
    candidate_groups: dict[str, list[Document]]
    selected_document_ids: list[str]
    rerank_history: list[RerankTrace]
    # 检索阶段的逐通道遥测（retrieve_queries 写入，诊断/归因读取）。
    retrieval_stage_history: list[dict[str, Any]]
    retrieval_round: int
    retrieved_docs: list[Document]
    answer_docs: list[Document]
    missing_evidence: list[str]
    evidence_sufficient: bool
    answer: str
    # 生成后闸门记录（见 FeatureConfig.answer_coverage_gate / answer_grounding_gate）。
    answer_coverage_missing: list[str]
    answer_grounding_violations: list[str]
    document_ids: list[str]
    model_calls: list[ModelCall]
    node_metrics: list[NodeMetric]
