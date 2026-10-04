from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


KNOWN_QUESTION_TYPES = frozenset(
    {
        "basic",
        "semantic",
        "intra_document_reasoning",
        "project_related",
        "constrained",
        "conflicting_info",
        "completeness",
        "miscellaneous",
        "high_level",
        "info_not_found",
    }
)


class DataConfig(FrozenConfigModel):
    name: str
    source_type: str | None = None
    expected_questions: int = Field(gt=0)
    archives_dir: str
    documents_dir: str
    gold_questions_file: str
    questions_file: str
    manifest_file: str
    # 分层采样配置（全部为 None 时保持旧单源行为）。
    source_scope: list[str] | None = None
    type_targets: dict[str, int] | None = None
    selection_seed: int | None = None
    # chunkce parent_block_llm 预建的父块索引文件路径（parent_blocks.py 产物）。
    # None = 不启用父块展开（旧 config 不受影响）。
    parent_blocks_file: str | None = None

    @model_validator(mode="after")
    def validate_selection(self) -> DataConfig:
        if self.type_targets is None and self.source_scope is None:
            return self
        if not self.source_scope:
            raise ValueError("source_scope is required when type_targets is set")
        if not self.type_targets:
            raise ValueError("type_targets is required when source_scope is set")
        if self.selection_seed is None:
            raise ValueError("selection_seed is required for reproducible stratified sampling")
        if sum(self.type_targets.values()) != self.expected_questions:
            raise ValueError("type_targets must sum to expected_questions")
        unknown = set(self.type_targets) - KNOWN_QUESTION_TYPES
        if unknown:
            raise ValueError(f"unknown question types in type_targets: {sorted(unknown)}")
        return self


class QdrantConfig(FrozenConfigModel):
    url: str
    api_key: str
    collection: str
    vector_size: int = Field(gt=0)
    distance: Literal["Cosine", "Dot", "Euclid", "Manhattan"]


class EmbeddingConfig(FrozenConfigModel):
    provider: str
    model: str
    api_key: str
    base_url: str


class LlmConfig(FrozenConfigModel):
    provider: str
    model: str
    api_key: str
    base_url: str
    temperature: float
    max_tokens: int = Field(gt=0)
    # plan_question 节点的输出上限覆盖；None 回落全局 max_tokens。
    planning_max_tokens: int | None = None
    # deepseek-v4-flash 的扩展思考会先占满 max_tokens 预算（thinking 块不计入 text），
    # 大上下文下输出被截断成空/残缺 JSON。默认关闭思考以保住结构化输出。
    thinking_disabled: bool = True


class RetrievalConfig(FrozenConfigModel):
    # 检索通道：dense / bm25 / rank_sum / rrf（rrf = 混合通道融合，默认）。
    mode: Literal["dense", "bm25", "rank_sum", "rrf"]
    fixed_document_budget: int = Field(gt=0)
    dense_candidate_k: int = Field(gt=0)
    bm25_candidate_k: int = Field(gt=0)
    hybrid_candidate_k: int = Field(gt=0)
    text_section_weight: float
    channel_rrf_k: int = Field(gt=0)
    max_queries: int = Field(gt=0)
    query_parallelism: int = Field(gt=0)
    rrf_k: int = Field(gt=0)
    candidate_documents: int = Field(gt=0)
    max_documents: int = Field(gt=0)
    chunks_per_document: int = Field(gt=0)
    # single 策略选档上限（planning.py 用它替代硬编码 budget=1）。默认 1 保持原行为；
    # 放宽到 4 可捞回 CE rank 2-4 的 gold 文档（sb4 采纳档）。
    single_document_budget: int = Field(default=1, gt=0)
    # 每个选中 doc 取多少父 chunk 进生成上下文（minimal 路径的上下文宽度旋钮）。
    max_parent_chunks: int = Field(gt=0)
    # ---- chunkce 路径专属(干净 chunk→CE,见 src/graphs/chunkce.py) ----
    # 检索池 union 去重后的 chunk 数封顶;None = 不封顶(仅防 CE 成本失控,非打分操作)。
    chunk_pool_cap: int | None = Field(default=None, gt=0)
    # 单个父块正文上限(含块内分隔符)。同一 (dsid,section) 叶子按序拼块,将超限就封块另起;
    # 单叶本就超限的极端独立成块。索引层预建父块时消费(scripts/build_parent_blocks.py)。
    parent_max_chars: int = Field(default=6000, gt=0)
    # 叶-CE 后**每子查询**取叶升父块的叶数上限。展开去重使每子查询贡献的父块数 ≤ K,
    # 合并后总块数随子查询数增长。太小(如 4-6)上下文窄,太大可能带进多 section 块。
    expand_after_ce_top_k: int = Field(default=6, gt=0)


class GraphConfig(FrozenConfigModel):
    recursion_limit: int = Field(gt=0)
    # 图变体：
    # - "minimal"（默认）= 3 阶段最小管线（plan → retrieve → CE top-k → 父展开 → generate），
    #   见 src/graphs/minimal.py。
    # - "chunkce" = 叶级 CE 干净路径（plan → retrieve → 叶 CE → 逐子查询 top-K 叶升父块 →
    #   generate），见 src/graphs/chunkce.py，需索引层预建父块 artifact。
    mode: Literal["minimal", "chunkce"] = "minimal"


class ExperimentConfig(FrozenConfigModel):
    variant: str


class FeatureConfig(FrozenConfigModel):
    adaptive_planning: bool
    # 意图路由（软路由）：classify_intent + 每意图检索策略覆盖（src/graphs/routing.py）。
    # planner 是 minimal/chunkce 共用的，改动它会改变两个基线（66.0 / 71.0）的可复现性。
    intent_routing: bool = True
    # answer: 生成后检查答案是否覆盖必需值型信号，缺失则记入 answer_coverage_missing 供诊断。
    answer_coverage_gate: bool = False
    # answer: 生成后检查答案中是否有不在 question ∪ evidence 里的技术标识符（捏造），
    # 记入 answer_grounding_violations 供诊断。
    answer_grounding_gate: bool = True


class CrossEncoderConfig(FrozenConfigModel):
    provider: Literal["local", "online"] = "online"
    model_path: str | None
    model_id: str
    model_tree_sha256: str | None
    device: Literal["cuda", "cpu"] | None
    base_url: str
    api_key: str
    batch_size: int = Field(gt=0)
    predict_timeout_seconds: int = Field(gt=0)
    # 在线 rerank 对瞬时连接错误/429/5xx 的重试：max_retries 为总尝试次数，指数退避基数为秒。
    max_retries: int = Field(default=3, ge=1)
    retry_backoff_seconds: float = Field(default=1.5, ge=0)

    @model_validator(mode="after")
    def validate_relationships(self) -> CrossEncoderConfig:
        if self.provider == "local" and not self.model_path:
            raise ValueError("model_path is required when cross-encoder provider is local")
        return self


class PricingConfig(FrozenConfigModel):
    input_per_million: float | None
    output_per_million: float | None


class OutputConfig(FrozenConfigModel):
    runs_dir: str


class AppConfig(FrozenConfigModel):
    run_name: str
    data: DataConfig
    qdrant: QdrantConfig
    embedding: EmbeddingConfig
    llm: LlmConfig
    retrieval: RetrievalConfig
    graph: GraphConfig
    experiment: ExperimentConfig
    features: FeatureConfig
    cross_encoder: CrossEncoderConfig
    pricing: PricingConfig
    output: OutputConfig
