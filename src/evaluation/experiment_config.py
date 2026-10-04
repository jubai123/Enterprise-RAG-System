from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from src.config import AppConfig


@dataclass(frozen=True)
class VariantSpec:
    section: Literal["retrieval", "features"]
    field: str
    value: Any


# 仅保留 minimal / chunkce 两个 graph.mode 真实开关的单因子消融：检索通道与
# FeatureConfig 现存字段。重排/父展开/证据追补/答案修复等开关已随 full 图下线删除。
ABLATION_VARIANTS: dict[str, VariantSpec | None] = {
    "main": None,
    "dense_only": VariantSpec("retrieval", "mode", "dense"),
    "bm25_only": VariantSpec("retrieval", "mode", "bm25"),
    "hybrid_rank_sum": VariantSpec("retrieval", "mode", "rank_sum"),
    "fixed_planning": VariantSpec("features", "adaptive_planning", False),
    "no_intent_routing": VariantSpec("features", "intent_routing", False),
}


def apply_experiment_variant(config: AppConfig, variant: str) -> AppConfig:
    """返回应用单字段消融后的新配置，不修改原始 AppConfig。"""
    if variant not in ABLATION_VARIANTS:
        raise ValueError(f"Unknown ablation variant: {variant}")
    spec = ABLATION_VARIANTS[variant]
    update: dict[str, Any] = {
        "experiment": config.experiment.model_copy(update={"variant": variant})
    }
    if spec is not None:
        section = getattr(config, spec.section)
        update[spec.section] = section.model_copy(update={spec.field: spec.value})
    return config.model_copy(update=update)
