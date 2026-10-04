from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

from src.config import PricingConfig
from src.evaluation.benchmark_report import write_jsonl


def calculate_model_cost(
    input_tokens: int,
    output_tokens: int,
    pricing: PricingConfig,
) -> float | None:
    input_price = pricing.input_per_million
    output_price = pricing.output_per_million
    if input_price is None or output_price is None:
        return None
    return round(
        input_tokens * float(input_price) / 1_000_000
        + output_tokens * float(output_price) / 1_000_000,
        8,
    )


def build_question_metrics(
    question_id: str,
    state: dict[str, Any],
    pricing: PricingConfig,
) -> dict[str, Any]:
    calls = state.get("model_calls", [])
    input_tokens = sum(int(call.get("input_tokens", 0)) for call in calls)
    output_tokens = sum(int(call.get("output_tokens", 0)) for call in calls)
    node_events = state.get("node_metrics", [])
    rerank_events = state.get("rerank_history", [])
    verification_triggered = False
    verification_group_calls = 0
    verification_final_calls = 0
    verification_search_candidates = 0
    ce_scored_rounds = 0
    ce_initial_scored_rounds = 0
    ce_deep_scored_rounds = 0
    ce_scored_documents = 0
    ce_latency_ms = 0.0
    ce_status_counts: dict[str, int] = {}
    for event in rerank_events:
        verification = event.get("verification")
        if isinstance(verification, dict):
            verification_triggered = (
                verification_triggered
                or bool(verification.get("verification_triggered"))
            )
            verification_group_calls += int(
                verification.get("group_call_count", 0)
            )
            verification_final_calls += int(
                verification.get("final_call_count", 0)
            )
            verification_search_candidates = max(
                verification_search_candidates,
                int(verification.get("search_candidate_count", 0)),
            )
        status = event.get("cross_encoder_status")
        if status is None:
            continue
        ce_status_counts[str(status)] = ce_status_counts.get(str(status), 0) + 1
        scored = int(event.get("cross_encoder_scored_count", 0) or 0)
        ce_latency_ms += float(event.get("cross_encoder_latency_ms") or 0.0)
        if scored <= 0:
            continue
        ce_scored_rounds += 1
        if event.get("retrieval_mode") == "initial":
            ce_initial_scored_rounds += 1
        else:
            ce_deep_scored_rounds += 1
        ce_scored_documents += scored
    return {
        "schema_version": 1,
        "question_id": question_id,
        "total_duration_ms": round(
            sum(float(event.get("duration_ms", 0.0)) for event in node_events), 3
        ),
        "node_metrics": node_events,
        "model_calls": calls,
        "llm_call_count": len(calls),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost": calculate_model_cost(input_tokens, output_tokens, pricing),
        "retrieval_rounds": int(state.get("retrieval_round", 0)),
        "followup_used": int(state.get("retrieval_round", 0)) > 1,
        "composite_rounds": (
            int(state.get("retrieval_round", 0))
            if state.get("plan", {}).get("intent") == "composite"
            else 0
        ),
        "composite_queries": len(state.get("executed_queries", [])),
        "verification": {
            "triggered": verification_triggered,
            "group_call_count": verification_group_calls,
            "final_call_count": verification_final_calls,
            "search_candidate_count": verification_search_candidates,
        },
        "cross_encoder": {
            "scored_rounds": ce_scored_rounds,
            "initial_scored_rounds": ce_initial_scored_rounds,
            "deep_scored_rounds": ce_deep_scored_rounds,
            "scored_documents": ce_scored_documents,
            "latency_ms": round(ce_latency_ms, 3),
            "status_counts": ce_status_counts,
        },
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_question_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [float(row.get("total_duration_ms", 0.0)) for row in rows]
    node_totals: dict[str, float] = {}
    for row in rows:
        for event in row.get("node_metrics", []):
            name = str(event.get("node", "unknown"))
            node_totals[name] = node_totals.get(name, 0.0) + float(
                event.get("duration_ms", 0.0)
            )
    costs = [row.get("cost") for row in rows]
    known_costs = [float(cost) for cost in costs if cost is not None]
    followup_rows = [row for row in rows if row.get("followup_used")]
    single_rows = [row for row in rows if not row.get("followup_used")]

    def average(rows_to_average: list[dict[str, Any]], key: str) -> float | None:
        values = [row.get(key) for row in rows_to_average]
        known = [float(value) for value in values if value is not None]
        return round(statistics.fmean(known), 4) if len(known) == len(values) and known else None

    followup_increment = {
        "duration_ms": (
            round(
                average(followup_rows, "total_duration_ms")
                - average(single_rows, "total_duration_ms"),
                3,
            )
            if followup_rows and single_rows
            else None
        ),
        "total_tokens": (
            round(
                average(followup_rows, "total_tokens")
                - average(single_rows, "total_tokens"),
                3,
            )
            if followup_rows and single_rows
            else None
        ),
        "cost": (
            round(average(followup_rows, "cost") - average(single_rows, "cost"), 8)
            if followup_rows
            and single_rows
            and average(followup_rows, "cost") is not None
            and average(single_rows, "cost") is not None
            else None
        ),
    }
    ce_status_counts: dict[str, int] = {}
    for row in rows:
        for status, count in row.get("cross_encoder", {}).get(
            "status_counts", {}
        ).items():
            ce_status_counts[str(status)] = ce_status_counts.get(str(status), 0) + int(
                count
            )
    return {
        "question_count": len(rows),
        "duration_ms": {
            "p50": round(_percentile(durations, 0.50), 3),
            "p95": round(_percentile(durations, 0.95), 3),
        },
        "node_duration_ms": {
            name: round(value, 3) for name, value in sorted(node_totals.items())
        },
        "llm_calls": sum(int(row.get("llm_call_count", 0)) for row in rows),
        "input_tokens": sum(int(row.get("input_tokens", 0)) for row in rows),
        "output_tokens": sum(int(row.get("output_tokens", 0)) for row in rows),
        "total_cost": round(sum(known_costs), 8) if len(known_costs) == len(rows) else None,
        "followup_questions": sum(bool(row.get("followup_used")) for row in rows),
        "followup_increment": followup_increment,
        "verification": {
            "triggered_questions": sum(
                bool(row.get("verification", {}).get("triggered")) for row in rows
            ),
            "group_calls": sum(
                int(row.get("verification", {}).get("group_call_count", 0))
                for row in rows
            ),
            "final_calls": sum(
                int(row.get("verification", {}).get("final_call_count", 0))
                for row in rows
            ),
        },
        "cross_encoder": {
            "initial_scored_rounds": sum(
                int(row.get("cross_encoder", {}).get("initial_scored_rounds", 0))
                for row in rows
            ),
            "deep_scored_rounds": sum(
                int(row.get("cross_encoder", {}).get("deep_scored_rounds", 0))
                for row in rows
            ),
            "scored_documents": sum(
                int(row.get("cross_encoder", {}).get("scored_documents", 0))
                for row in rows
            ),
            "latency_ms_total": round(
                sum(
                    float(row.get("cross_encoder", {}).get("latency_ms", 0.0))
                    for row in rows
                ),
                3,
            ),
            "status_counts": ce_status_counts,
        },
    }


def write_question_metrics_outputs(
    rows: list[dict[str, Any]], output_dir: str | Path
) -> dict[str, Any]:
    target = Path(output_dir)
    write_jsonl(rows, target / "question_metrics.jsonl")
    summary = summarize_question_metrics(rows)
    (target / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
