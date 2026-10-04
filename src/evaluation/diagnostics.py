from __future__ import annotations

import statistics
from typing import Any, Iterable


def _unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def _stage_document_ids(trace: dict[str, Any]) -> dict[str, list[str]]:
    stages: dict[str, list[str]] = {}
    for query_trace in trace.get("retrieval_stage_history", []):
        for stage, rows in query_trace.get("stages", {}).items():
            current = stages.setdefault(stage, [])
            current.extend(
                str(row.get("dsid")) if isinstance(row, dict) else str(row)
                for row in rows
                if (isinstance(row, str) and row) or (isinstance(row, dict) and row.get("dsid"))
            )
    stages = {name: _unique(ids) for name, ids in stages.items()}
    rerank = trace.get("rerank_history", [])
    if rerank:
        stages["document_fusion"] = _unique(rerank[-1].get("candidate_document_ids", []))
        stages["rerank"] = _unique(rerank[-1].get("selected_document_ids", []))
    stages["final"] = _unique(trace.get("selected_document_ids", []))
    return stages


def build_recall_funnel(
    questions: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    official_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    trace_by_id = {str(row["question_id"]): row for row in traces}
    result_by_id = {
        str(row["question_id"]): row
        for row in (official_results or {}).get("questions", [])
    }
    stage_rows: dict[str, list[dict[str, float]]] = {}
    per_question: list[dict[str, Any]] = []
    for question in questions:
        question_id = str(question["question_id"])
        expected = _unique(str(item) for item in question.get("expected_doc_ids", []))
        if not expected:
            continue
        stages = _stage_document_ids(trace_by_id.get(question_id, {}))
        metrics: dict[str, dict[str, float]] = {}
        expected_set = set(expected)
        for name, ids in stages.items():
            relevant = [item for item in ids if item in expected_set]
            first = next(
                (rank for rank, item in enumerate(ids, start=1) if item in expected_set),
                None,
            )
            metric = {
                "hit": float(bool(relevant)),
                "recall": len(set(relevant)) / len(expected),
                "mrr": 1.0 / first if first else 0.0,
            }
            metrics[name] = metric
            stage_rows.setdefault(name, []).append(metric)

        def stage_recall(stage: str) -> float:
            return len(expected_set & set(stages.get(stage, []))) / len(expected_set)

        answer_correct = bool(result_by_id.get(question_id, {}).get("answer_correct"))
        if "union" in stages and stage_recall("union") < 1.0:
            cause = "initial_recall_miss"
        elif (
            "union" in stages
            and "channel_fusion" in stages
            and stage_recall("channel_fusion") < stage_recall("union")
        ):
            cause = "channel_fusion_drop"
        elif (
            "channel_fusion" in stages
            and "document_fusion" in stages
            and stage_recall("document_fusion") < stage_recall("channel_fusion")
        ):
            cause = "document_fusion_drop"
        elif (
            "document_fusion" in stages
            and "rerank" in stages
            and stage_recall("rerank") < stage_recall("document_fusion")
            and stage_recall("final") < stage_recall("document_fusion")
        ):
            cause = "rerank_elimination"
        elif (
            "rerank" in stages
            and "final" in stages
            and stage_recall("final") < stage_recall("rerank")
        ):
            cause = "post_processing_drop"
        elif result_by_id and not answer_correct:
            cause = "answer_failure"
        else:
            cause = "success_or_unscored"
        per_question.append(
            {
                "question_id": question_id,
                "expected_doc_ids": expected,
                "stages": stages,
                "metrics": metrics,
                "primary_failure_cause": cause,
            }
        )

    overall = {
        stage: {
            metric: round(statistics.fmean(row[metric] for row in rows), 4)
            for metric in ("hit", "recall", "mrr")
        }
        for stage, rows in stage_rows.items()
    }
    return {
        # schema_version 保持 2：full 图独有的 fusion_guard / parent_expansion 阶段在
        # minimal / chunkce 运行里本就从未写入，删掉输出键不改变这两个图的输出形状。
        "schema_version": 2,
        "overall": overall,
        "questions": per_question,
    }
