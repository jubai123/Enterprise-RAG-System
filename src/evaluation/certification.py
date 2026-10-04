from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from src.config import AppConfig
from src.evaluation.benchmark_report import load_jsonl
from src.evaluation.reproducibility import (
    _external_git_commit,
    _git_state,
    sha256_file,
    sha256_tree,
    source_tree_sha256,
)

REQUIRED_PACKAGE_VERSIONS = {
    "langchain": "1.3.12",
    "langchain-core": "1.4.9",
    "langgraph": "1.2.8",
    "qdrant-client": "1.18.0",
}
# 仅本地 cross-encoder 需要；在线重排序（provider=online）不依赖 torch/sentence-transformers。
LOCAL_ONLY_PACKAGE_VERSIONS = {
    "sentence-transformers": "5.6.1",
    # 无 NVIDIA GPU 的本机使用 CPU 版 torch；CUDA 环境仍要求 +cu121。
    "torch": ["2.5.1+cu121", "2.5.1+cpu"],
}
ALLOWED_CE_STATUSES = {"skipped_initial_round", "success"}


def _project_file(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _gold_document_coverage(
    config: AppConfig, root: Path
) -> dict[str, Any] | None:
    """核对分层采样集的 gold 文档 dsid 是否已存在于解析出的 source 文件。

    仅当 data.source_scope 非空时启用；读 gold_questions_file 而不是 blind 文件
    （blind 文件已剔除 expected_doc_ids）。dsid 前缀与 source 文件名一一对应。
    """
    data = config.data
    if data.source_scope is None:
        return None
    gold_path = _project_file(root, data.gold_questions_file)
    blind_path = _project_file(root, data.questions_file)
    if not gold_path.is_file() or not blind_path.is_file():
        return None
    blind_ids = {str(row.get("question_id")) for row in load_jsonl(blind_path)}
    gold_ids: set[str] = set()
    for row in load_jsonl(gold_path):
        if str(row.get("question_id")) in blind_ids:
            gold_ids.update(str(item) for item in row.get("expected_doc_ids") or [])
    if not gold_ids:
        return {"covered": 0, "total": 0, "missing": [], "note": "no gold documents"}
    documents_root = Path(data.documents_dir)
    if not documents_root.is_dir():
        return {
            "covered": 0,
            "total": len(gold_ids),
            "missing": sorted(gold_ids),
            "note": f"documents dir not found: {documents_root}",
        }
    found: set[str] = set()
    for file in documents_root.rglob("*"):
        if file.is_file() and file.name.startswith("dsid_"):
            found.add(file.name.split("__", 1)[0])
    missing = sorted(gold_ids - found)
    return {
        "covered": len(gold_ids) - len(missing),
        "total": len(gold_ids),
        "missing": missing,
    }


def build_preflight_report(
    config: AppConfig,
    *,
    project_root: str | Path,
    require_clean: bool = True,
    check_qdrant: bool = True,
) -> dict[str, Any]:
    """检查正式认证所需的本机环境、模型、数据、索引与源码状态。"""
    root = Path(project_root).resolve()
    failures: list[str] = []
    package_versions: dict[str, str | None] = {}
    package_checks = dict(REQUIRED_PACKAGE_VERSIONS)
    if config.cross_encoder.provider == "local":
        package_checks.update(LOCAL_ONLY_PACKAGE_VERSIONS)
    for package, expected in package_checks.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        package_versions[package] = actual
        allowed = [expected] if isinstance(expected, str) else expected
        if actual not in allowed:
            failures.append(f"package {package}: expected {' or '.join(allowed)}, got {actual}")

    git = _git_state(root)
    if require_clean and git.get("dirty") is not False:
        failures.append("tracked certification paths are dirty")

    data = config.data
    questions_path = _project_file(root, data.questions_file)
    documents_path = _project_file(root, data.manifest_file)
    question_rows: list[dict[str, Any]] = []
    document_count = 0
    for label, path in (("questions", questions_path), ("documents", documents_path)):
        if not path.is_file():
            failures.append(f"{label} file not found: {path}")
    if questions_path.is_file():
        question_rows = load_jsonl(questions_path)
        question_ids = [str(row.get("question_id") or "") for row in question_rows]
        expected_questions = data.expected_questions
        if (
            len(question_ids) != expected_questions
            or len(set(question_ids)) != expected_questions
            or "" in question_ids
        ):
            failures.append(
                f"{data.name} blind questions must contain "
                f"{expected_questions} unique IDs"
            )
        forbidden = {"gold_answer", "expected_doc_ids"}
        if any(forbidden & set(row) for row in question_rows):
            failures.append("blind questions contain Gold fields")
    if documents_path.is_file():
        with documents_path.open("r", encoding="utf-8") as file:
            document_count = sum(bool(line.strip()) for line in file)

    cross_encoder = config.cross_encoder
    ce_report: dict[str, Any] = {"provider": cross_encoder.provider}
    if cross_encoder.provider == "online":
        if not cross_encoder.model_id:
            failures.append("CROSS_ENCODER_MODEL_ID is required for online reranker")
        if not cross_encoder.base_url:
            failures.append("CROSS_ENCODER_BASE_URL is required for online reranker")
        if not cross_encoder.api_key:
            failures.append("SILICONFLOW_API_KEY is required for online reranker")
        ce_report.update(
            {"model_id": cross_encoder.model_id, "base_url": cross_encoder.base_url}
        )
    else:
        model_value = cross_encoder.model_path
        model_path = Path(model_value) if model_value else None
        actual_model_hash = sha256_tree(model_path) if model_path else None
        expected_model_hash = cross_encoder.model_tree_sha256
        if model_path is None or not model_path.is_dir():
            failures.append("CROSS_ENCODER_MODEL_PATH does not point to a model directory")
        if not expected_model_hash:
            failures.append("CROSS_ENCODER_MODEL_SHA256 is required for certification")
        elif actual_model_hash != expected_model_hash:
            failures.append("cross-encoder model tree hash does not match")
        ce_report.update(
            {
                "path": str(model_path) if model_path else None,
                "tree_sha256": actual_model_hash,
                "expected_tree_sha256": expected_model_hash or None,
            }
        )

    qdrant_report: dict[str, Any] = {"checked": check_qdrant}
    if check_qdrant:
        qdrant = config.qdrant
        try:
            client = QdrantClient(
                url=qdrant.url, api_key=qdrant.api_key or None
            )
            collection = client.get_collection(qdrant.collection)
            count = client.count(qdrant.collection, exact=True).count
            vectors = collection.config.params.vectors
            vector_size = getattr(vectors, "size", None)
            distance = getattr(getattr(vectors, "distance", None), "value", None)
            qdrant_report.update(
                {"points_count": count, "vector_size": vector_size, "distance": distance}
            )
            if count != document_count:
                failures.append(
                    f"Qdrant point count {count} does not match manifest {document_count}"
                )
            if int(vector_size or 0) != qdrant.vector_size:
                failures.append("Qdrant vector size does not match config")
            if str(distance or "").lower() != qdrant.distance.lower():
                failures.append("Qdrant distance does not match config")
        except Exception as exc:
            qdrant_report["error"] = f"{type(exc).__name__}: {exc}"
            failures.append("Qdrant collection validation failed")

    official_commit = _external_git_commit(root / "external" / "EnterpriseRAG-Bench")
    if not official_commit:
        failures.append("official EnterpriseRAG-Bench commit could not be resolved")

    gold_coverage = _gold_document_coverage(config, root)
    warnings: list[str] = []
    if gold_coverage is not None and gold_coverage.get("missing"):
        warnings.append(
            f"gold document coverage {gold_coverage['covered']}/{gold_coverage['total']}: "
            f"missing {len(gold_coverage['missing'])} dsids (source not ingested yet)"
        )

    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "warnings": warnings,
        "packages": package_versions,
        "git": git,
        "source_tree_sha256": source_tree_sha256(root),
        "question_count": len(question_rows),
        "document_count": document_count,
        "gold_coverage": gold_coverage,
        "cross_encoder_model": ce_report,
        "qdrant": qdrant_report,
        "official_evaluator_commit": official_commit,
    }


def certify_run(
    run_dir: str | Path,
    *,
    project_root: str | Path,
    minimum_correct: int = 36,
) -> dict[str, Any]:
    """认证一次完整 N39 开发集运行；正确率是唯一质量阻断指标。"""
    run = Path(run_dir)
    root = Path(project_root).resolve()
    failures: list[str] = []
    required = {
        name: run / name
        for name in (
            "answers.jsonl",
            "official_results.json",
            "run_manifest.json",
            "run_summary.json",
        )
    }
    for name, path in required.items():
        if not path.is_file():
            failures.append(f"missing {name}")
    if failures:
        return {"status": "FAIL", "failures": failures, "correct": 0, "total": 0}

    answers = load_jsonl(required["answers.jsonl"])
    answer_ids = [str(row.get("question_id") or "") for row in answers]
    if len(answer_ids) != 39 or len(set(answer_ids)) != 39 or "" in answer_ids:
        failures.append("answers must contain exactly 39 unique question IDs")

    official = json.loads(required["official_results.json"].read_text(encoding="utf-8-sig"))
    questions = official.get("questions", [])
    official_ids = [str(row.get("question_id") or "") for row in questions]
    if len(official_ids) != 39 or set(official_ids) != set(answer_ids):
        failures.append("official results do not match the 39 answers")
    correct = sum(bool(row.get("answer_correct")) for row in questions)
    if correct < minimum_correct:
        failures.append(f"correct answers {correct}/39 are below {minimum_correct}/39")

    manifest = json.loads(required["run_manifest.json"].read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        failures.append("run manifest schema must be 2")
    if manifest.get("completed_question_count") != 39:
        failures.append("run manifest does not record 39 completed questions")
    if manifest.get("git", {}).get("dirty") is not False:
        failures.append("certification run used dirty tracked source")
    if manifest.get("source_tree_sha256") != source_tree_sha256(root):
        failures.append("source tree changed after the run")
    model = manifest.get("cross_encoder_model", {})
    if model.get("provider") == "online":
        if not model.get("model_id") or not model.get("base_url"):
            failures.append("run used an unverified online cross-encoder model")
    elif not model.get("expected_tree_sha256") or (
        model.get("tree_sha256") != model.get("expected_tree_sha256")
    ):
        failures.append("run used an unverified cross-encoder model")
    if not manifest.get("official_evaluator_commit"):
        failures.append("official evaluator commit was not recorded")
    if not manifest.get("observed_model_ids"):
        failures.append("no observed LLM model ID was recorded")

    summary = json.loads(required["run_summary.json"].read_text(encoding="utf-8"))
    status_counts = summary.get("cross_encoder", {}).get("status_counts", {})
    disallowed = sorted(set(status_counts) - ALLOWED_CE_STATUSES)
    if disallowed:
        failures.append(f"disallowed cross-encoder statuses: {', '.join(disallowed)}")

    aggregate = official.get("aggregate_stats", {})
    report = {
        "schema_version": 1,
        "status": "PASS" if not failures else "FAIL",
        "claim": (
            "GitHub N39 development set single-run accuracy reached at least 90%."
            if not failures
            else None
        ),
        "failures": failures,
        "correct": correct,
        "total": len(questions),
        "accuracy_pct": aggregate.get("average_correctness_pct"),
        "combined_score": aggregate.get(
            "combined_correctness_completeness_score"
        ),
        "document_recall_pct": aggregate.get("average_recall_pct"),
        "run_manifest_sha256": sha256_file(required["run_manifest.json"]),
    }
    return report
