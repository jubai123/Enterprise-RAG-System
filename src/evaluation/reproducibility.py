from __future__ import annotations

import hashlib
import importlib.metadata
import json
import random
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import AppConfig
from src.evaluation.benchmark_report import load_jsonl, matches_source_type

BLIND_QUESTION_FIELDS = ("question_id", "question", "question_type", "source_types")
REPRODUCIBLE_SOURCE_DIRS = ("src", "scripts", "configs", "tests")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_tree(path: str | Path) -> str | None:
    """对目录内文件名和内容做稳定哈希；目录不存在时返回 None。"""
    root = Path(path)
    if not root.exists():
        return None
    digest = hashlib.sha256()
    for file in sorted(item for item in root.rglob("*") if item.is_file()):
        if any(part in {"__pycache__", ".pytest_cache", ".pytest_tmp"} for part in file.parts):
            continue
        relative = file.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(file)))
    return digest.hexdigest()


def source_tree_sha256(project_root: str | Path) -> str:
    root = Path(project_root)
    digest = hashlib.sha256()
    for directory in REPRODUCIBLE_SOURCE_DIRS:
        tree_hash = sha256_tree(root / directory) or "missing"
        digest.update(f"{directory}:{tree_hash}\n".encode("utf-8"))
    return digest.hexdigest()


def prepare_blind_dataset(config: AppConfig) -> dict[str, Any]:
    """根据唯一 GitHub N39 配置生成不含 Gold 的问题文件。"""
    data = config.data
    if data.source_scope is not None:
        return _prepare_blind_dataset_stratified(config)
    selected = [
        row
        for row in load_jsonl(data.gold_questions_file)
        if matches_source_type(row, data.source_type)
    ]
    if len(selected) != data.expected_questions:
        raise ValueError(
            f"{data.name} expected {data.expected_questions} questions, "
            f"found {len(selected)}"
        )
    return _write_blind_questions(
        data, selected, {"source_type": data.source_type}
    )


def _prepare_blind_dataset_stratified(config: AppConfig) -> dict[str, Any]:
    """按题型分布从 source 白名单内分层采样生成不含 Gold 的问题文件。"""
    data = config.data
    scope = set(data.source_scope or [])
    pool_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(data.gold_questions_file):
        source_types = row.get("source_types") or []
        if source_types and not set(source_types) <= scope:
            continue
        pool_by_type[str(row.get("question_type"))].append(row)

    rng = random.Random(data.selection_seed)
    selected: list[dict[str, Any]] = []
    for question_type, count in (data.type_targets or {}).items():
        pool = pool_by_type.get(question_type, [])
        if len(pool) < count:
            raise ValueError(
                f"question_type {question_type!r}: need {count} in scope, "
                f"only {len(pool)} available"
            )
        selected.extend(rng.sample(pool, count))
    if len(selected) != data.expected_questions:
        raise ValueError(
            f"{data.name} expected {data.expected_questions} questions, "
            f"selected {len(selected)}"
        )
    selected.sort(key=lambda row: str(row.get("question_id")))
    return _write_blind_questions(
        data,
        selected,
        {
            "source_scope": sorted(scope),
            "type_targets": dict(data.type_targets or {}),
            "selection_seed": data.selection_seed,
            "selection_method": "stratified_random",
        },
    )


def _write_blind_questions(
    data: Any, selected: list[dict[str, Any]], extra: dict[str, Any]
) -> dict[str, Any]:
    blind_rows = [
        {field: row.get(field) for field in BLIND_QUESTION_FIELDS if field in row}
        for row in selected
    ]
    blind_path = Path(data.questions_file)
    serialized = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in blind_rows
    )
    if not blind_path.exists() or blind_path.read_text(encoding="utf-8") != serialized:
        blind_path.parent.mkdir(parents=True, exist_ok=True)
        blind_path.write_text(serialized, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "dataset": data.name,
        "role": "development",
        "question_count": len(blind_rows),
        "blind_questions_file": str(blind_path),
        "blind_questions_sha256": sha256_file(blind_path),
        "gold_is_excluded": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest.update(extra)
    manifest_path = blind_path.parent / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def _redact_secrets(value: Any, key: str = "") -> Any:
    if key.lower() in {"api_key", "authorization", "password", "secret"}:
        return "<redacted>" if value else value
    if isinstance(value, dict):
        return {name: _redact_secrets(item, name) for name, item in value.items()}
    if isinstance(value, list):
        return [_redact_secrets(item) for item in value]
    return value


def _git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_state(project_root: Path) -> dict[str, Any]:
    paths = list(REPRODUCIBLE_SOURCE_DIRS)
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", *paths],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        diff = subprocess.run(
            ["git", "diff", "HEAD", "--", *paths],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=False,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"dirty": None, "status": None, "diff_sha256": None}
    return {
        "dirty": bool(status.strip()),
        "status": status.splitlines(),
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def _external_git_commit(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        result = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={path.as_posix()}",
                "-C",
                str(path),
                "rev-parse",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_run_manifest(
    config: AppConfig,
    *,
    project_root: str | Path,
    prompt_texts: dict[str, str],
) -> dict[str, Any]:
    root = Path(project_root)
    data_config = config.data
    tracked_files = {
        "questions": data_config.questions_file,
        "documents": data_config.manifest_file,
    }
    file_hashes = {
        name: sha256_file(root / path)
        for name, path in tracked_files.items()
        if path and (root / path).exists()
    }
    packages = {}
    for package in (
        "langchain",
        "langchain-core",
        "langgraph",
        "qdrant-client",
        "sentence-transformers",
        "torch",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    safe_config = _redact_secrets(config.model_dump())
    cross_encoder = config.cross_encoder
    model_path_value = cross_encoder.model_path
    model_path = Path(model_path_value) if model_path_value else None
    git_state = _git_state(root)
    cuda: dict[str, Any] = {"available": None, "runtime": None}
    try:
        import torch

        cuda = {
            "available": bool(torch.cuda.is_available()),
            "runtime": getattr(torch.version, "cuda", None),
        }
    except ImportError:
        pass
    return {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(root),
        "git": git_state,
        "source_tree_sha256": source_tree_sha256(root),
        "python_version": sys.version.split()[0],
        "packages": packages,
        "cuda": cuda,
        "config": safe_config,
        "config_sha256": sha256_text(
            json.dumps(safe_config, ensure_ascii=False, sort_keys=True)
        ),
        "file_hashes": file_hashes,
        "prompt_hashes": {
            name: sha256_text(text) for name, text in sorted(prompt_texts.items())
        },
        "cross_encoder_model": {
            "provider": cross_encoder.provider,
            "model_id": cross_encoder.model_id,
            "path": str(model_path) if model_path is not None else None,
            "tree_sha256": sha256_tree(model_path) if model_path is not None else None,
            "expected_tree_sha256": cross_encoder.model_tree_sha256 or None,
        },
        "qdrant": {
            "collection": config.qdrant.collection,
            "vector_size": config.qdrant.vector_size,
            "distance": config.qdrant.distance,
        },
        "official_evaluator_commit": _external_git_commit(
            root / "external" / "EnterpriseRAG-Bench"
        ),
        "observed_model_ids": [],
    }


def write_run_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
