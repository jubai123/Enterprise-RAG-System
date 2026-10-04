from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import load_config
from src.config.models import DataConfig
from src.evaluation.reproducibility import (
    BLIND_QUESTION_FIELDS,
    prepare_blind_dataset,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _gold_rows() -> list[dict]:
    rows = []
    for index in range(5):
        rows.append(
            {
                "question_id": f"gb{index}",
                "question": f"basic github {index}",
                "question_type": "basic",
                "source_types": ["github"],
                "expected_doc_ids": [f"dsid_basic_{index}"],
                "gold_answer": f"answer {index}",
            }
        )
    for index in range(3):
        rows.append(
            {
                "question_id": f"gl{index}",
                "question": f"basic linear {index}",
                "question_type": "basic",
                "source_types": ["linear"],
                "expected_doc_ids": [f"dsid_linear_{index}"],
                "gold_answer": f"answer {index}",
            }
        )
    for index in range(3):
        rows.append(
            {
                "question_id": f"sg{index}",
                "question": f"semantic github {index}",
                "question_type": "semantic",
                "source_types": ["github"],
                "gold_answer": f"answer {index}",
            }
        )
    rows.append(
        {
            "question_id": "sbad",
            "question": "semantic slack (out of scope)",
            "question_type": "semantic",
            "source_types": ["slack"],
            "gold_answer": "other",
        }
    )
    rows.append(
        {
            "question_id": "hl1",
            "question": "high level synthesis",
            "question_type": "high_level",
            "source_types": [],
            "gold_answer": "N/A",
        }
    )
    rows.append(
        {
            "question_id": "nf1",
            "question": "not found anywhere",
            "question_type": "info_not_found",
            "source_types": [],
            "gold_answer": "N/A",
        }
    )
    return rows


def _stratified_config(tmp_path: Path, gold_path: Path, seed: int) -> DataConfig:
    base = load_config()
    return base.model_copy(
        update={
            "data": base.data.model_copy(
                update={
                    "name": "rag100_test",
                    "source_type": None,
                    "expected_questions": 8,
                    "gold_questions_file": str(gold_path),
                    "questions_file": str(tmp_path / "questions.blind.jsonl"),
                    "source_scope": ["github", "linear"],
                    "type_targets": {
                        "basic": 4,
                        "semantic": 2,
                        "high_level": 1,
                        "info_not_found": 1,
                    },
                    "selection_seed": seed,
                }
            )
        }
    )


def test_stratified_selection_honors_targets_scope_and_seed(tmp_path: Path) -> None:
    gold_path = tmp_path / "gold.jsonl"
    _write_jsonl(gold_path, _gold_rows())
    config = _stratified_config(tmp_path, gold_path, seed=7)

    manifest = prepare_blind_dataset(config)
    blind_path = tmp_path / "questions.blind.jsonl"
    blind = [
        json.loads(line)
        for line in blind_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(blind) == 8
    assert sum(row["question_type"] == "basic" for row in blind) == 4
    assert sum(row["question_type"] == "semantic" for row in blind) == 2
    assert sum(row["question_type"] == "high_level" for row in blind) == 1
    assert sum(row["question_type"] == "info_not_found" for row in blind) == 1
    # 越界 source（slack）的题必须被排除。
    assert all(row["question_id"] != "sbad" for row in blind)
    # blind 文件只含 BLIND 字段，无 gold。
    assert all(set(row) <= set(BLIND_QUESTION_FIELDS) for row in blind)
    assert all("expected_doc_ids" not in row and "gold_answer" not in row for row in blind)

    assert manifest["selection_method"] == "stratified_random"
    assert manifest["source_scope"] == ["github", "linear"]
    assert manifest["selection_seed"] == 7
    assert manifest["type_targets"]["basic"] == 4


def test_stratified_selection_is_deterministic(tmp_path: Path) -> None:
    gold_path = tmp_path / "gold.jsonl"
    _write_jsonl(gold_path, _gold_rows())

    first = _stratified_config(tmp_path, gold_path, seed=7)
    prepare_blind_dataset(first)
    first_blind = (tmp_path / "questions.blind.jsonl").read_text(encoding="utf-8")

    second = _stratified_config(tmp_path, gold_path, seed=7)
    prepare_blind_dataset(second)
    second_blind = (tmp_path / "questions.blind.jsonl").read_text(encoding="utf-8")

    assert first_blind == second_blind


def test_stratified_selection_different_seed_differs(tmp_path: Path) -> None:
    gold_path = tmp_path / "gold.jsonl"
    _write_jsonl(gold_path, _gold_rows())

    prepare_blind_dataset(_stratified_config(tmp_path, gold_path, seed=7))
    first = (tmp_path / "questions.blind.jsonl").read_text(encoding="utf-8")

    prepare_blind_dataset(_stratified_config(tmp_path, gold_path, seed=99))
    second = (tmp_path / "questions.blind.jsonl").read_text(encoding="utf-8")

    assert first != second


def test_stratified_selection_raises_when_type_under_supplied(tmp_path: Path) -> None:
    gold_path = tmp_path / "gold.jsonl"
    _write_jsonl(gold_path, _gold_rows())
    config = _stratified_config(tmp_path, gold_path, seed=7)
    config = config.model_copy(
        update={
            "data": config.data.model_copy(
                update={
                    "type_targets": {
                        "basic": 4,
                        "semantic": 2,
                        "high_level": 1,
                        "info_not_found": 1,
                        "conflicting_info": 4,
                    }
                }
            )
        }
    )

    with pytest.raises(ValueError, match="conflicting_info"):
        prepare_blind_dataset(config)


def _data_config(**overrides) -> DataConfig:
    kwargs = {
        "name": "t",
        "expected_questions": 100,
        "archives_dir": "archives",
        "documents_dir": "docs",
        "gold_questions_file": "g.jsonl",
        "questions_file": "q.jsonl",
        "manifest_file": "m.jsonl",
    }
    kwargs.update(overrides)
    return DataConfig(**kwargs)


def test_data_config_validates_stratified_selection() -> None:
    valid = {
        "source_scope": ["github"],
        "type_targets": {"basic": 50, "semantic": 50},
        "selection_seed": 42,
    }
    assert _data_config(**valid).source_scope == ["github"]

    with pytest.raises(ValueError, match="sum to expected_questions"):
        _data_config(
            source_scope=["github"],
            type_targets={"basic": 60, "semantic": 50},
            selection_seed=42,
        )
    with pytest.raises(ValueError, match="unknown question types"):
        _data_config(
            source_scope=["github"],
            type_targets={"basic": 60, "made_up_type": 40},
            selection_seed=42,
        )
    with pytest.raises(ValueError, match="selection_seed"):
        _data_config(
            source_scope=["github"],
            type_targets={"basic": 50, "semantic": 50},
            selection_seed=None,
        )
    with pytest.raises(ValueError, match="type_targets is required"):
        _data_config(source_scope=["github"])


def test_single_source_path_still_records_source_type(tmp_path: Path) -> None:
    gold_path = tmp_path / "gold.jsonl"
    _write_jsonl(
        gold_path,
        [
            {
                "question_id": "q1",
                "question": "x",
                "question_type": "basic",
                "source_types": ["confluence"],
            }
        ],
    )
    base = load_config()
    config = base.model_copy(
        update={
            "data": base.data.model_copy(
                update={
                    "name": "github_dev",
                    "source_type": "confluence",
                    "expected_questions": 1,
                    "gold_questions_file": str(gold_path),
                    "questions_file": str(tmp_path / "q.blind.jsonl"),
                }
            )
        }
    )

    manifest = prepare_blind_dataset(config)

    assert manifest["source_type"] == "confluence"
    assert manifest.get("selection_method") not in {"stratified_random"}
