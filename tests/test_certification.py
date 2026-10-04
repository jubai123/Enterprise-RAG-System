from __future__ import annotations

import json
from pathlib import Path

from src.config import load_config
from src.evaluation.certification import build_preflight_report, certify_run
from src.evaluation.reproducibility import source_tree_sha256


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _certification_fixture(tmp_path: Path, correct: int) -> Path:
    for directory in ("src", "scripts", "configs", "tests"):
        (tmp_path / directory).mkdir(parents=True)
    run = tmp_path / "run"
    run.mkdir()
    answers = [{"question_id": f"q{index:02d}", "answer": "x"} for index in range(39)]
    questions = [
        {"question_id": f"q{index:02d}", "answer_correct": index < correct}
        for index in range(39)
    ]
    _write_jsonl(run / "answers.jsonl", answers)
    (run / "official_results.json").write_text(
        json.dumps(
            {
                "aggregate_stats": {
                    "average_correctness_pct": round(correct / 39 * 100, 2),
                    "combined_correctness_completeness_score": 90.0,
                    "average_recall_pct": 89.0,
                },
                "questions": questions,
            }
        ),
        encoding="utf-8",
    )
    (run / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "completed_question_count": 39,
                "git": {"dirty": False},
                "source_tree_sha256": source_tree_sha256(tmp_path),
                "cross_encoder_model": {
                    "tree_sha256": "model-hash",
                    "expected_tree_sha256": "model-hash",
                },
                "official_evaluator_commit": "official-commit",
                "observed_model_ids": ["model"],
            }
        ),
        encoding="utf-8",
    )
    (run / "run_summary.json").write_text(
        json.dumps(
            {
                "cross_encoder": {
                    "status_counts": {"skipped_initial_round": 39, "success": 4}
                }
            }
        ),
        encoding="utf-8",
    )
    return run


def test_certification_fails_at_35_and_passes_at_36(tmp_path: Path) -> None:
    failed_root = tmp_path / "failed"
    passed_root = tmp_path / "passed"
    failed = certify_run(
        _certification_fixture(failed_root, 35), project_root=failed_root
    )
    passed = certify_run(
        _certification_fixture(passed_root, 36), project_root=passed_root
    )

    assert failed["status"] == "FAIL"
    assert passed["status"] == "PASS"
    assert passed["claim"] is not None


def test_certification_rejects_ce_degradation(tmp_path: Path) -> None:
    run = _certification_fixture(tmp_path, 39)
    (run / "run_summary.json").write_text(
        json.dumps({"cross_encoder": {"status_counts": {"inference_timeout": 1}}}),
        encoding="utf-8",
    )

    report = certify_run(run, project_root=tmp_path)

    assert report["status"] == "FAIL"
    assert any("inference_timeout" in failure for failure in report["failures"])


def test_preflight_reports_missing_model_and_hash(tmp_path: Path) -> None:
    questions = tmp_path / "questions.jsonl"
    documents = tmp_path / "documents.jsonl"
    _write_jsonl(
        questions,
        [{"question_id": f"q{index:02d}", "question": "x"} for index in range(39)],
    )
    _write_jsonl(documents, [{"chunk": "x"}])
    config = load_config()
    config = config.model_copy(
        update={
            "data": config.data.model_copy(
                update={
                    "questions_file": str(questions),
                    "manifest_file": str(documents),
                }
            ),
            "cross_encoder": config.cross_encoder.model_copy(
                update={
                    "provider": "local",
                    "model_path": "",
                    "model_tree_sha256": "",
                }
            ),
        }
    )

    report = build_preflight_report(
        config,
        project_root=tmp_path,
        require_clean=False,
        check_qdrant=False,
    )

    assert report["status"] == "FAIL"
    assert any("CROSS_ENCODER_MODEL_PATH" in item for item in report["failures"])
    assert any("CROSS_ENCODER_MODEL_SHA256" in item for item in report["failures"])


def test_preflight_online_requires_api_config(tmp_path: Path) -> None:
    questions = tmp_path / "questions.jsonl"
    documents = tmp_path / "documents.jsonl"
    _write_jsonl(
        questions,
        [{"question_id": f"q{index:02d}", "question": "x"} for index in range(39)],
    )
    _write_jsonl(documents, [{"chunk": "x"}])
    config = load_config()
    config = config.model_copy(
        update={
            "data": config.data.model_copy(
                update={
                    "questions_file": str(questions),
                    "manifest_file": str(documents),
                }
            ),
            "cross_encoder": config.cross_encoder.model_copy(
                update={"provider": "online", "base_url": "", "api_key": ""}
            ),
        }
    )

    report = build_preflight_report(
        config,
        project_root=tmp_path,
        require_clean=False,
        check_qdrant=False,
    )

    assert report["status"] == "FAIL"
    assert any("CROSS_ENCODER_BASE_URL" in item for item in report["failures"])
    assert any("SILICONFLOW_API_KEY" in item for item in report["failures"])
    assert report["cross_encoder_model"]["provider"] == "online"


def test_certify_online_run_verifies_endpoint(tmp_path: Path) -> None:
    run = _certification_fixture(tmp_path, 39)
    manifest_path = run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cross_encoder_model"] = {
        "provider": "online",
        "model_id": "BAAI/bge-reranker-v2-m3",
        "base_url": "https://api.siliconflow.com/v1",
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = certify_run(run, project_root=tmp_path)

    assert report["status"] == "PASS"
    assert "unverified" not in " ".join(report["failures"] or [])
