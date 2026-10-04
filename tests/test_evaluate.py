from __future__ import annotations

from pathlib import Path

import pytest

from src.config import LlmConfig
from src.evaluation.official import run_official_evaluation


def test_official_evaluation_maps_anthropic_compatible_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict = {}

    def fake_run(command, *, cwd, env, check):
        captured.update(command=command, cwd=cwd, env=env, check=check)

    monkeypatch.setattr("src.evaluation.official.subprocess.run", fake_run)
    run_official_evaluation(
        official_repo=tmp_path,
        answers_file=tmp_path / "answers.jsonl",
        questions_file=tmp_path / "questions.jsonl",
        results_file=tmp_path / "results.json",
        updated_questions_file=tmp_path / "updated.jsonl",
        parallelism=3,
        correction=False,
        resume=False,
        llm_config=LlmConfig(
            provider="anthropic_compatible",
            api_key="secret",
            model="deepseek-chat",
            base_url="https://api.deepseek.com/anthropic",
            temperature=0.0,
            max_tokens=1024,
        ),
    )

    assert captured["check"] is True
    assert captured["env"]["LLM_PROVIDER"] == "anthropic"
    assert captured["env"]["LLM_API_KEY"] == "secret"
    assert captured["env"]["LLM_MODEL_NAME"] == "deepseek-chat"
    assert captured["env"]["ANTHROPIC_BASE_URL"] == "https://api.deepseek.com/anthropic"


def test_official_evaluation_rejects_incomplete_llm_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="llm.api_key"):
        run_official_evaluation(
            official_repo=tmp_path,
            answers_file=tmp_path / "answers.jsonl",
            questions_file=tmp_path / "questions.jsonl",
            results_file=tmp_path / "results.json",
            updated_questions_file=tmp_path / "updated.jsonl",
            parallelism=1,
            correction=False,
            resume=False,
            llm_config=LlmConfig(
                provider="anthropic_compatible",
                api_key="",
                model="deepseek-chat",
                base_url="https://api.deepseek.com/anthropic",
                temperature=0.0,
                max_tokens=1024,
            ),
        )
