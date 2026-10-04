from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from src.config import LlmConfig


def run_official_evaluation(
    *,
    official_repo: Path,
    answers_file: Path,
    questions_file: Path,
    results_file: Path,
    updated_questions_file: Path,
    parallelism: int,
    correction: bool,
    resume: bool,
    llm_config: LlmConfig,
) -> None:
    command = [
        sys.executable,
        "-m",
        "src.scripts.answer_evaluation.metrics_based_eval",
        "--answers-file",
        str(answers_file),
        "--questions-file",
        str(questions_file),
        "--results-file",
        str(results_file),
        "--parallelism",
        str(parallelism),
    ]
    if correction:
        command.extend(
            [
                "--updated-questions-file",
                str(updated_questions_file),
                "--uuid-index-cache-file",
                str((official_repo / "generated_data" / "uuid_index.json").resolve()),
            ]
        )
    else:
        command.append("--no-correction")
    if resume:
        command.append("--resume")

    child_env = os.environ.copy()
    child_env["PYTHONUTF8"] = "1"
    if not llm_config.api_key or not llm_config.model or not llm_config.base_url:
        raise ValueError(
            "Official evaluation requires configured llm.api_key, llm.model and llm.base_url."
        )
    provider = llm_config.provider.lower()
    if provider in {"anthropic", "anthropic_compatible"}:
        child_env.update(
            {
                "LLM_PROVIDER": "anthropic",
                "LLM_API_KEY": llm_config.api_key,
                "LLM_MODEL_NAME": llm_config.model,
                "CHEAP_LLM_MODEL_NAME": llm_config.model,
                "ANTHROPIC_BASE_URL": llm_config.base_url,
            }
        )
    elif provider in {"openai", "openai_compatible"}:
        child_env.update(
            {
                "LLM_PROVIDER": "openai",
                "LLM_API_KEY": llm_config.api_key,
                "LLM_MODEL_NAME": llm_config.model,
                "CHEAP_LLM_MODEL_NAME": llm_config.model,
                "OPENAI_BASE_URL": llm_config.base_url,
            }
        )
    else:
        raise ValueError(f"Unsupported official-evaluation LLM provider: {provider}")
    subprocess.run(command, cwd=official_repo, env=child_env, check=True)
