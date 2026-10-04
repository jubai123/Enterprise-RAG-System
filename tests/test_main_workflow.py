from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import AppConfig, load_config
from src.config import loader as config_loader
from src.evaluation.reproducibility import sha256_text
from src.graphs.prompt_registry import benchmark_prompt_texts


def test_prompt_registry_hashes_match_the_frozen_baseline() -> None:
    hashes = {
        name: sha256_text(prompt) for name, prompt in benchmark_prompt_texts().items()
    }

    assert hashes == {
        "answer": "9b530a58c97b9b48c3bd10588be2998f5070882c3f7f8fb2a3e89e308e8facc3",
        "planning": "749149463cef55b5aac830d3b4746472b0611fd87b56388a0e7a7cd01a1a4fa6",
    }


def test_main_config_is_the_frozen_minimal_baseline() -> None:
    config = load_config()
    retrieval = config.retrieval

    assert config.graph.mode == "minimal"
    assert retrieval.mode == "rrf"
    assert retrieval.candidate_documents == 32
    assert retrieval.chunks_per_document == 2
    # sb4 采纳档：single 策略选档上限（捞回 CE rank 2-4 的 gold）。
    assert retrieval.single_document_budget == 4
    # minimal 路径的上下文宽度旋钮。
    assert retrieval.max_parent_chunks == 8
    # chunkce 路径旋钮在 main 上保持未启用/默认值。
    assert retrieval.chunk_pool_cap is None
    assert retrieval.expand_after_ce_top_k == 6
    assert config.cross_encoder.provider == "online"


def test_config_validation_rejects_local_ce_without_model_path() -> None:
    raw = load_config().model_dump()
    raw["cross_encoder"]["provider"] = "local"
    raw["cross_encoder"]["model_path"] = None

    with pytest.raises(ValidationError, match="model_path is required"):
        AppConfig.model_validate(raw)


def test_config_validation_rejects_missing_required_field() -> None:
    raw = load_config().model_dump()
    del raw["retrieval"]["rrf_k"]

    with pytest.raises(ValidationError, match="rrf_k"):
        AppConfig.model_validate(raw)


def test_custom_config_is_a_partial_override_of_main(tmp_path: Path) -> None:
    override = tmp_path / "override.yaml"
    override.write_text("retrieval:\n  max_queries: 7\n", encoding="utf-8")

    config = load_config(override)

    assert config.retrieval.max_queries == 7
    assert config.retrieval.candidate_documents == 32
    assert config.data.expected_questions == 39


def test_config_has_no_python_default_copy() -> None:
    source = Path(config_loader.__file__).read_text(encoding="utf-8")

    assert "DEFAULT_CONFIG" not in source
    assert config_loader.MAIN_CONFIG_PATH.is_file()


@pytest.mark.parametrize(
    "module",
    [
        "scripts.run_benchmark",
        "scripts.evaluate",
        "scripts.prepare_evaluation_protocol",
        "scripts.certify_baseline",
    ],
)
def test_public_clis_do_not_expose_dataset_switch(module: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        check=True,
        text=True,
    )

    assert "--dataset" not in completed.stdout
