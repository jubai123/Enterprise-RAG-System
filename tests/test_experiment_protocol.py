from __future__ import annotations

import json
from pathlib import Path

from src.config import PricingConfig, load_config
from src.evaluation.diagnostics import build_recall_funnel
from src.evaluation.experiment_config import (
    ABLATION_VARIANTS,
    apply_experiment_variant,
)
from src.evaluation.metrics import build_question_metrics, summarize_question_metrics
from src.evaluation.reproducibility import build_run_manifest, prepare_blind_dataset
from src.graphs.prompt_registry import benchmark_prompt_texts


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_blind_dataset_excludes_all_gold_fields(tmp_path: Path) -> None:
    questions = tmp_path / "questions.jsonl"
    _write_jsonl(
        questions,
        [
            {
                "question_id": "q1",
                "question": "What changed?",
                "question_type": "basic",
                "source_types": ["confluence"],
                "gold_answer": "secret",
                "expected_doc_ids": ["gold-doc"],
            },
            {
                "question_id": "q2",
                "question": "Ignored",
                "source_types": ["github"],
                "gold_answer": "other",
            },
        ],
    )
    blind_file = tmp_path / "questions.blind.jsonl"
    base = load_config()
    config = base.model_copy(
        update={
            "data": base.data.model_copy(
                update={
                    "name": "github_dev",
                    "source_type": "confluence",
                    "expected_questions": 1,
                    "gold_questions_file": str(questions),
                    "questions_file": str(blind_file),
                }
            )
        }
    )

    manifest = prepare_blind_dataset(config)
    blind = json.loads(blind_file.read_text(encoding="utf-8"))

    assert blind == {
        "question_id": "q1",
        "question": "What changed?",
        "question_type": "basic",
        "source_types": ["confluence"],
    }
    assert manifest["gold_is_excluded"] is True


def test_every_registered_ablation_changes_at_most_one_factor() -> None:
    for name, spec in ABLATION_VARIANTS.items():
        if name == "main":
            assert spec is None
        else:
            assert spec is not None


def test_variant_application_does_not_mutate_base_config() -> None:
    base = load_config()
    changed = apply_experiment_variant(base, "dense_only")

    assert base.retrieval.mode == "rrf"
    assert changed.retrieval.mode == "dense"
    assert changed.features.adaptive_planning is True


def test_intent_routing_variant_changes_only_that_switch() -> None:
    base = load_config()

    changed = apply_experiment_variant(base, "no_intent_routing")

    spec = ABLATION_VARIANTS["no_intent_routing"]
    assert spec is not None
    assert (spec.section, spec.field, spec.value) == (
        "features",
        "intent_routing",
        False,
    )
    assert changed.features.intent_routing is False
    assert changed.features.adaptive_planning is True
    assert base.features.intent_routing is True


def test_run_manifest_redacts_secrets_and_hashes_inputs(tmp_path: Path) -> None:
    questions = tmp_path / "questions.jsonl"
    documents = tmp_path / "documents.jsonl"
    questions.write_text("{}\n", encoding="utf-8")
    documents.write_text("{}\n", encoding="utf-8")
    base = load_config()
    config = base.model_copy(
        update={
            "data": base.data.model_copy(
                update={
                    "questions_file": str(questions),
                    "manifest_file": str(documents),
                }
            ),
            "llm": base.llm.model_copy(
                update={"api_key": "never-write-this", "model": "fake"}
            ),
        }
    )

    first = build_run_manifest(config, project_root=tmp_path, prompt_texts={"p": "same"})
    second = build_run_manifest(config, project_root=tmp_path, prompt_texts={"p": "same"})

    assert first["config"]["llm"]["api_key"] == "<redacted>"
    assert first["schema_version"] == 2
    assert first["config_sha256"] == second["config_sha256"]
    assert first["prompt_hashes"] == second["prompt_hashes"]
    assert first["file_hashes"] == second["file_hashes"]


def test_manifest_prompt_set_covers_only_the_live_graphs() -> None:
    prompts = benchmark_prompt_texts()

    # full 图独有的重排/证据判定 prompt 已随图下线，签名集只剩 planning + answer。
    assert "rerank" not in prompts
    assert "rerank_verify_version" not in prompts
    assert "rerank_verify_group" not in prompts
    assert "rerank_verify_final" not in prompts
    assert "evidence" not in prompts
    assert "answer_repair" not in prompts
    assert set(prompts) == {"planning", "answer"}


def test_question_metrics_keep_unknown_cost_as_null() -> None:
    state = {
        "model_calls": [
            {"input_tokens": 100, "output_tokens": 20},
            {"input_tokens": 50, "output_tokens": 10},
        ],
        "node_metrics": [{"node": "plan", "duration_ms": 12.5}],
        "retrieval_round": 2,
    }
    pricing = PricingConfig(input_per_million=None, output_per_million=None)
    row = build_question_metrics("q1", state, pricing)

    assert row["total_tokens"] == 180
    assert row["cost"] is None
    assert row["followup_used"] is True
    assert summarize_question_metrics([row])["duration_ms"]["p95"] == 12.5


def test_question_metrics_aggregate_ce_and_verification_events() -> None:
    state = {
        "model_calls": [],
        "node_metrics": [],
        "retrieval_round": 2,
        "rerank_history": [
            {
                "retrieval_mode": "initial",
                "cross_encoder_status": "skipped_initial_round",
                "cross_encoder_scored_count": 0,
                "cross_encoder_latency_ms": 0.0,
                "verification": {"verification_triggered": False},
            },
            {
                "retrieval_mode": "deep",
                "cross_encoder_status": "success",
                "cross_encoder_scored_count": 150,
                "cross_encoder_latency_ms": 3500.0,
                "verification": {
                    "verification_triggered": True,
                    "group_call_count": 4,
                    "final_call_count": 1,
                    "search_candidate_count": 20,
                },
            },
            {
                "retrieval_mode": "deep",
                "cross_encoder_status": "success",
                "cross_encoder_scored_count": 80,
                "cross_encoder_latency_ms": 4200.0,
                "verification": {"verification_triggered": False},
            },
        ],
    }

    pricing = PricingConfig(input_per_million=None, output_per_million=None)
    row = build_question_metrics("q1", state, pricing)

    assert row["cross_encoder"]["initial_scored_rounds"] == 0
    assert row["cross_encoder"]["deep_scored_rounds"] == 2
    assert row["cross_encoder"]["scored_documents"] == 230
    assert row["cross_encoder"]["status_counts"] == {
        "skipped_initial_round": 1,
        "success": 2,
    }
    assert row["verification"]["triggered"] is True
    assert row["verification"]["group_call_count"] == 4
    assert row["verification"]["final_call_count"] == 1
    assert row["verification"]["search_candidate_count"] == 20

    summary = summarize_question_metrics([row])
    assert summary["verification"]["triggered_questions"] == 1
    assert summary["verification"]["group_calls"] == 4
    assert summary["cross_encoder"]["initial_scored_rounds"] == 0
    assert summary["cross_encoder"]["deep_scored_rounds"] == 2
    assert summary["cross_encoder"]["scored_documents"] == 230
    assert summary["cross_encoder"]["status_counts"] == {
        "skipped_initial_round": 1,
        "success": 2,
    }


def test_recall_funnel_supports_multi_gold_and_rerank_attribution() -> None:
    questions = [
        {
            "question_id": "q1",
            "expected_doc_ids": ["gold-a", "gold-b"],
        }
    ]
    traces = [
        {
            "question_id": "q1",
            "retrieval_stage_history": [
                {
                    "stages": {
                        "dense": [{"dsid": "gold-a"}],
                        "bm25": [{"dsid": "gold-b"}],
                        "union": [{"dsid": "gold-a"}, {"dsid": "gold-b"}],
                        "channel_fusion": [
                            {"dsid": "gold-a"},
                            {"dsid": "gold-b"},
                        ],
                    }
                }
            ],
            "rerank_history": [
                {
                    "candidate_document_ids": ["gold-a", "gold-b"],
                    "selected_document_ids": ["gold-a"],
                }
            ],
            "selected_document_ids": ["gold-a"],
        }
    ]
    official = {"questions": [{"question_id": "q1", "answer_correct": False}]}

    report = build_recall_funnel(questions, traces, official)

    assert report["overall"]["union"]["recall"] == 1.0
    assert report["overall"]["rerank"]["recall"] == 0.5
    assert report["questions"][0]["primary_failure_cause"] == "rerank_elimination"
