from __future__ import annotations

from src.graphs.node_utils import format_retrieval_guidance


def test_retrieval_guidance_drops_overridden_llm_reason() -> None:
    guidance = format_retrieval_guidance(
        {
            "rerank_history": [
                {
                    "selection_reason": "The wrong tolerant-schema document is best.",
                    "guardrail_reason": "direct_evidence:http_status",
                }
            ]
        }
    )

    assert "wrong tolerant-schema" not in guidance
    assert "http_status" in guidance


def test_retrieval_guidance_keeps_unoverridden_reason() -> None:
    guidance = format_retrieval_guidance(
        {
            "rerank_history": [
                {
                    "selection_reason": "Both omitted and unset values use defaults.",
                    "guardrail_reason": None,
                }
            ]
        }
    )

    assert guidance == "Both omitted and unset values use defaults."


def test_retrieval_guidance_without_history_is_explicit() -> None:
    assert format_retrieval_guidance({}) == (
        "No additional retrieval assessment is available."
    )
