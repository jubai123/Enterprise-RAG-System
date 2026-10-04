from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.graphs.answer_nodes import generate_answer_node
from tests.rag_test_support import graph_config
from tests.rag_test_support import judge_state as _judge_state


def _sufficient_state() -> dict:
    state = _judge_state()
    state["question"] = "What is the --retry-loop timeout?"
    state["plan"] = {
        "requirements": ["the --retry-loop timeout value"],
        "retrieval_tasks": [],
    }
    state["evidence_sufficient"] = True
    state["answer_docs"] = state["retrieved_docs"]
    return state


def test_sufficient_evidence_generates_answer() -> None:
    # 生成授权只看 evidence_sufficient（minimal/chunkce 恒为 True，由选择层直接放行）；
    # 反捏造仍由 coverage/grounding gate 兜底。
    llm = FakeListChatModel(responses=["The exact limit is 10 MiB."])
    state = _sufficient_state()
    state["answer_docs"] = state["retrieved_docs"]

    result = generate_answer_node(llm)(state)

    assert "10 MiB" in result["answer"]
    assert result.get("model_calls", [])  # 生成器确实被调用
    assert result["answer_coverage_missing"] == []


def test_insufficient_evidence_refuses_without_calling_generator() -> None:
    llm = FakeListChatModel(responses=["must not be used"])
    state = {
        **_judge_state(),
        "evidence_sufficient": False,
        "missing_evidence": ["exact limit"],
    }

    result = generate_answer_node(llm)(state)

    assert "insufficient" in result["answer"].lower()
    assert result.get("model_calls", []) == []
    assert result["answer_coverage_missing"] == []


def test_answer_coverage_gate_records_missing_value() -> None:
    llm = FakeListChatModel(responses=["The --retry-loop retries indefinitely."])
    features = graph_config(features={"answer_coverage_gate": True}).features

    result = generate_answer_node(llm, features=features)(_sufficient_state())

    assert result["answer_coverage_missing"] == ["the --retry-loop timeout value"]


def test_answer_coverage_gate_passes_when_value_present() -> None:
    llm = FakeListChatModel(
        responses=["The --retry-loop retries 3 times and times out after 5 seconds."]
    )
    features = graph_config(features={"answer_coverage_gate": True}).features

    result = generate_answer_node(llm, features=features)(_sufficient_state())

    assert result["answer_coverage_missing"] == []


def test_answer_coverage_gate_ignores_identifier_only() -> None:
    llm = FakeListChatModel(responses=["It retries until success."])
    state = _sufficient_state()
    state["plan"]["requirements"] = ["what does --retry-loop do"]
    features = graph_config(features={"answer_coverage_gate": True}).features

    result = generate_answer_node(llm, features=features)(state)

    assert result["answer_coverage_missing"] == []


def test_answer_coverage_gate_disabled_by_default() -> None:
    llm = FakeListChatModel(responses=["The --retry-loop retries indefinitely."])
    features = graph_config(features={}).features

    result = generate_answer_node(llm, features=features)(_sufficient_state())

    assert result["answer_coverage_missing"] == []
