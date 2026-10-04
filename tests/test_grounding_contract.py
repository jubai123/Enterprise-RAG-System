from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.graphs.grounding import (
    GROUNDING_PREAMBLE,
    extract_ungrounded_identifiers,
    filter_ungrounded_queries,
    validate_prompt_grounding,
)
from src.graphs.planning import _ground_plan_task_queries, rewrite_colloquial_queries
from src.graphs.prompt_registry import (
    benchmark_prompt_texts,
    validate_grounding_contract,
)


def test_all_registered_prompts_carry_the_grounding_contract() -> None:
    """每个登记进 prompt registry 的 prompt 都必须带反捏造契约（preamble 或其占位符）。"""
    assert validate_grounding_contract() == []
    assert set(benchmark_prompt_texts()) == {"planning", "answer"}


def test_validate_prompt_grounding_accepts_preamble_or_placeholder() -> None:
    assert validate_prompt_grounding(GROUNDING_PREAMBLE + " extra")
    assert validate_prompt_grounding("prefix {grounding} suffix")
    assert not validate_prompt_grounding("no contract here")


def test_extract_ungrounded_identifiers_catches_fabrication() -> None:
    question = "What is the limit for the --max-batch flag?"
    evidence = "the --max-batch flag caps at 10."
    out = extract_ungrounded_identifiers(
        "the fix in PR 28564 sets --max-batch to 20",
        question,
        evidence,
    )
    # 幻觉 PR 号必须被捕获（qst_0207 失败模式）；来自问题/证据的标识符放行。
    assert "PR 28564" in out
    assert "--max-batch" not in out
    # 路径标识符（不在 question/evidence）也视为捏造。
    assert extract_ungrounded_identifiers(
        "logs are stored in /var/log/app.log",
        "where are logs stored?",
        "",
    ) == ["/var/log/app.log"]
    # 来自 question ∪ evidence 的标识符全部放行。
    assert extract_ungrounded_identifiers(
        "the --max-batch flag",
        question,
        evidence,
    ) == []


def test_filter_ungrounded_queries_blocks_hallucinated_queries() -> None:
    question = "What is the kill switch named obs.route_tags_tool_calls?"
    evidence = "gated behind obs.route_tags_tool_calls."
    kept = filter_ungrounded_queries(
        [
            "What is the rollout in PR 28564?",
            "What is obs.route_tags_tool_calls?",
            "Where is it documented?",
        ],
        question,
        evidence,
    )
    assert kept == [
        "What is obs.route_tags_tool_calls?",
        "Where is it documented?",
    ]


def test_ground_plan_task_queries_replaces_hallucinated_query() -> None:
    plan = {
        "retrieval_tasks": [
            {
                "task_id": "r1",
                "requirement": "q",
                "slot": "general",
                "query": "limit for --fake-flag",
            },
            {
                "task_id": "r2",
                "requirement": "q",
                "slot": "general",
                "query": "limit for --max-batch",
            },
        ],
        "queries": ["limit for --fake-flag", "limit for --max-batch"],
    }
    question = "what is the limit for --max-batch?"
    out = _ground_plan_task_queries(plan, question)
    assert out["retrieval_tasks"][0]["query"] == question
    assert out["retrieval_tasks"][1]["query"] == "limit for --max-batch"
    assert out["queries"][0] == question


def test_rewrite_falls_back_when_it_invents_an_identifier() -> None:
    llm = FakeListChatModel(responses=["what is the limit for --fake-flag"])
    plan = {
        "retrieval_tasks": [
            {"task_id": "r1", "requirement": "q", "slot": "general", "query": "original query"}
        ],
        "queries": ["original query"],
    }
    out, _calls = rewrite_colloquial_queries(
        llm,
        "what is the limit for --max-batch?",
        plan,
        {"model_calls": []},
    )
    # 改写引入不在问题中的标识符 → 回退原 task query（并补齐问题标识符），不扩散捏造。
    query = out["retrieval_tasks"][0]["query"]
    assert query.startswith("original query")
    assert "--fake-flag" not in query
