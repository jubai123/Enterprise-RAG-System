from __future__ import annotations

from src.graphs.planning import (
    _is_underspecified,
    classify_intent,
    normalize_plan,
    preserve_query_identifiers,
)


def test_normalize_plan_applies_adaptive_budget_and_keeps_original_query() -> None:
    plan = normalize_plan(
        {
            "strategy": "completeness",
            "queries": ["subquery one", "subquery two"],
            "document_budget": 2,
            "requirements": ["all rollout artifacts"],
        },
        question="original question",
        max_queries=5,
        max_documents=10,
    )

    assert plan["queries"][0] == "original question"
    assert plan["document_budget"] == 10
    assert plan["minimum_documents"] == 6

    single = normalize_plan(
        {
            "strategy": "single",
            "queries": ["unneeded rewrite"],
            "document_budget": 3,
        },
        question="exact lookup",
        max_queries=5,
        max_documents=10,
    )
    assert single["queries"] == ["exact lookup"]
    assert single["document_budget"] == 1


def test_normalize_plan_single_document_budget_override() -> None:
    single = normalize_plan(
        {"strategy": "single", "document_budget": 3},
        question="exact lookup",
        max_queries=5,
        max_documents=10,
        single_document_budget=4,
    )
    assert single["document_budget"] == 4
    assert single["minimum_documents"] == 1

    capped = normalize_plan(
        {"strategy": "single"},
        question="exact lookup",
        max_queries=5,
        max_documents=10,
        single_document_budget=12,
    )
    assert capped["document_budget"] == 10


def test_normalize_plan_builds_requirement_tasks_and_conflict_slots() -> None:
    completeness = normalize_plan(
        {
            "strategy": "multi_document",
            "document_budget": 6,
            "requirements": ["Go tickets", "Python tickets", "TypeScript tickets"],
            "retrieval_tasks": [
                {"requirement": "Go tickets", "slot": "go", "query": "Go auth tickets"},
                {
                    "requirement": "Python tickets",
                    "slot": "python",
                    "query": "Python auth tickets",
                },
                {
                    "requirement": "TypeScript tickets",
                    "slot": "typescript",
                    "query": "TypeScript auth tickets",
                },
            ],
        },
        question="Across all SDKs, which has the highest number and all corresponding tickets?",
        max_queries=10,
        max_documents=10,
    )

    assert completeness["strategy"] == "completeness"
    assert [task["task_id"] for task in completeness["retrieval_tasks"]] == [
        "r1",
        "r2",
        "r3",
    ]
    assert completeness["document_budget"] == 10

    conflicting = normalize_plan(
        {"strategy": "single", "requirements": ["compare behavior"]},
        question="What are the previous and current thresholds?",
        max_queries=10,
        max_documents=10,
    )

    assert conflicting["strategy"] == "conflicting"
    assert [task["slot"] for task in conflicting["retrieval_tasks"]] == [
        "previous",
        "current",
    ]

    same_document = normalize_plan(
        {
            "strategy": "multi_document",
            "source_scope": "single_source",
            "requirements": ["file limit", "request limit"],
            "retrieval_tasks": [
                {"requirement": "file limit", "query": "file limit"},
                {"requirement": "request limit", "query": "request limit"},
            ],
        },
        question="What are the file and request limits for multipart upload?",
        max_queries=10,
        max_documents=10,
    )

    assert same_document["strategy"] == "single"
    assert same_document["document_budget"] == 1
    assert same_document["queries"] == [
        "What are the file and request limits for multipart upload?"
    ]

    multi_without_flag = normalize_plan(
        {
            "strategy": "multi_document",
            "source_scope": "multiple_sources",
            "document_budget": 4,
            "requirements": ["rollout schedule", "kill switch name"],
            "retrieval_tasks": [
                {"requirement": "rollout schedule", "query": "rollout schedule"},
                {"requirement": "kill switch name", "query": "kill switch name"},
            ],
        },
        question=(
            "In the observability change for tool invocation tracking, what is the "
            "staged rollout schedule and the temporary kill switch name?"
        ),
        max_queries=10,
        max_documents=10,
    )
    # requirements 数量不等于文档数量：默认 minimum=1
    assert multi_without_flag["strategy"] == "multi_document"
    assert multi_without_flag["minimum_documents"] == 1
    assert multi_without_flag["requires_multiple_distinct_documents"] is False

    multi_with_flag = normalize_plan(
        {
            **multi_without_flag,
            "requires_multiple_distinct_documents": True,
        },
        question=multi_without_flag["queries"][0],
        max_queries=10,
        max_documents=10,
    )
    # requires_multiple 只保留可解释性标记，不再改变 hard minimum（R49a）
    assert multi_with_flag["minimum_documents"] == 1
    assert multi_with_flag["requires_multiple_distinct_documents"] is True

    parameter_states = normalize_plan(
        {
            "strategy": "conflicting",
            "source_scope": "multiple_sources",
            "requirements": ["explicit null", "omitted field"],
            "retrieval_tasks": [
                {"requirement": "explicit null", "query": "max_tokens null"},
                {"requirement": "omitted field", "query": "max_tokens omitted"},
            ],
        },
        question=(
            "How does the normalizer handle max_tokens as null compared to leaving it "
            "out entirely?"
        ),
        max_queries=10,
        max_documents=10,
    )

    assert parameter_states["strategy"] == "single"
    assert parameter_states["source_scope"] == "single_source"
    assert parameter_states["document_budget"] == 1

    release_components = normalize_plan(
        {
            "strategy": "multi_document",
            "source_scope": "multiple_sources",
            "requirements": ["config flag", "other release components"],
        },
        question=(
            "What config flag enables the TP fanout planner, and what other new "
            "components are called out in the release notes?"
        ),
        max_queries=10,
        max_documents=10,
    )

    assert release_components["strategy"] == "single"
    assert release_components["source_scope"] == "single_source"
    assert release_components["document_budget"] == 1

    observed_result = normalize_plan(
        {
            "strategy": "multi_document",
            "source_scope": "multiple_sources",
            "document_budget": 4,
            "requirements": ["implementation change", "observed memory reduction"],
            "retrieval_tasks": [
                {"requirement": "implementation change", "query": "buffer reuse"},
                {"requirement": "observed memory reduction", "query": "memory reduction"},
            ],
        },
        question=(
            "What change reused temporary buffers, and what memory reduction was "
            "observed in the benchmark?"
        ),
        max_queries=10,
        max_documents=10,
    )

    assert observed_result["strategy"] == "single"
    assert observed_result["source_scope"] == "single_source"
    assert observed_result["document_budget"] == 1

    independent_updates = normalize_plan(
        {
            "strategy": "multi_document",
            "source_scope": "multiple_sources",
            "requirements": ["Python update", "Go update"],
        },
        question="What changes were released across the Python and Go SDKs?",
        max_queries=10,
        max_documents=10,
    )

    assert independent_updates["strategy"] == "multi_document"
    assert independent_updates["source_scope"] == "multiple_sources"


def test_requires_multiple_never_changes_hard_minimum() -> None:
    plan = normalize_plan(
        {
            "strategy": "multi_document",
            "source_scope": "multiple_sources",
            "document_budget": 4,
            "requires_multiple_distinct_documents": True,
            "requirements": ["requirement one", "requirement two"],
        },
        question="Which independent projects contributed to the release?",
        max_queries=10,
        max_documents=10,
    )

    assert plan["strategy"] == "multi_document"
    assert plan["minimum_documents"] == 1
    assert plan["requires_multiple_distinct_documents"] is True


def test_follow_up_query_preserves_exact_question_identifiers() -> None:
    query = preserve_query_identifiers(
        "fanout planner configuration setting",
        "Which flag controls TP allreduce in runtime.bandwidth_fanout.enabled?",
    )

    assert query.endswith("TP runtime.bandwidth_fanout.enabled")


def test_underspecified_query_classifies_as_composite() -> None:
    assert classify_intent("what does it do after the fix?", {}) == "composite"
    assert classify_intent("is that still supported?", {}) == "composite"
    assert classify_intent("why did this happen?", {}) == "composite"


def test_underspecified_does_not_capture_simple_or_colloquial() -> None:
    # 无标识符但措辞完整、非指代：保持 simple。
    assert classify_intent("What is a distributed system?", {}) == "simple"
    # 口语化优先于欠定检测，走便宜的改写路径。
    assert (
        classify_intent("so basically does it handle retries now?", {})
        == "colloquial"
    )
    # 带精确标识符：不欠定。
    assert (
        classify_intent("How do I enable --feature-flag?", {}) == "simple"
    )


def test_composite_markers_still_take_priority() -> None:
    assert classify_intent("What was the root cause of the incident?", {}) == (
        "composite"
    )


def test_is_underspecified_heuristic() -> None:
    assert _is_underspecified("what does it do after the fix?")
    assert _is_underspecified("is that true?")
    assert not _is_underspecified("What is a distributed system?")
    assert not _is_underspecified("List all SLOs in the playbook")
