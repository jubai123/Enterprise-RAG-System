from __future__ import annotations

from src.evaluation.rejudge import (
    aggregate_votes,
    build_prompt,
    parse_aligned,
    run_majority,
)


class _Response:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLLM:
    """按调用序号循环吐出预设 content;invoke 参数(ChatPromptValue)不读。"""

    def __init__(self, contents: list[str]) -> None:
        self.contents = contents
        self.i = 0

    def invoke(self, _value) -> _Response:
        content = self.contents[self.i % len(self.contents)]
        self.i += 1
        return _Response(content)


_ROW = {
    "question_id": "qst_x",
    "question": "Who approves X?",
    "gold_answer": "cost-ops.",
    "candidate_answer": "cost-ops approves.",
}


def test_aggregate_majority_yes() -> None:
    agg = aggregate_votes(["yes", "yes", "no"])
    assert agg["majority"] == "yes"
    assert agg["yes"] == 2
    assert agg["no"] == 1
    assert agg["invalid"] == []


def test_aggregate_tie_ignores_invalid() -> None:
    agg = aggregate_votes(["yes", "no", "?", "ERR:x"])
    assert agg["majority"] == "tie"
    assert agg["invalid"] == ["?", "ERR:x"]


def test_aggregate_no_when_invalid_outnumber() -> None:
    agg = aggregate_votes(["no", "no", "?"])
    assert agg["majority"] == "no"


def test_parse_aligned() -> None:
    assert parse_aligned('{"aligned": "no"}') == "no"
    assert parse_aligned('{"aligned": "Yes"}') == "yes"
    assert parse_aligned("I think it is correct") is None


def test_run_majority_flips_on_2_of_3_yes() -> None:
    llm = _FakeLLM(['{"aligned": "yes"}', '{"aligned": "no"}', '{"aligned": "yes"}'])
    results = run_majority([_ROW], llm, flavor="strict", samples=3, workers=1)
    row = results[0]
    assert row["question_id"] == "qst_x"
    assert row["votes"] == ["yes", "no", "yes"]
    assert row["majority"] == "yes"
    assert row["flipped_to_correct"] is True


def test_run_majority_tie_does_not_flip() -> None:
    llm = _FakeLLM(['{"aligned": "yes"}', '{"aligned": "no"}', 'garbage-no-json'])
    results = run_majority([_ROW], llm, flavor="strict", samples=3, workers=1)
    assert results[0]["majority"] == "tie"
    assert results[0]["flipped_to_correct"] is False


def test_run_majority_unanimous_no_does_not_flip() -> None:
    llm = _FakeLLM(['{"aligned": "no"}'] * 3)
    results = run_majority([_ROW], llm, flavor="core", samples=3, workers=1)
    assert results[0]["majority"] == "no"
    assert results[0]["flipped_to_correct"] is False


def test_run_majority_concurrent_returns_all_rows() -> None:
    rows = [
        {**_ROW, "question_id": f"qst_{i}"} for i in range(4)
    ]
    llm = _FakeLLM(['{"aligned": "yes"}', '{"aligned": "no"}'])
    results = run_majority(rows, llm, flavor="strict", samples=2, workers=2)
    assert {r["question_id"] for r in results} == {f"qst_{i}" for i in range(4)}
    assert all(set(r["votes"]) <= {"yes", "no"} for r in results)


def test_build_prompt_rejects_unknown_flavor() -> None:
    try:
        build_prompt("bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown flavor")
