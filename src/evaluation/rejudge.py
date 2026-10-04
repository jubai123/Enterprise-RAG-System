"""N 次 majority 重判(离线修正单次 judge 的采样噪声)。

现役官方 judge(EnterpriseRAG-Bench metrics_based_eval)每题只调一次 DeepSeek,且温度走
provider 默认(非确定)——qst_0413 已证明单次会误杀。本模块对同一判据抽 N 次、多数票定论,
tie 不翻转(保守:保持原判错)。prompt 常量逐字来自 scratch/_tmp_judge_reeval_gen.py,
保证与 0413/0248 已有手工复核的数字可比。
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Sequence

from anthropic import APIConnectionError as AnthropicAPIConnectionError
from langchain_core.prompts import ChatPromptTemplate
from openai import APIConnectionError

JUDGE_STRICT = """You are a wholistic and detail-oriented answer evaluator. Given a query, a gold answer, and a candidate answer, evaluate if the candidate answer aligned with the gold answer.
The candidate answer must provide loosely the same information as the gold answer; core aspects must be addressed and must not conflict; specific quantities must match.

## Query
```
{query}
```

## Gold Answer
```
{gold_answer}
```

## Candidate Answer
```
{candidate_answer}
```

CRITICAL: Output only JSON: {{"aligned": "yes or no"}}"""

JUDGE_CORE = """You are grading a short factual QA answer. The gold answer states the expected facts. The candidate answer is the system's response.
Grade whether the candidate answers the CORE intent of the question correctly (same decisive answer as the gold: same role/person/value/decision on the central point).
Rules:
- If the candidate's central answer matches the gold's central answer, mark yes even if the candidate also adds extra context, qualifiers, or details that are factually supported by the documents it was given.
- Only mark no when the candidate's central answer actually conflicts with the gold's central answer, or omits it.
- Ignore wording differences and stylistic differences.

## Question
```
{query}
```

## Gold Answer
```
{gold_answer}
```

## Candidate Answer
```
{candidate_answer}
```

CRITICAL: Output only JSON: {{"aligned": "yes or no"}}"""

FLAVORS = {"strict": JUDGE_STRICT, "core": JUDGE_CORE}

_ALIGNED_RE = re.compile(r'"aligned"\s*:\s*"(yes|no)"', re.IGNORECASE)
_RETRYABLE = (APIConnectionError, AnthropicAPIConnectionError)

Vote = str  # "yes" | "no" | "?" | "ERR:<TypeName>"


def build_prompt(flavor: str = "strict") -> ChatPromptTemplate:
    """按 flavor 构建 human-only 判分 prompt(strict=现役 holistic 判法)。"""
    try:
        template = FLAVORS[flavor]
    except KeyError:
        raise ValueError(f"unknown judge flavor {flavor!r}; choose from {sorted(FLAVORS)}") from None
    return ChatPromptTemplate.from_messages([("human", template)])


def parse_aligned(content: str) -> str | None:
    """从 LLM 输出抽 aligned=yes/no;匹配不到返回 None。"""
    m = _ALIGNED_RE.search(content)
    return m.group(1).lower() if m else None


def invoke_vote(llm: Any, prompt: ChatPromptTemplate, row: dict[str, Any], retries: int = 2) -> Vote:
    """单次判分。连接类瞬时错误退避重试;解析失败返回 '?'。"""
    raw: str = ""
    for attempt in range(1, retries + 1):
        try:
            res = llm.invoke(
                prompt.invoke(
                    {
                        "query": row["question"],
                        "gold_answer": row["gold_answer"],
                        "candidate_answer": row["candidate_answer"],
                    }
                )
            )
            raw = res.content if hasattr(res, "content") else str(res)
            parsed = parse_aligned(raw)
            return parsed if parsed is not None else "?"
        except _RETRYABLE:
            if attempt < retries:
                import time

                time.sleep(1.5 * (2 ** (attempt - 1)))
                continue
            return "ERR:" + "connection"
        except Exception as exc:  # noqa: BLE001 —— 单样本失败不中断整轮,记 token 由多数票吸收
            return "ERR:" + type(exc).__name__
    return "?" if not raw else "?"


def run_majority(
    items: Sequence[dict[str, Any]],
    llm: Any,
    flavor: str = "strict",
    samples: int = 3,
    workers: int = 1,
) -> list[dict[str, Any]]:
    """对每题抽 samples 次多数票。返回与 items 对齐的结果行。

    每行:{question_id, votes, yes, no, invalid, majority, flipped_to_correct}。
    majority ∈ {yes, no, tie};tie = yes 数==no 数(无效样本不计)。flipped_to_correct
    以该题原判错(answer_correct 缺省 False)为前提:majority=yes 才算翻转。
    """
    prompt = build_prompt(flavor)

    def judge_one(row: dict[str, Any]) -> dict[str, Any]:
        votes = [invoke_vote(llm, prompt, row) for _ in range(samples)]
        agg = aggregate_votes(votes)
        return {
            "question_id": row["question_id"],
            **agg,
            "flipped_to_correct": agg["majority"] == "yes",
        }

    if workers > 1 and len(items) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(judge_one, row): row for row in items}
            return [f.result() for f in as_completed(futures)]
    return [judge_one(row) for row in items]


def aggregate_votes(votes: Sequence[Vote]) -> dict[str, Any]:
    """多数票聚合:只看有效 yes/no;平票判 tie(调用方保守处理为不翻转)。"""
    yes = sum(1 for v in votes if v == "yes")
    no = sum(1 for v in votes if v == "no")
    invalid = [v for v in votes if v not in ("yes", "no")]
    if yes == no:
        majority = "tie"
    else:
        majority = "yes" if yes > no else "no"
    return {"votes": list(votes), "yes": yes, "no": no, "invalid": invalid, "majority": majority}
