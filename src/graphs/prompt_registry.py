from __future__ import annotations

from src.chains.langchain_rag import ANSWER_PROMPT
from src.graphs.grounding import validate_prompt_grounding
from src.graphs.planning import PLAN_PROMPT


def benchmark_prompt_texts() -> dict[str, str]:
    """返回全部影响主流程行为的 prompt，供协议统一签名。"""
    return {
        "planning": PLAN_PROMPT,
        "answer": str(ANSWER_PROMPT),
    }


def validate_grounding_contract() -> list[str]:
    """校验全部影响主流程的 prompt 都含反捏造契约（preamble 或其 {grounding} 占位符）。

    返回未通过契约校验的 prompt 名列表；空列表 = 全部合规。版本签名串不参与校验。
    """
    return [
        name
        for name, text in benchmark_prompt_texts().items()
        if not validate_prompt_grounding(text)
    ]
