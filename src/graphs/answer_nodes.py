from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from src.chains.langchain_rag import ANSWER_PROMPT, format_context
from src.config import FeatureConfig
from src.graphs.grounding import (
    GROUNDING_PREAMBLE,
    extract_ungrounded_identifiers,
)
from src.graphs.node_utils import format_retrieval_guidance
from src.graphs.state import RagState
from src.observability.telemetry import invoke_text_model
from src.retrieval.candidate_text import (
    evidence_signal_matches,
    required_evidence_signals,
)


def compute_answer_coverage_gap(
    question: str,
    requirements: list[str],
    answer: str,
) -> list[str]:
    """返回答案缺失必需值型信号的需求列表。

    只查值型信号（time/http/size/three_item），纯 identifier 需求（答案没复读实体名）
    不触发，避免误报。需求无可核验信号直接跳过。
    """
    gap: list[str] = []
    for requirement in requirements:
        signals = required_evidence_signals(requirement)
        value_signals = {
            name: pattern
            for name, pattern in signals.items()
            if name != "identifier"
        }
        if not value_signals:
            continue
        if any(
            not evidence_signal_matches(name, pattern, requirement, answer)
            for name, pattern in value_signals.items()
        ):
            gap.append(requirement)
    return gap


def generate_answer_node(
    llm: BaseChatModel,
    *,
    features: FeatureConfig | None = None,
):
    def _node(state: RagState) -> RagState:
        # 生成授权 = evidence_sufficient（minimal/chunkce 恒为 True，由选择层直接放行）。
        if not state.get("evidence_sufficient", False):
            missing = "; ".join(state.get("missing_evidence", []))
            answer = "The available evidence is insufficient to answer reliably."
            if missing:
                answer += f" Missing evidence: {missing}."
            return {
                **state,
                "answer": answer,
                "answer_coverage_missing": [],
            }

        plan = state["plan"]
        prompt_value = ANSWER_PROMPT.invoke(
            {
                "question": state["question"],
                "requirements": "\n".join(f"- {x}" for x in plan["requirements"]),
                "retrieval_guidance": format_retrieval_guidance(state),
                "context": format_context(state.get("answer_docs", [])),
                "grounding": GROUNDING_PREAMBLE,
            }
        )
        answer, model_calls = invoke_text_model(
            llm,
            prompt_value,
            node="generate_answer",
            state=state,
        )
        if not answer.strip():
            # 思考截断/空输出：不写空答案，回退诚实的 insufficient 说明。
            answer = "The available evidence is insufficient to answer reliably."
            return {
                **state,
                "answer": answer,
                "answer_coverage_missing": [],
                "answer_grounding_violations": [],
                "model_calls": model_calls,
            }
        answer_coverage_missing: list[str] = []
        if features and features.answer_coverage_gate:
            answer_coverage_missing = compute_answer_coverage_gap(
                state["question"],
                plan["requirements"],
                answer,
            )
        answer_grounding_violations: list[str] = []
        if features and features.answer_grounding_gate:
            evidence_text = "\n".join(
                document.page_content for document in state.get("answer_docs", [])
            )
            answer_grounding_violations = extract_ungrounded_identifiers(
                answer,
                state["question"],
                evidence_text,
            )
        return {
            **state,
            "answer": answer,
            "answer_coverage_missing": answer_coverage_missing,
            "answer_grounding_violations": answer_grounding_violations,
            "model_calls": model_calls,
        }

    return _node
