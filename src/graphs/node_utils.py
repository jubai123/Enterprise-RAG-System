"""图节点共用的纯函数。"""
from __future__ import annotations

import json
import re
from typing import Any

from src.graphs.state import RagState


def _json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def format_retrieval_guidance(state: RagState) -> str:
    history = state.get("rerank_history", [])
    if not history:
        return "No additional retrieval assessment is available."
    latest = history[-1]
    guardrail_reason = latest.get("guardrail_reason")
    if guardrail_reason:
        signals = str(guardrail_reason).removeprefix("direct_evidence:")
        return (
            "The final document was selected because it contains the required direct "
            f"evidence signals: {signals}. Verify their exact values in the context."
        )
    reason = str(latest.get("selection_reason") or "").strip()
    return reason or "The selected documents best cover the retrieval requirements."
