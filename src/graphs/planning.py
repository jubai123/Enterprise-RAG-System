from __future__ import annotations

import re
from typing import Any

from langchain_core.language_models import BaseChatModel

from src.graphs.grounding import GROUNDING_PREAMBLE, filter_ungrounded_queries
from src.graphs.node_utils import _json_object
from src.graphs.retrieval_policy import DEFAULT_RETRIEVAL_POLICIES
from src.graphs.routing import INTENT_ROUTING
from src.graphs.state import RagState
from src.observability.telemetry import invoke_text_model
from src.retrieval.candidate_text import extract_exact_query_identifiers

PLAN_PROMPT = """You plan retrieval for an enterprise RAG system.
{grounding}
Classify the question as one strategy: single, semantic, multi_document,
conflicting, or completeness.

- single: one source document should contain the answer.
- semantic: one source is likely, but terminology is indirect or ambiguous.
- multi_document: several changes, projects, incidents, or artifacts are required.
- conflicting: versions, old/new behavior, or contradictory sources must be compared.
- completeness: the question asks for an exhaustive list or broad coverage.

Also classify source_scope as single_source or multiple_sources. Multiple requested
facts do not imply multiple sources: limits, labels, tiers, or enum values from one
feature/PR remain single_source. Comparisons between null, omitted, default, enabled,
or disabled states of the same parameter or feature also remain single_source. Use
multiple_sources only when independent projects, artifacts, incidents, versions, SDKs,
or linked records must each supply evidence.

Do not infer the number of required documents from the number of requirements. A single
PR or change often covers many facts (limits, names, schedules, flags, old/new behavior),
so requires_multiple_distinct_documents must be false unless independent projects,
versions, SDKs, artifacts, incidents, or contradictory sources must EACH supply evidence.

Create one retrieval task per independent evidence requirement. Preserve exact names,
numbers, API terms, quoted phrases, SDKs, repositories, tickets, and named entities.
For conflicting questions, create separate previous and current slots. For exhaustive
or cross-project counting questions, use completeness and create a task for every named
group. Do not answer the question.

Return exactly one JSON object with no markdown code fences, no explanations,
and no text before or after it:
{{"strategy":"multi_document","source_scope":"multiple_sources","document_budget":6,
"requires_multiple_distinct_documents":false,
"requirements":["fact that the final answer must cover"],
"retrieval_tasks":[{{"requirement":"...","slot":"general",
"query":"focused search query"}}]}}

Question:
{question}
"""


def preserve_query_identifiers(query: str, question: str) -> str:
    missing = [
        identifier
        for identifier in extract_exact_query_identifiers(question)
        if identifier.lower() not in query.lower()
    ]
    if not missing:
        return query
    return f"{query} {' '.join(missing)}"


_COLLOQUIAL_STRONG = (
    r"\bi was wondering\b",
    r"\bso basically\b",
    r"\bcan'?t remember\b",
    r"\btrying to remember\b",
    r"\btrying to figure out\b",
    r"\bpoking around\b",
    r"\bwhat'?s the deal with\b",
    r"\bjust curious\b",
    r"\bis it staged or what\b",
    r"\bwhat'?s that called again\b",
    r"\bif anyone knows\b",
    r"\bsorry if\b",
    r"\bkinda\b",
)

_COLLOQUIAL_WEAK = (
    r"\bum\b",
    r"\buh\b",
    r"\byou know\b",
    r"\bwhatever\b",
    r"\boh\b",
    r"\bhey\b",
    r"\bi've been\b",
    r"\bi think\b",
    r"\bright\?",
    r"\bkind of\b",
)


def is_colloquial(question: str) -> bool:
    lowered = question.lower()
    strong = sum(len(re.findall(marker, lowered)) for marker in _COLLOQUIAL_STRONG)
    weak = sum(len(re.findall(marker, lowered)) for marker in _COLLOQUIAL_WEAK)
    return strong >= 1 or weak >= 2


_COMPOSITE_MARKERS = (
    r"\bwhat caused\b",
    r"\b(?:what (?:was|is|were) (?:the )?(?:confirmed |underlying )?root cause|root cause of (?:the |this |that |their )?(?:incident|outage|issue|problem|failure|degradation))\b",
    r"\bunderlying cause\b",
    r"\b(?:dedicated |production |regional )?incident\b",
    r"\bplaybook\b",
    r"\bRCA\b",
    r"\bpostmortems?\b",
    r"\bretrospectives?\b",
    r"\bmitigations?\b",
    r"\b(?:recommended )?remediation\b",
    r"\bpreflight\b",
    r"\boncall\b",
    r"\bhow should (?:support|oncall|customers)\b",
    r"\bsupport (?:response|explain|handle|guided|can)\b",
    r"\bSLO\b",
    r"\bSLA\b",
    r"\benforced (?:differently|in|by|at)\b",
    r"\bwhat is the (?:canonical|recommended|underlying)\b",
    r"\bwhere (?:in|does) (?:the )?(?:gateway|sdk|python|installer|redwood|control)\b",
    r"\bacross (?:the )?(?:python|typescript|go|java|sdk|projects?|repositories?|releases?|versions?)\b",
    r"\bmultiple projects\b",
    r"\bdifferent projects\b",
    r"\bprevious (?:thresholds?|tiers?|values?|versions?|behavior)\b",
    r"\bin v\d+\b",
    r"\bbefore running\b",
)

_MULTI_PERSPECTIVE_MARKERS = (
    r"\bvs\.?\b",
    r"\bversus\b",
    r"\bcompared to\b",
    r"\bdifference between\b",
    r"\bdiffer (?:between|from)\b",
    r"\bhow does \w+ (?:handle|treat|process)\b",
    r"\bwhether .{0,60} or\b",
    r"\balternative(?:ly)?\b",
)

_INTERROGATIVE_RE = re.compile(
    r"\b(?:what|which|how|why|when|who|can|do|does|did|is|are|was|were)\b"
)
_ANAPHORA_RE = re.compile(r"\b(?:it|that|this|there|they|them)\b")


def _is_underspecified(question: str) -> bool:
    """欠定/缺背景查询：疑问句但没有精确标识符，且为指代式短查询。

    真正欠定的查询缺实体描述，必然很短（如 "what does it do?"、"is that true?"）。
    长查询即使含 that/it/they 等词，多为关系从句或句中指代（"…change that made…"、
    "…tier, how does it differ…"），实体已在句内给出，不属于欠定——避免把普通技术
    问题误路由到 composite 多轮检索（github 39 题曾因此误标 20 题）。
    """
    lowered = question.lower().strip()
    if not _INTERROGATIVE_RE.search(lowered):
        return False
    if extract_exact_query_identifiers(question):
        return False
    tokens = lowered.split()
    if len(tokens) <= 3:
        return True
    if len(tokens) <= 8 and _ANAPHORA_RE.search(lowered):
        return True
    return False


def classify_intent(question: str, payload: dict[str, Any]) -> str:
    """确定性意图分类，覆盖 LLM 输出。

    优先级 composite > colloquial > composite(欠定) > multi_perspective > simple。
    payload 的 cross-source 标记（multi_document/conflicting + multiple_sources）
    作为 composite 的次要证据，兜底 LLM 已正确识别多源但措辞不含特征词的情况。
    欠定检测放在 colloquial 之后：口语化查询仍走便宜的改写路径，仅正式措辞的
    缺背景查询升级 composite。
    """
    lowered = question.lower()
    if any(re.search(marker, lowered) for marker in _COMPOSITE_MARKERS):
        return "composite"
    if is_colloquial(question):
        return "colloquial"
    if _is_underspecified(question):
        return "composite"
    if any(re.search(marker, lowered) for marker in _MULTI_PERSPECTIVE_MARKERS):
        return "multi_perspective"
    if (
        payload.get("strategy") in ("multi_document", "conflicting")
        and payload.get("source_scope") == "multiple_sources"
    ):
        return "composite"
    return "simple"


COLLOQUIAL_REWRITE_PROMPT = """Rewrite the following casual user query into a precise,
formal search query for technical documentation retrieval. Strip conversational fillers,
recover the underlying information need, and use exact technical wording. Preserve every
specific entity verbatim (metric names, API/endpoint names, version numbers, config keys,
class/method names, PR references, numeric values, acronyms). {grounding}
Output ONLY the rewritten query.

Casual query:
{search_query}

Context question:
{question}

Precise search query:"""


def rewrite_colloquial_queries(
    llm: BaseChatModel,
    question: str,
    plan: dict[str, Any],
    state: RagState,
    max_tokens: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """B 链：把口语化查询改写为精确技术措辞，保留全部标识符（每 task 一次 LLM 调用）。"""
    tasks = list(plan["retrieval_tasks"])
    rewritten: list[dict[str, str]] = []
    current_state = state
    for task in tasks:
        prompt = COLLOQUIAL_REWRITE_PROMPT.format(
            search_query=task["query"],
            question=question,
            grounding=GROUNDING_PREAMBLE,
        )
        response, calls = invoke_text_model(
            llm,
            prompt,
            node="rewrite_colloquial_query",
            state=current_state,
            max_tokens=max_tokens,
        )
        current_state = {**current_state, "model_calls": calls}
        rewritten_text = response.strip()
        grounded = filter_ungrounded_queries([rewritten_text], question, "")
        if not grounded:
            # 改写引入不在问题中的技术标识符：回退原 task query，不扩散捏造。
            rewritten_text = task["query"]
        rewritten.append(
            {
                **task,
                "query": preserve_query_identifiers(rewritten_text, question),
            }
        )
    return {
        **plan,
        "queries": [task["query"] for task in rewritten],
        "retrieval_tasks": rewritten,
    }, current_state["model_calls"]


def _build_plan_tasks(
    payload: dict[str, Any],
    requirements: list[str],
    question: str,
    task_limit: int,
) -> list[dict[str, str]]:
    tasks: list[dict[str, str]] = []
    raw_tasks = payload.get("retrieval_tasks", [])
    if isinstance(raw_tasks, list):
        for item in raw_tasks:
            if len(tasks) >= task_limit:
                break
            if not isinstance(item, dict):
                continue
            query = item.get("query")
            if not isinstance(query, str) or not query.strip():
                continue
            requirement = item.get("requirement")
            slot = item.get("slot", "general")
            tasks.append(
                {
                    "task_id": f"r{len(tasks) + 1}",
                    "requirement": (
                        requirement.strip()
                        if isinstance(requirement, str) and requirement.strip()
                        else requirements[min(len(tasks), len(requirements) - 1)]
                    ),
                    "slot": slot.strip() if isinstance(slot, str) and slot.strip() else "general",
                    "query": query.strip(),
                }
            )
    raw_queries = payload.get("queries", [])
    if not tasks and isinstance(raw_queries, list):
        for query in [question, *raw_queries]:
            if len(tasks) >= task_limit:
                break
            if isinstance(query, str) and query.strip() and all(
                task["query"] != query.strip() for task in tasks
            ):
                tasks.append(
                    {
                        "task_id": f"r{len(tasks) + 1}",
                        "requirement": requirements[min(len(tasks), len(requirements) - 1)],
                        "slot": "general",
                        "query": query.strip(),
                    }
                )
    if not tasks:
        tasks = [
            {
                "task_id": "r1",
                "requirement": requirements[0],
                "slot": "general",
                "query": question,
            }
        ]
    return tasks


def normalize_plan(
    payload: dict[str, Any],
    question: str,
    max_queries: int,
    max_documents: int,
    *,
    intent_routing: bool = True,
    single_document_budget: int = 1,
) -> dict[str, Any]:
    strategy = str(payload.get("strategy", "single"))
    if strategy not in DEFAULT_RETRIEVAL_POLICIES:
        strategy = "single"
    lowered_question = question.lower()
    same_source_state_comparison = (
        any(
            term in lowered_question
            for term in (" null", "omitted", "leaving it out", "left out")
        )
        and any(term in lowered_question for term in ("compared to", "versus", " vs "))
        and not any(
            term in lowered_question
            for term in ("previous version", "current version", "old version", "new version")
        )
    )
    same_release_components = (
        "release notes" in lowered_question
        and any(phrase in lowered_question for phrase in ("config flag", "configuration flag"))
        and not any(
            term in lowered_question
            for term in ("across ", "different projects", "multiple releases")
        )
    )
    single_change_with_observed_result = (
        bool(
            re.search(
                r"\bwhat (?:change|mechanism|proposal|update)\b",
                lowered_question,
            )
        )
        and any(
            phrase in lowered_question
            for phrase in ("was observed", "were observed", "measured", "benchmark")
        )
        and not any(
            phrase in lowered_question
            for phrase in (
                "different projects",
                "separate projects",
                "each sdk",
                "each project",
                "respectively",
            )
        )
        and not re.search(
            r"\bacross (?:the )?(?:python|typescript|go|java|sdk|project|"
            r"repository|version|release)s?\b",
            lowered_question,
        )
    )
    collapsed_by_same_source = (
        same_source_state_comparison
        or same_release_components
        or single_change_with_observed_result
    )
    intent = classify_intent(question, payload) if intent_routing else "simple"
    if collapsed_by_same_source and intent in ("multi_perspective", "composite"):
        # 同源状态比较/单变更+可观察结果：本质是单文档问题，路由回退最小链。
        intent = "colloquial" if is_colloquial(question) else "simple"
    route = INTENT_ROUTING[intent]
    if (
        "complete list" in lowered_question
        or "all corresponding" in lowered_question
        or ("across " in lowered_question and "highest number" in lowered_question)
    ):
        strategy = "completeness"
    elif strategy == "single" and any(
        phrase in lowered_question
        for phrase in ("previous and current", "old and new", "no longer needed")
    ):
        strategy = "conflicting"
    raw_source_scope = payload.get("source_scope")
    source_scope = (
        raw_source_scope
        if raw_source_scope in {"single_source", "multiple_sources"}
        else (
            "multiple_sources"
            if strategy in {"conflicting", "completeness"}
            else "single_source"
        )
    )
    if (
        same_source_state_comparison
        or same_release_components
        or single_change_with_observed_result
    ):
        strategy = "single"
        source_scope = "single_source"
    elif strategy in {"conflicting", "completeness"}:
        source_scope = "multiple_sources"
    elif source_scope == "single_source" and strategy == "multi_document":
        strategy = "single"
    policy = DEFAULT_RETRIEVAL_POLICIES[strategy]
    requires_multiple = bool(
        payload.get("requires_multiple_distinct_documents", False)
    )

    raw_budget = payload.get("document_budget", policy.document_budget)
    try:
        budget = int(raw_budget)
    except (TypeError, ValueError):
        budget = policy.document_budget
    budget = max(
        policy.minimum_documents,
        min(budget, policy.document_budget, max_documents),
    )

    requirements = payload.get("requirements", [])
    if not isinstance(requirements, list):
        requirements = []
    requirements = [
        item.strip() for item in requirements if isinstance(item, str) and item.strip()
    ]
    if not requirements:
        requirements = [question]
    task_limit = min(max_queries, policy.max_queries)
    tasks = _build_plan_tasks(payload, requirements, question, task_limit)
    if strategy == "conflicting" and len(tasks) < 2:
        tasks = [
            {
                **tasks[0],
                "task_id": "r1",
                "slot": "previous",
                "query": f"{question} previous behavior",
            },
            {
                "task_id": "r2",
                "requirement": requirements[-1],
                "slot": "current",
                "query": f"{question} current behavior",
            },
        ]
    if strategy == "completeness":
        budget = min(policy.document_budget, max_documents)
    # 意图升级：仅当 LLM/启发式把 C/D 类压成 single 且不是同源比较时才抬升策略。
    if (
        intent_routing
        and intent in ("multi_perspective", "composite")
        and strategy == "single"
        and not collapsed_by_same_source
    ):
        strategy = route.strategy
        source_scope = route.source_scope
        task_limit = min(max_queries, route.max_queries)
        tasks = _build_plan_tasks(payload, requirements, question, task_limit)
        budget = min(route.document_budget, max_documents)
        minimum_override = route.minimum_documents
    else:
        minimum_override = None
    if strategy == "single":
        tasks = [
            {
                "task_id": "r1",
                "requirement": "\n".join(requirements),
                "slot": "general",
                "query": question,
            }
        ]
        # 选档上限：默认 1 保持历史行为；single_document_budget>1 时放宽到 CE rank
        # N 内的 gold（ce5 验证 qst_0020/0167/0315/0337 等 rank 2-4 的 gold 因此进窗口）。
        budget = min(single_document_budget, max_documents)
    if intent_routing and intent == "colloquial" and strategy == "single":
        # 口语化改写后单文档检索，budget 2 给 rerank_elimination 留余量。
        budget = min(route.document_budget, max_documents)
    if minimum_override is not None:
        minimum = minimum_override
    elif strategy == "conflicting":
        minimum = 2
    elif strategy == "completeness":
        minimum = policy.minimum_documents
    else:
        # single / semantic / multi_document：minimum 只作为硬下限，
        # 不再受 LLM 偶发的 requires_multiple 影响（qst_0232 翻转来源）。
        # 实际需要几篇由 parse_coverage_document_ids 按 coverage 决定。
        minimum = 1
    plan = {
        "strategy": strategy,
        "source_scope": source_scope,
        "queries": [task["query"] for task in tasks],
        "retrieval_tasks": tasks,
        "document_budget": budget,
        "minimum_documents": min(minimum, budget),
        "requires_multiple_distinct_documents": requires_multiple,
        "requirements": requirements,
    }
    if intent_routing:
        plan["intent"] = intent
        plan["routing"] = route.routing
    return plan


def _ground_plan_task_queries(plan: dict[str, Any], question: str) -> dict[str, Any]:
    """接地校验：retrieval_tasks 中查询含不在问题里的技术标识符则替换为原问题。"""
    tasks: list[dict[str, str]] = []
    for task in plan.get("retrieval_tasks", []):
        query = task["query"]
        if not filter_ungrounded_queries([query], question, ""):
            query = question
        tasks.append({**task, "query": query})
    return {
        **plan,
        "retrieval_tasks": tasks,
        "queries": [task["query"] for task in tasks],
    }


def plan_question_node(
    llm: BaseChatModel,
    max_queries: int,
    max_documents: int,
    *,
    adaptive: bool = True,
    fixed_document_budget: int = 4,
    intent_routing: bool = True,
    planning_max_tokens: int | None = None,
    single_document_budget: int = 1,
):
    def _node(state: RagState) -> RagState:
        if adaptive:
            response, model_calls = invoke_text_model(
                llm,
                PLAN_PROMPT.format(
                    question=state["question"],
                    grounding=GROUNDING_PREAMBLE,
                ),
                node="plan_question",
                state=state,
                max_tokens=planning_max_tokens,
            )
            plan = _ground_plan_task_queries(
                normalize_plan(
                    _json_object(response),
                    state["question"],
                    max_queries=max_queries,
                    max_documents=max_documents,
                    intent_routing=intent_routing,
                    single_document_budget=single_document_budget,
                ),
                state["question"],
            )
            if (
                intent_routing
                and plan.get("intent") == "colloquial"
                and plan.get("routing", {}).get("rewrite")
            ):
                plan, model_calls = rewrite_colloquial_queries(
                    llm,
                    state["question"],
                    plan,
                    {**state, "model_calls": model_calls},
                    max_tokens=planning_max_tokens,
                )
        else:
            budget = max(1, min(fixed_document_budget, max_documents))
            task = {
                "task_id": "r1",
                "requirement": state["question"],
                "slot": "general",
                "query": state["question"],
            }
            plan = {
                "strategy": "semantic",
                "source_scope": "single_source",
                "queries": [state["question"]],
                "retrieval_tasks": [task],
                "document_budget": budget,
                "minimum_documents": 1,
                "requirements": [state["question"]],
            }
            model_calls = list(state.get("model_calls", []))
        return {
            **state,
            "plan": plan,
            "pending_queries": plan["queries"],
            "pending_tasks": plan["retrieval_tasks"],
            "executed_queries": [],
            "executed_tasks": [],
            "query_results": [],
            "base_query_results": [],
            "result_tasks": [],
            "retrieval_round": 0,
            "rerank_history": [],
            "retrieval_stage_history": [],
            "model_calls": model_calls,
            "node_metrics": [],
        }

    return _node
