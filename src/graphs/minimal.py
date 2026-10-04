"""3 阶段最小管线（默认 graph.mode）：plan → retrieve → CE top-k → 父展开 → generate。

依据增量实验结论：混合检索之后只需 CE 重排 + 父文档展开，
去 selection set-cover / assess_evidence judge / repair / followup-reuse-probe 回环 / entity_links，
且答案上下文不二次截断（宽上下文，max_parent_chunks 每 doc 上限），全量 100 题 66.0%。

（当时的对照对象"完整管线 62.0%"已随 `full` 线于 2026-09-21 整体移除；本图自此为默认。）
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.graphs.answer_nodes import generate_answer_node
from src.graphs.planning import plan_question_node
from src.graphs.retrieval_nodes import expand_selected_documents_query_aware, retrieve_queries_node
from src.graphs.state import RagState
from src.retrieval.vector_retriever import reciprocal_rank_fuse


def fuse_ce_expand_node(
    *,
    scorer,
    parent_documents,
    pool_size: int,
    chunks_per_document: int,
    rrf_k: int,
    top_k: int,
    max_parent_chunks: int,
):
    """RRF doc 池 → CE 打分 → CE 序 top-k → 父文档展开（无选择层、无 cap、无 judge）。"""

    def _node(state: RagState) -> RagState:
        groups = reciprocal_rank_fuse(
            state.get("query_results", []),
            max_documents=pool_size,
            chunks_per_document=chunks_per_document,
            rrf_k=rrf_k,
        )
        docs_by_dsid = {
            dsid: parent_documents[dsid] for dsid in groups if dsid in parent_documents
        }
        scores = scorer.score_documents(state["question"], docs_by_dsid)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        top_dsids = [dsid for dsid, _ in ranked[:top_k]]
        requirements = state.get("plan", {}).get("requirements", [state["question"]])
        answer_docs = expand_selected_documents_query_aware(
            selected_ids=top_dsids,
            candidate_groups=groups,
            parent_documents=parent_documents,
            question=state["question"],
            requirements=requirements,
            max_parent_chunks=max_parent_chunks,
        )
        return {
            **state,
            "answer_docs": answer_docs,
            "retrieved_docs": answer_docs,
            "document_ids": top_dsids,
            "selected_document_ids": top_dsids,
            "candidate_groups": groups,
            "evidence_sufficient": True,
            "rerank_history": [
                {
                    "round": 1,
                    "retrieval_mode": "initial",
                    "candidate_document_ids": list(groups),
                    "selected_document_ids": top_dsids,
                    "cross_encoder_rankings": [
                        {"dsid": dsid, "score": round(s, 6), "rank": i + 1}
                        for i, (dsid, s) in enumerate(ranked)
                    ],
                }
            ],
        }

    return _node


def build_minimal_graph(config, dependencies):
    graph = StateGraph(RagState)
    retrieval = config.retrieval
    features = config.features
    graph.add_node(
        "plan_question",
        plan_question_node(
            dependencies.llm,
            max_queries=retrieval.max_queries,
            max_documents=retrieval.max_documents,
            adaptive=features.adaptive_planning,
            fixed_document_budget=retrieval.fixed_document_budget,
            intent_routing=features.intent_routing,
            planning_max_tokens=config.llm.planning_max_tokens,
            single_document_budget=retrieval.single_document_budget,
        ),
    )
    graph.add_node(
        "retrieve_queries",
        retrieve_queries_node(
            dependencies.retriever,
            max_concurrency=retrieval.query_parallelism,
        ),
    )
    graph.add_node(
        "fuse_ce_expand",
        fuse_ce_expand_node(
            scorer=dependencies.cross_encoder,
            parent_documents=dependencies.parent_documents,
            pool_size=retrieval.candidate_documents,
            chunks_per_document=retrieval.chunks_per_document,
            rrf_k=retrieval.rrf_k,
            top_k=retrieval.single_document_budget,
            max_parent_chunks=retrieval.max_parent_chunks,
        ),
    )
    graph.add_node("generate_answer", generate_answer_node(dependencies.llm, features=features))
    graph.add_edge(START, "plan_question")
    graph.add_edge("plan_question", "retrieve_queries")
    graph.add_edge("retrieve_queries", "fuse_ce_expand")
    graph.add_edge("fuse_ce_expand", "generate_answer")
    graph.add_edge("generate_answer", END)
    return graph.compile(name="minimal_3stage")
