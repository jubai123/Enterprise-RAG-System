from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.graphs.state import RagState
from src.retrieval.candidate_text import rank_candidate_chunks


def retrieve_queries_node(
    retriever: BaseRetriever,
    max_concurrency: int,
):
    def _node(state: RagState) -> RagState:
        tasks = state.get("pending_tasks", [])
        if not tasks:
            tasks = [
                {
                    "task_id": f"follow_up_{index + 1}",
                    "requirement": "\n".join(state.get("missing_evidence", [])),
                    "slot": "follow_up",
                    "query": query,
                }
                for index, query in enumerate(state.get("pending_queries", []))
            ]
        queries = [task["query"] for task in tasks]
        source_types = state.get("source_types")
        retrieval_config: dict = {"max_concurrency": max_concurrency}
        if source_types:
            retrieval_config["metadata"] = {"source_types": source_types}
        results = (
            retriever.batch(queries, config=retrieval_config)
            if queries
            else []
        )
        tagged_results: list[list[Document]] = []
        retrieval_stage_history = list(state.get("retrieval_stage_history", []))
        for task, documents in zip(tasks, results):
            retrieval_trace = (
                documents[0].metadata.get("_retrieval_trace") if documents else None
            )
            if isinstance(retrieval_trace, dict):
                retrieval_stage_history.append(
                    {
                        **retrieval_trace,
                        "round": state.get("retrieval_round", 0) + 1,
                        "task_id": task["task_id"],
                        "slot": task["slot"],
                    }
                )
            tagged_results.append(
                [
                    Document(
                        page_content=document.page_content,
                        metadata={
                            **{
                                key: value
                                for key, value in document.metadata.items()
                                if key != "_retrieval_trace"
                            },
                            "retrieval_task_ids": [task["task_id"]],
                            "retrieval_slot": task["slot"],
                        },
                    )
                    for document in documents
                ]
            )
        return {
            **state,
            "pending_queries": [],
            "pending_tasks": [],
            "executed_queries": state.get("executed_queries", []) + queries,
            "executed_tasks": state.get("executed_tasks", []) + tasks,
            "base_query_results": state.get("base_query_results", []) + tagged_results,
            "query_results": state.get("base_query_results", []) + tagged_results,
            "result_tasks": state.get("executed_tasks", []) + tasks,
            "retrieval_round": state.get("retrieval_round", 0) + 1,
            "retrieval_stage_history": retrieval_stage_history,
        }

    return _node


def expand_selected_documents_query_aware(
    selected_ids: list[str],
    candidate_groups: dict[str, list[Document]],
    parent_documents: dict[str, list[Document]],
    question: str,
    requirements: list[str],
    max_parent_chunks: int,
) -> list[Document]:
    """按问题相关性补齐父文档证据（R49c）。

    候选 chunk 始终优先；剩余父 chunk 去重后按 rank_candidate_chunks 相关性
    排序，而不是按父文档存储顺序截断，避免答案位于文档后部时被
    max_parent_chunks 挤出。与 _enrich_rerank_chunks 的语义保持一致。
    """
    selected: list[Document] = []
    for dsid in selected_ids:
        candidates = candidate_groups[dsid]
        if dsid not in parent_documents:
            selected.extend(candidates)
            continue
        existing_chunk_ids = {
            str(document.metadata.get("chunk_id")) for document in candidates
        }
        extra = [
            document
            for document in parent_documents[dsid]
            if str(document.metadata.get("chunk_id")) not in existing_chunk_ids
        ]
        combined = list(candidates)
        if extra:
            combined.extend(
                rank_candidate_chunks(question, requirements, extra)
            )
        selected.extend(combined[:max_parent_chunks])
    return selected


