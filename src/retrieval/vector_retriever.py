from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchAny

from src.config import QdrantConfig
from src.retrieval.lexical_retriever import build_bm25_retriever


def source_type_filter(source_types: list[str]) -> Filter | None:
    """构建 Qdrant payload 过滤：chunk 的 source_type 必须命中声明的来源集合。

    QdrantVectorStore 以 {page_content, metadata} 嵌套结构存储 payload，source_type
    落在 payload.metadata 下，因此 key 需用点路径 "metadata.source_type"。
    """
    if not source_types:
        return None
    return Filter(
        must=[
            FieldCondition(
                key="metadata.source_type",
                match=MatchAny(any=[str(value) for value in source_types]),
            )
        ]
    )


class SimilarityCandidateRetriever(BaseRetriever):
    """只负责高召回候选检索，规划、融合和重排由 LangGraph 编排。"""

    vector_store: QdrantVectorStore
    candidate_k: int = 30

    def _get_relevant_documents(self, query: str, *, run_manager) -> list[Document]:
        search_kwargs: dict = {"k": self.candidate_k}
        source_filter = source_type_filter(
            (run_manager.metadata or {}).get("source_types")
        )
        if source_filter is not None:
            search_kwargs["filter"] = source_filter
        documents = self.vector_store.similarity_search(query, **search_kwargs)
        return [
            Document(
                page_content=document.page_content,
                metadata={
                    **document.metadata,
                    "retrieval_channels": ["dense"],
                    "dense_rank": rank,
                },
            )
            for rank, document in enumerate(documents, start=1)
        ]


class HybridCandidateRetriever(BaseRetriever):
    """对 Dense 与 BM25 两路标准检索器执行 chunk 级 RRF。"""

    dense_retriever: BaseRetriever
    bm25_retriever: BaseRetriever
    candidate_k: int = 40
    rrf_k: int = 60
    text_section_weight: float = 0.8
    mode: Literal["dense", "bm25", "rank_sum", "rrf"] = "rrf"

    def _get_relevant_documents(self, query: str, *, run_manager) -> list[Document]:
        channel_results: dict[str, list[Document]] = {}
        source_types = (run_manager.metadata or {}).get("source_types")
        child_config: dict = {"callbacks": run_manager.get_child()}
        if source_types:
            child_config["metadata"] = {"source_types": source_types}
        if self.mode != "bm25":
            channel_results["dense"] = self.dense_retriever.invoke(
                query, config=child_config
            )
        if self.mode != "dense":
            channel_results["bm25"] = self.bm25_retriever.invoke(
                query, config=child_config
            )
        scores: dict[str, float] = defaultdict(float)
        documents: dict[str, Document] = {}
        channels: dict[str, list[str]] = defaultdict(list)
        channel_ranks: dict[str, dict[str, int]] = defaultdict(dict)

        for channel, results in channel_results.items():
            result_count = max(len(results), 1)
            for rank, document in enumerate(results, start=1):
                chunk_id = str(
                    document.metadata.get("chunk_id")
                    or f"{document.metadata.get('dsid', '')}::{document.page_content}"
                )
                section_weight = (
                    self.text_section_weight
                    if document.metadata.get("section") == "text"
                    else 1.0
                )
                if self.mode == "rank_sum":
                    scores[chunk_id] += section_weight * (
                        (result_count - rank + 1) / result_count
                    )
                else:
                    scores[chunk_id] += section_weight / (self.rrf_k + rank)
                documents.setdefault(chunk_id, document)
                channels[chunk_id].append(channel)
                channel_ranks[chunk_id][channel] = rank

        ordered_ids = sorted(scores, key=scores.get, reverse=True)[: self.candidate_k]
        union_ids = list(
            dict.fromkeys(
                str(
                    document.metadata.get("chunk_id")
                    or f"{document.metadata.get('dsid', '')}::{document.page_content}"
                )
                for results in channel_results.values()
                for document in results
            )
        )

        def trace_rows(ids: list[str]) -> list[dict[str, Any]]:
            return [
                {
                    "rank": rank,
                    "chunk_id": chunk_id,
                    "dsid": documents[chunk_id].metadata.get("dsid"),
                }
                for rank, chunk_id in enumerate(ids, start=1)
                if chunk_id in documents
            ]

        retrieval_trace = {
            "query": query,
            "mode": self.mode,
            "stages": {
                channel: [
                    {
                        "rank": rank,
                        "chunk_id": document.metadata.get("chunk_id"),
                        "dsid": document.metadata.get("dsid"),
                    }
                    for rank, document in enumerate(results, start=1)
                ]
                for channel, results in channel_results.items()
            }
            | {
                "union": trace_rows(union_ids),
                "channel_fusion": trace_rows(ordered_ids),
            },
        }
        return [
            Document(
                page_content=documents[chunk_id].page_content,
                metadata={
                    **documents[chunk_id].metadata,
                    "retrieval_channels": channels[chunk_id],
                    "retrieval_channel_ranks": channel_ranks[chunk_id],
                    "hybrid_rrf_score": scores[chunk_id],
                    # 只挂在第一条结果上，LangGraph 节点提取后立即移除。
                    **({"_retrieval_trace": retrieval_trace} if index == 0 else {}),
                },
            )
            for index, chunk_id in enumerate(ordered_ids)
        ]


def reciprocal_rank_fuse(
    query_results: list[list[Document]],
    max_documents: int,
    chunks_per_document: int,
    rrf_k: int = 60,
) -> dict[str, list[Document]]:
    """按文档执行 RRF；同一查询中的重复 chunk 只贡献该文档的最佳排名。"""
    scores: dict[str, float] = defaultdict(float)
    query_hits: dict[str, int] = defaultdict(int)
    chunks: dict[str, list[Document]] = defaultdict(list)
    seen_chunks: dict[str, set[str]] = defaultdict(set)
    task_ids: dict[str, set[str]] = defaultdict(set)
    slots: dict[str, set[str]] = defaultdict(set)

    for documents in query_results:
        best_rank: dict[str, int] = {}
        for rank, document in enumerate(documents, start=1):
            dsid = str(document.metadata.get("dsid") or "")
            if not dsid:
                continue
            best_rank.setdefault(dsid, rank)
            task_ids[dsid].update(document.metadata.get("retrieval_task_ids", []))
            if document.metadata.get("retrieval_slot"):
                slots[dsid].add(str(document.metadata["retrieval_slot"]))
            chunk_id = str(document.metadata.get("chunk_id") or document.page_content)
            if (
                chunk_id not in seen_chunks[dsid]
                and len(chunks[dsid]) < chunks_per_document
            ):
                seen_chunks[dsid].add(chunk_id)
                chunks[dsid].append(document)
        for dsid, rank in best_rank.items():
            scores[dsid] += 1.0 / (rrf_k + rank)
            query_hits[dsid] += 1

    ordered_ids = sorted(scores, key=lambda dsid: scores[dsid], reverse=True)
    groups: dict[str, list[Document]] = {}
    for document_rank, dsid in enumerate(ordered_ids[:max_documents], start=1):
        groups[dsid] = [
            Document(
                page_content=document.page_content,
                metadata={
                    **document.metadata,
                    "document_rrf_rank": document_rank,
                    "document_rrf_score": scores[dsid],
                    "query_hit_count": query_hits[dsid],
                    "retrieval_task_ids": sorted(task_ids[dsid]),
                    "retrieval_slots": sorted(slots[dsid]),
                },
            )
            for document in chunks[dsid]
        ]
    return groups


def build_parent_document_store(
    documents: list[Document],
) -> dict[str, list[Document]]:
    store: dict[str, list[Document]] = defaultdict(list)
    for document in documents:
        dsid = document.metadata.get("dsid")
        if dsid:
            store[str(dsid)].append(document)
    return dict(store)


def build_qdrant_client(config: QdrantConfig) -> QdrantClient:
    return QdrantClient(url=config.url, api_key=config.api_key or None)


def build_vector_store(
    qdrant_config: QdrantConfig,
    embeddings: Embeddings,
) -> QdrantVectorStore:
    client = build_qdrant_client(qdrant_config)
    return QdrantVectorStore(
        client=client,
        collection_name=qdrant_config.collection,
        embedding=embeddings,
    )


def build_candidate_retriever(
    qdrant_config: QdrantConfig,
    embeddings: Embeddings,
    candidate_k: int = 30,
) -> BaseRetriever:
    return SimilarityCandidateRetriever(
        vector_store=build_vector_store(qdrant_config, embeddings),
        candidate_k=candidate_k,
    )


def build_hybrid_candidate_retriever(
    qdrant_config: QdrantConfig,
    embeddings: Embeddings,
    documents: list[Document],
    dense_k: int = 30,
    bm25_k: int = 30,
    candidate_k: int = 40,
    rrf_k: int = 60,
    text_section_weight: float = 0.8,
    mode: Literal["dense", "bm25", "rank_sum", "rrf"] = "rrf",
) -> BaseRetriever:
    return HybridCandidateRetriever(
        dense_retriever=build_candidate_retriever(
            qdrant_config,
            embeddings,
            candidate_k=dense_k,
        ),
        bm25_retriever=build_bm25_retriever(
            documents,
            k=bm25_k,
            text_section_weight=text_section_weight,
        ),
        candidate_k=candidate_k,
        rrf_k=rrf_k,
        text_section_weight=text_section_weight,
        mode=mode,
    )
