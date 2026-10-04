"""chunk-CE 干净路径：plan → retrieve → 叶级 CE 直连 → 逐子查询父块展开 → 生成。

依据 ctx16 之后的归因重建：旧 minimal 的 CE 节点从未直接看到检索命中的
chunk——它 RRF 折叠成 doc 池后，对每 doc 的全部父 chunk 做词法重排 + 抽句，拼成
≤1000 字符混段再打分。这一层启发式把「检索漏 vs CE 掉 vs 生成败」混在一起，使旧实验
无法归因。本图把中间操作全部剥离：CE 直接对混合检索吐出的原始 chunk（完整问题配对）打分。

取法：**逐子查询各取 top-K 叶升父块，再在父块粒度去重合并**（`expand_after_ce_top_k`
即每子查询的叶预算）。CE 仍对原问题打分（子查询只当检索通道），故对合并池只打一次分
再按子查询切分即可。相较「CE 前就把各子查询的叶 union 成一个池、全局取 top-K」的旧做法，
本做法让每个子查询贡献自己的高分块，上下文宽度随子查询数增长，而不是被全局预算钉死
后靠抬高 K 吃进合并池 rank 7-16 的低分块。

父块由索引层预建（同一 (dsid,section) 的叶子按序拼成受控长度的块），见
src/ingestion/parent_blocks.py 与 scripts/build_parent_blocks.py。

不用 reciprocal_rank_fuse / rank_candidate_chunks / extract_relevant_window /
passage_chars —— chunkce 路径上不存在这些操作。
"""
from __future__ import annotations

from collections import defaultdict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from src.graphs.answer_nodes import generate_answer_node
from src.graphs.planning import plan_question_node
from src.graphs.retrieval_nodes import retrieve_queries_node
from src.graphs.state import RagState


def _chunk_key(document: Document) -> str:
    """chunk 的稳定去重键：优先 chunk_id，回落到正文。"""
    return str(document.metadata.get("chunk_id") or document.page_content)


def flatten_chunks(query_results: list[list[Document]]) -> list[Document]:
    """跨子查询 union 检索命中的原始 chunk，按 chunk_id 去重（保留首个）。

    这是 chunk→CE 之间唯一允许的操作——无损、只去重不重排不打分。顺序 = 子查询执行序。
    """
    seen: set[str] = set()
    pooled: list[Document] = []
    for results in query_results:
        for chunk in results:
            key = _chunk_key(chunk)
            if key in seen:
                continue
            seen.add(key)
            pooled.append(chunk)
    return pooled


def _leaf_to_block(parent_blocks, pool: list[Document]) -> dict[str, Document]:
    """叶 chunk_id → 其父块 Document 的映射。

    expand_chunks 对未命中叶返回其本身（identity），保证每叶都有对应；父块用
    child_chunk_ids 反查自己的叶子。
    """
    mapping: dict[str, Document] = {}
    for doc in parent_blocks.expand_chunks(pool) if parent_blocks else pool:
        if doc.metadata.get("kind") == "parent_block":
            for child in doc.metadata.get("child_chunk_ids", []):
                mapping[str(child)] = doc
        else:
            cid = doc.metadata.get("chunk_id")
            if cid:
                mapping.setdefault(str(cid), doc)
    return mapping


def _doc_rank_tables(
    ranked: list[tuple[Document, float]],
) -> tuple[dict[str, list[Document]], dict[str, float], list[tuple[str, float]]]:
    """从(按分降序的 chunk/块, score)构建 dsid 分组、每 doc max 分、doc 降序表。"""
    groups_by_dsid: dict[str, list[Document]] = defaultdict(list)
    doc_max: dict[str, float] = {}
    for chunk, score in ranked:
        dsid = str(chunk.metadata.get("dsid"))
        if not dsid:
            continue
        groups_by_dsid[dsid].append(chunk)
        doc_max[dsid] = max(doc_max.get(dsid, -1e9), score)
    ranked_docs = sorted(doc_max, key=lambda dsid: (-doc_max[dsid], dsid))
    return groups_by_dsid, doc_max, [(dsid, doc_max[dsid]) for dsid in ranked_docs]


def fuse_chunks_ce_node(
    *,
    scorer,
    pool_cap: int | None,
    parent_blocks=None,
    expand_after_ce_top_k: int = 6,
):
    """检索池 → 叶级 CE 打分 → 逐子查询 top-K 叶升父块合并（无 RRF/词法/拼贴）。"""

    def _node(state: RagState) -> RagState:
        question = state.get("question", "")
        pool = flatten_chunks(state.get("query_results", []))
        if pool_cap is not None and len(pool) > pool_cap:
            pool = pool[:pool_cap]

        # ---- 叶级 CE：CE 只看检索命中的原始 chunk，块分由块内最佳叶派生 ----
        scores = scorer.score_chunks(question, pool) if pool else []
        if len(scores) != len(pool):
            raise RuntimeError(
                "chunk-CE count mismatch: "
                f"{len(scores)} scores vs {len(pool)} chunks"
            )
        ranked = sorted(
            zip(pool, scores),
            key=lambda pair: (-pair[1], _chunk_key(pair[0])),
        )
        groups, doc_max, ranked_docs = _doc_rank_tables(ranked)
        leaf_to_block = _leaf_to_block(parent_blocks, pool)
        # 候选父块序由块内「最佳叶」的 CE rank 派生（CE 只对叶打分）。
        block_best: dict[str, tuple[Document, float, int]] = {}
        for rank, (leaf, _score) in enumerate(ranked, start=1):
            block = leaf_to_block.get(_chunk_key(leaf))
            if block is None:
                continue
            bid = _chunk_key(block)
            prev = block_best.get(bid)
            if prev is None or rank < prev[2]:
                block_best[bid] = (block, _score, rank)
        block_ordered = sorted(
            block_best.values(),
            key=lambda t: (t[2], _chunk_key(t[0])),
        )
        chunk_rankings = [
            {
                "parent_block_id": _chunk_key(b),
                "dsid": str(b.metadata.get("dsid")),
                "child_chunk_ids": list(b.metadata.get("child_chunk_ids", [])),
                "score": round(s, 6),
                "rank": i + 1,
                "kind": "parent_block",
                "best_leaf_rank": r,
            }
            for i, (b, s, r) in enumerate(block_ordered)
        ]
        # ---- 逐子查询各取 top-K 叶升父块，再跨子查询在父块粒度去重合并 ----
        # CE 对原问题打分且点对点，故复用合并池的分数，无需逐子查询重打。
        score_by_cid = {_chunk_key(d): s for d, s in zip(pool, scores)}
        selected: set[str] = set()
        for query_chunks in state.get("query_results", []):
            seen_in_query: set[str] = set()
            scored: list[tuple[Document, float]] = []
            for chunk in query_chunks:
                key = _chunk_key(chunk)
                # seen_in_query：子查询内去重；未打分：落在 pool_cap 截断之外。
                if key in seen_in_query or key not in score_by_cid:
                    continue
                seen_in_query.add(key)
                scored.append((chunk, score_by_cid[key]))
            scored.sort(key=lambda pair: (-pair[1], _chunk_key(pair[0])))
            for leaf, _score in scored[:expand_after_ce_top_k]:
                block = leaf_to_block.get(_chunk_key(leaf)) or leaf
                selected.add(_chunk_key(block))
        # 合并集按全局「最佳叶 CE 序」输出（与 chunk_rankings 同序，最强的在前）。
        answer_docs = [b for b, _s, _r in block_ordered if _chunk_key(b) in selected]
        top_dsids = list(
            dict.fromkeys(str(d.metadata.get("dsid")) for d in answer_docs)
        )

        doc_rankings = [
            {"dsid": dsid, "score": round(score, 6), "rank": i + 1}
            for i, (dsid, score) in enumerate(ranked_docs)
        ]
        trace: dict = {
            "round": 1,
            "retrieval_mode": "initial",
            "mode": "chunk_ce",
            "pool_size": len(pool),
            "per_query_leaf_top_k": expand_after_ce_top_k,
            "candidate_document_ids": [dsid for dsid, _score in ranked_docs],
            "cross_encoder_rankings": doc_rankings,
            "chunk_rankings": chunk_rankings,
            "doc_max_scores": {dsid: round(s, 6) for dsid, s in doc_max.items()},
        }
        return {
            **state,
            "answer_docs": answer_docs,
            "retrieved_docs": answer_docs,
            "document_ids": top_dsids,
            "selected_document_ids": top_dsids,
            "candidate_groups": dict(groups),
            "evidence_sufficient": True,
            "rerank_history": [trace],
        }

    return _node


def build_chunkce_graph(config, dependencies):
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
        "fuse_chunks_ce",
        fuse_chunks_ce_node(
            scorer=dependencies.cross_encoder,
            pool_cap=retrieval.chunk_pool_cap,
            parent_blocks=dependencies.parent_blocks,
            expand_after_ce_top_k=retrieval.expand_after_ce_top_k,
        ),
    )
    graph.add_node("generate_answer", generate_answer_node(dependencies.llm, features=features))
    graph.add_edge(START, "plan_question")
    graph.add_edge("plan_question", "retrieve_queries")
    graph.add_edge("retrieve_queries", "fuse_chunks_ce")
    graph.add_edge("fuse_chunks_ce", "generate_answer")
    graph.add_edge("generate_answer", END)
    return graph.compile(name="chunkce_3stage")
