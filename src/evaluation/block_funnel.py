"""parent_block 粒度的决策级错误归因(替代 dsid 级结论)。

dsid 漏斗的坑:gold doc 进池 ≠ 答案 block 进池。doc recall==100 时仍可能决定性 parent
block 从未进入检索池或 CE top-k,却被 doc 级分类误标成纯生成失败。本模块把每个错题的
「expected 证据块」(gold → 父块词法覆盖匹配出)放到引用上下文/CE 池/检索池三级里定位,
并结合 majority re-judge 先剥 judge 误杀,得到可驱动决策的桶:

  judge_flip_or_gold_contradiction  majority 判 yes(原判错 = 误杀/gold 矛盾)
  in_context_but_wrong              expected 块已在引用上下文仍错 → 真生成残差
  ce_select_drop                    expected 块进 CE 池但 rank>top_k,未选入上下文
  fusion_drop                       expected 块叶进 union 但 channel_fusion 丢掉
  retrieval_block_miss              expected 块连检索池(union)都没进
  no_lexical_evidence               gold 词法与 expected doc 全无重叠,词法判不了
  no_expected_doc                   gold 未标 expected doc,无法归因

expected 块匹配是启发式(bigram_cov>=0.15 或 token_cov>=0.50;不达标取 argmax),全量
cov 随行输出供审计,不下绝对结论。词法器与 scratch/_tmp_passage_presence.py 一致。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

# ---- 词法器(源自 scratch/_tmp_passage_presence.py)----
_STOP = set(
    "the and of to in for is are was were with on by at that this these those it its as from an a or "
    "not be been being can could should would do does did has have had what when where which who whom "
    "how why about into over under up then than so also per each their our your my his her".split()
)
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9%./+_:-]*")


def norm_tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 2 and t not in _STOP]


def token_coverage(gold_toks: list[str], haystack: str) -> float:
    if not gold_toks:
        return 1.0
    hs = set(norm_tokens(haystack))
    return sum(1 for t in gold_toks if t in hs) / len(gold_toks)


def bigram_coverage(gold_toks: list[str], haystack: str) -> float:
    if len(gold_toks) < 2:
        return 1.0 if not gold_toks else 0.0
    hs = set(norm_tokens(haystack))
    pairs = {(a, b) for a, b in zip(gold_toks, gold_toks[1:]) if a in hs and b in hs}
    return len(pairs) / (len(gold_toks) - 1)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _evidence_blocks_for_doc(
    gold_toks: list[str], blocks: list[dict[str, Any]]
) -> tuple[list[str], dict[str, list[float]], dict[str, Any]]:
    """单 dsid:返回(证据块 id 列表, cov_by_block, best)。

    判定:bigram_cov>=0.15 或 token_cov>=0.50 达标的块;都不达标则退回 argmax bigram(须>0)。
    """
    scored: list[tuple[str, float, float]] = []
    for block in blocks:
        text = block["text"]
        bi = bigram_coverage(gold_toks, text)
        tk = token_coverage(gold_toks, text)
        scored.append((block["parent_id"], bi, tk))
    cov_by_block = {pid: [round(tk, 3), round(bi, 3)] for pid, bi, tk in scored}
    qualifying = [pid for pid, bi, tk in scored if bi >= 0.15 or tk >= 0.50]
    if qualifying:
        chosen = qualifying
    else:
        best = max(scored, key=lambda s: s[1]) if scored else None
        chosen = [best[0]] if best and best[1] > 0.0 else []
    best = max(scored, key=lambda s: s[1]) if scored else None
    return chosen, cov_by_block, {"best_block": best[0] if best else None, "best_bi": best[1] if best else 0.0}


def load_parent_structure(
    parent_blocks_path: str | Path, need_dsids: set[str]
) -> tuple[dict[str, str], dict[str, list[dict[str, Any]]]]:
    """一趟扫父块文件:全量 leaf→parent 映射 + 只留需要 dsid 的块记录。"""
    leaf_to_parent: dict[str, str] = {}
    dsid_blocks: dict[str, list[dict[str, Any]]] = {}
    with Path(parent_blocks_path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            block = json.loads(line)
            pid = block.get("parent_id")
            dsid = block.get("dsid")
            if not pid:
                continue
            for child in block.get("child_chunk_ids", []):
                leaf_to_parent[child] = pid
            if dsid in need_dsids:
                dsid_blocks.setdefault(dsid, []).append(block)
    return leaf_to_parent, dsid_blocks


def _empty_trace(qid: str) -> dict[str, Any]:
    return {"question_id": qid, "bucket": "no_expected_doc", "note": "gold 无 expected_doc_ids"}


def build_block_funnel(
    gold_questions_file: str | Path,
    parent_blocks_file: str | Path,
    run_dir: str | Path,
    official_results_file: str | Path,
    rejudge_file: str | Path | None = None,
    recall_funnel_file: str | Path | None = None,
    context_top_k: int = 6,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    gold_by_id = {r["question_id"]: r for r in _load_jsonl(Path(gold_questions_file))}
    official = json.loads(Path(official_results_file).read_text(encoding="utf-8"))
    wrong_ids = [q["question_id"] for q in official["questions"] if not q.get("answer_correct")]

    rejudge: dict[str, dict[str, Any]] = {}
    if rejudge_file and Path(rejudge_file).exists():
        for r in json.loads(Path(rejudge_file).read_text(encoding="utf-8")).get("per_question", []):
            rejudge[r["question_id"]] = r

    old_cause: dict[str, str] = {}
    if recall_funnel_file and Path(recall_funnel_file).exists():
        old_cause = {
            q["question_id"]: q.get("primary_failure_cause", "")
            for q in json.loads(Path(recall_funnel_file).read_text(encoding="utf-8")).get("questions", [])
        }

    trace_by_id: dict[str, dict[str, Any]] = {}
    trace_file = run_dir / "graph_traces.jsonl"
    if trace_file.exists():
        trace_by_id = {r["question_id"]: r for r in _load_jsonl(trace_file)}

    need_dsids: set[str] = set()
    for qid in wrong_ids:
        q = gold_by_id.get(qid, {})
        need_dsids.update(str(d) for d in (q.get("expected_doc_ids") or []))

    leaf_to_parent, dsid_blocks = load_parent_structure(parent_blocks_file, need_dsids)

    official_row = {q["question_id"]: q for q in official["questions"]}
    per_question: list[dict[str, Any]] = []
    bucket_counts: dict[str, int] = {}

    for qid in sorted(wrong_ids):
        q = gold_by_id.get(qid, {})
        gold_toks = norm_tokens(q.get("gold_answer", ""))
        expected_ids = [str(d) for d in (q.get("expected_doc_ids") or [])]
        if not expected_ids or not q.get("gold_answer"):
            per_question.append(_empty_trace(qid))
            bucket_counts["no_expected_doc"] = bucket_counts.get("no_expected_doc", 0) + 1
            continue

        # 1) expected 证据块
        expected_block_ids: list[str] = []
        cov_by_block: dict[str, list[float]] = {}
        doc_best: dict[str, dict[str, Any]] = {}
        for dsid in expected_ids:
            blocks = dsid_blocks.get(dsid)
            if not blocks:
                doc_best[dsid] = {"best_block": None, "best_bi": 0.0}
                continue
            chosen, cov, best = _evidence_blocks_for_doc(gold_toks, blocks)
            expected_block_ids.extend(pid for pid in chosen if pid not in expected_block_ids)
            cov_by_block.update(cov)
            doc_best[dsid] = best
        no_evidence = not expected_block_ids

        # 2) 上下文 / CE 池 / 检索池
        trace = trace_by_id.get(qid, {})
        rerank = trace.get("rerank_history") or []
        chunk_rankings = rerank[-1].get("chunk_rankings", []) if rerank else []
        rank_by_block = {r.get("parent_block_id"): r.get("rank") for r in chunk_rankings if r.get("parent_block_id")}
        ce_pool_ids = set(rank_by_block)
        cited_ids = list(trace.get("answer_chunk_ids") or [])
        context_source = "answer_chunk_ids"
        if not cited_ids:
            ranked = sorted(rank_by_block.items(), key=lambda kv: kv[1])
            cited_ids = [pid for pid, _ in ranked[: min(context_top_k, len(ranked))]]
            context_source = "rerank_top_k_fallback"

        union_leaves: set[str] = set()
        fusion_leaves: set[str] = set()
        for qt in trace.get("retrieval_stage_history", []):
            for stage, rows in qt.get("stages", {}).items():
                leaves = {str(r["chunk_id"]) for r in rows if isinstance(r, dict) and r.get("chunk_id")}
                if stage == "union":
                    union_leaves |= leaves
                elif stage == "channel_fusion":
                    fusion_leaves |= leaves
        union_pool_ids = {leaf_to_parent[leaf] for leaf in union_leaves if leaf in leaf_to_parent}
        fusion_pool_ids = {leaf_to_parent[leaf] for leaf in fusion_leaves if leaf in leaf_to_parent}

        # 3) 分桶
        expected_set = set(expected_block_ids)
        in_context = expected_set & set(cited_ids)
        in_ce_pool = expected_set & ce_pool_ids
        in_union_pool = expected_set & union_pool_ids
        in_fusion_pool = expected_set & fusion_pool_ids

        rej = rejudge.get(qid, {})
        majority = rej.get("majority", "unjudged")
        flipped = bool(rej.get("flipped_to_correct", False))

        if flipped:
            bucket = "judge_flip_or_gold_contradiction"
        elif no_evidence:
            bucket = "no_lexical_evidence"
        elif in_context:
            bucket = "in_context_but_wrong"
        elif in_ce_pool:
            bucket = "ce_select_drop"
        elif in_union_pool:
            bucket = "fusion_drop" if not in_fusion_pool else "ce_select_drop"
        else:
            bucket = "retrieval_block_miss"
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

        snippet = ""
        if doc_best:
            best = next((d for d in doc_best.values() if d.get("best_block")), None)
            if best:
                blk_id = best["best_block"]
                for blocks in dsid_blocks.values():
                    for b in blocks:
                        if b["parent_id"] == blk_id:
                            snippet = b["text"][:200]
                            break
                    if snippet:
                        break

        expected_rank = {
            pid: rank_by_block.get(pid) for pid in expected_block_ids if pid in rank_by_block
        }
        # evidence_full_present:全部 expected 块都在 top-6 上下文(残余上界才接近「纯生成」);
        # 只部分在场 → 决定性子句可能仍在未取的兄弟块(0210/0248/0337 型),属证据选错。
        evidence_full_present = bool(expected_set) and expected_set <= set(cited_ids)
        per_question.append(
            {
                "question_id": qid,
                "question_type": q.get("question_type"),
                "expected_doc_ids": expected_ids,
                "expected_block_ids": expected_block_ids,
                "cov_by_block": cov_by_block,
                "doc_best": doc_best,
                "no_lexical_evidence": no_evidence,
                "context_source": context_source,
                "cited_block_ids": cited_ids,
                "expected_in_context": sorted(in_context),
                "expected_in_ce_pool": sorted(in_ce_pool),
                "expected_in_union_pool": sorted(in_union_pool),
                "expected_rank_in_ce_pool": expected_rank,
                "evidence_full_present": evidence_full_present,
                "bucket": bucket,
                "majority_verdict": majority,
                "flipped_to_correct": flipped,
                "doc_recall_pct": official_row.get(qid, {}).get("document_recall_pct"),
                "old_dsid_cause": old_cause.get(qid, ""),
                "evidence_snippet": snippet,
            }
        )

    ordered = [
        "judge_flip_or_gold_contradiction",
        "in_context_but_wrong",
        "ce_select_drop",
        "fusion_drop",
        "retrieval_block_miss",
        "no_lexical_evidence",
        "no_expected_doc",
    ]
    summary = {
        "total_wrong": len(wrong_ids),
        "bucket_counts": {k: bucket_counts.get(k, 0) for k in ordered if bucket_counts.get(k)},
        "true_residual_generation": bucket_counts.get("in_context_but_wrong", 0),
    }
    return {
        "schema_version": 1,
        "context_top_k": context_top_k,
        "summary": summary,
        "per_question": per_question,
    }
