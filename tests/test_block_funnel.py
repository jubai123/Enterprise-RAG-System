from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.block_funnel import build_block_funnel

# gold_answer 的判别性 token 放进 b_answer 的正文,词法能定位到该证据块。
GOLD = "The server racks use zeta cooling gamma fluid and omega pressure relief."
B_ANSWER_TEXT = "ops note: racks use zeta cooling gamma fluid; omega pressure relief active."
B_SIBLING_TEXT = "server fleet inventory and uptime figures for the quarter."
GOLD_TOKENS_PRESENT = True

DSID = "d1"


def _leaf(block_sec: str, idx: int, pid: str) -> dict:
    return {"rank": idx + 1, "chunk_id": f"{DSID}::{block_sec}::{idx}", "dsid": DSID}


def _parent_block(parent_id: str, text: str, child: str) -> dict:
    return {
        "parent_id": parent_id,
        "dsid": DSID,
        "section": parent_id.split("::")[1],
        "text": text,
        "child_chunk_ids": [child],
        "chars": len(text),
    }


def _write_gold(tmp: Path) -> None:
    with (tmp / "gold.jsonl").open("w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "question_id": "q1",
                    "question_type": "basic",
                    "question": "Which cooling fluid do the racks use?",
                    "gold_answer": GOLD,
                    "expected_doc_ids": [DSID],
                }
            )
            + "\n"
        )


def _write_parent_blocks(tmp: Path) -> None:
    records = [
        _parent_block(f"{DSID}::sec::block_0", B_ANSWER_TEXT, f"{DSID}::sec::0"),
        _parent_block(f"{DSID}::sec::block_1", B_SIBLING_TEXT, f"{DSID}::sec::1"),
    ]
    with (tmp / "blocks.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _write_official(tmp: Path, recall_pct: float = 100.0) -> None:
    with (tmp / "official.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "questions": [
                    {
                        "question_id": "q1",
                        "answer_correct": False,
                        "document_recall_pct": recall_pct,
                        "completeness_pct": 0.0,
                    }
                ]
            },
            f,
        )


def _write_trace(tmp: Path, *, union_children: list[str], fusion_children: list[str], ce_parents: list[str], cited: list[str]) -> None:
    """合成一条 chunkce trace:stage 行(叶子)+ chunk_rankings(parent)+ answer_chunk_ids。"""
    union_rows = [{"rank": i + 1, "chunk_id": c, "dsid": DSID} for i, c in enumerate(union_children)]
    fusion_rows = [{"rank": i + 1, "chunk_id": c, "dsid": DSID} for i, c in enumerate(fusion_children)]
    rankings = [
        {"parent_block_id": p, "dsid": DSID, "child_chunk_ids": [], "score": 1.0 / (i + 1), "rank": i + 1, "kind": "parent_block"}
        for i, p in enumerate(ce_parents)
    ]
    trace = {
        "question_id": "q1",
        "retrieval_stage_history": [
            {"query": "q", "mode": "rrf", "stages": {"dense": [], "bm25": [], "union": union_rows, "channel_fusion": fusion_rows}}
        ],
        "rerank_history": [{"round": 1, "chunk_rankings": rankings}],
        "answer_chunk_ids": cited,
        "selected_document_ids": [DSID],
    }
    with (tmp / "graph_traces.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps(trace) + "\n")


def _funnel(tmp: Path, rejudge: dict | None = None) -> dict:
    rejudge_file = None
    if rejudge is not None:
        rejudge_file = tmp / "rejudge.json"
        rejudge_file.write_text(
            json.dumps({"per_question": [{"question_id": "q1", "majority": rejudge["majority"], "flipped_to_correct": rejudge["flipped"]}]}),
            encoding="utf-8",
        )
    return build_block_funnel(
        tmp / "gold.jsonl",
        tmp / "blocks.jsonl",
        tmp,
        tmp / "official.json",
        rejudge_file=rejudge_file,
    )


def _row(funnel: dict) -> dict:
    return funnel["per_question"][0]


def test_doc_recall_100_but_answer_block_missing_is_retrieval_block_miss(tmp_path: Path) -> None:
    """dsid 口径误标场景:doc recall 100(兄弟叶进池)但答案 block 从未进池 → block 级硬漏。"""
    _write_gold(tmp_path)
    _write_parent_blocks(tmp_path)
    _write_official(tmp_path, recall_pct=100.0)
    _write_trace(
        tmp_path,
        union_children=[f"{DSID}::sec::1"],  # 只捞到兄弟叶
        fusion_children=[f"{DSID}::sec::1"],
        ce_parents=[f"{DSID}::sec::block_1"],
        cited=[f"{DSID}::sec::block_1"],
    )
    funnel = _funnel(tmp_path)
    row = _row(funnel)
    assert row["expected_block_ids"] == [f"{DSID}::sec::block_0"]
    assert row["bucket"] == "retrieval_block_miss"
    assert row["doc_recall_pct"] == 100.0


def test_evidence_block_in_pool_but_not_cited_is_ce_select_drop(tmp_path: Path) -> None:
    _write_gold(tmp_path)
    _write_parent_blocks(tmp_path)
    _write_official(tmp_path)
    _write_trace(
        tmp_path,
        union_children=[f"{DSID}::sec::0", f"{DSID}::sec::1"],
        fusion_children=[f"{DSID}::sec::0", f"{DSID}::sec::1"],
        ce_parents=[f"{DSID}::sec::block_1", f"{DSID}::sec::block_0"],  # b_answer 排第 2,但 top_k=6 内
        cited=[f"{DSID}::sec::block_1"],  # 引用里没有 b_answer
    )
    funnel = _funnel(tmp_path)
    assert _row(funnel)["bucket"] == "ce_select_drop"


def test_evidence_cited_but_still_wrong_is_in_context(tmp_path: Path) -> None:
    _write_gold(tmp_path)
    _write_parent_blocks(tmp_path)
    _write_official(tmp_path)
    _write_trace(
        tmp_path,
        union_children=[f"{DSID}::sec::0"],
        fusion_children=[f"{DSID}::sec::0"],
        ce_parents=[f"{DSID}::sec::block_0"],
        cited=[f"{DSID}::sec::block_0"],
    )
    funnel = _funnel(tmp_path)
    assert _row(funnel)["bucket"] == "in_context_but_wrong"
    assert funnel["summary"]["true_residual_generation"] == 1


def test_majority_flip_overrides_to_judge_flip(tmp_path: Path) -> None:
    _write_gold(tmp_path)
    _write_parent_blocks(tmp_path)
    _write_official(tmp_path)
    _write_trace(
        tmp_path,
        union_children=[f"{DSID}::sec::0"],
        fusion_children=[f"{DSID}::sec::0"],
        ce_parents=[f"{DSID}::sec::block_0"],
        cited=[f"{DSID}::sec::block_0"],
    )
    funnel = _funnel(tmp_path, rejudge={"majority": "yes", "flipped": True})
    assert _row(funnel)["bucket"] == "judge_flip_or_gold_contradiction"
    assert funnel["summary"]["true_residual_generation"] == 0
