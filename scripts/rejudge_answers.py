"""N× majority 重判一个 run 的全部错题,修正单次 judge 的采样噪声。

用法:
  python scripts/rejudge_answers.py [--config configs/rag100_dev_parentce.yaml]
      [--run rag100_dev_parentce_nodn] [--samples 3] [--flavor strict]
      [--qids qst_0413 qst_0248] [--parallelism 4] [--resume]

输入(错题集):优先 <runs>/<run>/failed_questions.jsonl;缺失时用 official_results.json
(错题)+ answers.jsonl + gold 合成。输出写 <results>/<run>/rejudge_majority_<flavor>_x<n>.json
(每题 votes/majority/flip)与 rejudge_summary_<flavor>_x<n>.json(修正正确率)。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_failed_rows(config: dict, run_dir: Path) -> list[dict]:
    """错题行(question/gold_answer/candidate_answer/question_type 自足)。"""
    failed_file = run_dir / "failed_questions.jsonl"
    if failed_file.exists():
        rows = load_jsonl(failed_file)
        return [r for r in rows if not r.get("answer_correct")]

    official_file = run_dir.parent / "results" / run_dir.name / "official_results.json"
    if official_file.exists() and (run_dir / "answers.jsonl").exists():
        official = json.loads(official_file.read_text(encoding="utf-8"))
        answers = {r["question_id"]: r.get("answer", "") for r in load_jsonl(run_dir / "answers.jsonl")}
        gold_file = Path(config["data"]["gold_questions_file"])
        gold = {r["question_id"]: r for r in load_jsonl(gold_file)}
        rows = []
        for q in official.get("questions", []):
            if q.get("answer_correct"):
                continue
            g = gold[q["question_id"]]
            rows.append(
                {
                    "question_id": q["question_id"],
                    "question_type": g.get("question_type"),
                    "question": g.get("question", ""),
                    "gold_answer": g.get("gold_answer", ""),
                    "candidate_answer": answers.get(q["question_id"], ""),
                }
            )
        return rows
    raise FileNotFoundError(f"no failed_questions.jsonl nor official_results.json for {run_dir}")


def load_official_baseline(config: dict, run_name: str) -> tuple[int, int, float | None]:
    """(total, raw_correct, raw_correctness_pct);找不到基线返回 total=0。"""
    official_file = ROOT / "results" / run_name / "official_results.json"
    if not official_file.exists():
        return 0, 0, None
    official = json.loads(official_file.read_text(encoding="utf-8"))
    questions = official.get("questions", [])
    if not questions:
        agg = official.get("aggregate_stats", {})
        return 0, 0, agg.get("average_correctness_pct")
    total = len(questions)
    correct = sum(1 for q in questions if q.get("answer_correct"))
    return total, correct, (100.0 * correct / total if total else None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rag100_dev_parentce.yaml")
    parser.add_argument("--run", default=None, help="run_name;默认用 config.run_name")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--flavor", choices=("strict", "core"), default="strict")
    parser.add_argument("--parallelism", type=int, default=4, help="判分线程数")
    parser.add_argument("--qids", nargs="*", default=None, help="只判这些 qid(默认全部错题)")
    parser.add_argument("--resume", action="store_true", help="沿用已有输出文件,跳过已判 qid")
    args = parser.parse_args()

    from src.config import load_config
    from src.evaluation.rejudge import run_majority
    from src.chains.langchain_rag import build_chat_model

    config = load_config(args.config)
    run_name = args.run or config.run_name
    run_dir = Path(config.output.runs_dir) / run_name
    out_dir = ROOT / "results" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    full_file = out_dir / f"rejudge_majority_{args.flavor}_x{args.samples}.json"

    rows = load_failed_rows(config.model_dump(), run_dir)
    if args.qids:
        wanted = set(args.qids)
        rows = [r for r in rows if r["question_id"] in wanted]
    rows.sort(key=lambda r: r["question_id"])

    existing: dict[str, dict] = {}
    if args.resume and full_file.exists():
        for r in json.loads(full_file.read_text(encoding="utf-8")).get("per_question", []):
            existing[r["question_id"]] = r
    todo = [r for r in rows if r["question_id"] not in existing]

    if todo:
        print(f"judging {len(todo)} questions x{args.samples} ({args.flavor}), workers={args.parallelism} ...")
        llm = build_chat_model(config.llm)
        for result in run_majority(todo, llm, flavor=args.flavor, samples=args.samples, workers=args.parallelism):
            existing[result["question_id"]] = result

    per_question = [existing[r["question_id"]] for r in rows if r["question_id"] in existing]

    flips = [r for r in per_question if r.get("flipped_to_correct")]
    ties = [r for r in per_question if r.get("majority") == "tie"]
    degraded = [r for r in per_question if any(v.startswith("ERR") for v in r.get("votes", []))]

    total, raw_correct, raw_pct = load_official_baseline(config.model_dump(), run_name)
    corrected = raw_correct + len(flips) if total else len(flips)
    corrected_pct = (100.0 * corrected / total) if total else None

    summary = {
        "run_name": run_name,
        "config": args.config,
        "flavor": args.flavor,
        "samples": args.samples,
        "judged_questions": len(per_question),
        "raw_correctness_pct": raw_pct,
        "raw_correct": raw_correct if total else None,
        "n_flip_to_correct": len(flips),
        "flips": [r["question_id"] for r in flips],
        "ties": [r["question_id"] for r in ties],
        "degraded": [r["question_id"] for r in degraded],
        "corrected_correctness_pct": round(corrected_pct, 2) if corrected_pct is not None else None,
        "corrected_correct": corrected,
    }

    full = {
        "run_name": run_name,
        "flavor": args.flavor,
        "samples": args.samples,
        "model": str(getattr(config.llm, "model", "")),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "per_question": per_question,
        "summary": summary,
    }
    full_file.write_text(json.dumps(full, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / f"rejudge_summary_{args.flavor}_x{args.samples}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    print("\n=== per-question ===")
    print(f"{'qid':<10}{'votes':<28}{'majority':<8}flip")
    for r in per_question:
        print(f"{r['question_id']:<10}{str(r['votes']):<28}{r['majority']:<8}{r.get('flipped_to_correct')}")
    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nwrote {full_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
