"""对一个 run 做 parent_block 粒度决策归因,写 results/<run>/block_funnel.json。

用法:
  python scripts/attribution_block.py --config configs/rag100_dev_parentce.yaml
      [--run rag100_dev_parentce_nodn] [--context-top-k 6] [--no-rejudge]

rejudge 输出若存在(<results>/<run>/rejudge_majority_strict_x3.json)自动并入,
judge 误杀(0413 式)先从漏斗里剥出,剩下的才是证据/选择/生成的残差。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rag100_dev_parentce.yaml")
    parser.add_argument("--run", default=None, help="run_name;默认用 config.run_name")
    parser.add_argument("--context-top-k", type=int, default=6)
    parser.add_argument(
        "--no-rejudge",
        action="store_true",
        help="不并入 majority rejudge 输出(保留全部为原判错)",
    )
    args = parser.parse_args()

    from src.config import load_config
    from src.evaluation.block_funnel import build_block_funnel

    config = load_config(args.config)
    run_name = args.run or config.run_name
    run_dir = Path(config.output.runs_dir) / run_name
    results_dir = ROOT / "results" / run_name
    gold_file = Path(config.data.gold_questions_file)
    parent_blocks_file = Path(config.data.parent_blocks_file)
    official_file = results_dir / "official_results.json"
    funnel_file = results_dir / "recall_funnel.json"
    rejudge_file = None
    if not args.no_rejudge:
        candidate = results_dir / "rejudge_majority_strict_x3.json"
        if candidate.exists():
            rejudge_file = candidate

    funnel = build_block_funnel(
        gold_file,
        parent_blocks_file,
        run_dir,
        official_file,
        rejudge_file=rejudge_file,
        recall_funnel_file=funnel_file if funnel_file.exists() else None,
        context_top_k=args.context_top_k,
    )
    out = results_dir / "block_funnel.json"
    out.write_text(json.dumps(funnel, ensure_ascii=False, indent=1), encoding="utf-8")

    summary = funnel["summary"]
    print(f"run={run_name} wrong={summary['total_wrong']} true_residual_generation={summary['true_residual_generation']}")
    print("bucket_counts:")
    for k, v in summary["bucket_counts"].items():
        print(f"  {k:<36}{v}")
    per = {r["question_id"]: r for r in funnel["per_question"]}
    print("\nper bucket:")
    for k in summary["bucket_counts"]:
        qids = [r["question_id"] for r in funnel["per_question"] if r["bucket"] == k]
        if qids:
            print(f"  {k:<36}{len(qids):<3}{qids}")

    old_af = [r for r in funnel["per_question"] if r.get("old_dsid_cause") == "answer_failure"]
    print("\ndsid-funnel 'answer_failure' 重归因:")
    for r in old_af:
        print(f"  {r['question_id']} -> {r['bucket']}  (majority={r['majority_verdict']})")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
