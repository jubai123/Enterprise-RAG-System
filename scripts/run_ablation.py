#!/usr/bin/env python3
"""Run ablation experiments for the RAG-Bench workflow.

For each ablation variant (default: the 5 single-factor ablations defined in
src.evaluation.experiment_config.ABLATION_VARIANTS), runs scripts.run_benchmark
with --variant, then scripts.evaluate (official DeepSeek judge), and finally
writes a summary CSV plus prints a comparison table. Ablations only touch
retrieval.mode / features.*, so the existing Qdrant index is reused.

Usage:
  python -m scripts.run_ablation                            # all 5 ablations
  python -m scripts.run_ablation --include-main             # + minimal baseline
  python -m scripts.run_ablation --variants dense_only fixed_planning
  python -m scripts.run_ablation --limit 2 --variants dense_only   # smoke test
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Running via `python -m scripts.run_ablation` puts only scripts/ on sys.path,
# so make the project root importable.
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.evaluation.experiment_config import ABLATION_VARIANTS

DEFAULT_EXCLUDED = {"main"}


def _log(message: str) -> None:
    print(f"\n===== {message} =====", flush=True)


def _run_module(module: str, args: list[str]) -> subprocess.CompletedProcess:
    _log(f"python -m {module} {' '.join(args)}")
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=PROJECT_ROOT,
        check=False,
    )


def select_variants(variants: list[str] | None, include_main: bool) -> list[str]:
    if variants:
        return list(variants)
    chosen = [name for name in ABLATION_VARIANTS if name not in DEFAULT_EXCLUDED]
    if include_main:
        chosen.insert(0, "main")
    return chosen


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def collect_row(variant: str, run_dir: Path, *, evaluate_ran: bool) -> dict[str, object]:
    row: dict[str, object] = {"variant": variant, "run_dir": str(run_dir)}
    official = run_dir / "official_results.json"
    if official.is_file():
        data = _read_json(official)
        agg = data.get("aggregate_stats", {})
        questions = data.get("questions", [])
        row.update(
            correct=sum(1 for q in questions if q.get("answer_correct")),
            total=len(questions),
            accuracy_pct=agg.get("average_correctness_pct"),
            combined_score=agg.get("combined_correctness_completeness_score"),
            recall_pct=agg.get("average_recall_pct"),
            status="ok",
        )
    else:
        row["status"] = "failed" if evaluate_ran else "no_eval"
    summary = run_dir / "run_summary.json"
    if summary.is_file():
        data = _read_json(summary)
        row["total_cost_usd"] = data.get("total_cost")
        row["input_tokens"] = data.get("input_tokens")
        row["output_tokens"] = data.get("output_tokens")
        row["duration_p50_ms"] = data.get("duration_ms", {}).get("p50")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(ABLATION_VARIANTS),
        default=None,
        help="Ablation variants to run (default: all single-factor ablations, excluding main).",
    )
    parser.add_argument(
        "--include-main",
        action="store_true",
        help="Also run the full-system 'main' baseline as the first row.",
    )
    parser.add_argument("--official-repo", default="external/EnterpriseRAG-Bench")
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Smoke test with N questions.")
    parser.add_argument("--resume", action="store_true", help="Pass --resume to evaluate.")
    parser.add_argument("--run-prefix", default="ablation")
    parser.add_argument("--output-csv", default="runs/ablation_summary.csv")
    parser.add_argument("--skip-evaluate", action="store_true")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    config = load_config(args.config)
    variants = select_variants(args.variants, args.include_main)
    runs_dir = PROJECT_ROOT / config.output.runs_dir
    _log(f"Running ablation variants: {', '.join(variants)}")

    rows: list[dict[str, object]] = []
    for variant in variants:
        run_name = f"{args.run_prefix}_{variant}"
        run_dir = runs_dir / run_name
        _log(f"Variant: {variant} (run: {run_name})")
        bm_cmd = [
            "scripts.run_benchmark",
            "--config", args.config,
            "--variant", variant,
            "--run-name", run_name,
        ]
        if args.limit:
            bm_cmd += ["--limit", str(args.limit)]
        if _run_module(bm_cmd[0], bm_cmd[1:]).returncode != 0:
            print(f"  [ablation] benchmark failed for {variant}", file=sys.stderr, flush=True)
            rows.append({"variant": variant, "run_dir": str(run_dir), "status": "failed"})
            continue
        evaluate_ran = not args.skip_evaluate
        if evaluate_ran:
            ev_cmd = [
                "scripts.evaluate",
                "--config", args.config,
                "--run-dir", str(run_dir),
                "--official-repo", str(PROJECT_ROOT / args.official_repo),
                "--parallelism", str(args.parallelism),
            ]
            if args.resume:
                ev_cmd.append("--resume")
            if _run_module(ev_cmd[0], ev_cmd[1:]).returncode != 0:
                print(f"  [ablation] evaluate failed for {variant}", file=sys.stderr, flush=True)
        rows.append(collect_row(variant, run_dir, evaluate_ran=evaluate_ran))

    _log("Ablation summary")
    headers = [
        "variant", "correct", "total", "accuracy_pct", "combined_score",
        "recall_pct", "total_cost_usd", "input_tokens", "output_tokens",
        "duration_p50_ms", "status",
    ]
    csv_path = PROJECT_ROOT / args.output_csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    def fmt(value: object) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    print(f"{'variant':<20}{'correct':>8}{'accuracy':>10}{'combined':>10}{'recall':>9}{'cost_usd':>12}{'tokens':>9}{'p50_ms':>11}  status")
    for row in rows:
        tokens = (row.get("input_tokens") or 0) + (row.get("output_tokens") or 0)
        print(
            f"{fmt(row.get('variant')):<20}"
            f"{fmt(row.get('correct')):>8}"
            f"{fmt(row.get('accuracy_pct')):>10}"
            f"{fmt(row.get('combined_score')):>10}"
            f"{fmt(row.get('recall_pct')):>9}"
            f"{fmt(row.get('total_cost_usd')):>12}"
            f"{str(tokens):>9}"
            f"{fmt(row.get('duration_p50_ms')):>11}"
            f"  {row.get('status')}"
        )
    print(f"\n  Summary CSV: {csv_path}")

    failed = [str(row.get("variant")) for row in rows if row.get("status") == "failed"]
    if failed:
        print(
            f"  {len(failed)} variant(s) failed: {', '.join(failed)}",
            file=sys.stderr,
        )
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
