from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.config import load_config
from src.evaluation.certification import build_preflight_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and certify the single-run GitHub N39 baseline.")
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--run-name", default="main_github_dev_certification")
    parser.add_argument("--parallelism", type=int, default=3)
    args = parser.parse_args()
    root = Path.cwd()
    config = load_config(args.config)
    preflight = build_preflight_report(config, project_root=root)
    if preflight["status"] != "PASS":
        for failure in preflight["failures"]:
            print(f"PRECHECK FAILED: {failure}")
        raise SystemExit(1)
    run_dir = Path(config.output.runs_dir) / args.run_name
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_benchmark",
            "--config",
            args.config,
            "--run-name",
            args.run_name,
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.evaluate",
            "--run-dir",
            str(run_dir),
            "--parallelism",
            str(args.parallelism),
        ],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "scripts.certify", "--run-dir", str(run_dir)],
        check=True,
    )


if __name__ == "__main__":
    main()
