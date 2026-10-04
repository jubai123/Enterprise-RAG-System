from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.certification import certify_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Certify one completed GitHub N39 run.")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    report = certify_run(run_dir, project_root=Path.cwd())
    target = run_dir / "certification_report.json"
    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
