from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import load_config
from src.evaluation.certification import build_preflight_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the local N39 certification environment.")
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--skip-qdrant", action="store_true")
    args = parser.parse_args()
    report = build_preflight_report(
        load_config(args.config),
        project_root=Path.cwd(),
        require_clean=not args.allow_dirty,
        check_qdrant=not args.skip_qdrant,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
