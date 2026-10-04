from __future__ import annotations

import argparse

from src.config import load_config
from src.evaluation.reproducibility import prepare_blind_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the Gold-free GitHub N39 question file.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    manifest = prepare_blind_dataset(config)
    print(
        f"Prepared {manifest['dataset']} with {manifest['question_count']} blind questions; "
        f"sha256={manifest['blind_questions_sha256']}"
    )


if __name__ == "__main__":
    main()
