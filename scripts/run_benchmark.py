from __future__ import annotations

import argparse

from src.config import load_config
from src.evaluation.experiment_config import ABLATION_VARIANTS, apply_experiment_variant
from src.runtime.benchmark import execute_benchmark
from src.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the R49c2 RAG benchmark graph.")
    parser.add_argument("--config", default=None, help="Optional YAML/JSON config override file.")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--variant", choices=tuple(ABLATION_VARIANTS), default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--question-source-type",
        default=None,
        help="Only run questions whose source_types exactly match this value.",
    )
    parser.add_argument(
        "--question-type",
        default=None,
        help="Select a benchmark question type for a focused experiment only.",
    )
    parser.add_argument("--all-questions", action="store_true")
    parser.add_argument(
        "--include-mixed-source-questions",
        action="store_true",
        help="Include questions that combine the selected source with other sources.",
    )
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    if args.run_name:
        config = config.model_copy(update={"run_name": args.run_name})
    if args.variant:
        config = apply_experiment_variant(config, args.variant)
    execute_benchmark(
        config,
        limit=args.limit,
        question_source_type=args.question_source_type,
        question_type=args.question_type,
        all_questions=args.all_questions,
        include_mixed_source_questions=args.include_mixed_source_questions,
    )


if __name__ == "__main__":
    main()
