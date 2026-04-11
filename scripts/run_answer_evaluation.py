"""Run retrieval, answer generation, and answer-level evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.config import load_config  # noqa: E402
from src.evaluation import run_answer_evaluation  # noqa: E402
from src.generation import LLMClientError  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run answer generation and answer-level evaluation across all retrieval methods."
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=None,
        help="Optional evaluation questions JSONL path. Defaults to data/evaluation/questions.jsonl.",
    )
    parser.add_argument(
        "--answers-output-root",
        type=Path,
        default=None,
        help="Optional root for saved generated answers. Defaults to results/answers/.",
    )
    parser.add_argument(
        "--evaluation-output-root",
        type=Path,
        default=None,
        help="Optional root for saved answer evaluations. Defaults to results/answer_evaluation/.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of retrieved chunks to pass into answer generation.",
    )
    parser.add_argument(
        "--question-ids",
        type=str,
        default=None,
        help="Optional comma-separated subset of question IDs to run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on how many questions to run after filtering.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    questions_path = args.questions or (config.paths.evaluation_data_dir / "questions.jsonl")
    answers_output_root = args.answers_output_root or config.paths.answers_dir
    evaluation_output_root = args.evaluation_output_root or config.paths.answer_evaluation_dir
    question_ids = (
        [item.strip() for item in args.question_ids.split(",") if item.strip()]
        if args.question_ids
        else None
    )

    try:
        summary, _ = run_answer_evaluation(
            questions_path=questions_path,
            answers_output_root=answers_output_root,
            evaluation_output_root=evaluation_output_root,
            config=config,
            top_k=args.top_k,
            question_ids=question_ids,
            question_limit=args.limit,
        )
    except (ValueError, LLMClientError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Answer evaluation succeeded: {summary.run_id}")
    print(f"Questions: {summary.question_count}")
    print(f"Top-k: {summary.top_k}")
    print(f"Generation model: {summary.generation_model}")
    print(f"Judge model: {summary.judge_model}")
    print(f"Answers output: {summary.answers_output_dir}")
    print(f"Evaluation output: {summary.evaluation_output_dir}")
    for method_summary in summary.method_summaries:
        print(
            f"- {method_summary.method_label}: overall={method_summary.mean_overall_score:.3f} "
            f"correctness={method_summary.mean_correctness:.3f} "
            f"completeness={method_summary.mean_completeness:.3f} "
            f"reasoning={method_summary.mean_reasoning:.3f} "
            f"groundedness={method_summary.mean_groundedness:.3f} "
            f"wins={method_summary.wins}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
