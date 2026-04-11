"""Run retrieval evaluation across the available baselines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.config import load_config  # noqa: E402
from src.evaluation.harness import run_retrieval_evaluation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic retrieval evaluation across all baselines."
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=None,
        help="Optional evaluation questions JSONL path. Defaults to data/evaluation/questions.jsonl.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional output root. Defaults to results/evaluation/.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of retrieved chunks to score per method.",
    )
    parser.add_argument(
        "--include-hybrid",
        action="store_true",
        help="Include the hybrid dense-plus-graph retrieval path in the evaluation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    questions_path = args.questions or (config.paths.evaluation_data_dir / "questions.jsonl")
    output_root = args.output_root or (config.paths.results_dir / "evaluation")

    summary, _ = run_retrieval_evaluation(
        questions_path=questions_path,
        output_root=output_root,
        config=config,
        top_k=args.top_k,
        include_hybrid=args.include_hybrid,
    )

    print(f"Evaluation succeeded: {summary.evaluation_id}")
    print(f"Questions: {summary.question_count}")
    print(f"Top-k: {summary.top_k}")
    print(f"Output directory: {summary.output_dir}")
    print(f"Category distribution: {summary.category_distribution}")
    print(f"Track distribution: {summary.evaluation_track_distribution}")
    print(
        "Explanation-oriented questions: "
        f"{summary.explanation_oriented_question_count}"
    )
    for method_summary in summary.method_summaries:
        print(
            f"- {method_summary.method_label}: overall={method_summary.mean_overall:.3f} "
            f"relevance={method_summary.mean_relevance:.3f} "
            f"directness={method_summary.mean_directness:.3f} "
            f"groundedness={method_summary.mean_groundedness:.3f} "
            f"correctness={method_summary.mean_correctness_of_evidence:.3f} "
            f"top1_doc={method_summary.top1_doc_accuracy:.3f} "
            f"topk_gold_chunk={method_summary.topk_gold_chunk_recall:.3f} "
            f"wins={method_summary.wins} "
            f"exact={method_summary.mean_overall_by_track.get('exact_retrieval', 0.0):.3f} "
            f"graph={method_summary.mean_overall_by_track.get('graph_stressing', 0.0):.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
