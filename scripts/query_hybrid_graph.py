"""Run one hybrid dense-plus-graph retrieval query."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.config import load_config  # noqa: E402
from src.graph_rag.hybrid_query import run_hybrid_graph_query  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a hybrid GraphRAG query with dense recall and graph-aware reranking."
    )
    parser.add_argument("--query", required=True, help="Question to retrieve against the corpus.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to return.")
    parser.add_argument(
        "--dense-candidates",
        type=int,
        default=40,
        help="Number of dense candidates to pull before fusion.",
    )
    parser.add_argument(
        "--graph-candidates",
        type=int,
        default=20,
        help="Number of graph candidates to pull before fusion.",
    )
    parser.add_argument(
        "--generate-answer",
        action="store_true",
        help="Attempt answer generation if OpenRouter is configured.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    run_record = run_hybrid_graph_query(
        query_text=args.query,
        dense_index_dir=config.paths.indexes_dir / "baseline_dense",
        graph_index_dir=config.paths.indexes_dir / "baseline_graph",
        output_dir=config.paths.runs_dir,
        top_k=args.top_k,
        llm_config=config.llm,
        generate_answer=args.generate_answer,
        project_root=config.paths.root_dir,
        dense_candidate_count=args.dense_candidates,
        graph_candidate_count=args.graph_candidates,
    )

    print(f"Run ID: {run_record.run_id}")
    print(f"Method: {run_record.method}")
    print(f"Embedding model: {run_record.embedding_model}")
    print(f"Dense index dir: {run_record.dense_index_dir}")
    print(f"Graph index dir: {run_record.graph_index_dir}")
    print(f"Output file: {config.paths.runs_dir / f'{run_record.run_id}.json'}")
    print("")

    for chunk in run_record.retrieved_chunks:
        print(
            f"[{chunk.rank}] score={chunk.score:.6f} "
            f"dense_rank={chunk.dense_rank} graph_rank={chunk.graph_rank} "
            f"chunk_id={chunk.chunk_id}"
        )
        print(
            f"    doc_id={chunk.doc_id} page={chunk.page_number} "
            f"title={chunk.title}"
        )
        print(f"    preview={chunk.text[:220].replace(chr(10), ' ')}")
        if chunk.contributing_relations:
            top_relation = chunk.contributing_relations[0]
            print(
                f"    top_relation={top_relation.source_name}-{top_relation.relation_type}->{top_relation.target_name}"
            )
        print("")

    if run_record.generation.status == "succeeded":
        print("Generated answer:")
        print(run_record.generation.answer)
    else:
        print(f"Generation: {run_record.generation.status} ({run_record.generation.reason})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
