"""Run graph-based retrieval over the saved GraphRAG index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.config import load_config  # noqa: E402
from src.graph_rag import GraphQueryError, run_graph_query  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query the deterministic GraphRAG index."
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Question to run against the graph index.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="Optional index directory. Defaults to results/indexes/baseline_graph.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to results/runs/.",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate an answer from retrieved chunks when OpenRouter config is available.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    index_dir = args.index_dir or (config.paths.indexes_dir / "baseline_graph")
    output_dir = args.output_dir or config.paths.runs_dir

    try:
        run_record = run_graph_query(
            query_text=args.query,
            index_dir=index_dir,
            output_dir=output_dir,
            top_k=args.top_k,
            llm_config=config.llm,
            generate_answer=args.generate,
            project_root=config.paths.root_dir,
        )
    except GraphQueryError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Graph query succeeded: {run_record.run_id}")
    print(f"Graph schema version: {run_record.embedding_model}")
    print(f"Top-k: {run_record.top_k}")
    print(
        "Query entities: "
        + ", ".join(
            f"{entity.canonical_name}<{entity.entity_type}>"
            for entity in run_record.query_entities
        )
    )
    for chunk in run_record.retrieved_chunks:
        preview = chunk.text[:180].replace("\n", " ")
        print(
            f"- rank={chunk.rank} score={chunk.score:.4f} "
            f"chunk_id={chunk.chunk_id} doc_id={chunk.doc_id} "
            f"page={chunk.page_number} preview={preview}"
        )
        entity_line = ", ".join(
            f"{entity.canonical_name}<{entity.entity_type}>:{entity.score_contribution:.2f}"
            for entity in chunk.contributing_entities
        )
        if entity_line:
            print(f"  entities={entity_line}")
        relation_line = ", ".join(
            f"{relation.source_name}-{relation.relation_type}->{relation.target_name}:{relation.score_contribution:.2f}"
            for relation in chunk.contributing_relations[:3]
        )
        if relation_line:
            print(f"  relations={relation_line}")
        if chunk.supporting_paths:
            for path in chunk.supporting_paths[:1]:
                chain = " -> ".join(path.relation_chain)
                print(
                    f"  path={path.source_name} via {path.via_name} to {path.target_name} "
                    f"[{chain}] score={path.score_contribution:.2f}"
                )

    print(
        f"Generation status: {run_record.generation.status}"
        + (
            f" ({run_record.generation.reason})"
            if run_record.generation.reason
            else ""
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
