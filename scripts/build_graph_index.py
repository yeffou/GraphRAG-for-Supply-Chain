"""Build the deterministic GraphRAG index from processed chunks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.config import load_config  # noqa: E402
from src.graph_rag import build_graph_index  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic GraphRAG index from chunk records."
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=None,
        help="Optional chunks JSONL path. Defaults to data/processed/chunks.jsonl.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="Optional index directory. Defaults to results/indexes/baseline_graph.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    chunks_path = args.chunks or (config.paths.processed_data_dir / "chunks.jsonl")
    index_dir = args.index_dir or (config.paths.indexes_dir / "baseline_graph")

    summary = build_graph_index(
        chunks_path=chunks_path,
        index_dir=index_dir,
        project_root=config.paths.root_dir,
    )

    print(f"Graph indexing succeeded: {summary.indexed_chunks} chunks indexed")
    print(f"Graph schema version: {summary.graph_schema_version}")
    print(f"Entity nodes: {summary.entity_nodes}")
    print(f"Sentence nodes: {summary.sentence_nodes}")
    print(f"Relation edges: {summary.relation_edges}")
    print(f"Average entities per chunk: {summary.average_entities_per_chunk:.3f}")
    print(f"Average relations per chunk: {summary.average_relations_per_chunk:.3f}")
    print(f"Index output path: {summary.index_dir}")
    print(f"Source chunks path: {summary.source_chunks_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
