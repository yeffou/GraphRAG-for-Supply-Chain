"""Build the dense semantic baseline index from processed chunks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.baseline_rag import (  # noqa: E402
    DEFAULT_DENSE_EMBEDDING_MODEL,
    BaselineIndexingError,
    build_dense_index,
)
from src.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a dense semantic index from chunk records."
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
        help="Optional index directory. Defaults to results/indexes/baseline_dense.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_DENSE_EMBEDDING_MODEL,
        help="Sentence-transformers model name used for dense embeddings.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    chunks_path = args.chunks or (config.paths.processed_data_dir / "chunks.jsonl")
    index_dir = args.index_dir or (config.paths.indexes_dir / "baseline_dense")

    try:
        summary = build_dense_index(
            chunks_path=chunks_path,
            index_dir=index_dir,
            embedding_model=args.embedding_model,
            batch_size=args.batch_size,
            project_root=config.paths.root_dir,
        )
    except BaselineIndexingError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Dense indexing succeeded: {summary.indexed_chunks} chunks indexed")
    print(f"Embedding model: {summary.embedding_model}")
    print(f"Index backend: {summary.index_backend}")
    print(f"Embedding dimension: {summary.embedding_dimension}")
    print(f"Index output path: {summary.index_dir}")
    print(f"Source chunks path: {summary.source_chunks_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
