"""Build the baseline vector index from processed chunks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.baseline_rag import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    BaselineIndexingError,
    build_baseline_index,
)
from src.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a baseline TF-IDF vector index from chunk records."
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
        help="Optional index directory. Defaults to results/indexes/baseline_tfidf.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="Label stored in index metadata for the embedding/vector model.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    chunks_path = args.chunks or (config.paths.processed_data_dir / "chunks.jsonl")
    index_dir = args.index_dir or (config.paths.indexes_dir / "baseline_tfidf")

    try:
        summary = build_baseline_index(
            chunks_path=chunks_path,
            index_dir=index_dir,
            embedding_model=args.embedding_model,
            project_root=config.paths.root_dir,
        )
    except BaselineIndexingError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Baseline indexing succeeded: {summary.indexed_chunks} chunks indexed")
    print(f"Embedding model: {summary.embedding_model}")
    print(f"Vocabulary size: {summary.vocabulary_size}")
    print(f"Index output path: {summary.index_dir}")
    print(f"Source chunks path: {summary.source_chunks_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
