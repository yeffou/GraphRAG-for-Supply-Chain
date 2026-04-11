"""Run chunking over extracted document records."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.config import load_config  # noqa: E402
from src.preprocessing import ChunkingError, chunk_documents_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk extracted page text from data/processed/documents.jsonl."
    )
    parser.add_argument(
        "--documents",
        type=Path,
        default=None,
        help="Optional documents JSONL path. Defaults to data/processed/documents.jsonl.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional chunks JSONL path. Defaults to data/processed/chunks.jsonl.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum chunk length in characters after normalization.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=150,
        help="Character overlap between consecutive chunks.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    documents_path = args.documents or (config.paths.processed_data_dir / "documents.jsonl")
    output_path = args.output or (config.paths.processed_data_dir / "chunks.jsonl")

    try:
        _, stats = chunk_documents_jsonl(
            documents_path=documents_path,
            output_path=output_path,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            project_root=config.paths.root_dir,
        )
    except ChunkingError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Chunking succeeded: output written to {output_path}")
    print(f"Documents: {stats.documents}")
    print(f"Pages processed: {stats.pages_processed}")
    print(f"Pages skipped for empty text: {stats.pages_skipped_empty}")
    print(f"Total chunks: {stats.total_chunks}")
    print(f"Average chunk length: {stats.average_chunk_length:.2f}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Chunk overlap: {args.chunk_overlap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
