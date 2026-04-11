"""Run manifest-driven PDF ingestion and save document-level JSONL output."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.config import load_config  # noqa: E402
from src.ingestion import (  # noqa: E402
    IngestionRunError,
    ManifestValidationError,
    ingest_manifest_documents,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract page-level text from manifest-declared PDFs."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest path. Defaults to data/manifest.jsonl.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. Defaults to data/processed/documents.jsonl.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    manifest_path = args.manifest or config.paths.manifest_path
    output_path = args.output or (config.paths.processed_data_dir / "documents.jsonl")

    try:
        documents = ingest_manifest_documents(
            manifest_path=manifest_path,
            output_path=output_path,
            project_root=config.paths.root_dir,
        )
    except (ManifestValidationError, IngestionRunError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    total_pages = sum(document.page_count for document in documents)
    documents_with_problems = [
        document for document in documents if document.extraction_problems
    ]

    print(
        f"Ingestion succeeded: {len(documents)} documents written to {output_path}"
    )
    print(f"Total pages extracted: {total_pages}")
    print(
        "Documents with page extraction problems: "
        f"{len(documents_with_problems)}"
    )

    for document in documents_with_problems:
        for problem in document.extraction_problems:
            print(
                f"- {document.doc_id} page {problem.page_number}: {problem.message}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
