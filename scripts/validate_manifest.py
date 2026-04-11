"""Validate the corpus manifest before ingestion is implemented."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.config import load_config  # noqa: E402
from src.ingestion import ManifestValidationError, load_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate data/manifest.jsonl")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional path to a manifest file. Defaults to the configured manifest path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    manifest_path = args.manifest or config.paths.manifest_path

    try:
        entries = load_manifest(
            manifest_path=manifest_path,
            project_root=config.paths.root_dir,
        )
    except ManifestValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Manifest is valid: {len(entries)} document entries found in {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
