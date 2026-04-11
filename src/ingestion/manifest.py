"""Schema and validation utilities for the corpus manifest."""

from __future__ import annotations

import json
from datetime import date
from json import JSONDecodeError
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationError, field_validator

from src.config import ROOT_DIR


class ManifestValidationError(Exception):
    """Raised when one or more manifest validation errors are found."""

    def __init__(self, manifest_path: Path, errors: list[str]) -> None:
        self.manifest_path = manifest_path
        self.errors = errors

        message_lines = [
            f"Manifest validation failed for {manifest_path} with {len(errors)} error(s):"
        ]
        message_lines.extend(f"- {error}" for error in errors)
        super().__init__("\n".join(message_lines))


class ManifestEntry(BaseModel):
    """Single document entry from ``data/manifest.jsonl``."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source_url: HttpUrl
    publisher: str = Field(min_length=1)
    year: int = Field(ge=1900, le=2100)
    accessed_at: date
    local_path: Path
    domain_tags: list[str] = Field(min_length=1)
    included_reason: str = Field(min_length=1)

    @field_validator("doc_id", "title", "publisher", "included_reason", mode="before")
    @classmethod
    def strip_text_fields(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("local_path", mode="before")
    @classmethod
    def normalize_local_path(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("domain_tags", mode="after")
    @classmethod
    def validate_domain_tags(cls, value: list[str]) -> list[str]:
        cleaned_tags = [tag.strip() for tag in value if isinstance(tag, str) and tag.strip()]
        if not cleaned_tags:
            raise ValueError("must contain at least one non-empty tag")
        return cleaned_tags

    def resolved_local_path(self, project_root: Path = ROOT_DIR) -> Path:
        """Resolve the entry's local path against the project root."""

        candidate = self.local_path
        if candidate.is_absolute():
            return candidate.resolve()
        return (project_root / candidate).resolve()


def load_manifest(
    manifest_path: Path | str,
    project_root: Path = ROOT_DIR,
) -> list[ManifestEntry]:
    """Load and validate the corpus manifest."""

    root_dir = project_root.resolve()
    manifest_file = Path(manifest_path)
    if not manifest_file.is_absolute():
        manifest_file = (root_dir / manifest_file).resolve()

    errors: list[str] = []
    entries: list[ManifestEntry] = []
    seen_doc_ids: dict[str, int] = {}

    if not manifest_file.exists():
        raise ManifestValidationError(
            manifest_file,
            ["manifest file does not exist"],
        )

    with manifest_file.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped_line = raw_line.strip()
            if not stripped_line:
                continue

            try:
                payload = json.loads(stripped_line)
            except JSONDecodeError as exc:
                errors.append(
                    f"line {line_number}: malformed JSON ({exc.msg})"
                )
                continue

            if not isinstance(payload, dict):
                errors.append(f"line {line_number}: expected a JSON object")
                continue

            try:
                entry = ManifestEntry.model_validate(payload)
            except ValidationError as exc:
                errors.extend(_format_model_errors(line_number, exc))
                continue

            previous_line = seen_doc_ids.get(entry.doc_id)
            if previous_line is not None:
                errors.append(
                    "line "
                    f"{line_number}: duplicate doc_id '{entry.doc_id}' "
                    f"(first defined on line {previous_line})"
                )
            else:
                seen_doc_ids[entry.doc_id] = line_number

            local_path_error = _validate_local_path(
                entry=entry,
                line_number=line_number,
                project_root=root_dir,
            )
            if local_path_error:
                errors.append(local_path_error)

            entries.append(entry)

    if errors:
        raise ManifestValidationError(manifest_file, errors)

    return entries


def _format_model_errors(line_number: int, exc: ValidationError) -> list[str]:
    formatted_errors: list[str] = []

    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        message = error["msg"]
        formatted_errors.append(f"line {line_number}: {location} {message}")

    return formatted_errors


def _validate_local_path(
    entry: ManifestEntry,
    line_number: int,
    project_root: Path,
) -> str | None:
    local_path = entry.local_path

    if local_path.is_absolute():
        return (
            f"line {line_number}: invalid local_path '{local_path}' "
            "because absolute paths are not allowed"
        )

    resolved_path = entry.resolved_local_path(project_root)

    try:
        resolved_path.relative_to(project_root)
    except ValueError:
        return (
            f"line {line_number}: invalid local_path '{local_path}' "
            "because it resolves outside the project root"
        )

    if not resolved_path.exists():
        return (
            f"line {line_number}: invalid local_path '{local_path}' "
            "because the file does not exist"
        )

    if not resolved_path.is_file():
        return (
            f"line {line_number}: invalid local_path '{local_path}' "
            "because it is not a file"
        )

    return None
