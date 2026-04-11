"""Manifest-driven PDF ingestion utilities."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from pypdf import PdfReader

from src.config import ROOT_DIR
from src.ingestion.manifest import ManifestEntry, load_manifest


class PDFIngestionError(Exception):
    """Raised when a manifest-declared PDF cannot be ingested."""

    def __init__(self, doc_id: str, message: str) -> None:
        self.doc_id = doc_id
        self.message = message
        super().__init__(f"[{doc_id}] {message}")


class IngestionRunError(Exception):
    """Raised when one or more documents fail during ingestion."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        message_lines = [f"PDF ingestion failed with {len(errors)} error(s):"]
        message_lines.extend(f"- {error}" for error in errors)
        super().__init__("\n".join(message_lines))


class ExtractionProblem(BaseModel):
    """Non-fatal page-level extraction issue."""

    page_number: int = Field(ge=1)
    message: str = Field(min_length=1)


class ExtractedPage(BaseModel):
    """Extracted content for one PDF page."""

    page_number: int = Field(ge=1)
    text: str
    char_count: int = Field(ge=0)


class ExtractedDocument(BaseModel):
    """Structured output for one manifest document."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    title: str
    publisher: str
    source_url: str
    year: int
    accessed_at: str
    local_path: str
    domain_tags: list[str]
    included_reason: str
    page_count: int = Field(ge=0)
    extracted_pages: list[ExtractedPage]
    extraction_problems: list[ExtractionProblem] = Field(default_factory=list)


def ingest_manifest_documents(
    manifest_path: Path | str,
    output_path: Path | str,
    project_root: Path = ROOT_DIR,
) -> list[ExtractedDocument]:
    """Load the manifest, extract all declared PDFs, and save JSONL output."""

    root_dir = project_root.resolve()
    entries = load_manifest(manifest_path=manifest_path, project_root=root_dir)

    documents: list[ExtractedDocument] = []
    errors: list[str] = []

    for entry in entries:
        try:
            documents.append(extract_document(entry=entry, project_root=root_dir))
        except PDFIngestionError as exc:
            errors.append(str(exc))

    if errors:
        raise IngestionRunError(errors)

    resolved_output = _resolve_path(output_path=output_path, project_root=root_dir)
    write_documents_jsonl(documents=documents, output_path=resolved_output)
    return documents


def extract_document(
    entry: ManifestEntry,
    project_root: Path = ROOT_DIR,
) -> ExtractedDocument:
    """Extract page-level text for a single manifest entry."""

    pdf_path = entry.resolved_local_path(project_root)
    if not pdf_path.exists():
        raise PDFIngestionError(entry.doc_id, f"PDF file does not exist: {pdf_path}")
    if not pdf_path.is_file():
        raise PDFIngestionError(entry.doc_id, f"PDF path is not a file: {pdf_path}")

    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:  # pragma: no cover - depends on file/parser behavior
        raise PDFIngestionError(
            entry.doc_id,
            f"unable to open PDF '{pdf_path}': {exc.__class__.__name__}: {exc}",
        ) from exc

    extracted_pages: list[ExtractedPage] = []
    extraction_problems: list[ExtractionProblem] = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_problem_recorded = False

        try:
            raw_text = page.extract_text()
        except Exception as exc:  # pragma: no cover - depends on page/parser behavior
            extraction_problems.append(
                ExtractionProblem(
                    page_number=page_number,
                    message=(
                        "page extraction failed: "
                        f"{exc.__class__.__name__}: {exc}"
                    ),
                )
            )
            raw_text = ""
            page_problem_recorded = True

        if raw_text is None:
            extraction_problems.append(
                ExtractionProblem(
                    page_number=page_number,
                    message="page extraction returned no text",
                )
            )
            raw_text = ""
            page_problem_recorded = True

        if raw_text == "" and not page_problem_recorded:
            extraction_problems.append(
                ExtractionProblem(
                    page_number=page_number,
                    message="page extraction produced empty text",
                )
            )

        extracted_pages.append(
            ExtractedPage(
                page_number=page_number,
                text=raw_text,
                char_count=len(raw_text),
            )
        )

    return ExtractedDocument(
        doc_id=entry.doc_id,
        title=entry.title,
        publisher=entry.publisher,
        source_url=str(entry.source_url),
        year=entry.year,
        accessed_at=entry.accessed_at.isoformat(),
        local_path=str(entry.local_path),
        domain_tags=entry.domain_tags,
        included_reason=entry.included_reason,
        page_count=len(extracted_pages),
        extracted_pages=extracted_pages,
        extraction_problems=extraction_problems,
    )


def write_documents_jsonl(
    documents: list[ExtractedDocument],
    output_path: Path,
) -> None:
    """Write extracted document records to a JSONL file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    with temp_path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document.model_dump(mode="json"), ensure_ascii=False))
            handle.write("\n")

    temp_path.replace(output_path)


def _resolve_path(output_path: Path | str, project_root: Path) -> Path:
    candidate = Path(output_path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (project_root / candidate).resolve()
