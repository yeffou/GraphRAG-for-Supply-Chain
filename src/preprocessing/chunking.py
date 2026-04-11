"""Preprocessing and chunking for extracted document records."""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.config import ROOT_DIR


WHITESPACE_RE = re.compile(r"\s+")


class ChunkingError(Exception):
    """Raised when document loading or chunking fails."""


class ExtractedPageRecord(BaseModel):
    """Page record produced by the ingestion layer."""

    model_config = ConfigDict(extra="forbid")

    page_number: int = Field(ge=1)
    text: str
    char_count: int = Field(ge=0)


class ExtractedDocumentRecord(BaseModel):
    """Document record loaded from ``data/processed/documents.jsonl``."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    year: int
    accessed_at: str = Field(min_length=1)
    local_path: str = Field(min_length=1)
    domain_tags: list[str]
    included_reason: str = Field(min_length=1)
    page_count: int = Field(ge=0)
    extracted_pages: list[ExtractedPageRecord]
    extraction_problems: list[dict] = Field(default_factory=list)


class ChunkRecord(BaseModel):
    """Chunked page segment for later retrieval."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    doc_id: str
    title: str
    publisher: str
    source_url: str
    year: int
    accessed_at: str
    local_path: str
    page_number: int = Field(ge=1)
    domain_tags: list[str]
    text: str
    char_count: int = Field(ge=0)
    token_estimate: int = Field(ge=0)
    chunk_index_within_page: int = Field(ge=0)


class ChunkingStats(BaseModel):
    """Summary stats for a chunking run."""

    documents: int = Field(ge=0)
    pages_processed: int = Field(ge=0)
    pages_skipped_empty: int = Field(ge=0)
    total_chunks: int = Field(ge=0)
    average_chunk_length: float = Field(ge=0.0)


def chunk_documents_jsonl(
    documents_path: Path | str,
    output_path: Path | str,
    chunk_size: int,
    chunk_overlap: int,
    project_root: Path = ROOT_DIR,
) -> tuple[list[ChunkRecord], ChunkingStats]:
    """Read extracted documents, create chunks, and write JSONL output."""

    _validate_chunk_config(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    root_dir = project_root.resolve()
    documents_file = _resolve_path(documents_path, root_dir)
    output_file = _resolve_path(output_path, root_dir)

    documents = load_documents_jsonl(documents_file)
    chunks, stats = build_chunks(
        documents=documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    write_chunks_jsonl(chunks=chunks, output_path=output_file)
    return chunks, stats


def load_documents_jsonl(documents_path: Path) -> list[ExtractedDocumentRecord]:
    """Load and validate extracted documents from JSONL."""

    if not documents_path.exists():
        raise ChunkingError(f"documents file does not exist: {documents_path}")
    if not documents_path.is_file():
        raise ChunkingError(f"documents path is not a file: {documents_path}")

    documents: list[ExtractedDocumentRecord] = []

    with documents_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue

            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ChunkingError(
                    f"malformed JSON in {documents_path} line {line_number}: {exc.msg}"
                ) from exc

            try:
                document = ExtractedDocumentRecord.model_validate(payload)
            except ValidationError as exc:
                raise ChunkingError(
                    f"invalid document schema in {documents_path} line {line_number}: {exc}"
                ) from exc

            documents.append(document)

    return documents


def build_chunks(
    documents: list[ExtractedDocumentRecord],
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[list[ChunkRecord], ChunkingStats]:
    """Split extracted pages into page-bounded chunks."""

    chunks: list[ChunkRecord] = []
    pages_processed = 0
    pages_skipped_empty = 0

    for document in documents:
        for page in document.extracted_pages:
            normalized_text = normalize_page_text(page.text)
            if not normalized_text:
                pages_skipped_empty += 1
                continue

            pages_processed += 1
            page_chunks = split_page_text(
                text=normalized_text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            for chunk_index, chunk_text in enumerate(page_chunks):
                chunks.append(
                    ChunkRecord(
                        chunk_id=build_chunk_id(
                            doc_id=document.doc_id,
                            page_number=page.page_number,
                            chunk_index=chunk_index,
                        ),
                        doc_id=document.doc_id,
                        title=document.title,
                        publisher=document.publisher,
                        source_url=document.source_url,
                        year=document.year,
                        accessed_at=document.accessed_at,
                        local_path=document.local_path,
                        page_number=page.page_number,
                        domain_tags=document.domain_tags,
                        text=chunk_text,
                        char_count=len(chunk_text),
                        token_estimate=estimate_tokens(chunk_text),
                        chunk_index_within_page=chunk_index,
                    )
                )

    average_chunk_length = (
        sum(chunk.char_count for chunk in chunks) / len(chunks) if chunks else 0.0
    )
    stats = ChunkingStats(
        documents=len(documents),
        pages_processed=pages_processed,
        pages_skipped_empty=pages_skipped_empty,
        total_chunks=len(chunks),
        average_chunk_length=average_chunk_length,
    )
    return chunks, stats


def normalize_page_text(text: str) -> str:
    """Conservatively normalize extracted page text."""

    return WHITESPACE_RE.sub(" ", text).strip()


def split_page_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split normalized text into overlapping character chunks within one page."""

    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        tentative_end = min(start + chunk_size, text_length)
        end = tentative_end

        if tentative_end < text_length:
            search_start = min(start + max(chunk_size // 2, 1), text_length)
            boundary = text.rfind(" ", search_start, tentative_end + 1)
            if boundary > start:
                end = boundary

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(chunk_text)

        if end >= text_length:
            break

        next_start = max(end - chunk_overlap, 0)
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def write_chunks_jsonl(chunks: list[ChunkRecord], output_path: Path) -> None:
    """Write chunk records to JSONL."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    with temp_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False))
            handle.write("\n")

    temp_path.replace(output_path)


def build_chunk_id(doc_id: str, page_number: int, chunk_index: int) -> str:
    """Create a stable chunk identifier."""

    return f"{doc_id}:p{page_number:04d}:c{chunk_index:03d}"


def estimate_tokens(text: str) -> int:
    """Estimate token count using whitespace tokenization."""

    return len(text.split())


def _validate_chunk_config(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ChunkingError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ChunkingError("chunk_overlap must be 0 or greater")
    if chunk_overlap >= chunk_size:
        raise ChunkingError("chunk_overlap must be smaller than chunk_size")


def _resolve_path(path_value: Path | str, project_root: Path) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (project_root / candidate).resolve()
