"""Baseline vector indexing over chunked document records."""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from scipy.sparse import csr_matrix, load_npz, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import ROOT_DIR
from src.preprocessing.chunking import ChunkRecord


DEFAULT_EMBEDDING_MODEL = "tfidf_word_unigram_bigram_l2_v1"


class BaselineIndexingError(Exception):
    """Raised when index building or loading fails."""


class IndexBuildSummary(BaseModel):
    """Summary information about a completed index build."""

    embedding_model: str
    indexed_chunks: int = Field(ge=0)
    vocabulary_size: int = Field(ge=0)
    index_dir: str
    source_chunks_path: str
    created_at_utc: str


class SavedIndexInfo(BaseModel):
    """On-disk description of the saved baseline index."""

    model_config = ConfigDict(extra="forbid")

    method: str
    embedding_model: str
    indexed_chunks: int = Field(ge=0)
    vocabulary_size: int = Field(ge=0)
    source_chunks_path: str
    created_at_utc: str
    vectorizer_path: str
    matrix_path: str
    metadata_path: str


def build_baseline_index(
    chunks_path: Path | str,
    index_dir: Path | str,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    project_root: Path = ROOT_DIR,
) -> IndexBuildSummary:
    """Build and save a baseline TF-IDF index from chunk records."""

    root_dir = project_root.resolve()
    resolved_chunks_path = _resolve_path(chunks_path, root_dir)
    resolved_index_dir = _resolve_path(index_dir, root_dir)
    resolved_index_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_chunk_records(resolved_chunks_path)
    if not chunks:
        raise BaselineIndexingError(f"no chunks found in {resolved_chunks_path}")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        norm="l2",
    )
    matrix = vectorizer.fit_transform(chunk.text for chunk in chunks)
    if not isinstance(matrix, csr_matrix):
        matrix = matrix.tocsr()

    vectorizer_path = resolved_index_dir / "vectorizer.pkl"
    matrix_path = resolved_index_dir / "chunk_vectors.npz"
    metadata_path = resolved_index_dir / "chunk_metadata.jsonl"
    info_path = resolved_index_dir / "index_info.json"

    with vectorizer_path.open("wb") as handle:
        pickle.dump(vectorizer, handle)

    save_npz(matrix_path, matrix)
    write_chunk_metadata(chunks=chunks, metadata_path=metadata_path)

    created_at_utc = _utc_now()
    index_info = SavedIndexInfo(
        method="baseline_rag",
        embedding_model=embedding_model,
        indexed_chunks=len(chunks),
        vocabulary_size=len(vectorizer.vocabulary_),
        source_chunks_path=str(resolved_chunks_path),
        created_at_utc=created_at_utc,
        vectorizer_path=str(vectorizer_path),
        matrix_path=str(matrix_path),
        metadata_path=str(metadata_path),
    )
    _write_json(index_info.model_dump(mode="json"), info_path)

    return IndexBuildSummary(
        embedding_model=embedding_model,
        indexed_chunks=len(chunks),
        vocabulary_size=len(vectorizer.vocabulary_),
        index_dir=str(resolved_index_dir),
        source_chunks_path=str(resolved_chunks_path),
        created_at_utc=created_at_utc,
    )


def load_saved_index(index_dir: Path | str, project_root: Path = ROOT_DIR) -> tuple[SavedIndexInfo, TfidfVectorizer, csr_matrix, list[ChunkRecord]]:
    """Load a previously saved baseline index."""

    root_dir = project_root.resolve()
    resolved_index_dir = _resolve_path(index_dir, root_dir)
    info_path = resolved_index_dir / "index_info.json"

    if not info_path.exists():
        raise BaselineIndexingError(f"index info file does not exist: {info_path}")

    try:
        info_payload = json.loads(info_path.read_text(encoding="utf-8"))
        info = SavedIndexInfo.model_validate(info_payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise BaselineIndexingError(f"invalid index info file: {info_path}") from exc

    vectorizer_path = Path(info.vectorizer_path)
    matrix_path = Path(info.matrix_path)
    metadata_path = Path(info.metadata_path)

    if not vectorizer_path.exists():
        raise BaselineIndexingError(f"missing saved vectorizer: {vectorizer_path}")
    if not matrix_path.exists():
        raise BaselineIndexingError(f"missing saved matrix: {matrix_path}")
    if not metadata_path.exists():
        raise BaselineIndexingError(f"missing saved metadata: {metadata_path}")

    with vectorizer_path.open("rb") as handle:
        vectorizer = pickle.load(handle)

    matrix = load_npz(matrix_path).tocsr()
    chunks = load_chunk_records(metadata_path)

    if matrix.shape[0] != len(chunks):
        raise BaselineIndexingError(
            "saved index is inconsistent: vector row count does not match metadata count"
        )

    return info, vectorizer, matrix, chunks


def load_chunk_records(chunks_path: Path) -> list[ChunkRecord]:
    """Load and validate chunk records from JSONL."""

    if not chunks_path.exists():
        raise BaselineIndexingError(f"chunks file does not exist: {chunks_path}")
    if not chunks_path.is_file():
        raise BaselineIndexingError(f"chunks path is not a file: {chunks_path}")

    chunks: list[ChunkRecord] = []
    with chunks_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue

            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise BaselineIndexingError(
                    f"malformed JSON in {chunks_path} line {line_number}: {exc.msg}"
                ) from exc

            try:
                chunk = ChunkRecord.model_validate(payload)
            except ValidationError as exc:
                raise BaselineIndexingError(
                    f"invalid chunk schema in {chunks_path} line {line_number}: {exc}"
                ) from exc

            chunks.append(chunk)

    return chunks


def write_chunk_metadata(chunks: list[ChunkRecord], metadata_path: Path) -> None:
    """Write chunk metadata JSONL for saved retrieval mapping."""

    with metadata_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False))
            handle.write("\n")


def _write_json(payload: dict, output_path: Path) -> None:
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve_path(path_value: Path | str, project_root: Path) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (project_root / candidate).resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
