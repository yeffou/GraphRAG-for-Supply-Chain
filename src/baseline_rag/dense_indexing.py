"""Dense semantic indexing over chunked document records."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sentence_transformers import SentenceTransformer

from src.baseline_rag.indexing import BaselineIndexingError, load_chunk_records, write_chunk_metadata
from src.config import ROOT_DIR
from src.preprocessing.chunking import ChunkRecord


DEFAULT_DENSE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class DenseIndexBuildSummary(BaseModel):
    """Summary information about a completed dense index build."""

    embedding_model: str
    index_backend: str
    indexed_chunks: int = Field(ge=0)
    embedding_dimension: int = Field(ge=1)
    index_dir: str
    source_chunks_path: str
    created_at_utc: str


class DenseSavedIndexInfo(BaseModel):
    """On-disk description of the saved dense index."""

    model_config = ConfigDict(extra="forbid")

    method: str
    embedding_model: str
    index_backend: str
    indexed_chunks: int = Field(ge=0)
    embedding_dimension: int = Field(ge=1)
    source_chunks_path: str
    created_at_utc: str
    embeddings_path: str
    metadata_path: str
    faiss_index_path: str | None = None
    notes: str | None = None


def build_dense_index(
    chunks_path: Path | str,
    index_dir: Path | str,
    embedding_model: str = DEFAULT_DENSE_EMBEDDING_MODEL,
    batch_size: int = 32,
    project_root: Path = ROOT_DIR,
) -> DenseIndexBuildSummary:
    """Build and save a dense semantic index from chunk records."""

    if batch_size <= 0:
        raise BaselineIndexingError("batch_size must be greater than 0")

    root_dir = project_root.resolve()
    resolved_chunks_path = _resolve_path(chunks_path, root_dir)
    resolved_index_dir = _resolve_path(index_dir, root_dir)
    resolved_index_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_chunk_records(resolved_chunks_path)
    if not chunks:
        raise BaselineIndexingError(f"no chunks found in {resolved_chunks_path}")

    model = load_sentence_transformer_model(
        embedding_model=embedding_model,
        local_files_only=False,
    )
    embeddings = model.encode(
        [chunk.text for chunk in chunks],
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    embeddings_path = resolved_index_dir / "chunk_embeddings.npy"
    metadata_path = resolved_index_dir / "chunk_metadata.jsonl"
    info_path = resolved_index_dir / "index_info.json"
    faiss_index_path = resolved_index_dir / "faiss.index"

    np.save(embeddings_path, embeddings)
    write_chunk_metadata(chunks=chunks, metadata_path=metadata_path)

    backend = "dense_cosine_numpy"
    notes = None
    saved_faiss_path: str | None = None

    try:
        import faiss

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, str(faiss_index_path))
        backend = "faiss_index_flat_ip"
        saved_faiss_path = str(faiss_index_path)
    except Exception as exc:  # pragma: no cover - depends on local faiss availability
        notes = f"FAISS unavailable; using dense cosine fallback over saved embeddings ({exc.__class__.__name__}: {exc})"

    created_at_utc = _utc_now()
    index_info = DenseSavedIndexInfo(
        method="baseline_dense",
        embedding_model=embedding_model,
        index_backend=backend,
        indexed_chunks=len(chunks),
        embedding_dimension=int(embeddings.shape[1]),
        source_chunks_path=str(resolved_chunks_path),
        created_at_utc=created_at_utc,
        embeddings_path=str(embeddings_path),
        metadata_path=str(metadata_path),
        faiss_index_path=saved_faiss_path,
        notes=notes,
    )
    info_path.write_text(
        json.dumps(index_info.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )

    return DenseIndexBuildSummary(
        embedding_model=embedding_model,
        index_backend=backend,
        indexed_chunks=len(chunks),
        embedding_dimension=int(embeddings.shape[1]),
        index_dir=str(resolved_index_dir),
        source_chunks_path=str(resolved_chunks_path),
        created_at_utc=created_at_utc,
    )


def load_dense_index(
    index_dir: Path | str,
    project_root: Path = ROOT_DIR,
) -> tuple[DenseSavedIndexInfo, np.ndarray, list[ChunkRecord], object | None]:
    """Load a previously saved dense index."""

    root_dir = project_root.resolve()
    resolved_index_dir = _resolve_path(index_dir, root_dir)
    info_path = resolved_index_dir / "index_info.json"

    if not info_path.exists():
        raise BaselineIndexingError(f"index info file does not exist: {info_path}")

    try:
        info_payload = json.loads(info_path.read_text(encoding="utf-8"))
        info = DenseSavedIndexInfo.model_validate(info_payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise BaselineIndexingError(f"invalid dense index info file: {info_path}") from exc

    embeddings_path = Path(info.embeddings_path)
    metadata_path = Path(info.metadata_path)

    if not embeddings_path.exists():
        raise BaselineIndexingError(f"missing saved embeddings: {embeddings_path}")
    if not metadata_path.exists():
        raise BaselineIndexingError(f"missing saved metadata: {metadata_path}")

    embeddings = np.load(embeddings_path).astype(np.float32)
    chunks = load_chunk_records(metadata_path)

    if embeddings.shape[0] != len(chunks):
        raise BaselineIndexingError(
            "saved dense index is inconsistent: embedding row count does not match metadata count"
        )

    faiss_index = None
    if info.index_backend == "faiss_index_flat_ip":
        if not info.faiss_index_path:
            raise BaselineIndexingError("dense index metadata says FAISS is used, but no faiss_index_path was saved")
        faiss_path = Path(info.faiss_index_path)
        if not faiss_path.exists():
            raise BaselineIndexingError(f"missing FAISS index file: {faiss_path}")
        try:
            import faiss

            faiss_index = faiss.read_index(str(faiss_path))
        except Exception as exc:  # pragma: no cover - depends on local faiss availability
            raise BaselineIndexingError(
                f"unable to load saved FAISS index: {exc.__class__.__name__}: {exc}"
            ) from exc

    return info, embeddings, chunks, faiss_index


def load_sentence_transformer_model(
    embedding_model: str,
    local_files_only: bool,
) -> SentenceTransformer:
    """Load a sentence-transformers model with explicit local/offline control."""

    try:
        return SentenceTransformer(
            embedding_model,
            local_files_only=local_files_only,
        )
    except TypeError:
        return SentenceTransformer(embedding_model)


def _resolve_path(path_value: Path | str, project_root: Path) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (project_root / candidate).resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
