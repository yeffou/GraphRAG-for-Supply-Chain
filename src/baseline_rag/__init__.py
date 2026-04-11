"""Baseline RAG exports."""

from src.baseline_rag.dense_indexing import (
    DEFAULT_DENSE_EMBEDDING_MODEL,
    DenseIndexBuildSummary,
    DenseSavedIndexInfo,
    build_dense_index,
    load_dense_index,
)
from src.baseline_rag.dense_query import run_dense_query
from src.baseline_rag.indexing import (
    DEFAULT_EMBEDDING_MODEL,
    BaselineIndexingError,
    IndexBuildSummary,
    SavedIndexInfo,
    build_baseline_index,
    load_saved_index,
)
from src.baseline_rag.query import (
    BaselineQueryError,
    GenerationResult,
    QueryRunRecord,
    RetrievedChunk,
    run_baseline_query,
)

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_DENSE_EMBEDDING_MODEL",
    "BaselineIndexingError",
    "BaselineQueryError",
    "DenseIndexBuildSummary",
    "DenseSavedIndexInfo",
    "GenerationResult",
    "IndexBuildSummary",
    "QueryRunRecord",
    "RetrievedChunk",
    "SavedIndexInfo",
    "build_baseline_index",
    "build_dense_index",
    "load_dense_index",
    "load_saved_index",
    "run_baseline_query",
    "run_dense_query",
]
