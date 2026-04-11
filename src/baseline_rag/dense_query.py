"""Dense semantic retrieval and optional answer generation."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.baseline_rag.dense_indexing import (
    DEFAULT_DENSE_EMBEDDING_MODEL,
    load_dense_index,
    load_sentence_transformer_model,
)
from src.baseline_rag.indexing import BaselineIndexingError
from src.baseline_rag.query import (
    BaselineQueryError,
    QueryRunRecord,
    RetrievedChunk,
    maybe_generate_answer,
)
from src.config import ROOT_DIR, LLMConfig


RUN_ID_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def run_dense_query(
    query_text: str,
    index_dir: Path | str,
    output_dir: Path | str,
    top_k: int,
    llm_config: LLMConfig,
    generate_answer: bool,
    project_root: Path = ROOT_DIR,
) -> QueryRunRecord:
    """Run dense retrieval over the saved semantic index and optionally generate an answer."""

    if top_k <= 0:
        raise BaselineQueryError("top_k must be greater than 0")
    if not query_text.strip():
        raise BaselineQueryError("query_text must not be empty")

    root_dir = project_root.resolve()
    resolved_output_dir = _resolve_path(output_dir, root_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        index_info, embeddings, chunks, faiss_index = load_dense_index(
            index_dir=index_dir,
            project_root=root_dir,
        )
    except BaselineIndexingError as exc:
        raise BaselineQueryError(str(exc)) from exc

    model = load_sentence_transformer_model(
        embedding_model=index_info.embedding_model,
        local_files_only=True,
    )
    query_embedding = model.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    result_count = min(top_k, embeddings.shape[0])
    if index_info.index_backend == "faiss_index_flat_ip" and faiss_index is not None:
        scores, indices = faiss_index.search(query_embedding, result_count)
        ranked_pairs = [
            (int(idx), float(score))
            for idx, score in zip(indices[0], scores[0], strict=False)
            if idx >= 0
        ]
    else:
        similarity_scores = embeddings @ query_embedding[0]
        ranked_indices = np.argsort(similarity_scores)[::-1][:result_count]
        ranked_pairs = [
            (int(idx), float(similarity_scores[idx]))
            for idx in ranked_indices
        ]

    retrieved_chunks = [
        RetrievedChunk(
            rank=rank + 1,
            score=score,
            chunk_id=chunks[idx].chunk_id,
            doc_id=chunks[idx].doc_id,
            title=chunks[idx].title,
            publisher=chunks[idx].publisher,
            page_number=chunks[idx].page_number,
            source_url=chunks[idx].source_url,
            local_path=chunks[idx].local_path,
            year=chunks[idx].year,
            text=chunks[idx].text,
            char_count=chunks[idx].char_count,
        )
        for rank, (idx, score) in enumerate(ranked_pairs)
    ]

    generation = maybe_generate_answer(
        query_text=query_text,
        retrieved_chunks=retrieved_chunks,
        llm_config=llm_config,
        generate_answer=generate_answer,
    )

    timestamp_utc = _utc_now()
    run_record = QueryRunRecord(
        run_id=build_dense_run_id(query_text=query_text, timestamp_utc=timestamp_utc),
        timestamp_utc=timestamp_utc,
        method="baseline_dense",
        embedding_model=index_info.embedding_model,
        query_text=query_text,
        top_k=top_k,
        index_dir=str(_resolve_path(index_dir, root_dir)),
        retrieved_chunks=retrieved_chunks,
        generation=generation,
    )

    output_path = resolved_output_dir / f"{run_record.run_id}.json"
    output_path.write_text(
        json.dumps(run_record.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return run_record


def build_dense_run_id(query_text: str, timestamp_utc: str) -> str:
    """Create a filesystem-safe run identifier for dense retrieval."""

    compact_timestamp = (
        timestamp_utc.replace("-", "")
        .replace(":", "")
        .replace("+00:00", "Z")
        .replace(".", "")
    )
    query_stub = RUN_ID_RE.sub("_", query_text.lower()).strip("_")[:50] or "query"
    return f"baseline_dense_{compact_timestamp}_{query_stub}"


def _resolve_path(path_value: Path | str, project_root: Path) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (project_root / candidate).resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
