"""Baseline retrieval and optional answer generation."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from src.baseline_rag.indexing import (
    BaselineIndexingError,
    DEFAULT_EMBEDDING_MODEL,
    load_saved_index,
)
from src.config import ROOT_DIR, LLMConfig


RUN_ID_RE = re.compile(r"[^a-zA-Z0-9_-]+")
DEFAULT_PROMPT_TEMPLATE_VERSION = "baseline_rag_answer_v1"


class GenerationResult(BaseModel):
    """Result of optional answer generation."""

    attempted: bool
    status: str
    reason: str | None = None
    model: str | None = None
    prompt_template_version: str | None = None
    answer: str | None = None
    error: str | None = None


class RetrievedChunk(BaseModel):
    """One retrieved chunk returned at query time."""

    rank: int = Field(ge=1)
    score: float
    chunk_id: str
    doc_id: str
    title: str
    publisher: str
    page_number: int = Field(ge=1)
    source_url: str
    local_path: str
    year: int
    text: str
    char_count: int = Field(ge=0)


class QueryRunRecord(BaseModel):
    """Saved retrieval run output."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    timestamp_utc: str
    method: str
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    query_text: str
    top_k: int = Field(ge=1)
    index_dir: str
    retrieved_chunks: list[RetrievedChunk]
    generation: GenerationResult


class BaselineQueryError(Exception):
    """Raised when query-time retrieval fails."""


def run_baseline_query(
    query_text: str,
    index_dir: Path | str,
    output_dir: Path | str,
    top_k: int,
    llm_config: LLMConfig,
    generate_answer: bool,
    project_root: Path = ROOT_DIR,
) -> QueryRunRecord:
    """Run retrieval over the saved baseline index and optionally generate an answer."""

    if top_k <= 0:
        raise BaselineQueryError("top_k must be greater than 0")
    if not query_text.strip():
        raise BaselineQueryError("query_text must not be empty")

    root_dir = project_root.resolve()
    resolved_output_dir = _resolve_path(output_dir, root_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        index_info, vectorizer, matrix, chunks = load_saved_index(
            index_dir=index_dir,
            project_root=root_dir,
        )
    except BaselineIndexingError as exc:
        raise BaselineQueryError(str(exc)) from exc

    query_vector = vectorizer.transform([query_text])
    scores = (query_vector @ matrix.T).toarray()[0]

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda idx: scores[idx],
        reverse=True,
    )[: min(top_k, len(scores))]

    retrieved_chunks = [
        RetrievedChunk(
            rank=rank + 1,
            score=float(scores[idx]),
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
        for rank, idx in enumerate(ranked_indices)
    ]

    generation = maybe_generate_answer(
        query_text=query_text,
        retrieved_chunks=retrieved_chunks,
        llm_config=llm_config,
        generate_answer=generate_answer,
    )

    timestamp_utc = _utc_now()
    run_record = QueryRunRecord(
        run_id=build_run_id(query_text=query_text, timestamp_utc=timestamp_utc),
        timestamp_utc=timestamp_utc,
        method="baseline_rag",
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


def maybe_generate_answer(
    query_text: str,
    retrieved_chunks: list[RetrievedChunk],
    llm_config: LLMConfig,
    generate_answer: bool,
) -> GenerationResult:
    """Generate an answer from retrieved chunks when credentials are configured."""

    if not generate_answer:
        return GenerationResult(
            attempted=False,
            status="skipped",
            reason="generation not requested",
        )

    if not llm_config.openrouter_api_key or not llm_config.openrouter_model:
        return GenerationResult(
            attempted=False,
            status="skipped",
            reason="OpenRouter API key or model not configured",
        )

    prompt = build_answer_prompt(query_text=query_text, retrieved_chunks=retrieved_chunks)
    endpoint = f"{llm_config.openrouter_base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": llm_config.openrouter_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Answer only from the provided retrieved evidence. "
                    "If the evidence is insufficient, say so clearly."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.0,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {llm_config.openrouter_api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return GenerationResult(
            attempted=True,
            status="failed",
            reason="OpenRouter request failed",
            model=llm_config.openrouter_model,
            prompt_template_version=DEFAULT_PROMPT_TEMPLATE_VERSION,
            error=f"{exc.__class__.__name__}: {exc}",
        )

    try:
        answer_text = response_payload["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, AttributeError) as exc:
        return GenerationResult(
            attempted=True,
            status="failed",
            reason="OpenRouter response format was not as expected",
            model=llm_config.openrouter_model,
            prompt_template_version=DEFAULT_PROMPT_TEMPLATE_VERSION,
            error=f"{exc.__class__.__name__}: {exc}",
        )

    return GenerationResult(
        attempted=True,
        status="succeeded",
        model=llm_config.openrouter_model,
        prompt_template_version=DEFAULT_PROMPT_TEMPLATE_VERSION,
        answer=answer_text,
    )


def build_answer_prompt(query_text: str, retrieved_chunks: list[RetrievedChunk]) -> str:
    """Build the prompt used for optional answer generation."""

    evidence_lines = []
    for chunk in retrieved_chunks:
        evidence_lines.append(
            f"[{chunk.rank}] chunk_id={chunk.chunk_id} doc_id={chunk.doc_id} "
            f"title={chunk.title} page={chunk.page_number} source_url={chunk.source_url}\n"
            f"{chunk.text}"
        )

    evidence_block = "\n\n".join(evidence_lines)
    return (
        f"Question:\n{query_text}\n\n"
        "Retrieved evidence:\n"
        f"{evidence_block}\n\n"
        "Provide a concise answer grounded only in the retrieved evidence. "
        "Cite the supporting chunk ranks in brackets."
    )


def build_run_id(query_text: str, timestamp_utc: str) -> str:
    """Create a filesystem-safe run identifier."""

    compact_timestamp = (
        timestamp_utc.replace("-", "")
        .replace(":", "")
        .replace("+00:00", "Z")
        .replace(".", "")
    )
    query_stub = RUN_ID_RE.sub("_", query_text.lower()).strip("_")[:50] or "query"
    return f"baseline_{compact_timestamp}_{query_stub}"


def _resolve_path(path_value: Path | str, project_root: Path) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (project_root / candidate).resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
