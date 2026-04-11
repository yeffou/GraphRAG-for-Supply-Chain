"""Hybrid GraphRAG retrieval: dense recall with graph-aware reranking."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.baseline_rag.dense_indexing import load_dense_index, load_sentence_transformer_model
from src.baseline_rag.query import GenerationResult, maybe_generate_answer
from src.config import ROOT_DIR, LLMConfig
from src.graph_rag.indexing import GraphChunkRecord, load_graph_index
from src.graph_rag.query import (
    GraphQueryError,
    GraphRetrievedChunk,
    _entity_chunk_mentions,
    build_generic_entity_id_set,
    detect_query_intent,
    query_entity_specificity,
    score_chunk,
)
from src.graph_rag.schema import GRAPH_SCHEMA_VERSION, EntityMention, extract_entities, query_terms


RUN_ID_RE = re.compile(r"[^a-zA-Z0-9_-]+")
DEFAULT_DENSE_CANDIDATES = 40
DEFAULT_GRAPH_CANDIDATES = 20
RRF_K = 10


class HybridScoreBreakdown(BaseModel):
    """Transparent fusion components for one hybrid-retrieved chunk."""

    model_config = ConfigDict(extra="forbid")

    dense_similarity: float
    graph_score: float
    dense_rank_signal: float
    graph_rank_signal: float
    entity_coverage: float
    relation_signal: float
    total: float


class HybridGraphRetrievedChunk(GraphRetrievedChunk):
    """Hybrid retrieval result with dense and graph provenance."""

    model_config = ConfigDict(extra="forbid")

    dense_rank: int | None = Field(default=None, ge=1)
    dense_score: float | None = None
    graph_rank: int | None = Field(default=None, ge=1)
    graph_score: float | None = None
    hybrid_score_breakdown: HybridScoreBreakdown


class HybridGraphQueryRunRecord(BaseModel):
    """Saved hybrid retrieval run output."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    timestamp_utc: str
    method: str
    embedding_model: str
    query_text: str
    top_k: int = Field(ge=1)
    index_dir: str
    dense_index_dir: str
    graph_index_dir: str
    query_entities: list[EntityMention]
    dense_candidate_count: int = Field(ge=1)
    graph_candidate_count: int = Field(ge=1)
    retrieved_chunks: list[HybridGraphRetrievedChunk]
    generation: GenerationResult


class HybridGraphQueryError(GraphQueryError):
    """Raised when hybrid GraphRAG retrieval fails."""


def run_hybrid_graph_query(
    query_text: str,
    dense_index_dir: Path | str,
    graph_index_dir: Path | str,
    output_dir: Path | str,
    top_k: int,
    llm_config: LLMConfig,
    generate_answer: bool,
    project_root: Path = ROOT_DIR,
    dense_candidate_count: int = DEFAULT_DENSE_CANDIDATES,
    graph_candidate_count: int = DEFAULT_GRAPH_CANDIDATES,
) -> HybridGraphQueryRunRecord:
    """Run dense-plus-graph hybrid retrieval and optionally answer generation."""

    if top_k <= 0:
        raise HybridGraphQueryError("top_k must be greater than 0")
    if dense_candidate_count <= 0:
        raise HybridGraphQueryError("dense_candidate_count must be greater than 0")
    if graph_candidate_count <= 0:
        raise HybridGraphQueryError("graph_candidate_count must be greater than 0")
    if not query_text.strip():
        raise HybridGraphQueryError("query_text must not be empty")

    root_dir = project_root.resolve()
    resolved_output_dir = _resolve_path(output_dir, root_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    resolved_dense_index_dir = _resolve_path(dense_index_dir, root_dir)
    resolved_graph_index_dir = _resolve_path(graph_index_dir, root_dir)

    dense_info, dense_ranked_pairs, dense_chunk_map = _retrieve_dense_candidates(
        query_text=query_text,
        index_dir=resolved_dense_index_dir,
        project_root=root_dir,
        candidate_count=dense_candidate_count,
    )
    (
        graph_ranked_chunks,
        graph_chunk_map,
        graph_scored_by_chunk_id,
        extracted_query_entities,
    ) = _retrieve_graph_candidates(
        query_text=query_text,
        index_dir=resolved_graph_index_dir,
        project_root=root_dir,
        candidate_count=graph_candidate_count,
    )

    chunk_map = dense_chunk_map
    if set(chunk_map) != set(graph_chunk_map):
        raise HybridGraphQueryError(
            "dense and graph indexes are inconsistent: chunk ids do not match"
        )

    dense_ranks = {
        chunk_id: rank
        for rank, (chunk_id, _) in enumerate(dense_ranked_pairs, start=1)
    }
    dense_scores = {
        chunk_id: score for chunk_id, score in dense_ranked_pairs
    }
    graph_ranks = {
        chunk.chunk_id: rank
        for rank, chunk in enumerate(graph_ranked_chunks, start=1)
    }

    candidate_ids = set(dense_ranks) | set(graph_ranks)
    if not candidate_ids:
        raise HybridGraphQueryError("no hybrid retrieval candidates were produced")

    dense_score_values = [
        dense_scores[chunk_id]
        for chunk_id in candidate_ids
        if chunk_id in dense_scores
    ]
    graph_score_values = [
        graph_scored_by_chunk_id[chunk_id].score_breakdown.total
        for chunk_id in candidate_ids
        if chunk_id in graph_scored_by_chunk_id
    ]

    query_entity_count = max(len(extracted_query_entities), 1)
    hybrid_chunks: list[HybridGraphRetrievedChunk] = []
    for chunk_id in candidate_ids:
        dense_rank = dense_ranks.get(chunk_id)
        graph_chunk = graph_scored_by_chunk_id.get(chunk_id)
        graph_rank = graph_ranks.get(chunk_id)

        dense_similarity = _min_max_normalize(
            dense_scores.get(chunk_id),
            dense_score_values,
        )
        graph_score = _min_max_normalize(
            graph_chunk.score_breakdown.total if graph_chunk is not None else None,
            graph_score_values,
        )
        dense_rank_signal = _rank_signal(dense_rank)
        graph_rank_signal = _rank_signal(graph_rank)
        entity_coverage = (
            min(len(graph_chunk.contributing_entities) / query_entity_count, 1.0)
            if graph_chunk is not None
            else 0.0
        )
        relation_signal = _relation_signal(graph_chunk)

        total = (
            0.50 * dense_similarity
            + 0.10 * graph_score
            + 0.10 * dense_rank_signal
            + 0.06 * graph_rank_signal
            + 0.12 * entity_coverage
            + 0.12 * relation_signal
        )

        base_chunk = chunk_map[chunk_id]
        graph_explanation = graph_chunk or GraphRetrievedChunk(
            rank=1,
            score=0.0,
            chunk_id=base_chunk.chunk_id,
            doc_id=base_chunk.doc_id,
            title=base_chunk.title,
            publisher=base_chunk.publisher,
            page_number=base_chunk.page_number,
            source_url=base_chunk.source_url,
            local_path=base_chunk.local_path,
            year=base_chunk.year,
            text=base_chunk.text,
            char_count=base_chunk.char_count,
            contributing_entities=[],
            contributing_relations=[],
            supporting_paths=[],
            score_breakdown={
                "entity_match": 0.0,
                "coverage_bonus": 0.0,
                "sentence_alignment": 0.0,
                "direct_relation": 0.0,
                "path_bonus": 0.0,
                "lexical_overlap": 0.0,
                "generic_penalty": 0.0,
                "query_miss_penalty": 0.0,
                "hub_penalty": 0.0,
                "total": 0.0,
            },
        )
        hybrid_chunks.append(
            HybridGraphRetrievedChunk(
                rank=1,
                score=round(total, 6),
                chunk_id=base_chunk.chunk_id,
                doc_id=base_chunk.doc_id,
                title=base_chunk.title,
                publisher=base_chunk.publisher,
                page_number=base_chunk.page_number,
                source_url=base_chunk.source_url,
                local_path=base_chunk.local_path,
                year=base_chunk.year,
                text=base_chunk.text,
                char_count=base_chunk.char_count,
                contributing_entities=graph_explanation.contributing_entities,
                contributing_relations=graph_explanation.contributing_relations,
                supporting_paths=graph_explanation.supporting_paths,
                score_breakdown=graph_explanation.score_breakdown,
                dense_rank=dense_rank,
                dense_score=round(dense_scores[chunk_id], 6) if chunk_id in dense_scores else None,
                graph_rank=graph_rank,
                graph_score=round(graph_explanation.score_breakdown.total, 6)
                if graph_chunk is not None
                else None,
                hybrid_score_breakdown=HybridScoreBreakdown(
                    dense_similarity=round(dense_similarity, 6),
                    graph_score=round(graph_score, 6),
                    dense_rank_signal=round(dense_rank_signal, 6),
                    graph_rank_signal=round(graph_rank_signal, 6),
                    entity_coverage=round(entity_coverage, 6),
                    relation_signal=round(relation_signal, 6),
                    total=round(total, 6),
                ),
            )
        )

    ranked_chunks = sorted(hybrid_chunks, key=lambda item: item.score, reverse=True)[
        : min(top_k, len(hybrid_chunks))
    ]
    for rank, chunk in enumerate(ranked_chunks, start=1):
        chunk.rank = rank

    generation = maybe_generate_answer(
        query_text=query_text,
        retrieved_chunks=ranked_chunks,
        llm_config=llm_config,
        generate_answer=generate_answer,
    )

    timestamp_utc = _utc_now()
    run_record = HybridGraphQueryRunRecord(
        run_id=build_hybrid_run_id(query_text=query_text, timestamp_utc=timestamp_utc),
        timestamp_utc=timestamp_utc,
        method="baseline_graph_hybrid",
        embedding_model=(
            f"{dense_info.embedding_model}+{GRAPH_SCHEMA_VERSION}"
        ),
        query_text=query_text,
        top_k=top_k,
        index_dir=(
            f"dense={resolved_dense_index_dir};graph={resolved_graph_index_dir}"
        ),
        dense_index_dir=str(resolved_dense_index_dir),
        graph_index_dir=str(resolved_graph_index_dir),
        query_entities=extracted_query_entities,
        dense_candidate_count=dense_candidate_count,
        graph_candidate_count=graph_candidate_count,
        retrieved_chunks=ranked_chunks,
        generation=generation,
    )

    output_path = resolved_output_dir / f"{run_record.run_id}.json"
    output_path.write_text(
        json.dumps(run_record.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return run_record


def _retrieve_dense_candidates(
    query_text: str,
    index_dir: Path,
    project_root: Path,
    candidate_count: int,
) -> tuple[object, list[tuple[str, float]], dict[str, object]]:
    info, embeddings, chunks, faiss_index = load_dense_index(
        index_dir=index_dir,
        project_root=project_root,
    )
    model = load_sentence_transformer_model(
        embedding_model=info.embedding_model,
        local_files_only=True,
    )
    query_embedding = model.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    result_count = min(candidate_count, embeddings.shape[0])
    if info.index_backend == "faiss_index_flat_ip" and faiss_index is not None:
        scores, indices = faiss_index.search(query_embedding, result_count)
        ranked_pairs = [
            (chunks[int(idx)].chunk_id, float(score))
            for idx, score in zip(indices[0], scores[0], strict=False)
            if idx >= 0
        ]
    else:
        similarity_scores = embeddings @ query_embedding[0]
        ranked_indices = np.argsort(similarity_scores)[::-1][:result_count]
        ranked_pairs = [
            (chunks[int(idx)].chunk_id, float(similarity_scores[int(idx)]))
            for idx in ranked_indices
        ]

    chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
    return info, ranked_pairs, chunk_map


def _retrieve_graph_candidates(
    query_text: str,
    index_dir: Path,
    project_root: Path,
    candidate_count: int,
) -> tuple[list[GraphRetrievedChunk], dict[str, object], dict[str, GraphRetrievedChunk], list[EntityMention]]:
    _, graph, chunks, chunk_graph_records, entity_specs = load_graph_index(
        index_dir=index_dir,
        project_root=project_root,
    )

    chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
    graph_record_map: dict[str, GraphChunkRecord] = {
        record.chunk_id: record for record in chunk_graph_records
    }

    extracted_query_entities = extract_entities(
        query_text,
        domain_tags=None,
        allow_domain_tag_backfill=False,
        entity_specs=entity_specs,
    )
    query_term_set = query_terms(query_text)
    intent = detect_query_intent(query_text)
    generic_entity_ids = build_generic_entity_id_set(
        graph=graph,
        total_chunks=len(chunks),
        query_entities=extracted_query_entities,
    )
    query_specificity = {
        mention.entity_id: query_entity_specificity(
            entity_type=mention.entity_type,
            mention_count=_entity_chunk_mentions(graph, mention.entity_id),
            is_generic=mention.entity_id in generic_entity_ids,
        )
        for mention in extracted_query_entities
        if graph.has_node(mention.entity_id)
    }
    query_entity_ids = set(query_specificity)
    query_entity_types = {
        mention.entity_id: mention.entity_type
        for mention in extracted_query_entities
        if mention.entity_id in query_entity_ids
    }

    scored_chunks: list[GraphRetrievedChunk] = []
    for chunk_id, chunk in chunk_map.items():
        graph_record = graph_record_map.get(chunk_id)
        if graph_record is None:
            continue
        scored = score_chunk(
            chunk=chunk,
            graph_record=graph_record,
            graph=graph,
            query_entity_ids=query_entity_ids,
            query_entity_types=query_entity_types,
            query_specificity=query_specificity,
            query_term_set=query_term_set,
            intent=intent,
            generic_entity_ids=generic_entity_ids,
        )
        if scored is not None:
            scored_chunks.append(scored)

    ranked = sorted(scored_chunks, key=lambda item: item.score, reverse=True)
    graph_scored_by_chunk_id = {chunk.chunk_id: chunk for chunk in ranked}
    return (
        ranked[: min(candidate_count, len(ranked))],
        chunk_map,
        graph_scored_by_chunk_id,
        extracted_query_entities,
    )


def build_hybrid_run_id(query_text: str, timestamp_utc: str) -> str:
    """Create a filesystem-safe run identifier for the hybrid query."""

    compact_timestamp = (
        timestamp_utc.replace("-", "")
        .replace(":", "")
        .replace("+00:00", "Z")
        .replace(".", "")
    )
    query_stub = RUN_ID_RE.sub("_", query_text.lower()).strip("_")[:50] or "query"
    return f"baseline_graph_hybrid_{compact_timestamp}_{query_stub}"


def _relation_signal(chunk: GraphRetrievedChunk | None) -> float:
    if chunk is None:
        return 0.0

    direct_pairs = sum(
        1 for relation in chunk.contributing_relations if relation.direct_query_pair
    )
    aligned_relations = sum(
        1 for relation in chunk.contributing_relations if relation.query_aligned
    )
    path_count = len(chunk.supporting_paths)
    return min(
        1.0,
        0.55 * min(direct_pairs, 1)
        + 0.25 * min(aligned_relations / 2.0, 1.0)
        + 0.20 * min(path_count / 2.0, 1.0),
    )


def _rank_signal(rank: int | None) -> float:
    if rank is None:
        return 0.0
    max_rrf = 1.0 / (RRF_K + 1)
    return (1.0 / (RRF_K + rank)) / max_rrf


def _min_max_normalize(value: float | None, values: list[float]) -> float:
    if value is None or not values:
        return 0.0
    minimum = min(values)
    maximum = max(values)
    if maximum <= minimum:
        return 1.0 if value > 0 else 0.0
    return (value - minimum) / (maximum - minimum)


def _resolve_path(path_value: Path | str, project_root: Path) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (project_root / candidate).resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
