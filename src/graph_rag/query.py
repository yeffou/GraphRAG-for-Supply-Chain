"""Pure GraphRAG retrieval with path-aware, sentence-grounded scoring."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from src.baseline_rag.query import GenerationResult, RetrievedChunk, maybe_generate_answer
from src.config import ROOT_DIR, LLMConfig
from src.graph_rag.indexing import GraphChunkRecord, load_graph_index
from src.graph_rag.schema import (
    ENTITY_TYPE_WEIGHTS,
    GRAPH_SCHEMA_VERSION,
    RELATION_TYPE_WEIGHTS,
    EntityMention,
    ExtractedRelation,
    SentenceExtractionRecord,
    extract_entities,
    query_terms,
)


RUN_ID_RE = re.compile(r"[^a-zA-Z0-9_-]+")
CAUSAL_QUERY_TERMS = {
    "how",
    "improve",
    "improves",
    "improved",
    "improving",
    "mechanism",
    "mitigate",
    "mitigates",
    "reduce",
    "reduces",
    "strengthen",
    "strengthens",
    "why",
}
TRADEOFF_QUERY_TERMS = {
    "balance",
    "balancing",
    "tradeoff",
    "tradeoffs",
    "trade-off",
    "trade-offs",
}


class GraphEntityContribution(BaseModel):
    """One matched query entity that contributed to chunk ranking."""

    model_config = ConfigDict(extra="forbid")

    entity_id: str
    canonical_name: str
    entity_type: str
    matched_alias: str
    specificity: float
    score_contribution: float
    is_generic: bool


class GraphRelationContribution(BaseModel):
    """One sentence-grounded relation that contributed to ranking."""

    model_config = ConfigDict(extra="forbid")

    relation_type: str
    source_entity_id: str
    source_name: str
    target_entity_id: str
    target_name: str
    trigger: str | None = None
    evidence_text: str
    sentence_index: int = Field(ge=0)
    support_count: int = Field(ge=1)
    direct_query_pair: bool
    query_aligned: bool
    score_contribution: float


class GraphPathContribution(BaseModel):
    """A short query-aligned path that helped rank the chunk."""

    model_config = ConfigDict(extra="forbid")

    source_entity_id: str
    source_name: str
    via_entity_id: str
    via_name: str
    target_entity_id: str
    target_name: str
    relation_chain: list[str]
    evidence_texts: list[str]
    score_contribution: float


class GraphScoreBreakdown(BaseModel):
    """Transparent score components for one retrieved chunk."""

    model_config = ConfigDict(extra="forbid")

    entity_match: float
    coverage_bonus: float
    sentence_alignment: float
    direct_relation: float
    path_bonus: float
    lexical_overlap: float
    generic_penalty: float
    query_miss_penalty: float
    hub_penalty: float
    total: float


class GraphRetrievedChunk(RetrievedChunk):
    """Retrieved chunk with GraphRAG-specific explanations."""

    model_config = ConfigDict(extra="forbid")

    contributing_entities: list[GraphEntityContribution]
    contributing_relations: list[GraphRelationContribution]
    supporting_paths: list[GraphPathContribution]
    score_breakdown: GraphScoreBreakdown


class GraphQueryRunRecord(BaseModel):
    """Saved GraphRAG retrieval run output."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    timestamp_utc: str
    method: str
    embedding_model: str
    query_text: str
    top_k: int = Field(ge=1)
    index_dir: str
    query_entities: list[EntityMention]
    retrieved_chunks: list[GraphRetrievedChunk]
    generation: GenerationResult


class GraphQueryError(Exception):
    """Raised when graph retrieval fails."""


@dataclass(frozen=True)
class QueryIntent:
    preferred_relation_types: tuple[str, ...]
    reward_multi_hop: bool
    causal_focus: bool
    explanation_focus: bool
    tradeoff_focus: bool


def run_graph_query(
    query_text: str,
    index_dir: Path | str,
    output_dir: Path | str,
    top_k: int,
    llm_config: LLMConfig,
    generate_answer: bool,
    project_root: Path = ROOT_DIR,
) -> GraphQueryRunRecord:
    """Run graph-based retrieval and optionally answer generation."""

    if top_k <= 0:
        raise GraphQueryError("top_k must be greater than 0")
    if not query_text.strip():
        raise GraphQueryError("query_text must not be empty")

    root_dir = project_root.resolve()
    resolved_output_dir = _resolve_path(output_dir, root_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        _, graph, chunks, chunk_graph_records, entity_specs = load_graph_index(
            index_dir=index_dir,
            project_root=root_dir,
        )
    except Exception as exc:
        raise GraphQueryError(str(exc)) from exc

    chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
    graph_record_map = {record.chunk_id: record for record in chunk_graph_records}

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

    scored_chunks: list[tuple[float, GraphRetrievedChunk]] = []
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
        if scored is None:
            continue
        scored_chunks.append((scored.score_breakdown.total, scored))

    if not scored_chunks:
        raise GraphQueryError("no graph retrieval candidates were produced")

    ranked_chunks = [
        retrieved_chunk
        for _, retrieved_chunk in sorted(
            scored_chunks,
            key=lambda item: item[0],
            reverse=True,
        )[: min(top_k, len(scored_chunks))]
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
    run_record = GraphQueryRunRecord(
        run_id=build_graph_run_id(query_text=query_text, timestamp_utc=timestamp_utc),
        timestamp_utc=timestamp_utc,
        method="baseline_graph",
        embedding_model=GRAPH_SCHEMA_VERSION,
        query_text=query_text,
        top_k=top_k,
        index_dir=str(_resolve_path(index_dir, root_dir)),
        query_entities=extracted_query_entities,
        retrieved_chunks=ranked_chunks,
        generation=generation,
    )

    output_path = resolved_output_dir / f"{run_record.run_id}.json"
    output_path.write_text(
        json.dumps(run_record.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return run_record


def score_chunk(
    chunk,
    graph_record: GraphChunkRecord,
    graph,
    query_entity_ids: set[str],
    query_entity_types: dict[str, str],
    query_specificity: dict[str, float],
    query_term_set: set[str],
    intent: QueryIntent,
    generic_entity_ids: set[str] | None = None,
) -> GraphRetrievedChunk | None:
    """Score one chunk using sentence-grounded graph evidence and short aligned paths."""

    generic_entity_ids = generic_entity_ids or set()
    mentions_by_id = {
        mention.entity_id: mention for mention in graph_record.extracted_entities
    }
    matched_entity_ids = sorted(query_entity_ids & set(mentions_by_id))
    matched_non_generic_ids = [
        entity_id for entity_id in matched_entity_ids if entity_id not in generic_entity_ids
    ]

    lexical_score = lexical_overlap_score(query_term_set, chunk.text) * 0.7
    if not matched_non_generic_ids and lexical_score < 0.18:
        return None

    contributing_entities = build_entity_contributions(
        mentions_by_id=mentions_by_id,
        matched_entity_ids=matched_entity_ids,
        query_specificity=query_specificity,
        generic_entity_ids=generic_entity_ids,
    )
    entity_match_score = sum(item.score_contribution for item in contributing_entities)
    coverage_bonus = coverage_bonus_for_query(
        query_entity_ids=query_entity_ids,
        matched_entity_ids=set(matched_entity_ids),
        generic_entity_ids=generic_entity_ids,
    )

    sentence_scores = []
    contributing_relations: list[GraphRelationContribution] = []
    for sentence_record in graph_record.sentence_records:
        sentence_scored = score_sentence(
            sentence_record=sentence_record,
            graph=graph,
            query_entity_ids=query_entity_ids,
            query_specificity=query_specificity,
            query_term_set=query_term_set,
            intent=intent,
            generic_entity_ids=generic_entity_ids,
        )
        if sentence_scored is None:
            continue
        sentence_scores.append(sentence_scored["score"])
        contributing_relations.extend(sentence_scored["relations"])

    if not sentence_scores and not matched_non_generic_ids:
        return None

    sentence_alignment_score = sum(sorted(sentence_scores, reverse=True)[:2])
    supporting_paths = score_supporting_paths(
        graph_record=graph_record,
        query_entity_ids=query_entity_ids,
        query_specificity=query_specificity,
        generic_entity_ids=generic_entity_ids,
        intent=intent,
    )
    path_bonus = sum(path.score_contribution for path in supporting_paths)
    direct_relation_score = sum(item.score_contribution for item in contributing_relations)

    high_signal_query_ids = {
        entity_id
        for entity_id, entity_type in query_entity_types.items()
        if entity_type in {"strategy", "policy", "capability", "risk", "sector", "actor"}
        and entity_id not in generic_entity_ids
    }
    missing_high_signal = high_signal_query_ids - set(matched_non_generic_ids)
    query_miss_penalty = 2.4 * len(missing_high_signal)
    if high_signal_query_ids and not contributing_relations and not supporting_paths:
        query_miss_penalty += 2.0

    generic_penalty = 0.0
    if matched_entity_ids and not matched_non_generic_ids:
        generic_penalty += 3.4
    generic_penalty += 0.25 * sum(
        1 for relation in contributing_relations if relation.direct_query_pair is False
    )
    hub_penalty = 0.10 * max(0, len(graph_record.extracted_entities) - 10)

    total_score = (
        entity_match_score
        + coverage_bonus
        + sentence_alignment_score
        + direct_relation_score
        + path_bonus
        + lexical_score
        - generic_penalty
        - query_miss_penalty
        - hub_penalty
    )
    if total_score <= 0:
        return None

    contributing_entities.sort(key=lambda item: item.score_contribution, reverse=True)
    contributing_relations.sort(key=lambda item: item.score_contribution, reverse=True)
    supporting_paths.sort(key=lambda item: item.score_contribution, reverse=True)

    return GraphRetrievedChunk(
        rank=1,
        score=float(total_score),
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        title=chunk.title,
        publisher=chunk.publisher,
        page_number=chunk.page_number,
        source_url=chunk.source_url,
        local_path=chunk.local_path,
        year=chunk.year,
        text=chunk.text,
        char_count=chunk.char_count,
        contributing_entities=contributing_entities[:6],
        contributing_relations=contributing_relations[:6],
        supporting_paths=supporting_paths[:4],
        score_breakdown=GraphScoreBreakdown(
            entity_match=round(entity_match_score, 6),
            coverage_bonus=round(coverage_bonus, 6),
            sentence_alignment=round(sentence_alignment_score, 6),
            direct_relation=round(direct_relation_score, 6),
            path_bonus=round(path_bonus, 6),
            lexical_overlap=round(lexical_score, 6),
            generic_penalty=round(generic_penalty, 6),
            query_miss_penalty=round(query_miss_penalty, 6),
            hub_penalty=round(hub_penalty, 6),
            total=round(total_score, 6),
        ),
    )


def build_entity_contributions(
    mentions_by_id: dict[str, EntityMention],
    matched_entity_ids: list[str],
    query_specificity: dict[str, float],
    generic_entity_ids: set[str],
) -> list[GraphEntityContribution]:
    contributions: list[GraphEntityContribution] = []
    for entity_id in matched_entity_ids:
        mention = mentions_by_id[entity_id]
        specificity = query_specificity.get(entity_id, 1.0)
        contribution = 4.0 * specificity
        if entity_id in generic_entity_ids:
            contribution *= 0.45
        contributions.append(
            GraphEntityContribution(
                entity_id=mention.entity_id,
                canonical_name=mention.canonical_name,
                entity_type=mention.entity_type,
                matched_alias=mention.matched_alias,
                specificity=specificity,
                score_contribution=round(contribution, 6),
                is_generic=entity_id in generic_entity_ids,
            )
        )
    return contributions


def coverage_bonus_for_query(
    query_entity_ids: set[str],
    matched_entity_ids: set[str],
    generic_entity_ids: set[str],
) -> float:
    if not query_entity_ids or not matched_entity_ids:
        return 0.0
    query_non_generic = query_entity_ids - generic_entity_ids
    matched_non_generic = matched_entity_ids - generic_entity_ids
    if query_non_generic:
        coverage_ratio = len(matched_non_generic) / len(query_non_generic)
        return 4.5 * coverage_ratio * max(len(matched_non_generic), 1)
    coverage_ratio = len(matched_entity_ids) / len(query_entity_ids)
    return 1.5 * coverage_ratio


def score_sentence(
    sentence_record: SentenceExtractionRecord,
    graph,
    query_entity_ids: set[str],
    query_specificity: dict[str, float],
    query_term_set: set[str],
    intent: QueryIntent,
    generic_entity_ids: set[str],
) -> dict | None:
    """Score one sentence as evidence for the current query."""

    sentence_entity_ids = {
        mention.entity_id for mention in sentence_record.extracted_entities
    }
    matched = query_entity_ids & sentence_entity_ids
    matched_non_generic = matched - generic_entity_ids
    lexical = lexical_overlap_score(query_term_set, sentence_record.text) * 0.5
    if not matched and lexical == 0.0:
        return None

    relation_contributions: list[GraphRelationContribution] = []
    relation_score = 0.0
    for relation in sentence_record.extracted_relations:
        relation_contribution = score_relation(
            relation=relation,
            graph=graph,
            query_entity_ids=query_entity_ids,
            query_specificity=query_specificity,
            intent=intent,
            generic_entity_ids=generic_entity_ids,
        )
        if relation_contribution is None:
            continue
        relation_contributions.append(relation_contribution)
        relation_score += relation_contribution.score_contribution

    if not matched_non_generic and relation_score == 0.0 and lexical < 0.20:
        return None

    entity_alignment = sum(
        1.8 * query_specificity.get(entity_id, 1.0)
        for entity_id in matched_non_generic
    )
    generic_only_bonus = 0.0
    if not matched_non_generic and matched:
        generic_only_bonus = 0.35 * len(matched)

    total = entity_alignment + generic_only_bonus + relation_score + lexical
    return {
        "score": total,
        "relations": relation_contributions,
    }


def score_relation(
    relation: ExtractedRelation,
    graph,
    query_entity_ids: set[str],
    query_specificity: dict[str, float],
    intent: QueryIntent,
    generic_entity_ids: set[str],
) -> GraphRelationContribution | None:
    """Score one sentence-grounded relation against the query intent."""

    source_is_query = relation.source_entity_id in query_entity_ids
    target_is_query = relation.target_entity_id in query_entity_ids
    if not source_is_query and not target_is_query:
        return None

    direct_query_pair = source_is_query and target_is_query
    non_generic_query_touch = (
        (source_is_query and relation.source_entity_id not in generic_entity_ids)
        or (target_is_query and relation.target_entity_id not in generic_entity_ids)
    )
    if not direct_query_pair and not non_generic_query_touch:
        return None

    relation_type = relation.relation_type
    relation_weight = RELATION_TYPE_WEIGHTS.get(relation_type, 0.4)
    query_aligned = relation_type in intent.preferred_relation_types
    support_count = relation_support_count(graph, relation)

    specificity_total = 0.0
    if source_is_query:
        specificity_total += query_specificity.get(relation.source_entity_id, 1.0)
    if target_is_query:
        specificity_total += query_specificity.get(relation.target_entity_id, 1.0)

    rarity_bonus = 1.2 / max(math.log1p(support_count), 1.0)
    score = relation_weight
    score += 1.35 * specificity_total
    score += 2.6 if direct_query_pair else 0.8
    score += 1.6 if query_aligned else 0.0
    score += rarity_bonus

    if relation_type in {"STRENGTHENS", "IMPROVES", "MITIGATES", "ENABLES"} and intent.causal_focus:
        score += 1.1
    if relation_type in {"EXPOSES", "CONSTRAINS"} and intent.explanation_focus:
        score += 0.9
    if relation_type == "TRADE_OFF_WITH":
        score += 1.0
    if intent.tradeoff_focus:
        if relation_type == "TRADE_OFF_WITH":
            score += 2.2
        elif relation_type in {"STRENGTHENS", "IMPROVES"}:
            score -= 2.4

    return GraphRelationContribution(
        relation_type=relation_type,
        source_entity_id=relation.source_entity_id,
        source_name=relation.source_name,
        target_entity_id=relation.target_entity_id,
        target_name=relation.target_name,
        trigger=relation.trigger,
        evidence_text=relation.evidence_text,
        sentence_index=relation.sentence_index,
        support_count=support_count,
        direct_query_pair=direct_query_pair,
        query_aligned=query_aligned,
        score_contribution=round(score, 6),
    )


def score_supporting_paths(
    graph_record: GraphChunkRecord,
    query_entity_ids: set[str],
    query_specificity: dict[str, float],
    generic_entity_ids: set[str],
    intent: QueryIntent,
) -> list[GraphPathContribution]:
    """Reward short aligned paths between non-generic query entities within a chunk."""

    if not intent.reward_multi_hop:
        return []

    relations = [
        relation
        for relation in graph_record.extracted_relations
        if relation.relation_type != "CO_OCCURS_WITH"
    ]
    adjacency: dict[str, list[tuple[str, ExtractedRelation]]] = defaultdict(list)
    for relation in relations:
        adjacency[relation.source_entity_id].append((relation.target_entity_id, relation))
        adjacency[relation.target_entity_id].append((relation.source_entity_id, relation))

    candidate_query_ids = sorted(query_entity_ids - generic_entity_ids)
    if len(candidate_query_ids) < 2:
        return []

    entity_name_map = {
        mention.entity_id: mention.canonical_name
        for mention in graph_record.extracted_entities
    }
    path_contributions: list[GraphPathContribution] = []
    seen_paths: set[tuple[str, str, str]] = set()

    for index, left_id in enumerate(candidate_query_ids):
        for right_id in candidate_query_ids[index + 1 :]:
            for via_id, left_relation in adjacency.get(left_id, []):
                if via_id in {left_id, right_id} or via_id in generic_entity_ids:
                    continue
                for maybe_right_id, right_relation in adjacency.get(via_id, []):
                    if maybe_right_id != right_id:
                        continue
                    relation_chain = [left_relation.relation_type, right_relation.relation_type]
                    aligned = sum(
                        1
                        for relation_type in relation_chain
                        if relation_type in intent.preferred_relation_types
                    )
                    if aligned == 0:
                        continue
                    path_key = tuple(sorted((left_id, via_id, right_id)))
                    if path_key in seen_paths:
                        continue
                    seen_paths.add(path_key)
                    specificity = (
                        query_specificity.get(left_id, 1.0)
                        + query_specificity.get(right_id, 1.0)
                    )
                    score = 1.4 * specificity + 1.1 * aligned
                    path_contributions.append(
                        GraphPathContribution(
                            source_entity_id=left_id,
                            source_name=entity_name_map.get(left_id, left_id),
                            via_entity_id=via_id,
                            via_name=entity_name_map.get(via_id, via_id),
                            target_entity_id=right_id,
                            target_name=entity_name_map.get(right_id, right_id),
                            relation_chain=relation_chain,
                            evidence_texts=[
                                left_relation.evidence_text,
                                right_relation.evidence_text,
                            ],
                            score_contribution=round(score, 6),
                        )
                    )
    return path_contributions


def detect_query_intent(query_text: str) -> QueryIntent:
    """Map question wording to relation preferences."""

    lowered = query_text.lower()
    relation_types: list[str] = []
    tradeoff_focus = any(term in lowered for term in TRADEOFF_QUERY_TERMS)
    causal_focus = any(term in lowered for term in CAUSAL_QUERY_TERMS)
    explanation_focus = lowered.startswith(("how", "why")) or "what role" in lowered or "what challenge" in lowered

    if tradeoff_focus:
        relation_types.extend(["TRADE_OFF_WITH", "CONSTRAINS", "EXPOSES"])

    if causal_focus and not tradeoff_focus:
        relation_types.extend(["STRENGTHENS", "IMPROVES", "MITIGATES", "ENABLES"])

    if any(term in lowered for term in {"disruption", "risk", "shock", "vulnerability"}):
        relation_types.extend(["MITIGATES", "AFFECTS", "EXPOSES", "CONSTRAINS"])

    if explanation_focus and not tradeoff_focus:
        relation_types.extend(["STRENGTHENS", "ENABLES", "EXPOSES", "CONSTRAINS", "TRADE_OFF_WITH"])

    if not relation_types:
        relation_types.extend(["STRENGTHENS", "ENABLES", "MITIGATES", "EXPOSES", "AFFECTS"])

    deduped = []
    for relation_type in relation_types:
        if relation_type not in deduped:
            deduped.append(relation_type)
    return QueryIntent(
        preferred_relation_types=tuple(deduped),
        reward_multi_hop=True,
        causal_focus=causal_focus,
        explanation_focus=explanation_focus,
        tradeoff_focus=tradeoff_focus,
    )


def lexical_overlap_score(query_term_set: set[str], text: str) -> float:
    """Compute a small lexical tie-break score."""

    if not query_term_set:
        return 0.0
    lowered = text.lower()
    overlap = sum(1 for term in query_term_set if term in lowered)
    return overlap / max(len(query_term_set), 1)


def query_entity_specificity(entity_type: str, mention_count: int, is_generic: bool) -> float:
    """Compute inverse-mention-frequency specificity for a query entity."""

    base_weight = ENTITY_TYPE_WEIGHTS.get(entity_type, 1.0)
    specificity = base_weight / max(_log_scale(mention_count), 1.0)
    if is_generic:
        specificity *= 0.40
    return specificity


def relation_support_count(graph, relation: ExtractedRelation) -> int:
    """Return the global support count for a saved relation."""

    direct_edge = graph.get_edge_data(
        relation.source_entity_id,
        relation.target_entity_id,
        key=relation.relation_type,
        default=None,
    )
    if direct_edge is not None:
        return int(direct_edge.get("support_count", 1))

    reverse_edge = graph.get_edge_data(
        relation.target_entity_id,
        relation.source_entity_id,
        key=relation.relation_type,
        default=None,
    )
    if reverse_edge is not None:
        return int(reverse_edge.get("support_count", 1))
    return 1


def build_generic_entity_id_set(
    graph,
    total_chunks: int,
    query_entities: list[EntityMention],
) -> set[str]:
    """Identify generic hub-like entities that should be downweighted."""

    generic_ids = {
        mention.entity_id for mention in query_entities if mention.is_generic
    }
    for mention in query_entities:
        mention_count = _entity_chunk_mentions(graph, mention.entity_id)
        chunk_ratio = mention_count / total_chunks if total_chunks else 0.0
        if chunk_ratio >= 0.10 and mention.entity_type in {"organization", "system", "capability"}:
            generic_ids.add(mention.entity_id)
    return generic_ids


def _entity_chunk_mentions(graph, entity_id: str) -> int:
    return sum(
        1
        for _, _, key in graph.out_edges(entity_id, keys=True)
        if key == "MENTIONED_IN"
    )


def _log_scale(value: float) -> float:
    return math.log1p(max(value, 1.0))


def build_graph_run_id(query_text: str, timestamp_utc: str) -> str:
    """Create a filesystem-safe run identifier for graph retrieval."""

    compact_timestamp = (
        timestamp_utc.replace("-", "")
        .replace(":", "")
        .replace("+00:00", "Z")
        .replace(".", "")
    )
    query_stub = RUN_ID_RE.sub("_", query_text.lower()).strip("_")[:50] or "query"
    return f"baseline_graph_{compact_timestamp}_{query_stub}"


def _resolve_path(path_value: Path | str, project_root: Path) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (project_root / candidate).resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
