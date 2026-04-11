"""Graph index build for deterministic, corpus-expanded GraphRAG."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.baseline_rag.indexing import BaselineIndexingError, load_chunk_records, write_chunk_metadata
from src.config import ROOT_DIR
from src.graph_rag.schema import (
    GRAPH_SCHEMA_VERSION,
    EntitySpec,
    EntityMention,
    ExtractedRelation,
    SentenceExtractionRecord,
    build_corpus_entity_specs,
    entity_id_for,
    extract_sentence_records,
    load_entity_specs,
    serialize_entity_specs,
)
from src.preprocessing.chunking import ChunkRecord


class GraphChunkRecord(BaseModel):
    """Auditable extraction record for one chunk."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    doc_id: str
    page_number: int = Field(ge=1)
    extracted_entities: list[EntityMention]
    extracted_relations: list[ExtractedRelation]
    sentence_records: list[SentenceExtractionRecord]


class GraphIndexInfo(BaseModel):
    """On-disk description of the saved graph index."""

    model_config = ConfigDict(extra="forbid")

    method: str
    graph_schema_version: str
    index_backend: str
    indexed_chunks: int = Field(ge=0)
    entity_nodes: int = Field(ge=0)
    chunk_nodes: int = Field(ge=0)
    sentence_nodes: int = Field(ge=0)
    generic_entity_nodes: int = Field(ge=0)
    mention_edges: int = Field(ge=0)
    relation_edges: int = Field(ge=0)
    average_entities_per_chunk: float = Field(ge=0.0)
    average_relations_per_chunk: float = Field(ge=0.0)
    source_chunks_path: str
    created_at_utc: str
    graph_path: str
    chunk_metadata_path: str
    chunk_graph_records_path: str
    entity_catalog_path: str
    relation_edges_path: str
    entity_specs_path: str


class GraphIndexBuildSummary(BaseModel):
    """Summary information about a completed graph build."""

    graph_schema_version: str
    indexed_chunks: int = Field(ge=0)
    entity_nodes: int = Field(ge=0)
    sentence_nodes: int = Field(ge=0)
    relation_edges: int = Field(ge=0)
    average_entities_per_chunk: float = Field(ge=0.0)
    average_relations_per_chunk: float = Field(ge=0.0)
    index_dir: str
    source_chunks_path: str
    created_at_utc: str


class EntityCatalogRecord(BaseModel):
    """Aggregated entity node metadata for auditability."""

    model_config = ConfigDict(extra="forbid")

    entity_id: str
    canonical_name: str
    entity_type: str
    source: str
    generic_hint: bool
    is_generic: bool
    mention_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    chunk_ratio: float = Field(ge=0.0)


class RelationEdgeRecord(BaseModel):
    """Aggregated relation edge metadata for auditability."""

    model_config = ConfigDict(extra="forbid")

    source_entity_id: str
    relation_type: str
    target_entity_id: str
    support_count: int = Field(ge=0)
    supporting_chunk_ids: list[str]
    supporting_sentence_ids: list[str]


def build_graph_index(
    chunks_path: Path | str,
    index_dir: Path | str,
    project_root: Path = ROOT_DIR,
) -> GraphIndexBuildSummary:
    """Build and save the deterministic GraphRAG index."""

    root_dir = project_root.resolve()
    resolved_chunks_path = _resolve_path(chunks_path, root_dir)
    resolved_index_dir = _resolve_path(index_dir, root_dir)
    resolved_index_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_chunk_records(resolved_chunks_path)
    if not chunks:
        raise BaselineIndexingError(f"no chunks found in {resolved_chunks_path}")

    entity_specs = build_corpus_entity_specs(chunks)
    spec_by_id = {entity_id_for(spec): spec for spec in entity_specs}

    graph = nx.MultiDiGraph()
    chunk_graph_records: list[GraphChunkRecord] = []
    entity_mentions_by_id: dict[str, set[str]] = defaultdict(set)
    relation_support: dict[tuple[str, str, str], set[tuple[str, str]]] = defaultdict(set)
    mention_edges = 0
    sentence_node_count = 0

    for chunk in chunks:
        chunk_node_id = chunk_node_id_for(chunk.chunk_id)
        graph.add_node(
            chunk_node_id,
            node_type="chunk",
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            title=chunk.title,
            page_number=chunk.page_number,
            source_url=chunk.source_url,
        )

        sentence_records = extract_sentence_records(
            text=chunk.text,
            entity_specs=entity_specs,
        )
        extracted_entities = _aggregate_entities(sentence_records)
        extracted_relations = _aggregate_relations(sentence_records)

        chunk_graph_records.append(
            GraphChunkRecord(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                page_number=chunk.page_number,
                extracted_entities=extracted_entities,
                extracted_relations=extracted_relations,
                sentence_records=sentence_records,
            )
        )

        for sentence_record in sentence_records:
            sentence_node_id = sentence_node_id_for(
                chunk_id=chunk.chunk_id,
                sentence_index=sentence_record.sentence_index,
            )
            graph.add_node(
                sentence_node_id,
                node_type="sentence",
                chunk_id=chunk.chunk_id,
                sentence_index=sentence_record.sentence_index,
                text=sentence_record.text,
            )
            graph.add_edge(
                chunk_node_id,
                sentence_node_id,
                key="HAS_SENTENCE",
                relation_type="HAS_SENTENCE",
                support_count=1,
            )
            sentence_node_count += 1

            for mention in sentence_record.extracted_entities:
                spec = spec_by_id[mention.entity_id]
                graph.add_node(
                    mention.entity_id,
                    node_type="entity",
                    entity_id=mention.entity_id,
                    canonical_name=mention.canonical_name,
                    entity_type=mention.entity_type,
                    source=mention.source,
                    generic_hint=mention.is_generic,
                )
                graph.add_edge(
                    mention.entity_id,
                    sentence_node_id,
                    key="MENTIONED_IN_SENTENCE",
                    relation_type="MENTIONED_IN_SENTENCE",
                    support_count=1,
                )
                entity_mentions_by_id[mention.entity_id].add(chunk.chunk_id)

            for relation in sentence_record.extracted_relations:
                edge_key = relation.relation_type
                existing = graph.get_edge_data(
                    relation.source_entity_id,
                    relation.target_entity_id,
                    key=edge_key,
                    default=None,
                )
                sentence_id = sentence_node_id_for(
                    chunk_id=chunk.chunk_id,
                    sentence_index=relation.sentence_index,
                )
                if existing is None:
                    graph.add_edge(
                        relation.source_entity_id,
                        relation.target_entity_id,
                        key=edge_key,
                        relation_type=relation.relation_type,
                        support_count=1,
                        supporting_chunk_ids=[chunk.chunk_id],
                        supporting_sentence_ids=[sentence_id],
                    )
                else:
                    existing["support_count"] += 1
                    if chunk.chunk_id not in existing["supporting_chunk_ids"]:
                        existing["supporting_chunk_ids"].append(chunk.chunk_id)
                    if sentence_id not in existing["supporting_sentence_ids"]:
                        existing["supporting_sentence_ids"].append(sentence_id)

                relation_support[
                    (
                        relation.source_entity_id,
                        relation.relation_type,
                        relation.target_entity_id,
                    )
                ].add((chunk.chunk_id, sentence_id))

        for mention in extracted_entities:
            graph.add_edge(
                mention.entity_id,
                chunk_node_id,
                key="MENTIONED_IN",
                relation_type="MENTIONED_IN",
                support_count=1,
            )
            mention_edges += 1

    graph_path = resolved_index_dir / "graph.json"
    chunk_metadata_path = resolved_index_dir / "chunk_metadata.jsonl"
    chunk_graph_records_path = resolved_index_dir / "chunk_graph_records.jsonl"
    entity_catalog_path = resolved_index_dir / "entity_catalog.jsonl"
    relation_edges_path = resolved_index_dir / "relation_edges.jsonl"
    entity_specs_path = resolved_index_dir / "entity_specs.jsonl"
    info_path = resolved_index_dir / "index_info.json"

    write_chunk_metadata(chunks=chunks, metadata_path=chunk_metadata_path)
    _write_jsonl(chunk_graph_records, chunk_graph_records_path)
    serialize_entity_specs(entity_specs, entity_specs_path)
    entity_catalog_records = _entity_catalog_records(
        entity_mentions_by_id=entity_mentions_by_id,
        spec_by_id=spec_by_id,
        indexed_chunks=len(chunks),
    )
    _write_jsonl(entity_catalog_records, entity_catalog_path)
    _write_jsonl(_relation_edge_records(relation_support), relation_edges_path)
    graph_path.write_text(
        json.dumps(json_graph.node_link_data(graph, edges="edges"), ensure_ascii=False),
        encoding="utf-8",
    )

    relation_edges = sum(
        1
        for _, _, key in graph.edges(keys=True)
        if key not in {"MENTIONED_IN", "MENTIONED_IN_SENTENCE", "HAS_SENTENCE"}
    )
    average_entities_per_chunk = (
        sum(len(record.extracted_entities) for record in chunk_graph_records) / len(chunk_graph_records)
        if chunk_graph_records
        else 0.0
    )
    average_relations_per_chunk = (
        sum(len(record.extracted_relations) for record in chunk_graph_records) / len(chunk_graph_records)
        if chunk_graph_records
        else 0.0
    )
    generic_entity_nodes = sum(1 for record in entity_catalog_records if record.is_generic)
    info = GraphIndexInfo(
        method="baseline_graph",
        graph_schema_version=GRAPH_SCHEMA_VERSION,
        index_backend="networkx_multidigraph_node_link_sentence_layer",
        indexed_chunks=len(chunks),
        entity_nodes=sum(1 for _, data in graph.nodes(data=True) if data.get("node_type") == "entity"),
        chunk_nodes=sum(1 for _, data in graph.nodes(data=True) if data.get("node_type") == "chunk"),
        sentence_nodes=sentence_node_count,
        generic_entity_nodes=generic_entity_nodes,
        mention_edges=mention_edges,
        relation_edges=relation_edges,
        average_entities_per_chunk=average_entities_per_chunk,
        average_relations_per_chunk=average_relations_per_chunk,
        source_chunks_path=str(resolved_chunks_path),
        created_at_utc=_utc_now(),
        graph_path=str(graph_path),
        chunk_metadata_path=str(chunk_metadata_path),
        chunk_graph_records_path=str(chunk_graph_records_path),
        entity_catalog_path=str(entity_catalog_path),
        relation_edges_path=str(relation_edges_path),
        entity_specs_path=str(entity_specs_path),
    )
    info_path.write_text(
        json.dumps(info.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )

    return GraphIndexBuildSummary(
        graph_schema_version=GRAPH_SCHEMA_VERSION,
        indexed_chunks=len(chunks),
        entity_nodes=info.entity_nodes,
        sentence_nodes=sentence_node_count,
        relation_edges=relation_edges,
        average_entities_per_chunk=average_entities_per_chunk,
        average_relations_per_chunk=average_relations_per_chunk,
        index_dir=str(resolved_index_dir),
        source_chunks_path=str(resolved_chunks_path),
        created_at_utc=info.created_at_utc,
    )


def load_graph_index(
    index_dir: Path | str,
    project_root: Path = ROOT_DIR,
) -> tuple[GraphIndexInfo, nx.MultiDiGraph, list[ChunkRecord], list[GraphChunkRecord], tuple[EntitySpec, ...]]:
    """Load a previously saved graph index."""

    root_dir = project_root.resolve()
    resolved_index_dir = _resolve_path(index_dir, root_dir)
    info_path = resolved_index_dir / "index_info.json"

    if not info_path.exists():
        raise BaselineIndexingError(f"graph index info file does not exist: {info_path}")

    try:
        info_payload = json.loads(info_path.read_text(encoding="utf-8"))
        info = GraphIndexInfo.model_validate(info_payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise BaselineIndexingError(f"invalid graph index info file: {info_path}") from exc

    graph_path = Path(info.graph_path)
    chunk_metadata_path = Path(info.chunk_metadata_path)
    chunk_graph_records_path = Path(info.chunk_graph_records_path)
    entity_specs_path = Path(info.entity_specs_path)

    if not graph_path.exists():
        raise BaselineIndexingError(f"missing saved graph: {graph_path}")
    if not chunk_metadata_path.exists():
        raise BaselineIndexingError(f"missing saved chunk metadata: {chunk_metadata_path}")
    if not chunk_graph_records_path.exists():
        raise BaselineIndexingError(
            f"missing saved chunk graph records: {chunk_graph_records_path}"
        )
    if not entity_specs_path.exists():
        raise BaselineIndexingError(f"missing saved entity specs: {entity_specs_path}")

    graph_payload = json.loads(graph_path.read_text(encoding="utf-8"))
    graph = json_graph.node_link_graph(graph_payload, edges="edges")
    chunks = load_chunk_records(chunk_metadata_path)
    chunk_graph_records = _read_jsonl_models(
        chunk_graph_records_path,
        GraphChunkRecord,
    )
    entity_specs = load_entity_specs(entity_specs_path)
    return info, graph, chunks, chunk_graph_records, entity_specs


def chunk_node_id_for(chunk_id: str) -> str:
    """Create a graph node id for a chunk."""

    return f"chunk::{chunk_id}"


def sentence_node_id_for(chunk_id: str, sentence_index: int) -> str:
    """Create a graph node id for a sentence evidence node."""

    return f"sentence::{chunk_id}::{sentence_index}"


def _aggregate_entities(sentence_records: list[SentenceExtractionRecord]) -> list[EntityMention]:
    mentions_by_id: dict[str, EntityMention] = {}
    for sentence_record in sentence_records:
        for mention in sentence_record.extracted_entities:
            mentions_by_id.setdefault(mention.entity_id, mention)
    return sorted(mentions_by_id.values(), key=lambda mention: mention.entity_id)


def _aggregate_relations(sentence_records: list[SentenceExtractionRecord]) -> list[ExtractedRelation]:
    relations: list[ExtractedRelation] = []
    seen_keys: set[tuple[str, str, str, int]] = set()
    for sentence_record in sentence_records:
        for relation in sentence_record.extracted_relations:
            key = (
                relation.source_entity_id,
                relation.relation_type,
                relation.target_entity_id,
                relation.sentence_index,
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            relations.append(relation)
    return relations


def _entity_catalog_records(
    entity_mentions_by_id: dict[str, set[str]],
    spec_by_id: dict[str, EntitySpec],
    indexed_chunks: int,
) -> list[EntityCatalogRecord]:
    records = []
    for entity_id, chunk_ids in sorted(entity_mentions_by_id.items()):
        spec = spec_by_id.get(entity_id)
        if spec is None:
            continue
        chunk_count = len(chunk_ids)
        chunk_ratio = chunk_count / indexed_chunks if indexed_chunks else 0.0
        is_generic = spec.generic_hint or (
            chunk_ratio >= 0.10 and spec.entity_type in {"organization", "system", "capability"}
        )
        records.append(
            EntityCatalogRecord(
                entity_id=entity_id,
                canonical_name=spec.canonical_name,
                entity_type=spec.entity_type,
                source=spec.source,
                generic_hint=spec.generic_hint,
                is_generic=is_generic,
                mention_count=chunk_count,
                chunk_count=chunk_count,
                chunk_ratio=chunk_ratio,
            )
        )
    return records


def _relation_edge_records(
    relation_support: dict[tuple[str, str, str], set[tuple[str, str]]]
) -> list[RelationEdgeRecord]:
    records = []
    for (source_entity_id, relation_type, target_entity_id), supports in sorted(
        relation_support.items()
    ):
        chunk_ids = sorted({chunk_id for chunk_id, _ in supports})
        sentence_ids = sorted({sentence_id for _, sentence_id in supports})
        records.append(
            RelationEdgeRecord(
                source_entity_id=source_entity_id,
                relation_type=relation_type,
                target_entity_id=target_entity_id,
                support_count=len(supports),
                supporting_chunk_ids=chunk_ids,
                supporting_sentence_ids=sentence_ids,
            )
        )
    return records


def _write_jsonl(records: list[BaseModel], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False))
            handle.write("\n")


def _read_jsonl_models(path: Path, model_type):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
                records.append(model_type.model_validate(payload))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise BaselineIndexingError(
                    f"invalid JSONL record in {path} line {line_number}"
                ) from exc
    return records


def _resolve_path(path_value: Path | str, project_root: Path) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (project_root / candidate).resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
