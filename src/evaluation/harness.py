"""Deterministic dual-track evaluation across retrieval baselines."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from src.baseline_rag.dense_query import run_dense_query
from src.baseline_rag.query import run_baseline_query
from src.config import ROOT_DIR, ProjectConfig
from src.evaluation.schema import (
    ChunkAssessment,
    EvaluationQuestion,
    EvaluationSummary,
    EvaluationTrack,
    MethodAggregateSummary,
    MethodQuestionEvaluation,
    PerQuestionComparison,
)
from src.graph_rag.hybrid_query import run_hybrid_graph_query
from src.graph_rag.indexing import GraphChunkRecord, load_graph_index
from src.graph_rag.query import detect_query_intent, run_graph_query
from src.graph_rag.schema import EntitySpec, extract_entities


SCORING_VERSION = "retrieval_eval_v2_dual_track"
RANK_WEIGHTS = (1.0, 0.7, 0.5, 0.35, 0.25)
WIN_EPSILON = 0.02
NORMALIZE_RE = re.compile(r"[^a-z0-9]+")

EXPLANATION_MARKERS = {
    "because",
    "cause",
    "causes",
    "causal",
    "due",
    "effect",
    "effects",
    "enable",
    "enables",
    "explain",
    "explains",
    "help",
    "helps",
    "how",
    "improve",
    "improves",
    "lead",
    "leads",
    "link",
    "linked",
    "mechanism",
    "mitigate",
    "mitigates",
    "role",
    "through",
    "trade-off",
    "trade-offs",
    "tradeoffs",
    "via",
    "why",
}

RELATIONALLY_RICH_TYPES = {
    "STRENGTHENS",
    "IMPROVES",
    "MITIGATES",
    "ENABLES",
    "TRADE_OFF_WITH",
    "EXPOSES",
    "CONSTRAINS",
    "AFFECTS",
}


@dataclass(frozen=True)
class GraphEvaluationSupport:
    """Graph metadata used to score retrieved chunks from any method."""

    chunk_graph_records: dict[str, GraphChunkRecord]
    entity_specs: tuple[EntitySpec, ...]


@dataclass(frozen=True)
class QuestionProfile:
    """Derived question metadata for fair deterministic scoring."""

    evaluation_track: EvaluationTrack
    is_explanation_oriented: bool
    is_cross_document: bool
    query_entity_ids: frozenset[str]
    non_generic_query_entity_ids: frozenset[str]
    normalized_key_concepts: tuple[str, ...]
    normalized_evidence_phrases: tuple[str, ...]
    preferred_relation_types: frozenset[str]


def load_questions(questions_path: Path | str) -> list[EvaluationQuestion]:
    """Load and validate the annotated evaluation set."""

    path = Path(questions_path)
    if not path.exists():
        raise FileNotFoundError(f"questions file does not exist: {path}")

    questions: list[EvaluationQuestion] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
                question = EvaluationQuestion.model_validate(payload)
            except (json.JSONDecodeError, ValidationError) as exc:
                raise ValueError(
                    f"invalid evaluation question at line {line_number} in {path}"
                ) from exc

            if question.question_id in seen_ids:
                raise ValueError(f"duplicate question_id in evaluation set: {question.question_id}")

            seen_ids.add(question.question_id)
            questions.append(question)

    if len(questions) < 20:
        raise ValueError("evaluation set must contain at least 20 questions")

    return questions


def run_retrieval_evaluation(
    questions_path: Path | str,
    output_root: Path | str,
    config: ProjectConfig,
    top_k: int = 3,
    include_hybrid: bool = False,
) -> tuple[EvaluationSummary, list[PerQuestionComparison]]:
    """Run all retrieval methods over the evaluation set and save artifacts."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    questions = load_questions(questions_path)
    graph_support = load_graph_evaluation_support(
        index_dir=config.paths.indexes_dir / "baseline_graph",
        project_root=config.paths.root_dir,
    )
    question_profiles = {
        question.question_id: build_question_profile(question, graph_support.entity_specs)
        for question in questions
    }

    evaluation_id = build_evaluation_id()
    output_dir = _resolve_path(output_root, config.paths.root_dir) / evaluation_id
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_dir = output_dir / "runs"
    tfidf_runs_dir = runs_dir / "tfidf"
    dense_runs_dir = runs_dir / "dense"
    graph_runs_dir = runs_dir / "graph"
    hybrid_runs_dir = runs_dir / "hybrid_graph"
    tfidf_runs_dir.mkdir(parents=True, exist_ok=True)
    dense_runs_dir.mkdir(parents=True, exist_ok=True)
    graph_runs_dir.mkdir(parents=True, exist_ok=True)
    if include_hybrid:
        hybrid_runs_dir.mkdir(parents=True, exist_ok=True)

    method_specs = [
        {
            "label": "tfidf",
            "runner": run_baseline_query,
            "index_dir": config.paths.indexes_dir / "baseline_tfidf",
            "output_dir": tfidf_runs_dir,
            "config_value": str(config.paths.indexes_dir / "baseline_tfidf"),
        },
        {
            "label": "dense",
            "runner": run_dense_query,
            "index_dir": config.paths.indexes_dir / "baseline_dense",
            "output_dir": dense_runs_dir,
            "config_value": str(config.paths.indexes_dir / "baseline_dense"),
        },
        {
            "label": "graph",
            "runner": run_graph_query,
            "index_dir": config.paths.indexes_dir / "baseline_graph",
            "output_dir": graph_runs_dir,
            "config_value": str(config.paths.indexes_dir / "baseline_graph"),
        },
    ]
    if include_hybrid:
        method_specs.append(
            {
                "label": "hybrid_graph",
                "runner": run_hybrid_graph_query,
                "output_dir": hybrid_runs_dir,
                "runner_kwargs": {
                    "dense_index_dir": config.paths.indexes_dir / "baseline_dense",
                    "graph_index_dir": config.paths.indexes_dir / "baseline_graph",
                },
                "config_value": {
                    "dense_index_dir": str(config.paths.indexes_dir / "baseline_dense"),
                    "graph_index_dir": str(config.paths.indexes_dir / "baseline_graph"),
                },
            }
        )

    per_question_comparisons: list[PerQuestionComparison] = []
    method_results_map: dict[str, list[MethodQuestionEvaluation]] = defaultdict(list)

    for question in questions:
        question_results: list[MethodQuestionEvaluation] = []
        profile = question_profiles[question.question_id]
        for method_spec in method_specs:
            runner_kwargs = dict(method_spec.get("runner_kwargs", {}))
            if "index_dir" in method_spec:
                runner_kwargs["index_dir"] = method_spec["index_dir"]

            run_record = method_spec["runner"](
                query_text=question.question,
                output_dir=method_spec["output_dir"],
                top_k=top_k,
                llm_config=config.llm,
                generate_answer=False,
                project_root=config.paths.root_dir,
                **runner_kwargs,
            )
            run_path = method_spec["output_dir"] / f"{run_record.run_id}.json"
            method_evaluation = evaluate_method_result(
                question=question,
                profile=profile,
                graph_support=graph_support,
                method_label=method_spec["label"],
                run_record=run_record,
                run_path=run_path,
                top_k=top_k,
            )
            question_results.append(method_evaluation)
            method_results_map[method_spec["label"]].append(method_evaluation)

        winner_methods = determine_winners(question_results)
        per_question_comparisons.append(
            PerQuestionComparison(
                question_id=question.question_id,
                category=question.category,
                evaluation_track=profile.evaluation_track,
                is_explanation_oriented=profile.is_explanation_oriented,
                question=question.question,
                winner_methods=winner_methods,
                method_results=question_results,
            )
        )

    summary = EvaluationSummary(
        evaluation_id=evaluation_id,
        timestamp_utc=_utc_now(),
        question_count=len(questions),
        top_k=top_k,
        scoring_version=SCORING_VERSION,
        questions_path=str(Path(questions_path).resolve()),
        output_dir=str(output_dir),
        category_distribution=dict(sorted(Counter(q.category for q in questions).items())),
        evaluation_track_distribution=dict(
            sorted(Counter(question_profiles[q.question_id].evaluation_track for q in questions).items())
        ),
        explanation_oriented_question_count=sum(
            1 for question in questions if question_profiles[question.question_id].is_explanation_oriented
        ),
        method_summaries=build_method_summaries(
            method_results_map=method_results_map,
            per_question_comparisons=per_question_comparisons,
        ),
    )

    write_evaluation_artifacts(
        output_dir=output_dir,
        questions=questions,
        question_profiles=question_profiles,
        per_question_comparisons=per_question_comparisons,
        summary=summary,
        top_k=top_k,
        method_specs=method_specs,
    )

    return summary, per_question_comparisons


def load_graph_evaluation_support(
    index_dir: Path | str,
    project_root: Path,
) -> GraphEvaluationSupport:
    """Load graph records once so all methods can be scored against the same structure."""

    _, _, _, chunk_graph_records, entity_specs = load_graph_index(
        index_dir=index_dir,
        project_root=project_root,
    )
    return GraphEvaluationSupport(
        chunk_graph_records={record.chunk_id: record for record in chunk_graph_records},
        entity_specs=entity_specs,
    )


def build_question_profile(
    question: EvaluationQuestion,
    entity_specs: tuple[EntitySpec, ...],
) -> QuestionProfile:
    """Build a scoring profile for one question."""

    combined_text = " ".join(
        [question.question, *question.key_concepts, *question.evidence_phrases]
    )
    query_entities = extract_entities(
        combined_text,
        domain_tags=None,
        allow_domain_tag_backfill=False,
        entity_specs=entity_specs,
    )
    evaluation_track = determine_evaluation_track(question)
    explanation_oriented = is_explanation_oriented(question)
    intent = detect_query_intent(question.question)

    return QuestionProfile(
        evaluation_track=evaluation_track,
        is_explanation_oriented=explanation_oriented,
        is_cross_document=len(question.gold_doc_ids) > 1,
        query_entity_ids=frozenset(mention.entity_id for mention in query_entities),
        non_generic_query_entity_ids=frozenset(
            mention.entity_id for mention in query_entities if not mention.is_generic
        ),
        normalized_key_concepts=tuple(normalize_text(item) for item in question.key_concepts if item),
        normalized_evidence_phrases=tuple(
            normalize_text(item) for item in question.evidence_phrases if item
        ),
        preferred_relation_types=frozenset(intent.preferred_relation_types),
    )


def determine_evaluation_track(question: EvaluationQuestion) -> EvaluationTrack:
    """Split the question set into exact-retrieval and graph-stressing tracks."""

    text = question.question.lower().strip()
    if question.category == "direct_factual":
        return "exact_retrieval"
    if question.category == "mitigation" and text.startswith("what strategies"):
        return "exact_retrieval"
    return "graph_stressing"


def is_explanation_oriented(question: EvaluationQuestion) -> bool:
    """Flag questions that ask for mechanisms, roles, or trade-off explanations."""

    text = question.question.lower()
    if question.category in {"causal", "trade_off"}:
        return True
    return any(
        marker in text
        for marker in (
            "how ",
            "how can",
            "how does",
            "why ",
            "why can",
            "what role",
            "what challenge",
            "why do they matter",
            "what trade-off",
            "what trade off",
        )
    )


def evaluate_method_result(
    question: EvaluationQuestion,
    profile: QuestionProfile,
    graph_support: GraphEvaluationSupport,
    method_label: str,
    run_record,
    run_path: Path,
    top_k: int,
) -> MethodQuestionEvaluation:
    """Score one method's retrieval output against annotated evidence."""

    chunk_assessments = [
        assess_chunk(
            question=question,
            profile=profile,
            chunk=chunk,
            graph_support=graph_support,
        )
        for chunk in run_record.retrieved_chunks[:top_k]
    ]

    if profile.evaluation_track == "exact_retrieval":
        relevance = aggregate_chunk_metric(
            [item.relevance_score for item in chunk_assessments]
        )
        directness = aggregate_chunk_metric(
            [item.directness_score for item in chunk_assessments]
        )
        groundedness = aggregate_chunk_metric(
            [item.groundedness_score for item in chunk_assessments]
        )
        correctness = aggregate_chunk_metric(
            [item.correctness_score for item in chunk_assessments]
        )
        collective_concept_score = collective_concept_coverage(
            question=question,
            chunk_assessments=chunk_assessments,
        )
        collective_relation_score = collective_relation_support(chunk_assessments)
        collective_reasoning_score = collective_reasoning_support(
            question=question,
            profile=profile,
            chunk_assessments=chunk_assessments,
            collective_concept=collective_concept_score,
            collective_relation=collective_relation_score,
        )
    else:
        base_relevance = aggregate_chunk_metric(
            [item.relevance_score for item in chunk_assessments]
        )
        base_directness = aggregate_chunk_metric(
            [item.directness_score for item in chunk_assessments]
        )
        base_groundedness = aggregate_chunk_metric(
            [item.groundedness_score for item in chunk_assessments]
        )
        base_correctness = aggregate_chunk_metric(
            [item.correctness_score for item in chunk_assessments]
        )
        collective_concept_score = collective_concept_coverage(
            question=question,
            chunk_assessments=chunk_assessments,
        )
        collective_relation_score = collective_relation_support(chunk_assessments)
        collective_reasoning_score = collective_reasoning_support(
            question=question,
            profile=profile,
            chunk_assessments=chunk_assessments,
            collective_concept=collective_concept_score,
            collective_relation=collective_relation_score,
        )

        max_doc = max((item.doc_match for item in chunk_assessments), default=0.0)
        max_chunk = max((item.chunk_match for item in chunk_assessments), default=0.0)
        evidence_signal = aggregate_chunk_metric(
            [item.evidence_match for item in chunk_assessments]
        )

        relevance = clamp_score(
            0.50 * base_relevance
            + 0.20 * collective_concept_score
            + 0.20 * collective_relation_score
            + 0.10 * max_doc
        )
        directness = clamp_score(
            0.40 * base_directness
            + 0.25 * collective_relation_score
            + 0.25 * collective_reasoning_score
            + 0.10 * max_chunk
        )
        groundedness = clamp_score(
            0.50 * base_groundedness
            + 0.20 * collective_concept_score
            + 0.15 * collective_relation_score
            + 0.15 * evidence_signal
        )
        correctness = clamp_score(
            0.45 * base_correctness
            + 0.20 * collective_concept_score
            + 0.20 * collective_relation_score
            + 0.15 * collective_reasoning_score
        )

    overall = round(
        (relevance + directness + groundedness + correctness) / 4.0,
        6,
    )

    return MethodQuestionEvaluation(
        question_id=question.question_id,
        category=question.category,
        evaluation_track=profile.evaluation_track,
        is_explanation_oriented=profile.is_explanation_oriented,
        question=question.question,
        method_label=method_label,
        retrieval_method=run_record.method,
        embedding_model=run_record.embedding_model,
        index_dir=run_record.index_dir,
        run_path=str(run_path),
        top_k=top_k,
        relevance=round(relevance, 6),
        directness=round(directness, 6),
        groundedness=round(groundedness, 6),
        correctness_of_evidence=round(correctness, 6),
        overall=overall,
        collective_concept_coverage=round(collective_concept_score, 6),
        collective_relation_support=round(collective_relation_score, 6),
        collective_reasoning_support=round(collective_reasoning_score, 6),
        top1_doc_match=chunk_assessments[0].doc_match if chunk_assessments else 0.0,
        any_gold_chunk_in_top_k=max(
            (item.chunk_match for item in chunk_assessments),
            default=0.0,
        ),
        failure_reason=detect_failure_reason(
            evaluation_track=profile.evaluation_track,
            chunk_assessments=chunk_assessments,
            relevance=relevance,
            directness=directness,
            groundedness=groundedness,
            collective_concept=collective_concept_score,
            collective_relation=collective_relation_score,
        ),
        retrieved_chunks=chunk_assessments,
    )


def assess_chunk(
    question: EvaluationQuestion,
    profile: QuestionProfile,
    chunk,
    graph_support: GraphEvaluationSupport,
) -> ChunkAssessment:
    """Assess one retrieved chunk against the question annotation."""

    normalized_text = normalize_text(chunk.text)
    doc_match = 1.0 if chunk.doc_id in question.gold_doc_ids else 0.0
    chunk_match = 1.0 if chunk.chunk_id in question.gold_chunk_ids else 0.0
    matched_concepts = matched_targets(
        normalized_text=normalized_text,
        targets=profile.normalized_key_concepts,
        original_targets=question.key_concepts,
    )
    matched_evidence = matched_targets(
        normalized_text=normalized_text,
        targets=profile.normalized_evidence_phrases,
        original_targets=question.evidence_phrases,
    )
    concept_match = len(matched_concepts) / len(profile.normalized_key_concepts) if profile.normalized_key_concepts else 1.0
    evidence_match = len(matched_evidence) / len(profile.normalized_evidence_phrases) if profile.normalized_evidence_phrases else 1.0

    graph_record = graph_support.chunk_graph_records.get(chunk.chunk_id)
    graph_concept_hits, graph_entity_hits = graph_concept_hits_for_chunk(
        graph_record=graph_record,
        normalized_key_concepts=profile.normalized_key_concepts,
        original_key_concepts=question.key_concepts,
    )
    graph_concept_support = (
        len(graph_concept_hits) / len(profile.normalized_key_concepts)
        if profile.normalized_key_concepts
        else 1.0
    )
    relation_support, matched_relation_types = relation_support_for_chunk(
        graph_record=graph_record,
        profile=profile,
        normalized_key_concepts=profile.normalized_key_concepts,
    )
    path_support, matched_path_relation_chains = path_support_for_chunk(
        graph_record=graph_record,
        profile=profile,
    )
    explanation_relevance = explanation_relevance_for_chunk(
        question=question,
        profile=profile,
        normalized_text=normalized_text,
        concept_match=concept_match,
        evidence_match=evidence_match,
        graph_concept_support=graph_concept_support,
        relation_support=relation_support,
        path_support=path_support,
    )

    if profile.evaluation_track == "exact_retrieval":
        relevance_score = clamp_score(
            0.40 * doc_match
            + 0.25 * chunk_match
            + 0.20 * concept_match
            + 0.15 * evidence_match
        )
        directness_score = clamp_score(
            0.50 * chunk_match
            + 0.20 * doc_match
            + 0.15 * concept_match
            + 0.15 * evidence_match
        )
        groundedness_score = clamp_score(
            0.35 * doc_match
            + 0.25 * chunk_match
            + 0.40 * evidence_match
        )
        correctness_score = clamp_score(
            0.30 * doc_match
            + 0.30 * chunk_match
            + 0.20 * concept_match
            + 0.20 * evidence_match
        )
    else:
        concept_support = max(concept_match, graph_concept_support)
        relation_or_path = max(relation_support, path_support)
        relevance_score = clamp_score(
            0.15 * doc_match
            + 0.10 * chunk_match
            + 0.25 * concept_support
            + 0.25 * relation_or_path
            + 0.15 * explanation_relevance
            + 0.10 * evidence_match
        )
        directness_score = clamp_score(
            0.10 * doc_match
            + 0.10 * chunk_match
            + 0.20 * concept_support
            + 0.30 * relation_or_path
            + 0.20 * explanation_relevance
            + 0.10 * evidence_match
        )
        groundedness_score = clamp_score(
            0.10 * doc_match
            + 0.10 * chunk_match
            + 0.25 * max(evidence_match, concept_match)
            + 0.20 * graph_concept_support
            + 0.20 * relation_support
            + 0.15 * path_support
        )
        correctness_score = clamp_score(
            0.10 * doc_match
            + 0.10 * chunk_match
            + 0.25 * concept_support
            + 0.20 * evidence_match
            + 0.20 * relation_support
            + 0.15 * path_support
        )

    return ChunkAssessment(
        rank=chunk.rank,
        score=float(chunk.score),
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        title=chunk.title,
        page_number=chunk.page_number,
        source_url=chunk.source_url,
        preview=chunk.text[:220].replace("\n", " "),
        evaluation_track=profile.evaluation_track,
        doc_match=doc_match,
        chunk_match=chunk_match,
        concept_match=round(concept_match, 6),
        evidence_match=round(evidence_match, 6),
        graph_concept_support=round(graph_concept_support, 6),
        relation_support=round(relation_support, 6),
        path_support=round(path_support, 6),
        explanation_relevance=round(explanation_relevance, 6),
        relevance_score=round(relevance_score, 6),
        directness_score=round(directness_score, 6),
        groundedness_score=round(groundedness_score, 6),
        correctness_score=round(correctness_score, 6),
        matched_concepts=matched_concepts,
        matched_evidence_phrases=matched_evidence,
        matched_graph_concepts=graph_concept_hits,
        matched_graph_entities=graph_entity_hits,
        matched_relation_types=matched_relation_types,
        matched_path_relation_chains=matched_path_relation_chains,
    )


def matched_targets(
    normalized_text: str,
    targets: tuple[str, ...],
    original_targets: list[str],
) -> list[str]:
    """Return the original target strings matched in the chunk text."""

    hits: list[str] = []
    for normalized_target, original_target in zip(targets, original_targets, strict=False):
        if normalized_target and normalized_target in normalized_text:
            hits.append(original_target)
    return hits


def graph_concept_hits_for_chunk(
    graph_record: GraphChunkRecord | None,
    normalized_key_concepts: tuple[str, ...],
    original_key_concepts: list[str],
) -> tuple[list[str], list[str]]:
    """Match question concepts against chunk-level graph entities."""

    if graph_record is None or not normalized_key_concepts:
        return [], []

    matched_graph_concepts: list[str] = []
    matched_entities: list[str] = []
    seen_entities: set[str] = set()
    for normalized_concept, original_concept in zip(
        normalized_key_concepts,
        original_key_concepts,
        strict=False,
    ):
        for mention in graph_record.extracted_entities:
            canonical = normalize_text(mention.canonical_name)
            alias = normalize_text(mention.matched_alias)
            if term_overlap(normalized_concept, canonical) or term_overlap(normalized_concept, alias):
                matched_graph_concepts.append(original_concept)
                if mention.canonical_name not in seen_entities:
                    seen_entities.add(mention.canonical_name)
                    matched_entities.append(mention.canonical_name)
                break

    return matched_graph_concepts, matched_entities


def relation_support_for_chunk(
    graph_record: GraphChunkRecord | None,
    profile: QuestionProfile,
    normalized_key_concepts: tuple[str, ...],
) -> tuple[float, list[str]]:
    """Score whether the chunk contains query-aligned relations."""

    if graph_record is None:
        return 0.0, []

    best_score = 0.0
    matched_relation_types: list[str] = []
    for relation in graph_record.extracted_relations:
        source_hit = relation.source_entity_id in profile.query_entity_ids
        target_hit = relation.target_entity_id in profile.query_entity_ids
        if not source_hit and not target_hit:
            continue

        direct_query_pair = source_hit and target_hit
        non_generic_touch = (
            (source_hit and relation.source_entity_id in profile.non_generic_query_entity_ids)
            or (target_hit and relation.target_entity_id in profile.non_generic_query_entity_ids)
        )
        source_name = normalize_text(relation.source_name)
        target_name = normalize_text(relation.target_name)
        concept_endpoint_bonus = 0.0
        if any(term_overlap(source_name, concept) for concept in normalized_key_concepts):
            concept_endpoint_bonus += 0.5
        if any(term_overlap(target_name, concept) for concept in normalized_key_concepts):
            concept_endpoint_bonus += 0.5

        aligned = relation.relation_type in profile.preferred_relation_types
        score = 0.0
        if direct_query_pair:
            score += 0.55
        elif non_generic_touch:
            score += 0.32
        else:
            score += 0.18
        score += 0.20 if aligned else 0.0
        score += 0.18 * min(concept_endpoint_bonus, 1.0)
        if relation.relation_type in RELATIONALLY_RICH_TYPES:
            score += 0.08
        if relation.relation_type == "CO_OCCURS_WITH" and not aligned:
            score *= 0.35

        score = clamp_score(score)
        if score >= 0.30 and relation.relation_type not in matched_relation_types:
            matched_relation_types.append(relation.relation_type)
        best_score = max(best_score, score)

    return best_score, matched_relation_types[:6]


def path_support_for_chunk(
    graph_record: GraphChunkRecord | None,
    profile: QuestionProfile,
) -> tuple[float, list[str]]:
    """Reward short query-aligned paths within a retrieved chunk."""

    if graph_record is None:
        return 0.0, []

    relations = [
        relation
        for relation in graph_record.extracted_relations
        if relation.relation_type != "CO_OCCURS_WITH"
    ]
    if not relations:
        return 0.0, []

    mention_by_id = {
        mention.entity_id: mention
        for mention in graph_record.extracted_entities
    }
    relevant_ids = set(profile.non_generic_query_entity_ids or profile.query_entity_ids)
    if len(relevant_ids) < 2:
        return 0.0, []

    adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)
    direct_best = 0.0
    chains: list[str] = []
    for relation in relations:
        adjacency[relation.source_entity_id].append((relation.target_entity_id, relation.relation_type))
        adjacency[relation.target_entity_id].append((relation.source_entity_id, relation.relation_type))
        if relation.source_entity_id in relevant_ids and relation.target_entity_id in relevant_ids:
            direct_score = 0.55
            if relation.relation_type in profile.preferred_relation_types:
                direct_score += 0.20
            direct_best = max(direct_best, clamp_score(direct_score))
            chain_name = relation.relation_type
            if chain_name not in chains:
                chains.append(chain_name)

    best_path_score = direct_best
    seen_path_keys: set[tuple[str, str, str]] = set()
    for left_id in sorted(relevant_ids):
        for via_id, left_relation_type in adjacency.get(left_id, []):
            if via_id == left_id:
                continue
            via_mention = mention_by_id.get(via_id)
            via_generic = via_mention.is_generic if via_mention is not None else False
            for right_id, right_relation_type in adjacency.get(via_id, []):
                if right_id == left_id or right_id not in relevant_ids:
                    continue
                path_key = tuple(sorted((left_id, via_id, right_id)))
                if path_key in seen_path_keys:
                    continue
                seen_path_keys.add(path_key)
                aligned_count = int(left_relation_type in profile.preferred_relation_types) + int(
                    right_relation_type in profile.preferred_relation_types
                )
                path_score = 0.62 + 0.12 * aligned_count + (0.10 if not via_generic else 0.0)
                best_path_score = max(best_path_score, clamp_score(path_score))
                chain_name = f"{left_relation_type} -> {right_relation_type}"
                if chain_name not in chains:
                    chains.append(chain_name)

    return best_path_score, chains[:6]


def explanation_relevance_for_chunk(
    question: EvaluationQuestion,
    profile: QuestionProfile,
    normalized_text: str,
    concept_match: float,
    evidence_match: float,
    graph_concept_support: float,
    relation_support: float,
    path_support: float,
) -> float:
    """Score whether the chunk helps answer mechanism-oriented questions."""

    marker_hits = sum(
        1 for marker in EXPLANATION_MARKERS if marker in normalized_text
    )
    marker_score = min(marker_hits / 4.0, 1.0)
    relation_or_path = max(relation_support, path_support)

    if not profile.is_explanation_oriented:
        return clamp_score(
            0.40 * relation_or_path
            + 0.30 * graph_concept_support
            + 0.20 * concept_match
            + 0.10 * marker_score
        )

    if question.category == "trade_off":
        marker_score = max(
            marker_score,
            1.0 if any(term in normalized_text for term in ("trade off", "trade-off", "trade-offs", "inverse relationship", "balance")) else 0.0,
        )

    return clamp_score(
        0.45 * relation_or_path
        + 0.20 * graph_concept_support
        + 0.15 * concept_match
        + 0.10 * evidence_match
        + 0.10 * marker_score
    )


def collective_concept_coverage(
    question: EvaluationQuestion,
    chunk_assessments: list[ChunkAssessment],
) -> float:
    """Measure how much of the question's concept set is covered across top-k."""

    if not question.key_concepts:
        return 1.0

    covered = {
        normalize_text(concept)
        for assessment in chunk_assessments
        for concept in assessment.matched_concepts + assessment.matched_graph_concepts
    }
    key_concepts = [normalize_text(concept) for concept in question.key_concepts if concept]
    hits = sum(1 for concept in key_concepts if concept in covered)
    return hits / len(key_concepts) if key_concepts else 1.0


def collective_relation_support(chunk_assessments: list[ChunkAssessment]) -> float:
    """Aggregate query-aligned relation/path support across top-k."""

    if not chunk_assessments:
        return 0.0

    relation_values = [max(item.relation_support, item.path_support) for item in chunk_assessments]
    peak = max(relation_values)
    aggregate = aggregate_chunk_metric(relation_values)
    return clamp_score(0.55 * peak + 0.45 * aggregate)


def collective_reasoning_support(
    question: EvaluationQuestion,
    profile: QuestionProfile,
    chunk_assessments: list[ChunkAssessment],
    collective_concept: float,
    collective_relation: float,
) -> float:
    """Reward top-k sets that jointly support a reasoning chain."""

    if not chunk_assessments:
        return 0.0

    best_single_concept = 0.0
    for assessment in chunk_assessments:
        matched = {
            normalize_text(concept)
            for concept in assessment.matched_concepts + assessment.matched_graph_concepts
        }
        if question.key_concepts:
            best_single_concept = max(best_single_concept, len(matched) / len(question.key_concepts))

    complementarity_bonus = 0.0
    if collective_concept > best_single_concept:
        remaining = max(1.0 - best_single_concept, 1e-9)
        complementarity_bonus = min((collective_concept - best_single_concept) / remaining, 1.0)

    cross_doc_bonus = 0.0
    if profile.is_cross_document:
        matched_gold_docs = {
            assessment.doc_id
            for assessment in chunk_assessments
            if assessment.doc_match > 0.0
        }
        if len(matched_gold_docs) >= 2:
            cross_doc_bonus = 1.0

    path_peak = max((assessment.path_support for assessment in chunk_assessments), default=0.0)
    return clamp_score(
        0.40 * collective_concept
        + 0.25 * collective_relation
        + 0.20 * path_peak
        + 0.10 * complementarity_bonus
        + 0.05 * cross_doc_bonus
    )


def aggregate_chunk_metric(values: list[float]) -> float:
    """Aggregate top-k chunk scores using rank weighting plus best-hit bonus."""

    if not values:
        return 0.0

    weights = RANK_WEIGHTS[: len(values)]
    weighted = sum(value * weight for value, weight in zip(values, weights, strict=False))
    weighted /= sum(weights)
    best = max(values)
    return clamp_score(0.6 * weighted + 0.4 * best)


def detect_failure_reason(
    evaluation_track: EvaluationTrack,
    chunk_assessments: list[ChunkAssessment],
    relevance: float,
    directness: float,
    groundedness: float,
    collective_concept: float,
    collective_relation: float,
) -> str:
    """Classify the main failure mode for one question/method result."""

    max_doc = max((item.doc_match for item in chunk_assessments), default=0.0)
    max_chunk = max((item.chunk_match for item in chunk_assessments), default=0.0)
    max_evidence = max((item.evidence_match for item in chunk_assessments), default=0.0)

    if max_doc == 0.0:
        return "wrong_document_cluster"
    if evaluation_track == "graph_stressing" and collective_concept < 0.35:
        return "concept_coverage_gap"
    if evaluation_track == "graph_stressing" and collective_relation < 0.30:
        return "missing_relation_chain"
    if max_chunk == 0.0 and max_evidence < 0.34:
        return "indirect_context_only"
    if relevance >= 0.55 and directness < 0.45:
        return "broad_context_not_direct"
    if groundedness < 0.5:
        return "weak_evidence_support"
    return "competitive"


def determine_winners(method_results: list[MethodQuestionEvaluation]) -> list[str]:
    """Determine which methods win or tie on one question."""

    if not method_results:
        return []

    best_score = max(result.overall for result in method_results)
    winners = [
        result.method_label
        for result in method_results
        if best_score - result.overall <= WIN_EPSILON
    ]
    return sorted(winners)


def build_method_summaries(
    method_results_map: dict[str, list[MethodQuestionEvaluation]],
    per_question_comparisons: list[PerQuestionComparison],
) -> list[MethodAggregateSummary]:
    """Aggregate scores and wins for each method."""

    wins_by_method: dict[str, list[str]] = defaultdict(list)
    wins_by_track: dict[str, Counter] = defaultdict(Counter)
    for comparison in per_question_comparisons:
        for winner in comparison.winner_methods:
            wins_by_method[winner].append(comparison.question_id)
            wins_by_track[winner][comparison.evaluation_track] += 1

    summaries: list[MethodAggregateSummary] = []
    for method_label in sorted(method_results_map):
        results = method_results_map[method_label]
        failure_reasons: dict[str, int] = defaultdict(int)
        by_category: dict[str, list[float]] = defaultdict(list)
        by_track: dict[str, list[float]] = defaultdict(list)

        for result in results:
            failure_reasons[result.failure_reason or "unknown"] += 1
            by_category[result.category].append(result.overall)
            by_track[result.evaluation_track].append(result.overall)

        summaries.append(
            MethodAggregateSummary(
                method_label=method_label,
                retrieval_method=results[0].retrieval_method,
                embedding_model=results[0].embedding_model,
                question_count=len(results),
                wins=len(wins_by_method.get(method_label, [])),
                wins_by_track=dict(sorted(wins_by_track.get(method_label, Counter()).items())),
                mean_relevance=round(mean(result.relevance for result in results), 6),
                mean_directness=round(mean(result.directness for result in results), 6),
                mean_groundedness=round(mean(result.groundedness for result in results), 6),
                mean_correctness_of_evidence=round(
                    mean(result.correctness_of_evidence for result in results),
                    6,
                ),
                mean_overall=round(mean(result.overall for result in results), 6),
                mean_collective_concept_coverage=round(
                    mean(result.collective_concept_coverage for result in results),
                    6,
                ),
                mean_collective_relation_support=round(
                    mean(result.collective_relation_support for result in results),
                    6,
                ),
                mean_collective_reasoning_support=round(
                    mean(result.collective_reasoning_support for result in results),
                    6,
                ),
                top1_doc_accuracy=round(mean(result.top1_doc_match for result in results), 6),
                topk_gold_chunk_recall=round(
                    mean(result.any_gold_chunk_in_top_k for result in results),
                    6,
                ),
                mean_overall_by_category={
                    category: round(mean(scores), 6)
                    for category, scores in sorted(by_category.items())
                },
                mean_overall_by_track={
                    track: round(mean(scores), 6)
                    for track, scores in sorted(by_track.items())
                },
                failure_reasons=dict(sorted(failure_reasons.items())),
                question_ids_won=sorted(wins_by_method.get(method_label, [])),
            )
        )

    return summaries


def write_evaluation_artifacts(
    output_dir: Path,
    questions: list[EvaluationQuestion],
    question_profiles: dict[str, QuestionProfile],
    per_question_comparisons: list[PerQuestionComparison],
    summary: EvaluationSummary,
    top_k: int,
    method_specs,
) -> None:
    """Write the evaluation outputs to disk."""

    (output_dir / "question_set.jsonl").write_text(
        "\n".join(
            json.dumps(question.model_dump(mode="json"), ensure_ascii=False)
            for question in questions
        )
        + "\n",
        encoding="utf-8",
    )

    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "evaluation_id": summary.evaluation_id,
                "timestamp_utc": summary.timestamp_utc,
                "top_k": top_k,
                "scoring_version": SCORING_VERSION,
                "category_distribution": summary.category_distribution,
                "evaluation_track_distribution": summary.evaluation_track_distribution,
                "explanation_oriented_question_count": summary.explanation_oriented_question_count,
                "methods": {
                    method_spec["label"]: method_spec["config_value"]
                    for method_spec in method_specs
                },
                "question_profiles": {
                    question.question_id: {
                        "evaluation_track": question_profiles[question.question_id].evaluation_track,
                        "is_explanation_oriented": question_profiles[question.question_id].is_explanation_oriented,
                        "is_cross_document": question_profiles[question.question_id].is_cross_document,
                    }
                    for question in questions
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    write_jsonl_models(
        path=output_dir / "per_question_results.jsonl",
        records=per_question_comparisons,
    )

    (output_dir / "aggregate_summary.json").write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )

    write_per_question_csv(
        output_path=output_dir / "per_question_comparison.csv",
        per_question_comparisons=per_question_comparisons,
        method_labels=[method_spec["label"] for method_spec in method_specs],
    )
    write_method_summary_csv(
        output_path=output_dir / "method_summary.csv",
        summary=summary,
    )
    (output_dir / "summary_report.md").write_text(
        build_summary_report(summary=summary, per_question_comparisons=per_question_comparisons),
        encoding="utf-8",
    )


def write_jsonl_models(path: Path, records) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False))
            handle.write("\n")


def write_per_question_csv(
    output_path: Path,
    per_question_comparisons: list[PerQuestionComparison],
    method_labels: list[str],
) -> None:
    fieldnames = [
        "question_id",
        "category",
        "evaluation_track",
        "is_explanation_oriented",
        "question",
        "winner_methods",
    ]
    for label in method_labels:
        fieldnames.extend(
            [
                f"{label}_overall",
                f"{label}_relevance",
                f"{label}_directness",
                f"{label}_groundedness",
                f"{label}_correctness",
                f"{label}_collective_concept",
                f"{label}_collective_relation",
                f"{label}_collective_reasoning",
            ]
        )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for comparison in per_question_comparisons:
            row = {
                "question_id": comparison.question_id,
                "category": comparison.category,
                "evaluation_track": comparison.evaluation_track,
                "is_explanation_oriented": comparison.is_explanation_oriented,
                "question": comparison.question,
                "winner_methods": ",".join(comparison.winner_methods),
            }
            by_method = {
                result.method_label: result for result in comparison.method_results
            }
            for label in method_labels:
                result = by_method.get(label)
                row[f"{label}_overall"] = result.overall if result else ""
                row[f"{label}_relevance"] = result.relevance if result else ""
                row[f"{label}_directness"] = result.directness if result else ""
                row[f"{label}_groundedness"] = result.groundedness if result else ""
                row[f"{label}_correctness"] = (
                    result.correctness_of_evidence if result else ""
                )
                row[f"{label}_collective_concept"] = (
                    result.collective_concept_coverage if result else ""
                )
                row[f"{label}_collective_relation"] = (
                    result.collective_relation_support if result else ""
                )
                row[f"{label}_collective_reasoning"] = (
                    result.collective_reasoning_support if result else ""
                )
            writer.writerow(row)


def write_method_summary_csv(output_path: Path, summary: EvaluationSummary) -> None:
    fieldnames = [
        "method_label",
        "retrieval_method",
        "embedding_model",
        "question_count",
        "wins",
        "mean_relevance",
        "mean_directness",
        "mean_groundedness",
        "mean_correctness_of_evidence",
        "mean_overall",
        "mean_collective_concept_coverage",
        "mean_collective_relation_support",
        "mean_collective_reasoning_support",
        "top1_doc_accuracy",
        "topk_gold_chunk_recall",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method_summary in summary.method_summaries:
            writer.writerow(method_summary.model_dump(include=set(fieldnames)))


def build_summary_report(
    summary: EvaluationSummary,
    per_question_comparisons: list[PerQuestionComparison],
) -> str:
    """Build a human-readable Markdown report."""

    lines: list[str] = []
    lines.append("# Retrieval Evaluation Summary")
    lines.append("")
    lines.append(f"- Evaluation ID: `{summary.evaluation_id}`")
    lines.append(f"- Questions: `{summary.question_count}`")
    lines.append(f"- Top-k compared: `{summary.top_k}`")
    lines.append(f"- Scoring version: `{summary.scoring_version}`")
    lines.append("")
    lines.append("## Question Set Analysis")
    lines.append("")
    lines.append(
        "- Category distribution: "
        + ", ".join(
            f"`{category}`={count}"
            for category, count in summary.category_distribution.items()
        )
    )
    lines.append(
        "- Track distribution: "
        + ", ".join(
            f"`{track}`={count}"
            for track, count in summary.evaluation_track_distribution.items()
        )
    )
    lines.append(
        f"- Explanation-oriented questions: `{summary.explanation_oriented_question_count}`"
    )
    lines.append("")
    lines.append("## Aggregate Scores")
    lines.append("")
    lines.append("| Method | Overall | Relevance | Directness | Groundedness | Correctness | Top-1 Doc | Top-k Gold Chunk | Wins |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for method_summary in sorted(
        summary.method_summaries,
        key=lambda item: item.mean_overall,
        reverse=True,
    ):
        lines.append(
            f"| {method_summary.method_label} | {method_summary.mean_overall:.3f} | "
            f"{method_summary.mean_relevance:.3f} | {method_summary.mean_directness:.3f} | "
            f"{method_summary.mean_groundedness:.3f} | "
            f"{method_summary.mean_correctness_of_evidence:.3f} | "
            f"{method_summary.top1_doc_accuracy:.3f} | "
            f"{method_summary.topk_gold_chunk_recall:.3f} | "
            f"{method_summary.wins} |"
        )

    lines.append("")
    lines.append("## Track Results")
    lines.append("")
    lines.append("| Method | Exact Retrieval | Graph-Stressing | Exact Wins | Graph Wins |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for method_summary in sorted(
        summary.method_summaries,
        key=lambda item: item.mean_overall,
        reverse=True,
    ):
        lines.append(
            f"| {method_summary.method_label} | "
            f"{method_summary.mean_overall_by_track.get('exact_retrieval', 0.0):.3f} | "
            f"{method_summary.mean_overall_by_track.get('graph_stressing', 0.0):.3f} | "
            f"{method_summary.wins_by_track.get('exact_retrieval', 0)} | "
            f"{method_summary.wins_by_track.get('graph_stressing', 0)} |"
        )

    lines.append("")
    lines.append("## Per-Question Winners")
    lines.append("")
    lines.append("| Question ID | Category | Track | Winners |")
    lines.append("| --- | --- | --- | --- |")
    for comparison in per_question_comparisons:
        lines.append(
            f"| {comparison.question_id} | {comparison.category} | {comparison.evaluation_track} | "
            f"{', '.join(comparison.winner_methods)} |"
        )

    lines.append("")
    lines.append("## Method Notes")
    lines.append("")
    for method_summary in sorted(
        summary.method_summaries,
        key=lambda item: item.mean_overall,
        reverse=True,
    ):
        strongest = sorted(
            method_summary.mean_overall_by_category.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:2]
        weakest = sorted(
            method_summary.mean_overall_by_category.items(),
            key=lambda item: item[1],
        )[:2]
        lines.append(f"### {method_summary.method_label}")
        lines.append(
            f"- Strongest categories: "
            + ", ".join(f"`{name}` ({score:.3f})" for name, score in strongest)
        )
        lines.append(
            f"- Weakest categories: "
            + ", ".join(f"`{name}` ({score:.3f})" for name, score in weakest)
        )
        lines.append(
            f"- Track performance: "
            + ", ".join(
                f"`{track}` ({score:.3f})"
                for track, score in method_summary.mean_overall_by_track.items()
            )
        )
        lines.append(
            f"- Collective support: concepts={method_summary.mean_collective_concept_coverage:.3f}, "
            f"relations={method_summary.mean_collective_relation_support:.3f}, "
            f"reasoning={method_summary.mean_collective_reasoning_support:.3f}"
        )
        lines.append(
            f"- Failure modes: "
            + ", ".join(
                f"`{name}`={count}"
                for name, count in sorted(
                    method_summary.failure_reasons.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )
        )
        lines.append(
            f"- Questions won: "
            + (", ".join(method_summary.question_ids_won) or "none")
        )
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def mean(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def clamp_score(value: float) -> float:
    return max(0.0, min(1.0, value))


def normalize_text(text: str) -> str:
    cleaned = NORMALIZE_RE.sub(" ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def term_overlap(left: str, right: str) -> bool:
    """Allow concept matching to tolerate phrase containment."""

    if not left or not right:
        return False
    return left == right or left in right or right in left


def build_evaluation_id() -> str:
    timestamp = _utc_now()
    compact = (
        timestamp.replace("-", "")
        .replace(":", "")
        .replace("+00:00", "Z")
        .replace(".", "")
    )
    return f"retrieval_eval_{compact}"


def _resolve_path(path_value: Path | str, project_root: Path = ROOT_DIR) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (project_root / candidate).resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
