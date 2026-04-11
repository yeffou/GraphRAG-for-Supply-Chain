"""Schemas for deterministic dual-track retrieval evaluation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


QuestionCategory = Literal[
    "direct_factual",
    "causal",
    "mitigation",
    "trade_off",
    "multi_hop",
]

EvaluationTrack = Literal["exact_retrieval", "graph_stressing"]


class EvaluationQuestion(BaseModel):
    """One annotated evaluation question grounded in the real corpus."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    category: QuestionCategory
    question: str
    gold_doc_ids: list[str] = Field(min_length=1)
    gold_chunk_ids: list[str] = Field(min_length=1)
    key_concepts: list[str] = Field(default_factory=list)
    evidence_phrases: list[str] = Field(default_factory=list)
    notes: str | None = None


class ChunkAssessment(BaseModel):
    """Per-chunk scoring details for one method/question pair."""

    model_config = ConfigDict(extra="forbid")

    rank: int = Field(ge=1)
    score: float
    chunk_id: str
    doc_id: str
    title: str
    page_number: int = Field(ge=1)
    source_url: str
    preview: str
    evaluation_track: EvaluationTrack
    doc_match: float = Field(ge=0.0, le=1.0)
    chunk_match: float = Field(ge=0.0, le=1.0)
    concept_match: float = Field(ge=0.0, le=1.0)
    evidence_match: float = Field(ge=0.0, le=1.0)
    graph_concept_support: float = Field(ge=0.0, le=1.0)
    relation_support: float = Field(ge=0.0, le=1.0)
    path_support: float = Field(ge=0.0, le=1.0)
    explanation_relevance: float = Field(ge=0.0, le=1.0)
    relevance_score: float = Field(ge=0.0, le=1.0)
    directness_score: float = Field(ge=0.0, le=1.0)
    groundedness_score: float = Field(ge=0.0, le=1.0)
    correctness_score: float = Field(ge=0.0, le=1.0)
    matched_concepts: list[str] = Field(default_factory=list)
    matched_evidence_phrases: list[str] = Field(default_factory=list)
    matched_graph_concepts: list[str] = Field(default_factory=list)
    matched_graph_entities: list[str] = Field(default_factory=list)
    matched_relation_types: list[str] = Field(default_factory=list)
    matched_path_relation_chains: list[str] = Field(default_factory=list)


class MethodQuestionEvaluation(BaseModel):
    """Evaluation result for one method on one question."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    category: QuestionCategory
    evaluation_track: EvaluationTrack
    is_explanation_oriented: bool
    question: str
    method_label: str
    retrieval_method: str
    embedding_model: str
    index_dir: str
    run_path: str
    top_k: int = Field(ge=1)
    relevance: float = Field(ge=0.0, le=1.0)
    directness: float = Field(ge=0.0, le=1.0)
    groundedness: float = Field(ge=0.0, le=1.0)
    correctness_of_evidence: float = Field(ge=0.0, le=1.0)
    overall: float = Field(ge=0.0, le=1.0)
    collective_concept_coverage: float = Field(ge=0.0, le=1.0)
    collective_relation_support: float = Field(ge=0.0, le=1.0)
    collective_reasoning_support: float = Field(ge=0.0, le=1.0)
    top1_doc_match: float = Field(ge=0.0, le=1.0)
    any_gold_chunk_in_top_k: float = Field(ge=0.0, le=1.0)
    failure_reason: str | None = None
    retrieved_chunks: list[ChunkAssessment]


class PerQuestionComparison(BaseModel):
    """Side-by-side comparison of all methods for one question."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    category: QuestionCategory
    evaluation_track: EvaluationTrack
    is_explanation_oriented: bool
    question: str
    winner_methods: list[str]
    method_results: list[MethodQuestionEvaluation]


class MethodAggregateSummary(BaseModel):
    """Aggregate metrics for one retrieval method."""

    model_config = ConfigDict(extra="forbid")

    method_label: str
    retrieval_method: str
    embedding_model: str
    question_count: int = Field(ge=0)
    wins: int = Field(ge=0)
    wins_by_track: dict[str, int]
    mean_relevance: float = Field(ge=0.0, le=1.0)
    mean_directness: float = Field(ge=0.0, le=1.0)
    mean_groundedness: float = Field(ge=0.0, le=1.0)
    mean_correctness_of_evidence: float = Field(ge=0.0, le=1.0)
    mean_overall: float = Field(ge=0.0, le=1.0)
    mean_collective_concept_coverage: float = Field(ge=0.0, le=1.0)
    mean_collective_relation_support: float = Field(ge=0.0, le=1.0)
    mean_collective_reasoning_support: float = Field(ge=0.0, le=1.0)
    top1_doc_accuracy: float = Field(ge=0.0, le=1.0)
    topk_gold_chunk_recall: float = Field(ge=0.0, le=1.0)
    mean_overall_by_category: dict[str, float]
    mean_overall_by_track: dict[str, float]
    failure_reasons: dict[str, int]
    question_ids_won: list[str]


class EvaluationSummary(BaseModel):
    """Top-level saved summary for one evaluation run."""

    model_config = ConfigDict(extra="forbid")

    evaluation_id: str
    timestamp_utc: str
    question_count: int = Field(ge=0)
    top_k: int = Field(ge=1)
    scoring_version: str
    questions_path: str
    output_dir: str
    category_distribution: dict[str, int]
    evaluation_track_distribution: dict[str, int]
    explanation_oriented_question_count: int = Field(ge=0)
    method_summaries: list[MethodAggregateSummary]
