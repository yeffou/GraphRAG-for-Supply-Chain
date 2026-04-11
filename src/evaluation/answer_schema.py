"""Schemas for answer generation and answer-level evaluation artifacts."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RetrievedContextRecord(BaseModel):
    """Retrieved chunk context stored with a generated answer."""

    model_config = ConfigDict(extra="forbid")

    rank: int = Field(ge=1)
    chunk_id: str
    doc_id: str
    title: str
    page_number: int = Field(ge=1)
    source_url: str
    text: str


class AnswerGenerationRecord(BaseModel):
    """Saved generated answer for one question/method pair."""

    model_config = ConfigDict(extra="forbid")

    answer_id: str
    timestamp_utc: str
    question_id: str
    category: str
    method_label: str
    retrieval_method: str
    query: str
    top_k: int = Field(ge=1)
    retrieval_run_id: str
    retrieval_run_path: str
    retrieved_chunk_ids: list[str]
    retrieved_context: list[RetrievedContextRecord]
    generation_model: str
    prompt_template_version: str
    prompt: dict
    answer_text: str
    cited_chunk_ids: list[str]
    generation_metadata: dict


class DeterministicAnswerChecks(BaseModel):
    """Inspectably computed answer-level checks."""

    model_config = ConfigDict(extra="forbid")

    citation_count: int = Field(ge=0)
    valid_citation_count: int = Field(ge=0)
    citation_precision: float = Field(ge=0.0, le=1.0)
    answer_key_concept_coverage: float = Field(ge=0.0, le=1.0)
    answer_evidence_phrase_coverage: float = Field(ge=0.0, le=1.0)
    answer_mentions_gold_doc: bool
    answer_mentions_unknown_citation: bool


class AnswerJudgeScores(BaseModel):
    """LLM-judge scores for one answer."""

    model_config = ConfigDict(extra="forbid")

    correctness: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0)
    reasoning: float = Field(ge=0.0, le=1.0)
    groundedness: float = Field(ge=0.0, le=1.0)
    clarity: float = Field(ge=0.0, le=1.0)


class AnswerEvaluationRecord(BaseModel):
    """Saved answer-level evaluation artifact for one question/method pair."""

    model_config = ConfigDict(extra="forbid")

    evaluation_id: str
    timestamp_utc: str
    question_id: str
    category: str
    method_label: str
    retrieval_method: str
    query: str
    answer_id: str
    answer_path: str
    judge_model: str
    judge_prompt_template_version: str
    judge_prompt: dict
    scores: AnswerJudgeScores
    weighted_judge_score: float = Field(ge=0.0, le=1.0)
    score_cap: float = Field(ge=0.0, le=1.0)
    overall_score: float = Field(ge=0.0, le=1.0)
    deterministic_checks: DeterministicAnswerChecks
    calibration_notes: list[str]
    judge_explanation: str
    raw_judge_output: dict


class AnswerMethodQuestionResult(BaseModel):
    """One method's answer result for one question."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    category: str
    question: str
    method_label: str
    retrieval_method: str
    answer_path: str
    evaluation_path: str
    overall_score: float = Field(ge=0.0, le=1.0)
    correctness: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0)
    reasoning: float = Field(ge=0.0, le=1.0)
    groundedness: float = Field(ge=0.0, le=1.0)
    clarity: float = Field(ge=0.0, le=1.0)
    citation_precision: float = Field(ge=0.0, le=1.0)
    answer_key_concept_coverage: float = Field(ge=0.0, le=1.0)
    answer_evidence_phrase_coverage: float = Field(ge=0.0, le=1.0)
    winner: bool = False


class AnswerPerQuestionComparison(BaseModel):
    """Side-by-side answer comparison for one question."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    category: str
    question: str
    winner_methods: list[str]
    method_results: list[AnswerMethodQuestionResult]


class AnswerMethodAggregateSummary(BaseModel):
    """Aggregate answer-level summary for one retrieval method."""

    model_config = ConfigDict(extra="forbid")

    method_label: str
    retrieval_method: str
    question_count: int = Field(ge=0)
    wins: int = Field(ge=0)
    mean_overall_score: float = Field(ge=0.0, le=1.0)
    mean_correctness: float = Field(ge=0.0, le=1.0)
    mean_completeness: float = Field(ge=0.0, le=1.0)
    mean_reasoning: float = Field(ge=0.0, le=1.0)
    mean_groundedness: float = Field(ge=0.0, le=1.0)
    mean_clarity: float = Field(ge=0.0, le=1.0)
    mean_citation_precision: float = Field(ge=0.0, le=1.0)
    mean_answer_key_concept_coverage: float = Field(ge=0.0, le=1.0)
    mean_answer_evidence_phrase_coverage: float = Field(ge=0.0, le=1.0)
    mean_overall_by_category: dict[str, float]
    question_ids_won: list[str]


class AnswerEvaluationSummary(BaseModel):
    """Top-level summary for a full answer-evaluation run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    timestamp_utc: str
    question_count: int = Field(ge=0)
    top_k: int = Field(ge=1)
    generation_model: str
    judge_model: str
    questions_path: str
    answers_output_dir: str
    evaluation_output_dir: str
    method_summaries: list[AnswerMethodAggregateSummary]
