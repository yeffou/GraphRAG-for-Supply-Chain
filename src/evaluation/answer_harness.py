"""Answer generation and answer-level evaluation over the four retrieval methods."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src.baseline_rag.dense_query import run_dense_query
from src.baseline_rag.query import run_baseline_query
from src.config import ROOT_DIR, ProjectConfig
from src.evaluation.answer_schema import (
    AnswerEvaluationRecord,
    AnswerEvaluationSummary,
    AnswerGenerationRecord,
    AnswerJudgeScores,
    AnswerMethodAggregateSummary,
    AnswerMethodQuestionResult,
    AnswerPerQuestionComparison,
    DeterministicAnswerChecks,
    RetrievedContextRecord,
)
from src.evaluation.harness import load_questions, normalize_text
from src.evaluation.schema import EvaluationQuestion
from src.generation import (
    LLMClientError,
    build_answer_judge_prompt,
    build_grounded_answer_prompt,
    call_chat_completion,
    parse_json_object_from_text,
    prompt_to_messages,
    require_generation_config,
    require_judge_config,
)
from src.graph_rag import run_graph_query
from src.graph_rag.hybrid_query import run_hybrid_graph_query


ANSWER_EVAL_VERSION = "answer_eval_v2_calibrated"
WIN_EPSILON = 0.02
CITATION_RE = re.compile(r"\[([^\[\]]+)\]")

ANSWER_SCORE_WEIGHTS = {
    "correctness": 0.35,
    "completeness": 0.25,
    "groundedness": 0.25,
    "reasoning": 0.10,
    "clarity": 0.05,
}


def run_answer_evaluation(
    *,
    questions_path: Path | str,
    answers_output_root: Path | str,
    evaluation_output_root: Path | str,
    config: ProjectConfig,
    top_k: int = 3,
    question_ids: list[str] | None = None,
    question_limit: int | None = None,
) -> tuple[AnswerEvaluationSummary, list[AnswerPerQuestionComparison]]:
    """Run retrieval, answer generation, judging, and summarization."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    generation_model = require_generation_config(config.llm)
    judge_model = require_judge_config(config.llm)

    questions = load_questions(questions_path)
    if question_ids:
        wanted = set(question_ids)
        questions = [question for question in questions if question.question_id in wanted]
        missing = wanted - {question.question_id for question in questions}
        if missing:
            raise ValueError(f"unknown question_ids requested: {', '.join(sorted(missing))}")
    if question_limit is not None:
        if question_limit <= 0:
            raise ValueError("question_limit must be greater than 0 when provided")
        questions = questions[:question_limit]
    if not questions:
        raise ValueError("no evaluation questions selected for answer evaluation")

    run_id = build_answer_run_id()
    answers_output_dir = _resolve_path(answers_output_root, config.paths.root_dir) / run_id
    evaluation_output_dir = _resolve_path(
        evaluation_output_root,
        config.paths.root_dir,
    ) / run_id
    answers_output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_output_dir.mkdir(parents=True, exist_ok=True)

    retrieval_runs_dir = answers_output_dir / "retrieval_runs"
    generated_answers_dir = answers_output_dir / "generated"
    judged_answers_dir = evaluation_output_dir / "judged"
    retrieval_runs_dir.mkdir(parents=True, exist_ok=True)
    generated_answers_dir.mkdir(parents=True, exist_ok=True)
    judged_answers_dir.mkdir(parents=True, exist_ok=True)

    method_specs = build_method_specs(config, retrieval_runs_dir)
    per_question_comparisons: list[AnswerPerQuestionComparison] = []
    method_results_map: dict[str, list[AnswerMethodQuestionResult]] = defaultdict(list)

    for question in questions:
        question_results: list[AnswerMethodQuestionResult] = []
        for method_spec in method_specs:
            runner_kwargs = dict(method_spec.get("runner_kwargs", {}))
            if "index_dir" in method_spec:
                runner_kwargs["index_dir"] = method_spec["index_dir"]

            retrieval_run = method_spec["runner"](
                query_text=question.question,
                output_dir=method_spec["output_dir"],
                top_k=top_k,
                llm_config=config.llm,
                generate_answer=False,
                project_root=config.paths.root_dir,
                **runner_kwargs,
            )
            retrieval_run_path = method_spec["output_dir"] / f"{retrieval_run.run_id}.json"
            answer_output_dir = generated_answers_dir / method_spec["label"]
            answer_output_dir.mkdir(parents=True, exist_ok=True)
            answer_record, answer_path = generate_answer_for_method(
                question=question,
                method_label=method_spec["label"],
                retrieval_run=retrieval_run,
                retrieval_run_path=retrieval_run_path,
                generation_model=generation_model,
                answer_output_dir=answer_output_dir,
                config=config,
            )
            judge_output_dir = judged_answers_dir / method_spec["label"]
            judge_output_dir.mkdir(parents=True, exist_ok=True)
            evaluation_record, evaluation_path = judge_generated_answer(
                question=question,
                answer_record=answer_record,
                answer_path=answer_path,
                judge_model=judge_model,
                judge_output_dir=judge_output_dir,
                config=config,
            )
            result = AnswerMethodQuestionResult(
                question_id=question.question_id,
                category=question.category,
                question=question.question,
                method_label=method_spec["label"],
                retrieval_method=retrieval_run.method,
                answer_path=str(answer_path),
                evaluation_path=str(evaluation_path),
                overall_score=evaluation_record.overall_score,
                correctness=evaluation_record.scores.correctness,
                completeness=evaluation_record.scores.completeness,
                reasoning=evaluation_record.scores.reasoning,
                groundedness=evaluation_record.scores.groundedness,
                clarity=evaluation_record.scores.clarity,
                citation_precision=evaluation_record.deterministic_checks.citation_precision,
                answer_key_concept_coverage=evaluation_record.deterministic_checks.answer_key_concept_coverage,
                answer_evidence_phrase_coverage=evaluation_record.deterministic_checks.answer_evidence_phrase_coverage,
            )
            question_results.append(result)
            method_results_map[method_spec["label"]].append(result)

        winner_methods = determine_winners(question_results)
        for result in question_results:
            result.winner = result.method_label in winner_methods

        per_question_comparisons.append(
            AnswerPerQuestionComparison(
                question_id=question.question_id,
                category=question.category,
                question=question.question,
                winner_methods=winner_methods,
                method_results=question_results,
            )
        )

    summary = AnswerEvaluationSummary(
        run_id=run_id,
        timestamp_utc=_utc_now(),
        question_count=len(questions),
        top_k=top_k,
        generation_model=generation_model,
        judge_model=judge_model,
        questions_path=str(Path(questions_path).resolve()),
        answers_output_dir=str(answers_output_dir),
        evaluation_output_dir=str(evaluation_output_dir),
        method_summaries=build_method_summaries(
            method_results_map=method_results_map,
            per_question_comparisons=per_question_comparisons,
        ),
    )

    write_answer_evaluation_artifacts(
        answers_output_dir=answers_output_dir,
        evaluation_output_dir=evaluation_output_dir,
        questions=questions,
        per_question_comparisons=per_question_comparisons,
        summary=summary,
        top_k=top_k,
        method_specs=method_specs,
    )

    return summary, per_question_comparisons


def build_method_specs(config: ProjectConfig, retrieval_runs_dir: Path) -> list[dict]:
    """Define the four answer-evaluation retrieval paths."""

    return [
        {
            "label": "tfidf",
            "runner": run_baseline_query,
            "index_dir": config.paths.indexes_dir / "baseline_tfidf",
            "output_dir": retrieval_runs_dir / "tfidf",
            "config_value": str(config.paths.indexes_dir / "baseline_tfidf"),
        },
        {
            "label": "dense",
            "runner": run_dense_query,
            "index_dir": config.paths.indexes_dir / "baseline_dense",
            "output_dir": retrieval_runs_dir / "dense",
            "config_value": str(config.paths.indexes_dir / "baseline_dense"),
        },
        {
            "label": "graph",
            "runner": run_graph_query,
            "index_dir": config.paths.indexes_dir / "baseline_graph",
            "output_dir": retrieval_runs_dir / "graph",
            "config_value": str(config.paths.indexes_dir / "baseline_graph"),
        },
        {
            "label": "hybrid_graph",
            "runner": run_hybrid_graph_query,
            "output_dir": retrieval_runs_dir / "hybrid_graph",
            "runner_kwargs": {
                "dense_index_dir": config.paths.indexes_dir / "baseline_dense",
                "graph_index_dir": config.paths.indexes_dir / "baseline_graph",
            },
            "config_value": {
                "dense_index_dir": str(config.paths.indexes_dir / "baseline_dense"),
                "graph_index_dir": str(config.paths.indexes_dir / "baseline_graph"),
            },
        },
    ]


def generate_answer_for_method(
    *,
    question: EvaluationQuestion,
    method_label: str,
    retrieval_run,
    retrieval_run_path: Path,
    generation_model: str,
    answer_output_dir: Path,
    config: ProjectConfig,
) -> tuple[AnswerGenerationRecord, Path]:
    """Generate one grounded answer from retrieved chunks."""

    prompt = build_grounded_answer_prompt(
        question=question.question,
        retrieved_chunks=retrieval_run.retrieved_chunks,
        method_label=method_label,
    )
    completion = call_chat_completion(
        llm_config=config.llm,
        model=generation_model,
        messages=prompt_to_messages(prompt),
        temperature=0.0,
        max_tokens=900,
    )
    cited_chunk_ids = extract_cited_chunk_ids(
        answer_text=completion.content,
        valid_chunk_ids=[chunk.chunk_id for chunk in retrieval_run.retrieved_chunks],
    )
    answer_record = AnswerGenerationRecord(
        answer_id=build_answer_record_id(question.question_id, method_label),
        timestamp_utc=_utc_now(),
        question_id=question.question_id,
        category=question.category,
        method_label=method_label,
        retrieval_method=retrieval_run.method,
        query=question.question,
        top_k=retrieval_run.top_k,
        retrieval_run_id=retrieval_run.run_id,
        retrieval_run_path=str(retrieval_run_path),
        retrieved_chunk_ids=[chunk.chunk_id for chunk in retrieval_run.retrieved_chunks],
        retrieved_context=[
            RetrievedContextRecord(
                rank=chunk.rank,
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                title=chunk.title,
                page_number=chunk.page_number,
                source_url=chunk.source_url,
                text=chunk.text,
            )
            for chunk in retrieval_run.retrieved_chunks
        ],
        generation_model=generation_model,
        prompt_template_version=prompt.template_version,
        prompt=prompt.model_dump(mode="json"),
        answer_text=completion.content,
        cited_chunk_ids=cited_chunk_ids,
        generation_metadata={
            "latency_seconds": completion.latency_seconds,
            "usage": completion.usage,
            "raw_response": completion.raw_response,
        },
    )
    output_path = answer_output_dir / f"{question.question_id}.json"
    output_path.write_text(
        json.dumps(answer_record.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return answer_record, output_path


def judge_generated_answer(
    *,
    question: EvaluationQuestion,
    answer_record: AnswerGenerationRecord,
    answer_path: Path,
    judge_model: str,
    judge_output_dir: Path,
    config: ProjectConfig,
) -> tuple[AnswerEvaluationRecord, Path]:
    """Judge one generated answer with an LLM rubric and deterministic checks."""

    deterministic_checks = compute_deterministic_checks(
        question=question,
        answer_record=answer_record,
    )
    prompt = build_answer_judge_prompt(
        question_id=question.question_id,
        question=question.question,
        category=question.category,
        answer_text=answer_record.answer_text,
        retrieved_chunks=answer_record.retrieved_context,
        gold_doc_ids=question.gold_doc_ids,
        gold_chunk_ids=question.gold_chunk_ids,
        key_concepts=question.key_concepts,
        evidence_phrases=question.evidence_phrases,
        citation_count=deterministic_checks.citation_count,
        valid_citation_count=deterministic_checks.valid_citation_count,
        citation_precision=deterministic_checks.citation_precision,
        answer_key_concept_coverage=deterministic_checks.answer_key_concept_coverage,
        answer_evidence_phrase_coverage=deterministic_checks.answer_evidence_phrase_coverage,
        answer_mentions_gold_doc=deterministic_checks.answer_mentions_gold_doc,
    )
    completion = call_chat_completion(
        llm_config=config.llm,
        model=judge_model,
        messages=prompt_to_messages(prompt),
        temperature=0.0,
        max_tokens=700,
    )
    judge_payload = parse_json_object_from_text(completion.content)
    try:
        scores = AnswerJudgeScores(
            correctness=float(judge_payload["correctness"]),
            completeness=float(judge_payload["completeness"]),
            reasoning=float(judge_payload["reasoning"]),
            groundedness=float(judge_payload["groundedness"]),
            clarity=float(judge_payload.get("clarity", 0.5)),
        )
        judge_explanation = str(judge_payload["judge_explanation"]).strip()
    except (KeyError, TypeError, ValueError) as exc:
        raise LLMClientError(
            "Judge output did not match the required JSON schema."
        ) from exc
    weighted_judge_score = round(
        compute_weighted_judge_score(scores),
        6,
    )
    score_cap, calibration_notes = determine_score_cap(
        question=question,
        scores=scores,
        deterministic_checks=deterministic_checks,
    )
    overall_score = round(min(weighted_judge_score, score_cap), 6)
    evaluation_record = AnswerEvaluationRecord(
        evaluation_id=build_answer_judgment_id(question.question_id, answer_record.method_label),
        timestamp_utc=_utc_now(),
        question_id=question.question_id,
        category=question.category,
        method_label=answer_record.method_label,
        retrieval_method=answer_record.retrieval_method,
        query=question.question,
        answer_id=answer_record.answer_id,
        answer_path=str(answer_path),
        judge_model=judge_model,
        judge_prompt_template_version=prompt.template_version,
        judge_prompt=prompt.model_dump(mode="json"),
        scores=scores,
        weighted_judge_score=weighted_judge_score,
        score_cap=round(score_cap, 6),
        overall_score=overall_score,
        deterministic_checks=deterministic_checks,
        calibration_notes=calibration_notes,
        judge_explanation=judge_explanation,
        raw_judge_output={
            "parsed_output": judge_payload,
            "raw_content": completion.content,
            "raw_response": completion.raw_response,
            "latency_seconds": completion.latency_seconds,
            "usage": completion.usage,
        },
    )
    output_path = judge_output_dir / f"{question.question_id}.json"
    output_path.write_text(
        json.dumps(evaluation_record.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return evaluation_record, output_path


def compute_deterministic_checks(
    *,
    question: EvaluationQuestion,
    answer_record: AnswerGenerationRecord,
) -> DeterministicAnswerChecks:
    """Compute transparent answer checks alongside the LLM judge."""

    valid_citations = set(answer_record.retrieved_chunk_ids)
    cited_chunk_ids = answer_record.cited_chunk_ids
    valid_cited = [chunk_id for chunk_id in cited_chunk_ids if chunk_id in valid_citations]
    answer_text_normalized = normalize_text(answer_record.answer_text)

    normalized_key_concepts = [
        normalize_text(concept) for concept in question.key_concepts if concept
    ]
    normalized_evidence_phrases = [
        normalize_text(phrase) for phrase in question.evidence_phrases if phrase
    ]

    answer_key_concept_coverage = coverage_ratio(
        answer_text_normalized,
        normalized_key_concepts,
    )
    answer_evidence_phrase_coverage = coverage_ratio(
        answer_text_normalized,
        normalized_evidence_phrases,
    )

    cited_docs = {
        context.doc_id
        for context in answer_record.retrieved_context
        if context.chunk_id in valid_cited
    }
    return DeterministicAnswerChecks(
        citation_count=len(cited_chunk_ids),
        valid_citation_count=len(valid_cited),
        citation_precision=(
            len(valid_cited) / len(cited_chunk_ids) if cited_chunk_ids else 0.0
        ),
        answer_key_concept_coverage=answer_key_concept_coverage,
        answer_evidence_phrase_coverage=answer_evidence_phrase_coverage,
        answer_mentions_gold_doc=bool(cited_docs & set(question.gold_doc_ids)),
        answer_mentions_unknown_citation=any(
            chunk_id not in valid_citations for chunk_id in cited_chunk_ids
        ),
    )


def extract_cited_chunk_ids(answer_text: str, valid_chunk_ids: list[str]) -> list[str]:
    """Extract cited chunk ids from the generated answer."""

    valid = set(valid_chunk_ids)
    cited = []
    for match in CITATION_RE.findall(answer_text):
        candidate = match.strip()
        if candidate in valid and candidate not in cited:
            cited.append(candidate)
    return cited


def coverage_ratio(normalized_text: str, normalized_targets: list[str]) -> float:
    """Compute substring coverage against expected concepts or phrases."""

    if not normalized_targets:
        return 1.0
    hits = sum(1 for target in normalized_targets if target and target in normalized_text)
    return hits / len(normalized_targets)


def compute_weighted_judge_score(scores: AnswerJudgeScores) -> float:
    """Aggregate judge scores with explicit weights.

    This keeps correctness, completeness, and groundedness dominant while
    reducing the penalty for concise factual answers.
    """

    return (
        scores.correctness * ANSWER_SCORE_WEIGHTS["correctness"]
        + scores.completeness * ANSWER_SCORE_WEIGHTS["completeness"]
        + scores.groundedness * ANSWER_SCORE_WEIGHTS["groundedness"]
        + scores.reasoning * ANSWER_SCORE_WEIGHTS["reasoning"]
        + scores.clarity * ANSWER_SCORE_WEIGHTS["clarity"]
    )


def determine_score_cap(
    *,
    question: EvaluationQuestion,
    scores: AnswerJudgeScores,
    deterministic_checks: DeterministicAnswerChecks,
) -> tuple[float, list[str]]:
    """Apply light deterministic caps so trust signals affect the final score."""

    score_cap = 1.0
    notes: list[str] = []

    if deterministic_checks.citation_precision == 0.0:
        score_cap = min(score_cap, 0.8)
        notes.append("cap_0.80_no_valid_citations")
    elif deterministic_checks.citation_precision < 1.0:
        score_cap = min(score_cap, 0.9)
        notes.append("cap_0.90_imperfect_citation_precision")

    if question.key_concepts:
        if deterministic_checks.answer_key_concept_coverage == 0.0:
            score_cap = min(score_cap, 0.75)
            notes.append("cap_0.75_zero_key_concept_coverage")
        elif deterministic_checks.answer_key_concept_coverage < 0.5:
            score_cap = min(score_cap, 0.9)
            notes.append("cap_0.90_low_key_concept_coverage")

    if (
        question.evidence_phrases
        and deterministic_checks.answer_key_concept_coverage == 0.0
        and deterministic_checks.answer_evidence_phrase_coverage == 0.0
    ):
        score_cap = min(score_cap, 0.7)
        notes.append("cap_0.70_no_key_concepts_or_evidence_phrases")

    if (
        question.category == "direct_factual"
        and scores.correctness >= 0.9
        and scores.completeness >= 0.9
        and scores.groundedness >= 0.9
        and scores.reasoning < 0.5
    ):
        notes.append("direct_factual_brevity_not_over_penalized")

    return score_cap, notes


def determine_winners(method_results: list[AnswerMethodQuestionResult]) -> list[str]:
    """Determine which methods win or tie for one answer question."""

    if not method_results:
        return []
    best_score = max(result.overall_score for result in method_results)
    return sorted(
        result.method_label
        for result in method_results
        if best_score - result.overall_score <= WIN_EPSILON
    )


def build_method_summaries(
    *,
    method_results_map: dict[str, list[AnswerMethodQuestionResult]],
    per_question_comparisons: list[AnswerPerQuestionComparison],
) -> list[AnswerMethodAggregateSummary]:
    """Aggregate answer-level scores by method."""

    wins_by_method: dict[str, list[str]] = defaultdict(list)
    for comparison in per_question_comparisons:
        for winner in comparison.winner_methods:
            wins_by_method[winner].append(comparison.question_id)

    summaries: list[AnswerMethodAggregateSummary] = []
    for method_label in sorted(method_results_map):
        results = method_results_map[method_label]
        by_category: dict[str, list[float]] = defaultdict(list)
        for result in results:
            by_category[result.category].append(result.overall_score)

        summaries.append(
            AnswerMethodAggregateSummary(
                method_label=method_label,
                retrieval_method=results[0].retrieval_method,
                question_count=len(results),
                wins=len(wins_by_method.get(method_label, [])),
                mean_overall_score=round(mean(result.overall_score for result in results), 6),
                mean_correctness=round(mean(result.correctness for result in results), 6),
                mean_completeness=round(mean(result.completeness for result in results), 6),
                mean_reasoning=round(mean(result.reasoning for result in results), 6),
                mean_groundedness=round(mean(result.groundedness for result in results), 6),
                mean_clarity=round(mean(result.clarity for result in results), 6),
                mean_citation_precision=round(
                    mean(result.citation_precision for result in results),
                    6,
                ),
                mean_answer_key_concept_coverage=round(
                    mean(result.answer_key_concept_coverage for result in results),
                    6,
                ),
                mean_answer_evidence_phrase_coverage=round(
                    mean(result.answer_evidence_phrase_coverage for result in results),
                    6,
                ),
                mean_overall_by_category={
                    category: round(mean(scores), 6)
                    for category, scores in sorted(by_category.items())
                },
                question_ids_won=sorted(wins_by_method.get(method_label, [])),
            )
        )
    return summaries


def write_answer_evaluation_artifacts(
    *,
    answers_output_dir: Path,
    evaluation_output_dir: Path,
    questions: list[EvaluationQuestion],
    per_question_comparisons: list[AnswerPerQuestionComparison],
    summary: AnswerEvaluationSummary,
    top_k: int,
    method_specs: list[dict],
) -> None:
    """Write answer-generation and answer-evaluation summary artifacts."""

    (answers_output_dir / "question_set.jsonl").write_text(
        "\n".join(
            json.dumps(question.model_dump(mode="json"), ensure_ascii=False)
            for question in questions
        )
        + "\n",
        encoding="utf-8",
    )
    (answers_output_dir / "config.json").write_text(
        json.dumps(
            {
                "answer_evaluation_version": ANSWER_EVAL_VERSION,
                "run_id": summary.run_id,
                "timestamp_utc": summary.timestamp_utc,
                "top_k": top_k,
                "generation_model": summary.generation_model,
                "judge_model": summary.judge_model,
                "overall_score_weights": ANSWER_SCORE_WEIGHTS,
                "methods": {
                    method_spec["label"]: method_spec["config_value"]
                    for method_spec in method_specs
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (evaluation_output_dir / "aggregate_summary.json").write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    write_jsonl_models(
        evaluation_output_dir / "per_question_results.jsonl",
        per_question_comparisons,
    )
    write_per_question_csv(
        evaluation_output_dir / "per_question_comparison.csv",
        per_question_comparisons,
        [method_spec["label"] for method_spec in method_specs],
    )
    write_method_summary_csv(
        evaluation_output_dir / "method_summary.csv",
        summary,
    )
    (evaluation_output_dir / "summary_report.md").write_text(
        build_summary_report(summary, per_question_comparisons),
        encoding="utf-8",
    )


def write_jsonl_models(path: Path, records) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False))
            handle.write("\n")


def write_per_question_csv(
    output_path: Path,
    per_question_comparisons: list[AnswerPerQuestionComparison],
    method_labels: list[str],
) -> None:
    """Write per-question answer results as CSV."""

    fieldnames = ["question_id", "category", "question", "winner_methods"]
    for label in method_labels:
        fieldnames.extend(
            [
                f"{label}_overall",
                f"{label}_correctness",
                f"{label}_completeness",
                f"{label}_reasoning",
                f"{label}_groundedness",
                f"{label}_clarity",
                f"{label}_citation_precision",
            ]
        )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for comparison in per_question_comparisons:
            row = {
                "question_id": comparison.question_id,
                "category": comparison.category,
                "question": comparison.question,
                "winner_methods": ",".join(comparison.winner_methods),
            }
            by_method = {result.method_label: result for result in comparison.method_results}
            for label in method_labels:
                result = by_method.get(label)
                row[f"{label}_overall"] = result.overall_score if result else ""
                row[f"{label}_correctness"] = result.correctness if result else ""
                row[f"{label}_completeness"] = result.completeness if result else ""
                row[f"{label}_reasoning"] = result.reasoning if result else ""
                row[f"{label}_groundedness"] = result.groundedness if result else ""
                row[f"{label}_clarity"] = result.clarity if result else ""
                row[f"{label}_citation_precision"] = result.citation_precision if result else ""
            writer.writerow(row)


def write_method_summary_csv(output_path: Path, summary: AnswerEvaluationSummary) -> None:
    """Write answer-level aggregate summary as CSV."""

    fieldnames = [
        "method_label",
        "retrieval_method",
        "question_count",
        "wins",
        "mean_overall_score",
        "mean_correctness",
        "mean_completeness",
        "mean_reasoning",
        "mean_groundedness",
        "mean_clarity",
        "mean_citation_precision",
        "mean_answer_key_concept_coverage",
        "mean_answer_evidence_phrase_coverage",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method_summary in summary.method_summaries:
            writer.writerow(method_summary.model_dump(include=set(fieldnames)))


def build_summary_report(
    summary: AnswerEvaluationSummary,
    per_question_comparisons: list[AnswerPerQuestionComparison],
) -> str:
    """Build a compact markdown report for answer-level comparison."""

    lines: list[str] = []
    lines.append("# Answer Evaluation Summary")
    lines.append("")
    lines.append(f"- Run ID: `{summary.run_id}`")
    lines.append(f"- Questions: `{summary.question_count}`")
    lines.append(f"- Top-k retrieved per answer: `{summary.top_k}`")
    lines.append(f"- Generation model: `{summary.generation_model}`")
    lines.append(f"- Judge model: `{summary.judge_model}`")
    lines.append(f"- Answer evaluation version: `{ANSWER_EVAL_VERSION}`")
    lines.append(
        "- Weighted overall: correctness 0.35, completeness 0.25, groundedness 0.25, reasoning 0.10, clarity 0.05"
    )
    lines.append("")
    lines.append("## Aggregate Scores")
    lines.append("")
    lines.append("| Method | Overall | Correctness | Completeness | Reasoning | Groundedness | Clarity | Wins |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for method_summary in sorted(
        summary.method_summaries,
        key=lambda item: item.mean_overall_score,
        reverse=True,
    ):
        lines.append(
            f"| {method_summary.method_label} | {method_summary.mean_overall_score:.3f} | "
            f"{method_summary.mean_correctness:.3f} | {method_summary.mean_completeness:.3f} | "
            f"{method_summary.mean_reasoning:.3f} | {method_summary.mean_groundedness:.3f} | "
            f"{method_summary.mean_clarity:.3f} | {method_summary.wins} |"
        )
    lines.append("")
    lines.append("## Per-Question Winners")
    lines.append("")
    lines.append("| Question ID | Category | Winners |")
    lines.append("| --- | --- | --- |")
    for comparison in per_question_comparisons:
        lines.append(
            f"| {comparison.question_id} | {comparison.category} | {', '.join(comparison.winner_methods)} |"
        )
    lines.append("")
    lines.append("## Method Notes")
    lines.append("")
    for method_summary in sorted(
        summary.method_summaries,
        key=lambda item: item.mean_overall_score,
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
            "- Strongest categories: "
            + ", ".join(f"`{name}` ({score:.3f})" for name, score in strongest)
        )
        lines.append(
            "- Weakest categories: "
            + ", ".join(f"`{name}` ({score:.3f})" for name, score in weakest)
        )
        lines.append(
            f"- Citation precision: `{method_summary.mean_citation_precision:.3f}`"
        )
        lines.append(
            f"- Answer concept coverage: `{method_summary.mean_answer_key_concept_coverage:.3f}`"
        )
        lines.append(
            f"- Questions won: {', '.join(method_summary.question_ids_won) or 'none'}"
        )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def build_answer_run_id() -> str:
    """Create a filesystem-safe run id for answer evaluation."""

    timestamp = _utc_now()
    compact = (
        timestamp.replace("-", "")
        .replace(":", "")
        .replace("+00:00", "Z")
        .replace(".", "")
    )
    return f"answer_eval_{compact}"


def build_answer_record_id(question_id: str, method_label: str) -> str:
    """Stable answer id for one question/method pair."""

    return f"{method_label}__{question_id}"


def build_answer_judgment_id(question_id: str, method_label: str) -> str:
    """Stable answer-evaluation id for one question/method pair."""

    return f"judge__{method_label}__{question_id}"


def mean(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def _resolve_path(path_value: Path | str, project_root: Path = ROOT_DIR) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (project_root / candidate).resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
