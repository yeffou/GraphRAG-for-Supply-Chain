"""Prompt builders for grounded answer generation and answer judging."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from src.generation.client import ChatMessage


ANSWER_PROMPT_TEMPLATE_VERSION = "grounded_answer_v1"
JUDGE_PROMPT_TEMPLATE_VERSION = "answer_judge_v2_calibrated"


class PromptBundle(BaseModel):
    """Structured prompt payload saved with artifacts."""

    model_config = ConfigDict(extra="forbid")

    template_version: str
    system_prompt: str
    user_prompt: str


def build_grounded_answer_prompt(
    *,
    question: str,
    retrieved_chunks: Sequence,
    method_label: str,
) -> PromptBundle:
    """Create a grounded answer prompt from retrieved evidence."""

    evidence_lines = []
    for chunk in retrieved_chunks:
        evidence_lines.append(
            f"[{chunk.chunk_id}] doc_id={chunk.doc_id} title={chunk.title} "
            f"page={chunk.page_number} source_url={chunk.source_url}\n"
            f"{chunk.text}"
        )

    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Retrieval method: {method_label}\n\n"
        "Retrieved evidence:\n"
        + "\n\n".join(evidence_lines)
        + "\n\n"
        "Write a concise answer using only the retrieved evidence.\n"
        "Rules:\n"
        "1. Do not use outside knowledge.\n"
        "2. If the evidence is insufficient or conflicting, say so clearly.\n"
        "3. Support each substantive claim with chunk-id citations like [chunk_id].\n"
        "4. Do not cite chunks that do not support the claim.\n"
    )
    system_prompt = (
        "You are answering a supply-chain risk and resilience question from retrieved evidence. "
        "Stay grounded in the provided context and prefer precise, supported statements."
    )
    return PromptBundle(
        template_version=ANSWER_PROMPT_TEMPLATE_VERSION,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


def build_answer_judge_prompt(
    *,
    question_id: str,
    question: str,
    category: str,
    answer_text: str,
    retrieved_chunks: Sequence,
    gold_doc_ids: Sequence[str],
    gold_chunk_ids: Sequence[str],
    key_concepts: Sequence[str],
    evidence_phrases: Sequence[str],
    citation_count: int,
    valid_citation_count: int,
    citation_precision: float,
    answer_key_concept_coverage: float,
    answer_evidence_phrase_coverage: float,
    answer_mentions_gold_doc: bool,
) -> PromptBundle:
    """Create a rubric-driven judging prompt with strict JSON output instructions."""

    retrieved_context_lines = []
    for chunk in retrieved_chunks:
        retrieved_context_lines.append(
            f"[{chunk.chunk_id}] doc_id={chunk.doc_id} title={chunk.title} "
            f"page={chunk.page_number} source_url={chunk.source_url}\n"
            f"{chunk.text}"
        )

    user_prompt = (
        "Evaluate one generated answer for a domain QA system.\n\n"
        f"Question ID: {question_id}\n"
        f"Category: {category}\n"
        f"Question: {question}\n\n"
        "Retrieved context:\n"
        + "\n\n".join(retrieved_context_lines)
        + "\n\n"
        f"Reference gold documents: {', '.join(gold_doc_ids)}\n"
        f"Reference gold chunks: {', '.join(gold_chunk_ids)}\n"
        f"Key concepts: {', '.join(key_concepts)}\n"
        f"Evidence phrases: {', '.join(evidence_phrases)}\n\n"
        "Deterministic answer checks:\n"
        f"- citation_count: {citation_count}\n"
        f"- valid_citation_count: {valid_citation_count}\n"
        f"- citation_precision: {citation_precision:.3f}\n"
        f"- answer_key_concept_coverage: {answer_key_concept_coverage:.3f}\n"
        f"- answer_evidence_phrase_coverage: {answer_evidence_phrase_coverage:.3f}\n"
        f"- answer_mentions_gold_doc: {str(answer_mentions_gold_doc).lower()}\n\n"
        "Generated answer:\n"
        f"{answer_text}\n\n"
        "Score the answer from 0.0 to 1.0 on:\n"
        "- correctness: factual accuracy relative to the evidence and reference evidence\n"
        "- completeness: whether the main requested points are covered\n"
        "- reasoning: whether the answer responds appropriately for the question type. "
        "For direct factual questions, a concise correct answer can still score highly on reasoning; "
        "do not require unnecessary explanation.\n"
        "- groundedness: whether the answer stays faithful to the retrieved context\n"
        "- clarity: whether the answer is readable and well-structured\n\n"
        "Use these scoring anchors:\n"
        "- 1.0 only for answers that are fully correct, complete, strongly grounded, and well cited.\n"
        "- 0.9 for very strong answers with only a minor omission.\n"
        "- 0.75 for good but clearly imperfect answers.\n"
        "- 0.5 for partial or mixed answers.\n"
        "- 0.25 for weak answers.\n"
        "- 0.0 for unsupported or incorrect answers.\n\n"
        "Hard guardrails:\n"
        "- Do not give any category a 1.0 if citation_precision is below 1.0.\n"
        "- Do not give completeness or groundedness a 1.0 if answer_key_concept_coverage is below 0.5.\n"
        "- Do not give overall-perfect category scores when the answer misses most key concepts or evidence phrases.\n"
        "- If the answer is concise but directly and correctly answers a factual question from the evidence, that is acceptable and should not be penalized merely for brevity.\n"
        "- Be conservative about 1.0 scores. Most acceptable answers should score below 1.0.\n\n"
        "The final overall score will be computed downstream with these weights:\n"
        "- correctness: 0.35\n"
        "- completeness: 0.25\n"
        "- groundedness: 0.25\n"
        "- reasoning: 0.10\n"
        "- clarity: 0.05\n\n"
        "Return JSON only with this schema:\n"
        "{\n"
        '  "correctness": 0.0,\n'
        '  "completeness": 0.0,\n'
        '  "reasoning": 0.0,\n'
        '  "groundedness": 0.0,\n'
        '  "clarity": 0.0,\n'
        '  "judge_explanation": "short explanation"\n'
        "}\n"
        "Use numeric scores only, no markdown, no extra keys."
    )
    system_prompt = (
        "You are a strict evaluator for grounded question answering. "
        "Judge only from the retrieved context and the provided reference evidence summary. "
        "Be conservative about giving high scores when support, concept coverage, or citation quality is weak."
    )
    return PromptBundle(
        template_version=JUDGE_PROMPT_TEMPLATE_VERSION,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


def prompt_to_messages(prompt: PromptBundle) -> list[ChatMessage]:
    """Convert a structured prompt into chat messages."""

    return [
        ChatMessage(role="system", content=prompt.system_prompt),
        ChatMessage(role="user", content=prompt.user_prompt),
    ]
