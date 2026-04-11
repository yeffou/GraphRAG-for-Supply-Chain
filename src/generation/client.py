"""Small OpenRouter-compatible client for answer generation and judging."""

from __future__ import annotations

import json
import re
import time
from typing import Any

import requests
from pydantic import BaseModel, ConfigDict, Field

from src.config import LLMConfig


JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)
MAX_LLM_RETRIES = 3
RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


class LLMClientError(Exception):
    """Raised when answer generation or judging cannot proceed."""


class ChatMessage(BaseModel):
    """One chat-completion message."""

    model_config = ConfigDict(extra="forbid")

    role: str
    content: str


class ChatCompletionResult(BaseModel):
    """Normalized response from the configured LLM provider."""

    model_config = ConfigDict(extra="forbid")

    model: str
    content: str
    raw_response: dict[str, Any]
    latency_seconds: float = Field(ge=0.0)
    usage: dict[str, Any] | None = None


def require_generation_config(llm_config: LLMConfig) -> str:
    """Validate that answer generation config is present."""

    if not llm_config.openrouter_api_key:
        raise LLMClientError(
            "Missing OPENROUTER_API_KEY. Add it to a local .env file or export it in your shell."
        )
    if not llm_config.openrouter_model:
        raise LLMClientError(
            "Missing OPENROUTER_MODEL. Add it to a local .env file or export it in your shell."
        )
    return llm_config.openrouter_model


def require_judge_config(llm_config: LLMConfig) -> str:
    """Validate that judge-model config is present."""

    if not llm_config.openrouter_api_key:
        raise LLMClientError(
            "Missing OPENROUTER_API_KEY. Add it to a local .env file or export it in your shell."
        )
    judge_model = llm_config.openrouter_judge_model or llm_config.openrouter_model
    if not judge_model:
        raise LLMClientError(
            "Missing OPENROUTER_JUDGE_MODEL or OPENROUTER_MODEL. "
            "Configure at least one of them in .env."
        )
    return judge_model


def call_chat_completion(
    *,
    llm_config: LLMConfig,
    model: str,
    messages: list[ChatMessage],
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> ChatCompletionResult:
    """Call the configured OpenRouter-compatible chat endpoint."""

    if not llm_config.openrouter_api_key:
        raise LLMClientError(
            "Missing OPENROUTER_API_KEY. Add it to a local .env file or export it in your shell."
        )

    endpoint = f"{llm_config.openrouter_base_url.rstrip('/')}/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [message.model_dump(mode="json") for message in messages],
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {llm_config.openrouter_api_key}",
    }

    last_error: Exception | None = None
    response = None
    response_payload: dict[str, Any] | None = None
    latency_seconds = 0.0

    for attempt in range(1, MAX_LLM_RETRIES + 1):
        started_at = time.perf_counter()
        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=llm_config.llm_timeout_seconds,
            )
        except requests.RequestException as exc:
            last_error = exc
            if attempt == MAX_LLM_RETRIES:
                raise LLMClientError(
                    f"LLM request failed: {exc.__class__.__name__}: {exc}"
                ) from exc
            time.sleep(2 ** (attempt - 1))
            continue

        latency_seconds = time.perf_counter() - started_at

        try:
            response_payload = response.json()
        except json.JSONDecodeError as exc:
            raise LLMClientError(
                f"LLM response was not valid JSON (status={response.status_code})."
            ) from exc

        if response.status_code >= 400:
            provider_error = response_payload.get("error")
            if response.status_code in RETRYABLE_STATUS_CODES and attempt < MAX_LLM_RETRIES:
                time.sleep(2 ** (attempt - 1))
                continue
            raise LLMClientError(
                "LLM request failed with status "
                f"{response.status_code}: {provider_error or response.text}"
            )

        break

    if response is None or response_payload is None:
        if last_error is not None:
            raise LLMClientError(f"LLM request failed: {last_error.__class__.__name__}: {last_error}")
        raise LLMClientError("LLM request failed before a response was received.")

    try:
        content = response_payload["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, AttributeError) as exc:
        raise LLMClientError("LLM response format was not as expected.") from exc

    return ChatCompletionResult(
        model=model,
        content=content,
        raw_response=response_payload,
        latency_seconds=latency_seconds,
        usage=response_payload.get("usage"),
    )


def parse_json_object_from_text(text: str) -> dict[str, Any]:
    """Extract a JSON object from model output."""

    stripped = text.strip()
    match = JSON_BLOCK_RE.search(stripped)
    candidate = match.group(1) if match else stripped

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise LLMClientError("Judge output did not contain a parseable JSON object.")
        try:
            return json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMClientError("Judge output JSON could not be parsed.") from exc
