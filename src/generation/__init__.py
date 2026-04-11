"""Generation helpers for grounded answering and judging."""

from src.generation.client import (
    ChatCompletionResult,
    ChatMessage,
    LLMClientError,
    call_chat_completion,
    parse_json_object_from_text,
    require_generation_config,
    require_judge_config,
)
from src.generation.prompts import (
    ANSWER_PROMPT_TEMPLATE_VERSION,
    JUDGE_PROMPT_TEMPLATE_VERSION,
    PromptBundle,
    build_answer_judge_prompt,
    build_grounded_answer_prompt,
    prompt_to_messages,
)

__all__ = [
    "ANSWER_PROMPT_TEMPLATE_VERSION",
    "JUDGE_PROMPT_TEMPLATE_VERSION",
    "ChatCompletionResult",
    "ChatMessage",
    "LLMClientError",
    "PromptBundle",
    "build_answer_judge_prompt",
    "build_grounded_answer_prompt",
    "call_chat_completion",
    "parse_json_object_from_text",
    "prompt_to_messages",
    "require_generation_config",
    "require_judge_config",
]
