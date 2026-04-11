"""Central configuration for project paths and environment settings."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - fallback for pre-install validation
    def load_dotenv(*args: object, **kwargs: object) -> bool:
        return False


ROOT_DIR = Path(__file__).resolve().parents[1]


class ProjectPaths(BaseModel):
    """Filesystem locations used by the pipeline."""

    root_dir: Path = ROOT_DIR
    data_dir: Path = ROOT_DIR / "data"
    raw_data_dir: Path = ROOT_DIR / "data" / "raw"
    processed_data_dir: Path = ROOT_DIR / "data" / "processed"
    evaluation_data_dir: Path = ROOT_DIR / "data" / "evaluation"
    manifest_path: Path = ROOT_DIR / "data" / "manifest.jsonl"
    results_dir: Path = ROOT_DIR / "results"
    indexes_dir: Path = ROOT_DIR / "results" / "indexes"
    graphs_dir: Path = ROOT_DIR / "results" / "graphs"
    runs_dir: Path = ROOT_DIR / "results" / "runs"
    evaluations_dir: Path = ROOT_DIR / "results" / "evaluations"
    answers_dir: Path = ROOT_DIR / "results" / "answers"
    answer_evaluation_dir: Path = ROOT_DIR / "results" / "answer_evaluation"


class LLMConfig(BaseModel):
    """Environment-backed LLM settings.

    These values are read but not used until generation or extraction modules
    are implemented.
    """

    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str | None = None
    openrouter_judge_model: str | None = None
    llm_timeout_seconds: int = 120


class ProjectConfig(BaseModel):
    """Top-level project configuration."""

    paths: ProjectPaths = Field(default_factory=ProjectPaths)
    llm: LLMConfig = Field(default_factory=LLMConfig)


def load_config(env_file: Path | None = None) -> ProjectConfig:
    """Load project configuration from the environment."""

    load_dotenv(env_file or ROOT_DIR / ".env")

    return ProjectConfig(
        llm=LLMConfig(
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY") or None,
            openrouter_base_url=os.getenv(
                "OPENROUTER_BASE_URL",
                "https://openrouter.ai/api/v1",
            ),
            openrouter_model=os.getenv("OPENROUTER_MODEL") or None,
            openrouter_judge_model=os.getenv("OPENROUTER_JUDGE_MODEL") or None,
            llm_timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", "120")),
        )
    )
