"""Analysis configuration: the default analyst model, input and output limits, and the editable
prompt templates (`config/analysis.toml`).

Constants:
    PLACEHOLDERS: the placeholders a prompt template may use.
Classes:
    AnalysisConfig: the whole file (both templates validated against PLACEHOLDERS). `temperature` is
        left out of the request when the model's catalog entry does not list it.
Functions:
    check_prompt: validate one (possibly user-edited) template against PLACEHOLDERS.
    load_analysis_config: parse and validate the TOML file.
"""

import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from jev_bench.templates import check_template

__all__ = [
    "PLACEHOLDERS",
    "AnalysisConfig",
    "check_prompt",
    "load_analysis_config",
]

PLACEHOLDERS = ("generations", "runs", "questions", "report", "emails", "disputed")


def check_prompt(template: str) -> str:
    return check_template(template, PLACEHOLDERS)


class AnalysisConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    default_model: str = Field(min_length=1)
    max_disputed_emails: int = Field(default=12, ge=0, le=100)
    max_output_tokens: int = Field(default=8000, ge=1)
    temperature: float | None = Field(default=0.2, ge=0)
    timeout_s: float = Field(default=900.0, gt=0)
    system_prompt: str
    user_prompt: str

    @field_validator("system_prompt", "user_prompt")
    @classmethod
    def _valid_template(cls, value: str) -> str:
        return check_prompt(value)


def load_analysis_config(path: Path) -> AnalysisConfig:
    return AnalysisConfig.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))
