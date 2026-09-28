"""Benchmark configuration (`config/benchmark.toml`): columns and per-kind request parameters.

Types:
    ColumnKind, ChatMode, Modality
Constants:
    EMAIL_PLACEHOLDERS, OPTION_PLACEHOLDERS: placeholders allowed in embedding templates.
Classes:
    ColumnConfig: one UI column (kind, catalog filter, default model, card slot: columns sharing a
        slot share one Benchmark card; `effective_slot` defaults to the column id).
    JevParams, LlmParams, EmbeddingParams: per-kind request parameters (snapshotted into runs;
        an LlmParams `temperature` / `reasoning_enabled` of None means "not sent" because the
        model does not support it).
    TokenParams: token-estimate and fallback-limit settings.
    BenchmarkConfig: the whole file.
Functions:
    load_benchmark_config: parse and validate the TOML file.
"""

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from jev_bench.templates import check_template

__all__ = [
    "EMAIL_PLACEHOLDERS",
    "OPTION_PLACEHOLDERS",
    "BenchmarkConfig",
    "ChatMode",
    "ColumnConfig",
    "ColumnKind",
    "EmbeddingParams",
    "JevParams",
    "LlmParams",
    "Modality",
    "TokenParams",
    "load_benchmark_config",
]

type ColumnKind = Literal["decisions", "chat", "embeddings"]
type ChatMode = Literal["per_email", "all_in_one"]
type Modality = Literal["text", "decisions", "embeddings"]

EMAIL_PLACEHOLDERS: tuple[str, ...] = ("sent_at", "from", "to", "cc", "subject", "body")
OPTION_PLACEHOLDERS: tuple[str, ...] = ("instructions", "option", "description")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


_SLUG = r"^[a-z][a-z0-9_]*$"


class ColumnConfig(_Frozen):
    id: str = Field(pattern=_SLUG)
    title: str
    kind: ColumnKind
    modality: Modality
    prefix: str | None = None
    default_model: str
    slot: str | None = Field(default=None, pattern=_SLUG)
    cache_system_prompt: bool = False

    @property
    def effective_slot(self) -> str:
        return self.slot or self.id


class JevParams(_Frozen):
    kind: Literal["decisions"] = "decisions"
    concurrency: int = Field(default=8, ge=1)


class LlmParams(_Frozen):
    kind: Literal["chat"] = "chat"
    system_prompt: str
    system_prompt_all_in_one: str
    temperature: float | None = Field(default=0.0, ge=0)
    reasoning_enabled: bool | None = False
    concurrency: int = Field(default=8, ge=1)
    all_in_one_timeout_s: float = Field(default=1800.0, gt=0)
    est_output_tokens_per_email: int = Field(default=400, ge=1)

    @field_validator("system_prompt", "system_prompt_all_in_one")
    @classmethod
    def _prompt_placeholders(cls, value: str) -> str:
        return check_template(value, ("questions",))


class EmbeddingParams(_Frozen):
    kind: Literal["embeddings"] = "embeddings"
    email_template: str
    option_template: str
    temperature: float = Field(default=0.05, gt=0)
    emails_per_request: int = Field(default=32, ge=1)
    max_request_tokens: int = Field(default=100_000, ge=1)
    concurrency: int = Field(default=4, ge=1)

    @field_validator("email_template")
    @classmethod
    def _email_placeholders(cls, value: str) -> str:
        return check_template(value, EMAIL_PLACEHOLDERS)

    @field_validator("option_template")
    @classmethod
    def _option_placeholders(cls, value: str) -> str:
        return check_template(value, OPTION_PLACEHOLDERS)


class TokenParams(_Frozen):
    bytes_per_token: float = Field(default=3.0, gt=0)
    jev_output_reserve: int = Field(default=1000, ge=0)
    fallback_context_length: int = Field(default=32_000, ge=1)
    fallback_max_completion_tokens: int = Field(default=4_096, ge=1)


class BenchmarkConfig(_Frozen):
    jev: JevParams = JevParams()
    llm: LlmParams
    embeddings: EmbeddingParams
    tokens: TokenParams = TokenParams()
    columns: tuple[ColumnConfig, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_columns(self) -> BenchmarkConfig:
        ids = [column.id for column in self.columns]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate column ids")
        return self

    def column(self, column_id: str) -> ColumnConfig:
        for column in self.columns:
            if column.id == column_id:
                return column
        raise KeyError(column_id)


def load_benchmark_config(path: Path) -> BenchmarkConfig:
    return BenchmarkConfig.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))
