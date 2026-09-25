"""Validates a run request, builds its classifier and meta, and starts it as a background job.

Classes:
    RunRequest: what the UI / CLI asks for.
    RunLaunchError: invalid request (unknown column/generation/model, empty email set, bad mode).
Functions:
    launch_run: persist the initial RunMeta and start the job; returns (meta, task). Raises
        RunLaunchError for a blank api_key, before any meta is saved.
    build_classifier: column + model + mode -> Classifier. Chat parameters the model's catalog
        entry does not list (`temperature`, `reasoning`) are not sent, and the run snapshot
        records them as None; with no catalog entry everything configured is sent.
"""

import asyncio
import tomllib
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from jev_bench.benchmark_config import (
    BenchmarkConfig,
    ChatMode,
    ColumnConfig,
    EmbeddingParams,
    JevParams,
    LlmParams,
)
from jev_bench.catalog import Catalog, ModelInfo
from jev_bench.classifiers.base import Classifier
from jev_bench.classifiers.embeddings import EmbeddingClassifier
from jev_bench.classifiers.jev import JevClassifier
from jev_bench.classifiers.llm import LlmClassifier
from jev_bench.emails import Email
from jev_bench.ids import new_id
from jev_bench.openrouter import OpenRouterError
from jev_bench.questions import QuestionSet
from jev_bench.runner import execute_run
from jev_bench.store.runs import RunMeta, RunMode

__all__ = [
    "RunLaunchError",
    "RunRequest",
    "build_classifier",
    "launch_run",
]

if TYPE_CHECKING:
    from jev_bench.services import Services


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str
    model: str | None = None
    generation_ids: tuple[str, ...] = Field(min_length=1)
    mode: ChatMode | None = None


class RunLaunchError(ValueError):
    pass


async def launch_run(
    request: RunRequest, api_key: str, services: Services, *, now: datetime | None = None
) -> tuple[RunMeta, asyncio.Task[None]]:
    if not api_key.strip():
        raise RunLaunchError("an OpenRouter API key is required")
    config = _load_config(services.benchmark_config)
    column = _column(config, request.column)
    mode = _mode(column, request.mode)
    model = request.model or column.default_model
    emails = _emails(services, request.generation_ids)
    info = await _model_info(services.catalog, column, model)
    questions = _load_config(services.question_set)
    classifier = build_classifier(column, model, mode, info, questions, config, services, api_key)
    params = _params(column, config, model, info)
    meta = _new_meta(
        column, model, mode, request.generation_ids, questions, params, classifier, len(emails), now
    )
    services.runs.save(meta)
    task = services.jobs.start(
        meta.id,
        len(emails),
        lambda progress: execute_run(meta, emails, classifier, services.runs, progress),
    )
    return meta, task


def _load_config[T](loader: Callable[[], T]) -> T:
    try:
        return loader()
    except (ValidationError, tomllib.TOMLDecodeError) as exc:
        raise RunLaunchError(f"invalid config: {exc}") from exc


def _column(config: BenchmarkConfig, column_id: str) -> ColumnConfig:
    try:
        return config.column(column_id)
    except KeyError as exc:
        raise RunLaunchError(f"unknown column {column_id!r}") from exc


def _mode(column: ColumnConfig, requested: ChatMode | None) -> RunMode:
    if column.kind == "chat":
        return requested or "per_email"
    if requested is not None:
        raise RunLaunchError(f"mode is only accepted for chat columns, not {column.id!r}")
    return "batched" if column.kind == "embeddings" else "per_email"


def _emails(services: Services, generation_ids: Sequence[str]) -> list[Email]:
    try:
        emails = services.generations.emails_for(generation_ids)
    except KeyError as exc:
        raise RunLaunchError(f"unknown generation {exc.args[0]!r}") from exc
    if not emails:
        raise RunLaunchError("the selected generations contain no emails")
    return emails


async def _model_info(catalog: Catalog, column: ColumnConfig, model: str) -> ModelInfo | None:
    try:
        models = await catalog.for_column(column)
    except OpenRouterError as exc:
        logger.bind(column=column.id).warning("catalog unavailable, using fallback limits: {}", exc)
        return None
    found = next((candidate for candidate in models if candidate.id == model), None)
    if found is None:
        raise RunLaunchError(f"model {model!r} is not available for column {column.id!r}")
    return found


def build_classifier(
    column: ColumnConfig,
    model: str,
    mode: RunMode,
    model_info: ModelInfo | None,
    questions: QuestionSet,
    config: BenchmarkConfig,
    services: Services,
    api_key: str,
) -> Classifier:
    client = services.client
    if column.kind == "decisions":
        return JevClassifier(
            client=client,
            api_key=api_key,
            model=model,
            model_info=model_info,
            questions=questions,
            params=config.jev,
            tokens=config.tokens,
        )
    if column.kind == "embeddings":
        cache = services.embedding_caches.for_model(model)
        return EmbeddingClassifier(
            client=client,
            api_key=api_key,
            model=model,
            model_info=model_info,
            questions=questions,
            params=config.embeddings,
            tokens=config.tokens,
            cache=cache,
        )
    chat_mode: ChatMode = "all_in_one" if mode == "all_in_one" else "per_email"
    return LlmClassifier(
        client=client,
        api_key=api_key,
        model=model,
        model_info=model_info,
        questions=questions,
        params=_fit_chat_params(config.llm, model_info),
        tokens=config.tokens,
        mode=chat_mode,
        cache_system_prompt=column.cache_system_prompt,
    )


_OPTIONAL_CHAT_PARAMETERS = {"temperature": "temperature", "reasoning_enabled": "reasoning"}


def _params(
    column: ColumnConfig, config: BenchmarkConfig, model: str, model_info: ModelInfo | None
) -> JevParams | LlmParams | EmbeddingParams:
    if column.kind == "decisions":
        return config.jev
    if column.kind == "embeddings":
        return config.embeddings
    fitted = _fit_chat_params(config.llm, model_info)
    unsent = [name for name in _OPTIONAL_CHAT_PARAMETERS if getattr(fitted, name) is None]
    if fitted != config.llm:
        logger.bind(model=model).info(
            "{} does not accept {}; not sending it", model, ", ".join(unsent)
        )
    return fitted


def _fit_chat_params(params: LlmParams, model_info: ModelInfo | None) -> LlmParams:
    supported = set(model_info.supported_parameters) if model_info else set()
    if not supported:
        return params
    unsupported = {
        field: None for field, name in _OPTIONAL_CHAT_PARAMETERS.items() if name not in supported
    }
    return params.model_copy(update=unsupported)


def _new_meta(
    column: ColumnConfig,
    model: str,
    mode: RunMode,
    generation_ids: Sequence[str],
    questions: QuestionSet,
    params: JevParams | LlmParams | EmbeddingParams,
    classifier: Classifier,
    n_emails: int,
    now: datetime | None,
) -> RunMeta:
    created = now or datetime.now(UTC)
    return RunMeta(
        id=new_id(f"{column.id}-{model}", created),
        column=column.id,
        column_config=column,
        kind=column.kind,
        model=model,
        generation_ids=tuple(dict.fromkeys(generation_ids)),
        mode=mode,
        emails_per_request=classifier.emails_per_request,
        question_set=questions,
        params=params,
        concurrency=classifier.concurrency,
        created_at=created,
        n_emails=n_emails,
    )
