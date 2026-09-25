"""Validates an analysis request, estimates its prompt and cost, and starts it as a background job.

Classes:
    AnalysisRequest: the runs to analyse plus an optional analyst model, prompt templates, label
        threshold and disputed-email count (the config's defaults otherwise).
    AnalysisEstimate: input tokens (the pessimistic byte-based estimate of `tokens.py`), the
        output-token budget (the config's, capped by the model's), the upper-bound cost (None
        without catalog prices) and whether both fit the model's context.
    AnalysisLaunchError: invalid request (unreadable config, unknown model, invalid template, a
        prompt that does not fit the model's context).
Functions:
    estimate_analysis: AnalysisEstimate for a request over a source.
    launch_analysis: persist the initial record and start the job; returns (meta, task).
        `temperature` is sent only when the model's catalog entry lists it (or the catalog is
        unavailable); `reasoning` is left at the model's default.
"""

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, NamedTuple

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from jev_bench.analysis.config import AnalysisConfig, check_prompt
from jev_bench.analysis.context import PromptInputs, prompt_inputs, render_prompts
from jev_bench.analysis.job import CHARS_PER_TOKEN, AnalysisDeps, execute_analysis
from jev_bench.analysis.source import AnalysisSource
from jev_bench.benchmark_config import TokenParams
from jev_bench.catalog import Catalog, ModelInfo
from jev_bench.ids import new_id
from jev_bench.openrouter import JsonObject, OpenRouterError
from jev_bench.store.analyses import AnalysisMeta
from jev_bench.tokens import estimate_tokens

if TYPE_CHECKING:
    from jev_bench.services import Services

__all__ = [
    "AnalysisEstimate",
    "AnalysisLaunchError",
    "AnalysisRequest",
    "estimate_analysis",
    "launch_analysis",
]


class AnalysisLaunchError(ValueError):
    pass


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runs: tuple[Annotated[str, Field(min_length=1)], ...] = Field(min_length=1, max_length=50)
    model: str | None = Field(default=None, min_length=1)
    system_prompt: str | None = None
    user_prompt: str | None = None
    threshold: float | None = Field(default=None, gt=0, le=1, allow_inf_nan=False)
    max_disputed_emails: int | None = Field(default=None, ge=0, le=100)


class AnalysisEstimate(BaseModel):
    model: str
    input_tokens: int
    max_output_tokens: int
    context_length: int
    fits: bool
    cost: float | None
    n_emails: int
    n_disputed: int


class _Prepared(NamedTuple):
    config: AnalysisConfig
    info: ModelInfo | None
    system: str
    user: str
    inputs: PromptInputs
    estimate: AnalysisEstimate


async def estimate_analysis(
    request: AnalysisRequest, source: AnalysisSource, services: Services
) -> AnalysisEstimate:
    return (await _prepare(request, source, services)).estimate


async def launch_analysis(
    request: AnalysisRequest,
    source: AnalysisSource,
    api_key: str,
    services: Services,
    *,
    now: datetime | None = None,
) -> tuple[AnalysisMeta, asyncio.Task[None]]:
    prepared = await _prepare(request, source, services)
    if not prepared.estimate.fits:
        raise AnalysisLaunchError(_too_long(prepared.estimate))
    meta = _new_meta(request, source, prepared, now or datetime.now(UTC))
    services.analyses.save(meta)
    config = prepared.config
    deps = AnalysisDeps(
        services.client, SecretStr(api_key), services.analyses, prepared.info, config.timeout_s
    )
    body = _body(prepared, meta)
    total = meta.max_output_tokens * CHARS_PER_TOKEN
    task = services.jobs.start(
        meta.id, total, lambda progress: execute_analysis(meta, body, deps, progress)
    )
    return meta, task


async def _prepare(
    request: AnalysisRequest, source: AnalysisSource, services: Services
) -> _Prepared:
    config, limits = _configs(services)
    model = request.model or config.default_model
    info = await _model_info(services.catalog, model)
    disputed = config.max_disputed_emails
    inputs = prompt_inputs(source, max_disputed=_given(request.max_disputed_emails, disputed))
    system_template = _template(request.system_prompt, config.system_prompt)
    user_template = _template(request.user_prompt, config.user_prompt)
    system, user = render_prompts(system_template, user_template, inputs.values)
    counts = (len(source.emails), inputs.n_disputed)
    estimate = _estimate(model, info, limits, config.max_output_tokens, system + user, counts)
    return _Prepared(config, info, system, user, inputs, estimate)


def _configs(services: Services) -> tuple[AnalysisConfig, TokenParams]:
    try:
        return services.analysis_config(), services.benchmark_config().tokens
    except (ValueError, OSError) as exc:
        raise AnalysisLaunchError(f"invalid analysis or benchmark config: {exc}") from exc


def _given(value: int | None, default: int) -> int:
    return default if value is None else value


async def _model_info(catalog: Catalog, model: str) -> ModelInfo | None:
    try:
        info = await catalog.find("text", model)
    except OpenRouterError as exc:
        logger.bind(model=model).warning("catalog unavailable, using fallback limits: {}", exc)
        return None
    if info is None:
        raise AnalysisLaunchError(f"model {model!r} is not in the OpenRouter catalog")
    return info


def _template(given: str | None, default: str) -> str:
    if given is None:
        return default
    try:
        return check_prompt(given)
    except ValueError as exc:
        raise AnalysisLaunchError(str(exc)) from exc


def _estimate(
    model: str,
    info: ModelInfo | None,
    limits: TokenParams,
    budget: int,
    prompt: str,
    counts: tuple[int, int],
) -> AnalysisEstimate:
    input_tokens = estimate_tokens(prompt, limits.bytes_per_token)
    output = min(budget, info.max_completion_tokens or budget) if info else budget
    context = (info.context_length if info else None) or limits.fallback_context_length
    return AnalysisEstimate(
        model=model,
        input_tokens=input_tokens,
        max_output_tokens=output,
        context_length=context,
        fits=input_tokens + output <= context,
        cost=info.estimate_cost(input_tokens, output) if info else None,
        n_emails=counts[0],
        n_disputed=counts[1],
    )


def _too_long(estimate: AnalysisEstimate) -> str:
    return (
        f"the prompt (~{estimate.input_tokens:,} tokens) plus the output budget "
        f"({estimate.max_output_tokens:,}) does not fit {estimate.model}'s context of "
        f"{estimate.context_length:,} tokens: include fewer disputed emails or runs, or choose a "
        "model with a larger context"
    )


def _temperature(config: AnalysisConfig, info: ModelInfo | None) -> float | None:
    supported = set(info.supported_parameters) if info else set()
    return config.temperature if not supported or "temperature" in supported else None


def _new_meta(
    request: AnalysisRequest, source: AnalysisSource, prepared: _Prepared, created: datetime
) -> AnalysisMeta:
    return AnalysisMeta(
        id=new_id("analysis", created),
        created_at=created,
        model=prepared.estimate.model,
        run_ids=tuple(meta.id for meta in source.metas),
        generation_ids=tuple(generation.id for generation in source.generations),
        threshold=request.threshold,
        system_prompt=prepared.system,
        user_prompt=prepared.user,
        email_refs=prepared.inputs.email_refs,
        n_emails=len(source.emails),
        n_disputed=prepared.inputs.n_disputed,
        max_output_tokens=prepared.estimate.max_output_tokens,
        temperature=_temperature(prepared.config, prepared.info),
    )


def _body(prepared: _Prepared, meta: AnalysisMeta) -> JsonObject:
    messages = [
        {"role": "system", "content": prepared.system},
        {"role": "user", "content": prepared.user},
    ]
    body: JsonObject = {
        "model": meta.model,
        "messages": messages,
        "max_tokens": meta.max_output_tokens,
    }
    return body if meta.temperature is None else {**body, "temperature": meta.temperature}
