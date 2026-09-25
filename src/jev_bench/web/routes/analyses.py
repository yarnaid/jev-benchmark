"""Analysis routes: defaults, estimate, create (background job), list, read with live progress,
cancel. Building the comparison behind an analysis runs in a worker thread (bootstrap CIs are
CPU-bound).

Constants:
    LIGHT: record fields left out of list and create responses (prompts, result, refs).
    NO_PROMPTS: the prompts, left out of the record a running analysis is polled for.
Classes:
    AnalysisDefaults: the config's analyst model, prompt templates and limits, plus the
        placeholders a template may use.
    AnalysisView: record + live progress (characters received so far).
    AnalysisPrompts: the prompts actually sent (read on demand).
Functions:
    analysis_defaults, estimate, create_analysis, list_analyses, get_analysis, analysis_prompts,
        cancel_analysis: route handlers. An invalid request is HTTP 400, an unknown run or
        analysis 404.
"""

import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from jev_bench.analysis.config import PLACEHOLDERS, AnalysisConfig
from jev_bench.analysis.launcher import (
    AnalysisEstimate,
    AnalysisLaunchError,
    AnalysisRequest,
    estimate_analysis,
    launch_analysis,
)
from jev_bench.analysis.source import AnalysisSource
from jev_bench.jobs import ProgressView
from jev_bench.services import Services
from jev_bench.store.analyses import AnalysisMeta
from jev_bench.web.deps import ApiKeyDep, ServicesDep
from jev_bench.web.loaders import load_analysis_source
from jev_bench.web.views import CancelView, load_or_404

__all__ = [
    "LIGHT",
    "NO_PROMPTS",
    "AnalysisDefaults",
    "AnalysisPrompts",
    "AnalysisView",
    "analysis_defaults",
    "analysis_prompts",
    "cancel_analysis",
    "create_analysis",
    "estimate",
    "get_analysis",
    "list_analyses",
    "router",
]

router = APIRouter(tags=["analyses"])
LIGHT = {"meta": {"system_prompt", "user_prompt", "result", "email_refs"}}
NO_PROMPTS = {"meta": {"system_prompt", "user_prompt"}}


class AnalysisDefaults(BaseModel):
    default_model: str
    system_prompt: str
    user_prompt: str
    max_disputed_emails: int
    max_output_tokens: int
    placeholders: tuple[str, ...]


class AnalysisView(BaseModel):
    meta: AnalysisMeta
    progress: ProgressView | None


class AnalysisPrompts(BaseModel):
    system_prompt: str
    user_prompt: str


def _view(services: Services, meta: AnalysisMeta) -> AnalysisView:
    return AnalysisView(meta=meta, progress=services.jobs.progress(meta.id))


def _config(services: Services) -> AnalysisConfig:
    try:
        return services.analysis_config()
    except (ValueError, OSError) as exc:
        detail = f"config/analysis.toml is invalid: {exc}"
        raise HTTPException(status_code=500, detail=detail) from exc


async def _source(services: Services, request: AnalysisRequest) -> AnalysisSource:
    return await asyncio.to_thread(
        load_analysis_source, services, list(request.runs), request.threshold
    )


@router.get("/analysis/defaults")
def analysis_defaults(services: ServicesDep) -> AnalysisDefaults:
    config = _config(services)
    return AnalysisDefaults(
        default_model=config.default_model,
        system_prompt=config.system_prompt,
        user_prompt=config.user_prompt,
        max_disputed_emails=config.max_disputed_emails,
        max_output_tokens=config.max_output_tokens,
        placeholders=PLACEHOLDERS,
    )


@router.post("/analyses/estimate")
async def estimate(request: AnalysisRequest, services: ServicesDep) -> AnalysisEstimate:
    source = await _source(services, request)
    try:
        return await estimate_analysis(request, source, services)
    except AnalysisLaunchError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/analyses", status_code=202, response_model_exclude=LIGHT)
async def create_analysis(
    request: AnalysisRequest, services: ServicesDep, api_key: ApiKeyDep
) -> AnalysisView:
    source = await _source(services, request)
    try:
        meta, _ = await launch_analysis(request, source, api_key, services)
    except AnalysisLaunchError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _view(services, meta)


@router.get("/analyses", response_model_exclude={"__all__": LIGHT})
def list_analyses(services: ServicesDep) -> list[AnalysisView]:
    return [_view(services, meta) for meta in services.analyses.list_metas()]


@router.get("/analyses/{analysis_id}", response_model_exclude=NO_PROMPTS)
def get_analysis(analysis_id: str, services: ServicesDep) -> AnalysisView:
    return _view(services, load_or_404(services.analyses.get, analysis_id, "analysis"))


@router.get("/analyses/{analysis_id}/prompts")
def analysis_prompts(analysis_id: str, services: ServicesDep) -> AnalysisPrompts:
    meta = load_or_404(services.analyses.get, analysis_id, "analysis")
    return AnalysisPrompts(system_prompt=meta.system_prompt, user_prompt=meta.user_prompt)


@router.post("/analyses/{analysis_id}/cancel")
def cancel_analysis(analysis_id: str, services: ServicesDep) -> CancelView:
    load_or_404(services.analyses.get, analysis_id, "analysis")
    return CancelView(cancelled=services.jobs.cancel(analysis_id))
