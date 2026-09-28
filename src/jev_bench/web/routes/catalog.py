"""Catalog route: every benchmark column with its selectable models (OpenRouter, or static for
Kev) and card slot.

Classes:
    CatalogModel, CatalogColumn
Functions:
    catalog: GET /catalog
"""

import tomllib

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ValidationError

from jev_bench.benchmark_config import BenchmarkConfig, ColumnConfig, ColumnKind
from jev_bench.catalog import ModelInfo
from jev_bench.openrouter import OpenRouterError
from jev_bench.services import Services
from jev_bench.web.deps import ServicesDep

__all__ = [
    "CatalogColumn",
    "CatalogModel",
    "catalog",
    "router",
]

router = APIRouter(tags=["catalog"])


class CatalogModel(BaseModel):
    id: str
    name: str
    prompt_price_per_m: float
    completion_price_per_m: float
    context_length: int | None
    max_completion_tokens: int | None


class CatalogColumn(BaseModel):
    id: str
    title: str
    kind: ColumnKind
    default_model: str
    models: list[CatalogModel]
    error: str | None = None
    embedding_temperature: float | None = None
    emails_per_request: int | None = None
    slot: str
    calibrated: bool | None = None


@router.get("/catalog")
async def catalog(services: ServicesDep) -> list[CatalogColumn]:
    config = _benchmark_config(services)
    return [await _column(services, column, config) for column in config.columns]


def _benchmark_config(services: Services) -> BenchmarkConfig:
    try:
        return services.benchmark_config()
    except (ValidationError, tomllib.TOMLDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid config: {exc}") from exc


async def _column(
    services: Services, column: ColumnConfig, config: BenchmarkConfig
) -> CatalogColumn:
    models, error = await _models(services, column)
    embeddings = column.kind == "embeddings"
    return CatalogColumn(
        id=column.id,
        title=column.title,
        kind=column.kind,
        default_model=column.default_model,
        models=_with_default(models, column.default_model),
        error=error,
        embedding_temperature=config.embeddings.temperature if embeddings else None,
        emails_per_request=config.embeddings.emails_per_request if embeddings else None,
        slot=column.effective_slot,
        calibrated=config.kev.calibrated if column.kind == "kev" else None,
    )


async def _models(services: Services, column: ColumnConfig) -> tuple[list[ModelInfo], str | None]:
    try:
        return await services.catalog.for_column(column), None
    except OpenRouterError as exc:
        return [], str(exc)


def _with_default(models: list[ModelInfo], default: str) -> list[CatalogModel]:
    views = [_view(model) for model in models]
    if all(view.id != default for view in views):
        views.insert(0, _view(ModelInfo(id=default, name=f"{default} (default)")))
    return views


def _view(model: ModelInfo) -> CatalogModel:
    return CatalogModel(
        id=model.id,
        name=model.name,
        prompt_price_per_m=model.prompt_price * 1_000_000,
        completion_price_per_m=model.completion_price * 1_000_000,
        context_length=model.context_length,
        max_completion_tokens=model.max_completion_tokens,
    )
