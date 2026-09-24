"""OpenRouter model catalog: fetch per modality, filter per column, TTL cache, price lookups.

Classes:
    ModelInfo: one catalog entry reduced to what the benchmark needs.
    Catalog: TTL-cached catalog per modality.
Functions:
    parse_model: raw catalog entry -> ModelInfo.
    select_models: filter by id prefix (aliases included), drop `:batch` variants,
        require structured outputs.
"""

import asyncio
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict

from jev_bench.benchmark_config import ColumnConfig, Modality
from jev_bench.openrouter import OpenRouterClient

MODELS_PATH = "/v1/models"


class ModelInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    prompt_price: float = 0.0
    completion_price: float = 0.0
    context_length: int | None = None
    max_completion_tokens: int | None = None
    supported_parameters: tuple[str, ...] = ()

    def estimate_cost(self, input_tokens: float, output_tokens: float) -> float:
        return input_tokens * self.prompt_price + output_tokens * self.completion_price


def parse_model(raw: Mapping[str, Any]) -> ModelInfo:
    pricing = raw.get("pricing") or {}
    top = raw.get("top_provider") or {}
    return ModelInfo(
        id=raw["id"],
        name=raw.get("name") or raw["id"],
        prompt_price=_price(pricing.get("prompt")),
        completion_price=_price(pricing.get("completion")),
        context_length=raw.get("context_length"),
        max_completion_tokens=top.get("max_completion_tokens"),
        supported_parameters=tuple(raw.get("supported_parameters") or ()),
    )


def _price(value: object) -> float:
    if not isinstance(value, str | int | float) or isinstance(value, bool):
        return 0.0
    try:
        return max(0.0, float(value))
    except ValueError:
        return 0.0


def select_models(
    models: Iterable[ModelInfo], *, prefix: str | None, require_structured: bool
) -> list[ModelInfo]:
    chosen = (model for model in models if _selected(model, prefix, require_structured))
    return sorted(chosen, key=lambda model: model.id)


def _selected(model: ModelInfo, prefix: str | None, require_structured: bool) -> bool:
    if model.id.endswith(":batch"):
        return False
    if prefix is not None and not model.id.lstrip("~").startswith(prefix):
        return False
    return not require_structured or "structured_outputs" in model.supported_parameters


class Catalog:
    def __init__(
        self,
        client: OpenRouterClient,
        *,
        ttl_s: float = 3600.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._client = client
        self._ttl = ttl_s
        self._clock = clock
        self._cache: dict[str, tuple[float, tuple[ModelInfo, ...]]] = {}
        self._lock = asyncio.Lock()

    async def models(self, modality: Modality) -> tuple[ModelInfo, ...]:
        async with self._lock:
            cached = self._cache.get(modality)
            if cached is not None and self._clock() - cached[0] < self._ttl:
                return cached[1]
            models = await self._fetch(modality)
            self._cache[modality] = (self._clock(), models)
            return models

    async def find(self, modality: Modality, model_id: str) -> ModelInfo | None:
        return next((model for model in await self.models(modality) if model.id == model_id), None)

    async def for_column(self, column: ColumnConfig) -> list[ModelInfo]:
        models = await self.models(column.modality)
        return select_models(models, prefix=column.prefix, require_structured=column.kind == "chat")

    async def _fetch(self, modality: Modality) -> tuple[ModelInfo, ...]:
        params = None if modality == "text" else {"output_modalities": modality}
        body = await self._client.get_json(MODELS_PATH, params)
        return tuple(
            parse_model(raw)
            for raw in body.get("data", [])
            if isinstance(raw, dict) and "id" in raw
        )
