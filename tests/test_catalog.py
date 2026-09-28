"""Tests for jev_bench.catalog."""

from typing import Any

import httpx2
import pytest
from tests.factories import ClientFactory

from jev_bench.benchmark_config import ColumnConfig
from jev_bench.catalog import Catalog, ModelInfo, parse_model, select_models

_STRUCTURED = ["response_format", "structured_outputs", "reasoning"]
_SONNET: dict[str, Any] = {
    "id": "anthropic/claude-sonnet-5",
    "name": "Anthropic: Claude Sonnet 5",
    "pricing": {"prompt": "0.000002", "completion": "0.00001"},
    "context_length": 1000000,
    "top_provider": {"max_completion_tokens": 128000},
    "supported_parameters": _STRUCTURED,
}
_SONNET_BATCH = {**_SONNET, "id": "anthropic/claude-sonnet-5:batch"}
_ALIAS = {**_SONNET, "id": "~anthropic/claude-sonnet-latest"}
_OLD = {
    "id": "anthropic/claude-3-haiku",
    "name": "Haiku 3",
    "pricing": {"prompt": "0.00000025"},
    "supported_parameters": [],
}
_GPT = {**_SONNET, "id": "openai/gpt-5.6-terra", "name": "OpenAI: GPT-5.6 Terra"}
_EMBEDDING = {
    "id": "openai/text-embedding-3-large",
    "name": "Embedding 3 Large",
    "pricing": {"prompt": "0.00000013"},
    "context_length": 8192,
}
_JEV = {
    "id": "typesafe/jev-1.13",
    "name": "Jev 1.13",
    "pricing": {"prompt": "0.000000042"},
    "context_length": 32000,
}


def _column(**fields: Any) -> ColumnConfig:
    return ColumnConfig.model_validate({"id": "c", "title": "C", "default_model": "x", **fields})


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(
            _SONNET,
            ModelInfo(
                id="anthropic/claude-sonnet-5",
                name="Anthropic: Claude Sonnet 5",
                prompt_price=0.000002,
                completion_price=0.00001,
                context_length=1000000,
                max_completion_tokens=128000,
                supported_parameters=tuple(_STRUCTURED),
            ),
            id="full",
        ),
        pytest.param({"id": "x/y"}, ModelInfo(id="x/y", name="x/y"), id="minimal"),
        pytest.param(
            {"id": "x/y", "pricing": {"prompt": "-1", "completion": "abc"}},
            ModelInfo(id="x/y", name="x/y"),
            id="unusable-prices",
        ),
    ],
)
def test_parse_model(raw: dict[str, Any], expected: ModelInfo) -> None:
    assert parse_model(raw) == expected


def test_estimate_cost() -> None:
    assert parse_model(_SONNET).estimate_cost(1000, 100) == pytest.approx(0.003)


@pytest.mark.parametrize(
    ("prefix", "structured", "expected"),
    [
        pytest.param(
            "anthropic/",
            True,
            ["anthropic/claude-sonnet-5", "~anthropic/claude-sonnet-latest"],
            id="chat-prefix",
        ),
        pytest.param(
            "anthropic/",
            False,
            [
                "anthropic/claude-3-haiku",
                "anthropic/claude-sonnet-5",
                "~anthropic/claude-sonnet-latest",
            ],
            id="no-structured-requirement",
        ),
        pytest.param(
            None,
            True,
            [
                "anthropic/claude-sonnet-5",
                "openai/gpt-5.6-terra",
                "~anthropic/claude-sonnet-latest",
            ],
            id="no-prefix",
        ),
    ],
)
def test_select_models(prefix: str | None, structured: bool, expected: list[str]) -> None:
    models = [parse_model(raw) for raw in (_SONNET, _SONNET_BATCH, _ALIAS, _OLD, _GPT)]
    result = [m.id for m in select_models(models, prefix=prefix, require_structured=structured)]
    assert result == expected


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _catalog_handler(calls: list[httpx2.Request]) -> Any:
    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        modality = request.url.params.get("output_modalities")
        data = {
            "decisions": [_JEV],
            "embeddings": [_EMBEDDING],
        }.get(modality or "", [_SONNET, _OLD, _GPT])
        return httpx2.Response(200, json={"data": [*data, {"no_id": True}]})

    return handler


async def test_catalog_caches_per_modality_until_ttl(make_client: ClientFactory) -> None:
    calls: list[httpx2.Request] = []
    clock = FakeClock()
    catalog = Catalog(make_client(_catalog_handler(calls)), ttl_s=10, clock=clock)
    text_models = [m.id for m in await catalog.models("text")]
    assert text_models == [
        "anthropic/claude-sonnet-5",
        "anthropic/claude-3-haiku",
        "openai/gpt-5.6-terra",
    ]
    await catalog.models("text")
    decisions_models = [m.id for m in await catalog.models("decisions")]
    assert decisions_models == ["typesafe/jev-1.13"]
    assert len(calls) == 2
    clock.now = 11
    await catalog.models("text")
    assert len(calls) == 3
    assert "output_modalities" not in calls[0].url.params
    assert calls[1].url.params["output_modalities"] == "decisions"


async def test_find(make_client: ClientFactory) -> None:
    catalog = Catalog(make_client(_catalog_handler([])))
    found = await catalog.find("embeddings", "openai/text-embedding-3-large")
    assert found is not None
    assert found.context_length == 8192
    assert await catalog.find("embeddings", "missing/model") is None


@pytest.mark.parametrize(
    ("column", "expected", "calls"),
    [
        pytest.param(
            _column(kind="chat", modality="text", prefix="anthropic/"),
            ["anthropic/claude-sonnet-5"],
            1,
            id="chat",
        ),
        pytest.param(
            _column(kind="embeddings", modality="embeddings"),
            ["openai/text-embedding-3-large"],
            1,
            id="embeddings",
        ),
        pytest.param(
            _column(kind="decisions", modality="decisions"),
            ["typesafe/jev-1.13"],
            1,
            id="decisions",
        ),
        pytest.param(
            _column(kind="kev", models=["Kev-4B", "Kev-0.8B"], default_model="Kev-4B"),
            ["Kev-4B", "Kev-0.8B"],
            0,
            id="kev-static-models-without-a-request",
        ),
    ],
)
async def test_for_column(
    make_client: ClientFactory, column: ColumnConfig, expected: list[str], calls: int
) -> None:
    requests: list[httpx2.Request] = []
    catalog = Catalog(make_client(_catalog_handler(requests)))
    result = await catalog.for_column(column)
    assert [m.id for m in result] == expected
    assert len(requests) == calls
    if column.kind == "kev":
        assert result == [ModelInfo(id=model, name=model) for model in expected]
