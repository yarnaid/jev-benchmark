"""Tests for jev_bench.run_launcher."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx2
import pytest
from pydantic import ValidationError
from tests.factories import FakeOpenRouter, ServicesFactory, seed_generation

from jev_bench.benchmark_config import LlmParams
from jev_bench.run_launcher import RunLaunchError, RunRequest, launch_run

_NOW = datetime(2026, 9, 24, 15, 30, 12, tzinfo=UTC)


@pytest.mark.parametrize(
    ("column", "mode", "expected_mode", "per_request", "requests"),
    [
        pytest.param("jev", None, "per_email", 1, 2, id="jev"),
        pytest.param("anthropic", None, "per_email", 1, 2, id="chat-per-email"),
        pytest.param("anthropic", "all_in_one", "all_in_one", None, 1, id="chat-all-in-one"),
        pytest.param("embeddings", None, "batched", 2, 1, id="embeddings"),
        pytest.param("kev", None, "per_email", 1, 2, id="kev-through-openrouter-decisions"),
    ],
)
async def test_launch_run_for_every_column_kind(
    make_services: ServicesFactory,
    column: str,
    mode: str | None,
    expected_mode: str,
    per_request: int | None,
    requests: int,
) -> None:
    fake = FakeOpenRouter()
    services = make_services(fake)
    generation_id = seed_generation(services)
    request = RunRequest.model_validate(
        {"column": column, "generation_ids": [generation_id], "mode": mode}
    )
    meta, task = await launch_run(request, "sk-test", services, now=_NOW)
    await task
    final = services.runs.get(meta.id)
    assert meta.id.startswith(f"20260924-153012-{column}-")
    assert (meta.mode, meta.emails_per_request) == (expected_mode, per_request)
    assert meta.column_config is not None
    assert meta.column_config.id == column
    assert final.status == "completed"
    assert (final.n_done, final.n_errors, final.n_requests) == (2, 0, requests)
    assert final.generation_ids == (generation_id,)
    posts = [sent for sent in fake.requests if sent.method == "POST"]
    assert posts
    assert all(sent.headers["authorization"] == "Bearer sk-test" for sent in posts)


@pytest.mark.parametrize(
    "headers",
    [
        pytest.param({}, id="missing"),
        pytest.param({"Authorization": "Bearer "}, id="empty-token"),
        pytest.param({"Authorization": "Basic sk-test"}, id="wrong-scheme"),
    ],
)
def test_fake_openrouter_rejects_unauthenticated_posts(headers: dict[str, str]) -> None:
    fake = FakeOpenRouter()
    request = httpx2.Request(
        "POST", "https://openrouter.test/api/v1/chat/completions", headers=headers, json={}
    )
    response = fake(request)
    assert response.status_code == 401
    assert fake.requests == [request]


async def test_catalog_outage_falls_back_to_default_limits(make_services: ServicesFactory) -> None:
    services = make_services(FakeOpenRouter(models_status=503))
    generation_id = seed_generation(services)
    meta, task = await launch_run(
        RunRequest(column="jev", generation_ids=(generation_id,)), "sk-test", services
    )
    await task
    assert services.runs.get(meta.id).status == "completed"


def _request(**fields: Any) -> RunRequest:
    return RunRequest.model_validate(
        {"column": "jev", "generation_ids": ["20260924-100000-seed-abcd"], **fields}
    )


@pytest.mark.parametrize(
    ("request_fields", "seed", "api_key", "message"),
    [
        pytest.param({"column": "missing"}, True, "sk-test", "unknown column", id="unknown-column"),
        pytest.param(
            {"mode": "all_in_one"}, True, "sk-test", "only accepted for chat", id="mode-on-jev"
        ),
        pytest.param(
            {"generation_ids": ["20260924-100000-none-abcd"]},
            True,
            "sk-test",
            "unknown generation",
            id="unknown-generation",
        ),
        pytest.param({}, False, "sk-test", "unknown generation", id="nothing-seeded"),
        pytest.param(
            {"model": "typesafe/not-a-model"},
            True,
            "sk-test",
            "not available",
            id="model-not-in-catalog",
        ),
        pytest.param({}, True, "   ", "an OpenRouter API key is required", id="blank-api-key"),
        pytest.param({}, True, "", "an OpenRouter API key is required", id="empty-api-key"),
    ],
)
async def test_invalid_requests_raise_launch_errors(
    make_services: ServicesFactory,
    request_fields: dict[str, Any],
    seed: bool,
    api_key: str,
    message: str,
) -> None:
    services = make_services(FakeOpenRouter())
    if seed:
        seed_generation(services)
    with pytest.raises(RunLaunchError, match=message):
        await launch_run(_request(**request_fields), api_key, services)
    assert services.runs.list_metas() == []


async def test_invalid_benchmark_config_becomes_a_launch_error(
    tmp_path: Path, make_services: ServicesFactory
) -> None:
    services = make_services(FakeOpenRouter())
    (tmp_path / "config" / "benchmark.toml").write_text("not [valid", encoding="utf-8")
    with pytest.raises(RunLaunchError, match="invalid config"):
        await launch_run(_request(), "sk-test", services)


async def test_generation_without_emails_is_rejected(make_services: ServicesFactory) -> None:
    services = make_services(FakeOpenRouter())
    generation_id = seed_generation(services, emails=0)
    with pytest.raises(RunLaunchError, match="contain no emails"):
        await launch_run(_request(generation_ids=[generation_id]), "sk-test", services)


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"column": "jev", "generation_ids": []}, id="no-generations"),
        pytest.param({"column": "jev", "generation_ids": ["g"], "extra": 1}, id="extra-field"),
        pytest.param({"column": "jev", "generation_ids": ["g"], "mode": "batched"}, id="bad-mode"),
    ],
)
def test_run_request_validation(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        RunRequest.model_validate(payload)


@pytest.mark.parametrize(
    ("fake", "temperature", "reasoning"),
    [
        pytest.param(FakeOpenRouter(), None, False, id="temperature-unsupported-is-not-sent"),
        pytest.param(
            FakeOpenRouter(chat_parameters=("reasoning", "structured_outputs", "temperature")),
            0.0,
            False,
            id="supported-temperature-is-sent",
        ),
        pytest.param(
            FakeOpenRouter(chat_parameters=("structured_outputs",)),
            None,
            None,
            id="reasoning-unsupported-is-not-sent",
        ),
        pytest.param(FakeOpenRouter(models_status=503), 0.0, False, id="catalog-down-sends-all"),
    ],
)
async def test_chat_params_follow_the_models_supported_parameters(
    make_services: ServicesFactory,
    fake: FakeOpenRouter,
    temperature: float | None,
    reasoning: bool | None,
) -> None:
    services = make_services(fake)
    generation_id = seed_generation(services)
    request = RunRequest.model_validate({"column": "anthropic", "generation_ids": [generation_id]})
    meta, task = await launch_run(request, "sk-test", services, now=_NOW)
    await task
    assert isinstance(meta.params, LlmParams)
    assert (meta.params.temperature, meta.params.reasoning_enabled) == (temperature, reasoning)
    bodies = [
        json.loads(sent.content) for sent in fake.requests if sent.url.path.endswith("/completions")
    ]
    assert bodies
    assert all(body.get("temperature") == temperature for body in bodies)
    expected = None if reasoning is None else {"enabled": reasoning}
    assert all(body.get("reasoning") == expected for body in bodies)
    assert services.runs.get(meta.id).status == "completed"
