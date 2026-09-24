"""Tests for jev_bench.generation.launcher."""

from datetime import UTC, datetime
from typing import Any

import httpx2
import pytest
from pydantic import ValidationError
from tests.factories import ServicesFactory, chat_body, generator_output

from jev_bench.generation.launcher import GenerationRequest, launch_generation


def _server(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(200, json=chat_body(generator_output()))


async def test_launch_generation_runs_to_completion(make_services: ServicesFactory) -> None:
    services = make_services(_server)
    now = datetime(2026, 9, 24, 15, 30, 12, tzinfo=UTC)
    meta, task = await launch_generation(
        GenerationRequest(name="Spam test", count=4, seed=9), "sk-test", services, now=now
    )
    assert services.generations.get(meta.id).status == "running"
    await task
    final = services.generations.get(meta.id)
    assert meta.id.startswith("20260924-153012-spam-test-")
    assert (final.status, final.done, final.seed, final.models) == (
        "completed",
        4,
        9,
        ("gen/a", "gen/b"),
    )
    assert len(services.generations.emails(meta.id)) == 4
    assert final.question_set.name == "mini"


async def test_request_models_override_config_and_seed_is_recorded(
    make_services: ServicesFactory,
) -> None:
    services = make_services(_server)
    meta, task = await launch_generation(
        GenerationRequest(count=1, models=("custom/model",)), "sk-test", services
    )
    await task
    assert meta.models == ("custom/model",)
    assert 0 <= meta.seed < 2**31
    assert services.generations.get(meta.id).seed == meta.seed


@pytest.mark.parametrize(
    "supplied",
    [
        pytest.param("", id="empty"),
        pytest.param("   ", id="blank"),
    ],
)
async def test_launch_generation_requires_an_api_key(
    make_services: ServicesFactory, supplied: str
) -> None:
    services = make_services(_server)
    with pytest.raises(ValueError, match="API key"):
        await launch_generation(GenerationRequest(count=1), supplied, services)
    assert services.generations.list_metas() == []


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"count": 0}, id="zero"),
        pytest.param({"count": 2001}, id="too-many"),
        pytest.param({"count": 1, "unknown": True}, id="extra-field"),
        pytest.param({"count": 1, "name": ""}, id="empty-name"),
        pytest.param({"count": 1, "seed": -1}, id="negative-seed"),
        pytest.param({"count": 1, "models": ()}, id="empty-models"),
        pytest.param({"count": 1, "models": ("",)}, id="blank-model"),
    ],
)
def test_invalid_requests(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        GenerationRequest.model_validate(payload)
