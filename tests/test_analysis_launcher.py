"""Tests for jev_bench.analysis.launcher."""

import json
from datetime import UTC, datetime
from typing import Any

import pytest
from tests.analysis_data import GENERATION, RUN_A, RUN_B, build_source
from tests.factories import CHAT_PARAMETERS, FakeOpenRouter, ServicesFactory

from jev_bench.analysis.launcher import (
    AnalysisLaunchError,
    AnalysisRequest,
    estimate_analysis,
    launch_analysis,
)
from jev_bench.questions import QuestionSet

_NOW = datetime(2026, 9, 25, 12, tzinfo=UTC)


def _request(**fields: Any) -> AnalysisRequest:
    return AnalysisRequest.model_validate({"runs": [RUN_A, RUN_B], **fields})


async def test_estimate_uses_catalog_prices(
    make_services: ServicesFactory, multi_questions: QuestionSet
) -> None:
    services = make_services(FakeOpenRouter())
    estimate = await estimate_analysis(_request(), build_source(multi_questions), services)
    assert estimate.model == "anthropic/claude-sonnet-5"
    assert (estimate.max_output_tokens, estimate.context_length) == (500, 1_000_000)
    assert (estimate.n_emails, estimate.n_disputed, estimate.fits) == (3, 1, True)
    assert estimate.input_tokens > 100
    assert estimate.cost == pytest.approx(estimate.input_tokens * 0.000002 + 500 * 0.00001)


async def test_estimate_without_catalog_uses_fallbacks(
    make_services: ServicesFactory, multi_questions: QuestionSet
) -> None:
    services = make_services(FakeOpenRouter(models_status=500))
    request = _request(system_prompt="x" * 120_000, max_disputed_emails=0)
    estimate = await estimate_analysis(request, build_source(multi_questions), services)
    assert (estimate.cost, estimate.context_length, estimate.fits) == (None, 32_000, False)
    assert (estimate.n_disputed, estimate.max_output_tokens) == (0, 500)


@pytest.mark.parametrize(
    ("request_fields", "message"),
    [
        pytest.param({"model": "nobody/unknown"}, "not in the OpenRouter catalog", id="model"),
        pytest.param({"user_prompt": "$nope"}, "placeholders", id="user-template"),
        pytest.param({"system_prompt": "costs $5"}, "placeholders", id="system-template"),
    ],
)
async def test_invalid_requests(
    make_services: ServicesFactory,
    multi_questions: QuestionSet,
    request_fields: dict[str, Any],
    message: str,
) -> None:
    services = make_services(FakeOpenRouter())
    with pytest.raises(AnalysisLaunchError, match=message):
        await estimate_analysis(_request(**request_fields), build_source(multi_questions), services)


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param({"runs": []}, id="no-runs"),
        pytest.param({"threshold": 0}, id="zero-threshold"),
        pytest.param({"threshold": float("nan")}, id="nan-threshold"),
        pytest.param({"max_disputed_emails": 101}, id="too-many-disputed"),
        pytest.param({"model": ""}, id="blank-model"),
        pytest.param({"extra": 1}, id="unknown-field"),
    ],
)
def test_request_validation(fields: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="validation error"):
        _request(**fields)


async def test_launch_runs_the_analysis(
    make_services: ServicesFactory, multi_questions: QuestionSet
) -> None:
    fake = FakeOpenRouter()
    services = make_services(fake)
    request = _request(threshold=0.5, user_prompt="Data: $emails\nDisputed: $disputed")
    meta, task = await launch_analysis(
        request, build_source(multi_questions), "sk-k", services, now=_NOW
    )
    assert (meta.status, meta.run_ids, meta.generation_ids) == (
        "running",
        (RUN_A, RUN_B),
        (GENERATION,),
    )
    counts = (meta.threshold, meta.n_emails, meta.n_disputed, meta.max_output_tokens)
    assert counts == (0.5, 3, 1, 500)
    assert meta.email_refs["e002"] == f"{GENERATION}.0002"
    assert meta.id.startswith("20260925-120000-analysis-")
    await task
    stored = services.analyses.get(meta.id)
    assert (stored.status, stored.result) == (
        "completed",
        "## Executive summary\n- Jev agrees with the reference.",
    )
    body = json.loads(fake.requests[-1].content)
    assert [message["role"] for message in body["messages"]] == ["system", "user"]
    assert body["messages"][1]["content"] == stored.user_prompt
    assert stored.user_prompt.startswith("Data: ### category (choice)")
    assert (body["max_tokens"], "temperature" in body, stored.temperature) == (500, False, None)


async def test_launch_sends_temperature_when_supported(
    make_services: ServicesFactory, multi_questions: QuestionSet
) -> None:
    fake = FakeOpenRouter(chat_parameters=(*CHAT_PARAMETERS, "temperature"))
    services = make_services(fake)
    meta, task = await launch_analysis(_request(), build_source(multi_questions), "sk-k", services)
    await task
    assert json.loads(fake.requests[-1].content)["temperature"] == 0.2
    assert meta.temperature == 0.2


async def test_launch_refuses_a_prompt_that_does_not_fit(
    make_services: ServicesFactory, multi_questions: QuestionSet
) -> None:
    services = make_services(FakeOpenRouter(models_status=500))
    request = _request(system_prompt="x" * 120_000)
    with pytest.raises(AnalysisLaunchError, match="does not fit"):
        await launch_analysis(request, build_source(multi_questions), "sk-k", services)
    assert services.analyses.list_metas() == []
