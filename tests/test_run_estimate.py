"""Tests for jev_bench.run_estimate."""

from datetime import UTC, datetime

import pytest
from tests.factories import FakeOpenRouter, ServicesFactory, seed_generation

from jev_bench.benchmark_config import JevParams, TokenParams
from jev_bench.run_estimate import estimate_run
from jev_bench.run_launcher import RunLaunchError, RunRequest
from jev_bench.services import Services
from jev_bench.store.runs import RunMeta, RunMode
from jev_bench.store.status import JobStatus

_JEV_PROMPT_PRICE = 0.000000042
_CHAT_PRICES = (0.000002, 0.00001)


def _past_run(
    services: Services,
    column: str,
    model: str,
    mode: RunMode,
    past: tuple[float, int, JobStatus],
    cold_cost: float = 0.0,
) -> None:
    total_cost, n_done, status = past
    meta = RunMeta(
        id=f"20260924-10000{len(services.runs.list_metas())}-{column}-x-0001",
        column=column,
        kind="embeddings" if column == "embeddings" else "decisions",
        model=model,
        generation_ids=("g",),
        mode=mode,
        emails_per_request=1,
        question_set=services.question_set(),
        params=JevParams(),
        concurrency=1,
        created_at=datetime(2026, 9, 24, tzinfo=UTC),
        n_emails=4,
        n_done=n_done,
        status=status,
        total_cost=total_cost,
        cold_cost=cold_cost,
    )
    services.runs.save(meta)


async def test_token_estimate_prices_the_runners_own_plan(make_services: ServicesFactory) -> None:
    services = make_services(FakeOpenRouter())
    generation_id = seed_generation(services, emails=3)
    estimate = await estimate_run(
        RunRequest(column="jev", generation_ids=(generation_id,)), services
    )
    assert (estimate.n_emails, estimate.n_requests) == (3, 3)
    assert estimate.input_tokens > 0
    assert estimate.output_tokens == 3 * 1000
    assert estimate.token_cost == pytest.approx(estimate.input_tokens * _JEV_PROMPT_PRICE)
    assert (estimate.history_cost, estimate.history_emails) == (None, 0)


@pytest.mark.parametrize(
    ("mode", "requests"),
    [
        pytest.param(None, 3, id="per-email"),
        pytest.param("all_in_one", 1, id="all-in-one"),
    ],
)
async def test_chat_estimate_follows_the_request_mode(
    make_services: ServicesFactory, mode: str | None, requests: int
) -> None:
    services = make_services(FakeOpenRouter())
    generation_id = seed_generation(services, emails=3)
    request = RunRequest.model_validate(
        {"column": "anthropic", "generation_ids": [generation_id], "mode": mode}
    )
    estimate = await estimate_run(request, services)
    prompt, completion = _CHAT_PRICES
    assert estimate.n_requests == requests
    assert estimate.output_tokens == 3 * 400
    assert estimate.token_cost == pytest.approx(
        estimate.input_tokens * prompt + estimate.output_tokens * completion
    )


@pytest.mark.parametrize(
    ("past", "expected"),
    [
        pytest.param([(0.01, 4, "completed")], 3 * 0.01 / 4, id="one-completed-run"),
        pytest.param(
            [(0.01, 4, "completed"), (0.05, 1, "completed")], 3 * 0.06 / 5, id="pooled-over-runs"
        ),
        pytest.param(
            [(0.0, 4, "completed"), (0.02, 4, "failed")],
            None,
            id="zero-cost-and-unfinished-runs-ignored",
        ),
    ],
)
async def test_history_estimate_uses_comparable_past_runs(
    make_services: ServicesFactory,
    past: list[tuple[float, int, JobStatus]],
    expected: float | None,
) -> None:
    services = make_services(FakeOpenRouter())
    generation_id = seed_generation(services, emails=3)
    for totals in past:
        _past_run(services, "jev", "typesafe/jev-1.13", "per_email", totals)
    _past_run(services, "jev", "other/model", "per_email", (9.0, 1, "completed"))
    estimate = await estimate_run(
        RunRequest(column="jev", generation_ids=(generation_id,)), services
    )
    assert estimate.history_cost == (None if expected is None else pytest.approx(expected))


async def test_embeddings_history_uses_the_cold_cost(make_services: ServicesFactory) -> None:
    services = make_services(FakeOpenRouter())
    generation_id = seed_generation(services, emails=2)
    _past_run(
        services,
        "embeddings",
        "openai/text-embedding-3-large",
        "batched",
        (0.0, 4, "completed"),
        cold_cost=0.004,
    )
    request = RunRequest(column="embeddings", generation_ids=(generation_id,))
    estimate = await estimate_run(request, services)
    assert estimate.history_cost == pytest.approx(2 * 0.004 / 4)
    assert (estimate.output_tokens, estimate.n_requests) == (0, 1)


async def test_unknown_column_is_a_launch_error(make_services: ServicesFactory) -> None:
    services = make_services(FakeOpenRouter())
    generation_id = seed_generation(services)
    with pytest.raises(RunLaunchError, match="unknown column"):
        await estimate_run(RunRequest(column="nope", generation_ids=(generation_id,)), services)


async def test_unpriced_model_has_no_token_cost(make_services: ServicesFactory) -> None:
    services = make_services(FakeOpenRouter(models_status=503))
    generation_id = seed_generation(services)
    estimate = await estimate_run(
        RunRequest(column="jev", generation_ids=(generation_id,)), services
    )
    assert estimate.token_cost is None
    assert estimate.input_tokens > 0


@pytest.mark.parametrize("emails", [pytest.param(1, id="one"), pytest.param(3, id="three")])
async def test_kev_estimate_is_free(make_services: ServicesFactory, emails: int) -> None:
    services = make_services(FakeOpenRouter())
    generation_id = seed_generation(services, emails=emails)
    estimate = await estimate_run(
        RunRequest(column="kev", generation_ids=(generation_id,)), services
    )
    assert (estimate.n_emails, estimate.n_requests) == (emails, emails)
    assert (estimate.token_cost, estimate.history_cost) == (0.0, None)
    assert estimate.output_tokens == emails * TokenParams().jev_output_reserve
