"""Opt-in smoke test against the real OpenRouter API: one hand-written email per column and mode.

Run with: uv run pytest -m integration tests/test_integration_openrouter.py
(needs OPENROUTER_API_KEY).
"""

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from jev_bench.emails import Email, Party
from jev_bench.generation.launcher import GenerationRequest, launch_generation
from jev_bench.openrouter import build_http_client
from jev_bench.run_launcher import RunRequest, launch_run
from jev_bench.services import Services
from jev_bench.settings import Settings, load_settings
from jev_bench.store.generations import GenerationMeta

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]
CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
GENERATION_ID = "20260924-000000-smoke-0001"


@pytest.fixture
async def live(tmp_path: Path) -> AsyncIterator[tuple[Services, str]]:
    key = load_settings().server_api_key()
    if not key:
        pytest.skip("OPENROUTER_API_KEY is not set")
    settings = Settings.model_validate(
        {
            "openrouter_api_key": key,
            "data_dir": tmp_path / "data",
            "config_dir": CONFIG_DIR,
        }
    )
    http = build_http_client(settings)
    try:
        yield Services(settings, http), key
    finally:
        await http.aclose()


def _seed(services: Services) -> None:
    meta = GenerationMeta(
        id=GENERATION_ID,
        name="smoke",
        created_at=datetime.now(UTC),
        status="completed",
        requested=1,
        done=1,
        seed=0,
        models=("manual",),
        question_set=services.question_set(),
        config=services.generation_config(),
    )
    services.generations.save(meta)
    email = Email(
        id=f"{GENERATION_ID}.0001",
        sent_at=datetime(2026, 9, 24, 8, 15, tzinfo=UTC),
        sender=Party(name="Security Team", address="security@paypa1-support.test"),
        to=(Party(name="Ann Lee", address="ann.lee@example.org"),),
        subject="Urgent: your account will be suspended in 24 hours",
        body=(
            "We detected unusual activity. Confirm your password within 24 hours at "
            "http://paypa1-support.test/verify or your account will be closed."
        ),
        generator_model="manual",
    )
    services.generations.append_email(GENERATION_ID, email)


@pytest.mark.parametrize(
    ("column", "mode"),
    [
        pytest.param("jev", None, id="jev"),
        pytest.param("anthropic", "per_email", id="anthropic-per-email"),
        pytest.param("anthropic", "all_in_one", id="anthropic-all-in-one"),
        pytest.param("openai", "per_email", id="openai-per-email"),
        pytest.param("embeddings", None, id="embeddings"),
    ],
)
async def test_one_email_per_column(
    live: tuple[Services, str], column: str, mode: str | None
) -> None:
    services, key = live
    _seed(services)
    request = RunRequest.model_validate(
        {"column": column, "generation_ids": [GENERATION_ID], "mode": mode}
    )
    meta, task = await launch_run(request, key, services)
    await asyncio.wait_for(task, timeout=240)
    final = services.runs.get(meta.id)
    assert final.status == "completed", final.error
    assert final.n_errors == 0, services.runs.predictions(meta.id)
    assert final.total_cost > 0


async def test_generation_of_one_email(live: tuple[Services, str]) -> None:
    services, key = live
    request = GenerationRequest(name="smoke", count=1, seed=1)
    meta, task = await launch_generation(request, key, services)
    await asyncio.wait_for(task, timeout=240)
    final = services.generations.get(meta.id)
    assert (final.status, final.done, final.errors) == ("completed", 1, 0)
