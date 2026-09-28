"""Tests for jev_bench.services."""

from datetime import UTC, datetime

import httpx2
import pytest
from tests.factories import ServicesFactory

from jev_bench.benchmark_config import JevParams
from jev_bench.store.analyses import AnalysisMeta
from jev_bench.store.generations import GenerationMeta
from jev_bench.store.runs import RunMeta


def _unused(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(500)


@pytest.mark.parametrize("secret", ["api_key", "hf_token"])
@pytest.mark.parametrize(
    ("server_key", "supplied", "expected"),
    [
        pytest.param("sk-server", "sk-browser", "sk-browser", id="browser-key-overrides-server"),
        pytest.param("sk-server", None, "sk-server", id="server-key-without-browser-key"),
        pytest.param("sk-server", "   ", "sk-server", id="blank-browser-key-falls-back"),
        pytest.param(None, "  sk-browser  ", "sk-browser", id="browser-stripped"),
        pytest.param(None, None, None, id="none"),
        pytest.param(None, "   ", None, id="blank"),
    ],
)
async def test_secret_resolution(
    make_services: ServicesFactory,
    secret: str,
    server_key: str | None,
    supplied: str | None,
    expected: str | None,
) -> None:
    services = make_services(_unused, **{secret: server_key})
    assert getattr(services, secret)(supplied) == expected


async def test_configs_are_read_from_the_config_dir(make_services: ServicesFactory) -> None:
    services = make_services(_unused)
    assert services.question_set().name == "mini"
    assert [column.id for column in services.benchmark_config().columns] == [
        "jev",
        "anthropic",
        "embeddings",
        "kev",
    ]
    assert services.generation_config().models == ("gen/a", "gen/b")
    assert services.analysis_config().max_output_tokens == 500


async def test_sweep_interrupted_marks_orphans(make_services: ServicesFactory) -> None:
    services = make_services(_unused)
    questions = services.question_set()
    created = datetime(2026, 9, 24, tzinfo=UTC)
    run = RunMeta(
        id="20260924-100000-jev-x-0001",
        column="jev",
        kind="decisions",
        model="m",
        generation_ids=("g",),
        mode="per_email",
        emails_per_request=1,
        question_set=questions,
        params=JevParams(),
        concurrency=1,
        created_at=created,
        n_emails=0,
    )
    generation = GenerationMeta(
        id="20260924-100000-gen-0001",
        name="g",
        created_at=created,
        requested=1,
        seed=1,
        models=("m",),
        question_set=questions,
        config=services.generation_config(),
    )
    analysis = AnalysisMeta(
        id="20260924-100000-analysis-0001",
        created_at=created,
        model="m",
        run_ids=(run.id,),
        generation_ids=(generation.id,),
        system_prompt="S",
        user_prompt="U",
        email_refs={},
        n_emails=0,
        n_disputed=0,
        max_output_tokens=1,
    )
    services.runs.save(run)
    services.generations.save(generation)
    services.analyses.save(analysis)
    assert sorted(services.sweep_interrupted()) == sorted([run.id, generation.id, analysis.id])
    assert services.runs.get(run.id).status == "interrupted"
    assert services.generations.get(generation.id).status == "interrupted"
    assert services.analyses.get(analysis.id).status == "interrupted"
