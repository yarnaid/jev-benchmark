"""Tests for jev_bench.web.loaders."""

from collections.abc import Callable
from datetime import UTC, datetime

import httpx2
import pytest
from fastapi import HTTPException
from tests.factories import ServicesFactory, seed_generation

from jev_bench.benchmark_config import JevParams
from jev_bench.questions import QuestionSet
from jev_bench.services import Services
from jev_bench.store.runs import RunMeta
from jev_bench.web.loaders import (
    generation_snapshots,
    generations_of,
    load_comparison,
    load_email,
    load_emails,
    load_runs,
    run_raters,
)

type ServicesCall = Callable[[Services], object]


def _unused(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(500)


def _run(run_id: str, generation_ids: tuple[str, ...], questions: QuestionSet) -> RunMeta:
    return RunMeta(
        id=run_id,
        column="jev",
        kind="decisions",
        model="m",
        generation_ids=generation_ids,
        mode="per_email",
        emails_per_request=1,
        question_set=questions,
        params=JevParams(),
        concurrency=1,
        created_at=datetime(2026, 9, 24, tzinfo=UTC),
        n_emails=0,
    )


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda s: load_runs(s, ["20260924-100000-missing-0001"]), id="run"),
        pytest.param(lambda s: load_emails(s, ["20260924-100000-missing-0001"]), id="generation"),
        pytest.param(lambda s: load_email(s, "20260924-100000-missing-0001.0001"), id="email"),
        pytest.param(lambda s: load_email(s, "../../etc"), id="malformed-email"),
        pytest.param(
            lambda s: generation_snapshots(s, ["20260924-100000-missing-0001"]),
            id="generation-snapshot",
        ),
        pytest.param(
            lambda s: load_comparison(s, ["20260924-100000-missing-0001"]), id="comparison-run"
        ),
    ],
)
async def test_unknown_ids_are_404(make_services: ServicesFactory, call: ServicesCall) -> None:
    with pytest.raises(HTTPException) as info:
        call(make_services(_unused))
    assert info.value.status_code == 404


async def test_loaders_return_data(make_services: ServicesFactory) -> None:
    services = make_services(_unused)
    generation_id = seed_generation(services)
    run = _run(
        "20260924-100000-jev-x-0001", (generation_id, generation_id), services.question_set()
    )
    services.runs.save(run)
    assert [meta.id for meta in load_runs(services, [run.id])] == [run.id]
    assert len(load_emails(services, [generation_id])) == 2
    assert load_email(services, f"{generation_id}.0002").id == f"{generation_id}.0002"
    assert run_raters(services, [run])[0].answers == {}
    assert generations_of([run, run]) == [generation_id]
    snapshots = generation_snapshots(services, [generation_id])
    assert snapshots == {generation_id: services.question_set()}


async def test_load_comparison_needs_run_ids(make_services: ServicesFactory) -> None:
    with pytest.raises(HTTPException) as info:
        load_comparison(make_services(_unused), [])
    assert info.value.status_code == 400


@pytest.mark.parametrize(
    ("labelled", "raters"),
    [
        pytest.param(False, ["20260924-100000-jev-x-0001", "reference"], id="without-labels"),
        pytest.param(True, ["20260924-100000-jev-x-0001", "reference", "human"], id="with-labels"),
    ],
)
async def test_load_comparison(
    make_services: ServicesFactory, labelled: bool, raters: list[str]
) -> None:
    services = make_services(_unused)
    generation_id = seed_generation(services)
    older = _run("20260924-090000-jev-old-0001", (generation_id,), services.question_set())
    newer = _run("20260924-100000-jev-x-0001", (generation_id,), services.question_set())
    newer_questions = newer.question_set.model_copy(update={"name": "newest"})
    services.runs.save(older.model_copy(update={"created_at": datetime(2026, 9, 23, tzinfo=UTC)}))
    services.runs.save(newer.model_copy(update={"question_set": newer_questions}))
    if labelled:
        await services.labels.update(f"{generation_id}.0001", {"category": "spam"})
    inputs = load_comparison(services, [newer.id])
    assert [meta.id for meta in inputs.metas] == [newer.id]
    assert len(inputs.emails) == 2
    assert [rater.id for rater in inputs.raters()] == raters
    assert load_comparison(services, [older.id, newer.id]).base.name == "newest"
