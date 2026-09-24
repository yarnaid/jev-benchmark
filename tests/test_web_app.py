"""Tests for jev_bench.web.app."""

import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import httpx2
import pytest
from fastapi.testclient import TestClient
from loguru import logger
from tests.factories import AppFactory, FakeOpenRouter, mini_settings

from jev_bench.benchmark_config import JevParams
from jev_bench.questions import load_question_set
from jev_bench.store.runs import RunMeta, RunStore
from jev_bench.web import app as web_app
from jev_bench.web.app import STATIC_DIR, create_app, create_default_app


def test_static_index_is_served(make_app: AppFactory) -> None:
    response = make_app(FakeOpenRouter()).get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert (STATIC_DIR / "index.html").exists()


def test_unknown_api_route_is_404(make_app: AppFactory) -> None:
    assert make_app(FakeOpenRouter()).get("/api/does-not-exist").status_code == 404


def test_lifespan_sweeps_interrupted_runs(tmp_path: Path) -> None:
    settings = mini_settings(tmp_path)
    run = RunMeta(
        id="20260924-100000-jev-x-0001",
        column="jev",
        kind="decisions",
        model="m",
        generation_ids=("g",),
        mode="per_email",
        emails_per_request=1,
        question_set=load_question_set(settings.config_dir / "questions.toml"),
        params=JevParams(),
        concurrency=1,
        created_at=datetime(2026, 9, 24, tzinfo=UTC),
        n_emails=0,
    )
    RunStore(settings.data_dir / "runs").save(run)
    http = httpx2.AsyncClient(transport=httpx2.MockTransport(FakeOpenRouter()))
    with TestClient(create_app(settings, http=http)):
        assert RunStore(settings.data_dir / "runs").get(run.id).status == "interrupted"


def test_create_default_app_builds_an_app() -> None:
    assert create_default_app().title == "jev-bench"


@pytest.fixture
def _restore_default_logging() -> Iterator[None]:
    try:
        yield
    finally:
        logger.remove()
        logger.add(sys.stderr)


@pytest.mark.usefixtures("_restore_default_logging")
def test_create_default_app_configures_logging_before_building_the_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(web_app, "load_settings", lambda: mini_settings(tmp_path))
    create_default_app()
    handlers = list(cast(Any, logger)._core.handlers.values())
    assert len(handlers) == 1
    assert handlers[0]._exception_formatter._diagnose is False
