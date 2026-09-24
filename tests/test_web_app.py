"""Tests for jev_bench.web.app."""

import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn

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


@pytest.mark.parametrize(
    ("owned", "expect_closed"),
    [
        pytest.param(True, True, id="owned-client-closed-on-failed-startup"),
        pytest.param(False, False, id="injected-client-not-closed-on-failed-startup"),
    ],
)
def test_lifespan_closes_only_the_owned_client_on_failed_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, owned: bool, expect_closed: bool
) -> None:
    settings = mini_settings(tmp_path)
    closed = {"value": False}
    client = httpx2.AsyncClient(transport=httpx2.MockTransport(FakeOpenRouter()))
    real_aclose = client.aclose

    async def spy_aclose() -> None:
        closed["value"] = True
        await real_aclose()

    monkeypatch.setattr(client, "aclose", spy_aclose)
    monkeypatch.setattr(web_app, "build_http_client", lambda _settings: client)

    def _raise(*args: object, **kwargs: object) -> NoReturn:
        raise RuntimeError("boom-startup")

    monkeypatch.setattr(web_app, "Services", _raise)

    app = create_app(settings, http=None if owned else client)
    with pytest.raises(RuntimeError, match="boom-startup"), TestClient(app):
        pass
    assert closed["value"] is expect_closed


@pytest.fixture
def _restore_default_logging() -> Iterator[None]:
    try:
        yield
    finally:
        logger.remove()
        logger.add(sys.stderr)


def _fail_with_api_key(api_key: str) -> None:
    raise RuntimeError("boom")


def _call_with_local_secret(secret: str) -> None:
    _fail_with_api_key(api_key=secret)


@pytest.mark.usefixtures("_restore_default_logging")
def test_create_default_app_configures_logging_before_building_the_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(web_app, "load_settings", lambda: mini_settings(tmp_path))
    create_default_app()
    secret = "sk-or-v1-SUPERSECRET"
    try:
        _call_with_local_secret(secret)
    except RuntimeError as exc:
        logger.opt(exception=exc).error("job failed")
    captured = capsys.readouterr()
    assert "job failed" in captured.err
    assert secret not in captured.err
