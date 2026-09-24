"""Tests for jev_bench.web.security."""

from pathlib import Path

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient
from tests.factories import AppFactory, FakeOpenRouter, mini_settings

from jev_bench.web import app as web_app
from jev_bench.web.app import create_app
from jev_bench.web.security import CSP


@pytest.mark.parametrize(
    "path",
    [
        pytest.param("/api/status", id="api"),
        pytest.param("/", id="static-index"),
        pytest.param("/api/missing", id="api-404"),
    ],
)
def test_security_headers_on_every_response(make_app: AppFactory, path: str) -> None:
    response = make_app(FakeOpenRouter()).get(path)
    assert response.headers["Content-Security-Policy"] == CSP
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Referrer-Policy"] == "no-referrer"


def test_security_headers_on_an_unhandled_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    boom_router = APIRouter()

    @boom_router.get("/boom")
    def boom() -> None:
        raise RuntimeError("secret-ish detail")

    monkeypatch.setattr(web_app, "ROUTERS", [*web_app.ROUTERS, boom_router])
    app = create_app(mini_settings(tmp_path))
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/boom")
    assert response.status_code == 500
    assert response.headers["Content-Security-Policy"] == CSP
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert "secret-ish" not in response.text


def _directive(entry: str) -> tuple[str, str]:
    name, value = entry.split(" ", 1)
    return name, value


def test_csp_forbids_inline_scripts_and_foreign_connections() -> None:
    directives = dict(_directive(part) for part in CSP.split("; "))
    assert directives["script-src"] == "'self' https://cdn.jsdelivr.net"
    assert directives["connect-src"] == "'self'"
    assert directives["object-src"] == "'none'"
    assert "'unsafe-inline'" not in directives["script-src"]
