"""Tests for jev_bench.web.security."""

import pytest
from tests.factories import AppFactory, FakeOpenRouter

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


def _directive(entry: str) -> tuple[str, str]:
    name, value = entry.split(" ", 1)
    return name, value


def test_csp_forbids_inline_scripts_and_foreign_connections() -> None:
    directives = dict(_directive(part) for part in CSP.split("; "))
    assert directives["script-src"] == "'self' https://cdn.jsdelivr.net"
    assert directives["connect-src"] == "'self'"
    assert directives["object-src"] == "'none'"
    assert "'unsafe-inline'" not in directives["script-src"]
