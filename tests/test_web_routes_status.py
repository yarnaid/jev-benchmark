"""Tests for jev_bench.web.routes.status."""

import pytest
from tests.factories import AppFactory, FakeOpenRouter


@pytest.mark.parametrize(
    ("api_key", "expected"),
    [
        pytest.param("sk-server", True, id="server-key"),
        pytest.param(None, False, id="no-key"),
    ],
)
def test_status(make_app: AppFactory, api_key: str | None, expected: bool) -> None:
    assert make_app(FakeOpenRouter(), api_key=api_key).get("/api/status").json() == {
        "server_key": expected
    }
