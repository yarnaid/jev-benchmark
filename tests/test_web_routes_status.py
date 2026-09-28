"""Tests for jev_bench.web.routes.status."""

import pytest
from tests.factories import AppFactory, FakeOpenRouter


@pytest.mark.parametrize(
    ("secrets", "expected"),
    [
        pytest.param(
            {"api_key": "sk-server"},
            {"server_key": True, "server_hf_token": False},
            id="server-key",
        ),
        pytest.param(
            {"hf_token": "hf_server"},
            {"server_key": False, "server_hf_token": True},
            id="server-hf-token",
        ),
        pytest.param({}, {"server_key": False, "server_hf_token": False}, id="nothing"),
    ],
)
def test_status(make_app: AppFactory, secrets: dict[str, str], expected: dict[str, bool]) -> None:
    assert make_app(FakeOpenRouter(), **secrets).get("/api/status").json() == expected
