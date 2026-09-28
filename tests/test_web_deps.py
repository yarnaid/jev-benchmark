"""Tests for jev_bench.web.deps."""

import httpx2
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tests.factories import ServicesFactory

from jev_bench.web.deps import NO_KEY_DETAIL, ApiKeyDep, split_ids


def _unused(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(500)


@pytest.mark.parametrize(
    ("server_key", "header", "status", "expected"),
    [
        pytest.param("sk-server", "sk-browser", 200, "sk-browser", id="header-overrides-server"),
        pytest.param("sk-server", None, 200, "sk-server", id="server-without-header"),
        pytest.param(None, "sk-browser", 200, "sk-browser", id="header-used"),
        pytest.param(None, None, 400, NO_KEY_DETAIL, id="missing"),
        pytest.param(None, "   ", 400, NO_KEY_DETAIL, id="blank-header"),
    ],
)
async def test_require_api_key(
    make_services: ServicesFactory,
    server_key: str | None,
    header: str | None,
    status: int,
    expected: str,
) -> None:
    app = FastAPI()
    app.state.services = make_services(_unused, api_key=server_key)

    @app.get("/key")
    def key(api_key: ApiKeyDep) -> dict[str, str]:
        return {"key": api_key}

    headers = {"X-OpenRouter-Key": header} if header is not None else {}
    response = TestClient(app).get("/key", headers=headers)
    assert response.status_code == status
    body = response.json()
    assert (body.get("key") or body.get("detail")) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(None, [], id="none"),
        pytest.param("", [], id="empty"),
        pytest.param(" a, b ,,a ", ["a", "b"], id="trim-dedupe"),
    ],
)
def test_split_ids(value: str | None, expected: list[str]) -> None:
    assert split_ids(value) == expected
