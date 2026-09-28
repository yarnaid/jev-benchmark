"""Tests for jev_bench.web.deps."""

import httpx2
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tests.factories import ServicesFactory

from jev_bench.web.deps import (
    HF_FORMAT_DETAIL,
    NO_KEY_DETAIL,
    ApiKeyDep,
    HfTokenDep,
    OptionalApiKeyDep,
    split_ids,
)


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
    ("server", "headers", "expected"),
    [
        pytest.param({}, {}, {"key": None, "hf": None}, id="nothing"),
        pytest.param(
            {"api_key": "sk-server", "hf_token": "hf_server"},
            {},
            {"key": "sk-server", "hf": "hf_server"},
            id="server-secrets",
        ),
        pytest.param(
            {"hf_token": "hf_server"},
            {"X-HF-Token": "hf_browser"},
            {"key": None, "hf": "hf_browser"},
            id="browser-token-overrides",
        ),
        pytest.param(
            {"hf_token": "hf_server"},
            {"X-HF-Token": "   "},
            {"key": None, "hf": "hf_server"},
            id="blank-header-falls-back",
        ),
        pytest.param(
            {},
            {"X-HF-Token": "sk-or-v1-pasted-into-the-wrong-field"},
            {"detail": HF_FORMAT_DETAIL},
            id="openrouter-key-in-the-hf-header-is-refused",
        ),
        pytest.param(
            {"hf_token": "not-a-token"},
            {},
            {"detail": HF_FORMAT_DETAIL},
            id="malformed-server-token-is-refused",
        ),
    ],
)
async def test_optional_secrets(
    make_services: ServicesFactory,
    server: dict[str, str],
    headers: dict[str, str],
    expected: dict[str, str | None],
) -> None:
    app = FastAPI()
    app.state.services = make_services(_unused, **server)

    @app.get("/secrets")
    def secrets(key: OptionalApiKeyDep, hf: HfTokenDep) -> dict[str, str | None]:
        return {"key": key, "hf": hf}

    assert TestClient(app).get("/secrets", headers=headers).json() == expected


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
