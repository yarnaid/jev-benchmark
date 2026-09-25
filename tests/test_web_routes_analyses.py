"""Tests for jev_bench.web.routes.analyses."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx2
import pytest
from fastapi.testclient import TestClient
from tests.factories import AppFactory, FakeOpenRouter, poll, seed_generation, services_of

from jev_bench.analysis.config import PLACEHOLDERS
from jev_bench.web.deps import NO_KEY_DETAIL

if TYPE_CHECKING:
    from loguru import Message

_KEY = {"X-OpenRouter-Key": "sk-browser"}
_SENTINEL = "sk-or-v1-SENTINEL-4242"
_MISSING = "20260925-100000-missing-0001"


def _two_runs(client: TestClient) -> list[str]:
    generation_id = seed_generation(services_of(client))
    body = {"generation_ids": [generation_id]}
    run_ids = [
        client.post("/api/runs", json={**body, "column": column}, headers=_KEY).json()["meta"]["id"]
        for column in ("jev", "anthropic")
    ]
    for run_id in run_ids:
        poll(client, f"/api/runs/{run_id}", until=lambda view: view["meta"]["status"] != "running")
    return run_ids


def _finished(view: dict[str, Any]) -> bool:
    return view["meta"]["status"] != "running"


def test_defaults(make_app: AppFactory) -> None:
    defaults = make_app(FakeOpenRouter()).get("/api/analysis/defaults").json()
    assert defaults["default_model"] == "anthropic/claude-sonnet-5"
    assert (defaults["max_disputed_emails"], defaults["max_output_tokens"]) == (2, 500)
    assert defaults["user_prompt"].startswith("$generations")
    assert tuple(defaults["placeholders"]) == PLACEHOLDERS


def test_defaults_with_a_broken_config(make_app: AppFactory) -> None:
    client = make_app(FakeOpenRouter())
    (services_of(client).settings.config_dir / "analysis.toml").write_text(
        "x = [", encoding="utf-8"
    )
    response = client.get("/api/analysis/defaults")
    assert response.status_code == 500
    assert response.json()["detail"].startswith("config/analysis.toml is invalid")


def test_estimate_needs_no_key(make_app: AppFactory) -> None:
    client = make_app(FakeOpenRouter(), api_key="sk-server")
    run_ids = _two_runs(client)
    response = client.post("/api/analyses/estimate", json={"runs": run_ids})
    assert response.status_code == 200
    estimate = response.json()
    assert (estimate["n_emails"], estimate["n_disputed"], estimate["fits"]) == (2, 2, True)
    assert estimate["cost"] > 0


@pytest.mark.parametrize(
    ("body", "status"),
    [
        pytest.param({"runs": [_MISSING]}, 404, id="unknown-run"),
        pytest.param({"runs": []}, 422, id="no-runs"),
        pytest.param({"runs": ["x"], "threshold": 2}, 422, id="bad-threshold"),
    ],
)
def test_estimate_errors(make_app: AppFactory, body: dict[str, Any], status: int) -> None:
    assert (
        make_app(FakeOpenRouter()).post("/api/analyses/estimate", json=body).status_code == status
    )


def test_invalid_template_is_400(make_app: AppFactory) -> None:
    client = make_app(FakeOpenRouter(), api_key="sk-server")
    body = {"runs": _two_runs(client), "user_prompt": "$nope"}
    response = client.post("/api/analyses/estimate", json=body)
    assert response.status_code == 400
    assert "placeholders" in response.json()["detail"]


def test_create_analysis_runs_in_the_background(make_app: AppFactory) -> None:
    client = make_app(FakeOpenRouter(), api_key="sk-server")
    run_ids = _two_runs(client)
    response = client.post("/api/analyses", json={"runs": run_ids, "threshold": 0.5}, headers=_KEY)
    assert response.status_code == 202
    created = response.json()["meta"]
    assert (created["status"], created["threshold"], created["run_ids"]) == (
        "running",
        0.5,
        run_ids,
    )
    assert {"user_prompt", "system_prompt", "result", "email_refs"}.isdisjoint(created)
    final = poll(client, f"/api/analyses/{created['id']}", until=_finished)
    assert final["meta"]["status"] == "completed"
    assert final["meta"]["result"].startswith("## Executive summary")
    assert final["meta"]["email_refs"]["e001"].endswith(".0001")
    assert {"user_prompt", "system_prompt"}.isdisjoint(final["meta"])
    prompts = client.get(f"/api/analyses/{created['id']}/prompts").json()
    assert prompts["system_prompt"].startswith("Analyse.")
    assert [name for name in PLACEHOLDERS if f"${name}" in prompts["user_prompt"]] == []
    listed = client.get("/api/analyses").json()
    assert [view["meta"]["id"] for view in listed] == [created["id"]]
    assert "result" not in listed[0]["meta"]
    cancel = client.post(f"/api/analyses/{created['id']}/cancel").json()
    assert cancel == {"cancelled": False}


def test_create_analysis_without_key_is_rejected(make_app: AppFactory) -> None:
    response = make_app(FakeOpenRouter()).post("/api/analyses", json={"runs": [_MISSING]})
    assert (response.status_code, response.json()["detail"]) == (400, NO_KEY_DETAIL)


def test_create_analysis_with_an_invalid_request_is_400(make_app: AppFactory) -> None:
    client = make_app(FakeOpenRouter(), api_key="sk-server")
    body = {"runs": _two_runs(client), "model": "nobody/unknown"}
    response = client.post("/api/analyses", json=body, headers=_KEY)
    assert response.status_code == 400
    assert "not in the OpenRouter catalog" in response.json()["detail"]
    assert client.get("/api/analyses").json() == []


@pytest.mark.parametrize(
    ("method", "path"),
    [
        pytest.param("GET", f"/api/analyses/{_MISSING}", id="get"),
        pytest.param("GET", f"/api/analyses/{_MISSING}/prompts", id="prompts"),
        pytest.param("POST", f"/api/analyses/{_MISSING}/cancel", id="cancel"),
        pytest.param("GET", "/api/analyses/..%2Fetc", id="unsafe-id"),
    ],
)
def test_unknown_analysis_is_404(make_app: AppFactory, method: str, path: str) -> None:
    assert make_app(FakeOpenRouter()).request(method, path).status_code == 404


class _Analyst:
    def __init__(self, *, echo_key: bool) -> None:
        self.upstream = FakeOpenRouter()
        self.echo_key = echo_key
        self.analysis_requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        if (
            not request.url.path.endswith("/v1/chat/completions")
            or b"response_format" in request.content
        ):
            return self.upstream(request)
        self.analysis_requests.append(request)
        if self.echo_key:
            key = request.headers["authorization"].removeprefix("Bearer ")
            return httpx2.Response(401, json={"error": {"message": f"bad key {key}", "code": 401}})
        return self.upstream(request)


@pytest.mark.parametrize(
    ("echo_key", "status"),
    [
        pytest.param(False, "completed", id="success"),
        pytest.param(True, "failed", id="provider-echoes-the-key"),
    ],
)
def test_browser_key_is_used_but_never_persisted_or_logged(
    make_app: AppFactory, tmp_path: Path, log_records: list[Message], echo_key: bool, status: str
) -> None:
    analyst = _Analyst(echo_key=echo_key)
    client = make_app(analyst, api_key="sk-server")
    run_ids = _two_runs(client)
    headers = {"X-OpenRouter-Key": _SENTINEL}
    created = client.post("/api/analyses", json={"runs": run_ids}, headers=headers).json()
    final = poll(client, f"/api/analyses/{created['meta']['id']}", until=_finished)
    assert final["meta"]["status"] == status
    assert [r.headers["authorization"] for r in analyst.analysis_requests] == [
        f"Bearer {_SENTINEL}"
    ]
    stored = [path.read_text(encoding="utf-8") for path in (tmp_path / "data").rglob("*.json")]
    assert all(_SENTINEL not in text for text in stored)
    assert all(_SENTINEL not in str(message) for message in log_records)
