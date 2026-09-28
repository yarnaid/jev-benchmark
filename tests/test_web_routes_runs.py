"""Tests for jev_bench.web.routes.runs."""

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import httpx2
import pytest
from tests.factories import AppFactory, FakeOpenRouter, poll, seed_generation, services_of

if TYPE_CHECKING:
    from loguru import Message

_SENTINEL = "sk-or-v1-SENTINEL-4242"


def _done(body: dict[str, object]) -> bool:
    meta = body["meta"]
    return isinstance(meta, dict) and meta["status"] != "running"


def test_create_run_and_poll_to_completion(make_app: AppFactory) -> None:
    client = make_app(FakeOpenRouter(), api_key="sk-server")
    generation_id = seed_generation(services_of(client))
    payload = {"column": "anthropic", "generation_ids": [generation_id], "mode": "all_in_one"}
    response = client.post("/api/runs", json=payload)
    assert response.status_code == 202
    run_id = response.json()["meta"]["id"]
    final = poll(client, f"/api/runs/{run_id}", until=_done)
    outcome = (final["meta"]["status"], final["meta"]["n_done"], final["meta"]["mode"])
    assert outcome == ("completed", 2, "all_in_one")
    assert "question_set" not in final["meta"]
    assert "params" not in final["meta"]
    assert final["progress"] is None


def test_list_runs_filters_by_exact_generation_set(make_app: AppFactory) -> None:
    client = make_app(FakeOpenRouter(), api_key="sk-server")
    services = services_of(client)
    first = seed_generation(services)
    second = seed_generation(services, generation_id="20260924-110000-seed-bcde")
    one_body = {"column": "jev", "generation_ids": [first]}
    both_body = {"column": "jev", "generation_ids": [first, second]}
    one = client.post("/api/runs", json=one_body).json()["meta"]["id"]
    both = client.post("/api/runs", json=both_body).json()["meta"]["id"]
    poll(client, f"/api/runs/{both}", until=_done)
    poll(client, f"/api/runs/{one}", until=_done)
    only_first = client.get("/api/runs", params={"generations": first}).json()
    assert [view["meta"]["id"] for view in only_first] == [one]
    both_filter = {"generations": f"{second},{first}"}
    both_listed = client.get("/api/runs", params=both_filter).json()
    assert [view["meta"]["id"] for view in both_listed] == [both]
    assert len(client.get("/api/runs").json()) == 2


_KEY_HEADER = {"X-OpenRouter-Key": "k"}
_ONE_GENERATION = ["20260924-100000-seed-abcd"]


@pytest.mark.parametrize(
    ("payload", "headers", "status", "detail"),
    [
        pytest.param({"column": "jev"}, _KEY_HEADER, 422, None, id="no-generations-field"),
        pytest.param(
            {"column": "missing", "generation_ids": _ONE_GENERATION},
            _KEY_HEADER,
            400,
            "unknown column",
            id="unknown-column",
        ),
        pytest.param(
            {"column": "jev", "generation_ids": _ONE_GENERATION, "mode": "all_in_one"},
            _KEY_HEADER,
            400,
            "only accepted for chat",
            id="mode-on-jev",
        ),
        pytest.param(
            {"column": "jev", "generation_ids": _ONE_GENERATION},
            {},
            400,
            "API key",
            id="no-key",
        ),
    ],
)
def test_invalid_run_requests(
    make_app: AppFactory,
    payload: dict[str, object],
    headers: dict[str, str],
    status: int,
    detail: str | None,
) -> None:
    client = make_app(FakeOpenRouter())
    seed_generation(services_of(client))
    response = client.post("/api/runs", json=payload, headers=headers)
    assert response.status_code == status
    if detail is not None:
        assert detail in response.json()["detail"]


def test_unknown_run_is_404(make_app: AppFactory) -> None:
    client = make_app(FakeOpenRouter())
    assert client.get("/api/runs/20260924-100000-missing-0001").status_code == 404
    assert client.post("/api/runs/20260924-100000-missing-0001/cancel").status_code == 404


def _refusing(inner: FakeOpenRouter, *, status: int) -> Callable[[httpx2.Request], httpx2.Response]:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/alpha/decisions"):
            inner.requests.append(request)
            return httpx2.Response(status, json={"error": {"message": "refused", "code": status}})
        return inner(request)

    return handler


@pytest.mark.parametrize(
    ("refuse", "expected_status"),
    [
        pytest.param(False, "completed", id="success"),
        pytest.param(True, "failed", id="provider-refusal-402"),
    ],
)
def test_browser_key_is_used_but_never_persisted_or_logged(
    make_app: AppFactory,
    tmp_path: Path,
    log_records: list[Message],
    refuse: bool,
    expected_status: str,
) -> None:
    upstream = FakeOpenRouter()
    handler = _refusing(upstream, status=402) if refuse else upstream
    client = make_app(handler)
    generation_id = seed_generation(services_of(client))
    created = client.post(
        "/api/runs",
        json={"column": "jev", "generation_ids": [generation_id]},
        headers={"X-OpenRouter-Key": _SENTINEL},
    )
    final = poll(client, f"/api/runs/{created.json()['meta']['id']}", until=_done)
    assert final["meta"]["status"] == expected_status
    decisions = [
        request for request in upstream.requests if request.url.path.endswith("/alpha/decisions")
    ]
    assert decisions
    assert all(request.headers["Authorization"] == f"Bearer {_SENTINEL}" for request in decisions)
    data_files = (path for path in (tmp_path / "data").rglob("*") if path.is_file())
    stored = [path.read_text(encoding="utf-8") for path in data_files]
    assert stored
    assert all(_SENTINEL not in text for text in stored)
    assert all(_SENTINEL not in str(message) for message in log_records)


def test_cancel_running_run(make_app: AppFactory) -> None:
    client = make_app(FakeOpenRouter(), api_key="sk-server")
    generation_id = seed_generation(services_of(client), emails=1)
    body = {"column": "jev", "generation_ids": [generation_id]}
    run_id = client.post("/api/runs", json=body).json()["meta"]["id"]
    result = client.post(f"/api/runs/{run_id}/cancel").json()
    final = poll(client, f"/api/runs/{run_id}", until=_done)
    assert final["meta"]["status"] in ({"cancelled"} if result["cancelled"] else {"completed"})


@pytest.mark.parametrize(
    ("body", "status"),
    [
        pytest.param({"column": "jev"}, 200, id="jev"),
        pytest.param({"column": "anthropic", "mode": "all_in_one"}, 200, id="chat-all-in-one"),
        pytest.param({"column": "nope"}, 400, id="unknown-column"),
        pytest.param({"column": "jev", "mode": "all_in_one"}, 400, id="mode-for-non-chat"),
    ],
)
def test_estimate_run_needs_no_key(make_app: AppFactory, body: dict[str, str], status: int) -> None:
    client = make_app(FakeOpenRouter())
    generation_id = seed_generation(services_of(client), emails=2)
    response = client.post("/api/runs/estimate", json={**body, "generation_ids": [generation_id]})
    assert response.status_code == status
    if status == 200:
        estimate = response.json()
        assert estimate["n_emails"] == 2
        assert estimate["token_cost"] > 0
        assert estimate["history_cost"] is None
    assert services_of(client).runs.list_metas() == []
