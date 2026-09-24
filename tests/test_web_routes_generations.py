"""Tests for jev_bench.web.routes.generations."""

from pathlib import Path

import pytest
from tests.factories import AppFactory, FakeOpenRouter, poll

from jev_bench.web.deps import NO_KEY_DETAIL

_KEY = {"X-OpenRouter-Key": "sk-browser"}


def _done(body: dict[str, object]) -> bool:
    meta = body["meta"]
    return isinstance(meta, dict) and meta["status"] != "running"


def test_create_generation_runs_in_the_background(make_app: AppFactory) -> None:
    client = make_app(FakeOpenRouter())
    payload = {"name": "Spam", "count": 2, "seed": 3}
    response = client.post("/api/generations", json=payload, headers=_KEY)
    assert response.status_code == 202
    created = response.json()
    assert created["meta"]["status"] == "running"
    assert "config" not in created["meta"]
    assert "question_set" not in created["meta"]
    final = poll(client, f"/api/generations/{created['meta']['id']}", until=_done)
    statuses = (final["meta"]["status"], final["meta"]["done"], final["meta"]["seed"])
    assert statuses == ("completed", 2, 3)
    listed = client.get("/api/generations").json()
    assert [view["meta"]["id"] for view in listed] == [created["meta"]["id"]]
    assert "config" not in listed[0]["meta"]


def test_create_generation_without_key_is_rejected(make_app: AppFactory) -> None:
    response = make_app(FakeOpenRouter()).post("/api/generations", json={"count": 1})
    assert (response.status_code, response.json()["detail"]) == (400, NO_KEY_DETAIL)


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"count": 0}, id="zero"),
        pytest.param({"count": 1, "bogus": 1}, id="extra-field"),
    ],
)
def test_invalid_generation_payload_is_422(
    make_app: AppFactory, payload: dict[str, object]
) -> None:
    client = make_app(FakeOpenRouter())
    response = client.post("/api/generations", json=payload, headers=_KEY)
    assert response.status_code == 422


def test_launch_value_error_is_400(make_app: AppFactory, tmp_path: Path) -> None:
    client = make_app(FakeOpenRouter())
    config = tmp_path / "config" / "generation.toml"
    original = config.read_text(encoding="utf-8")
    broken = original.replace('question = "category"', 'question = "missing"')
    config.write_text(broken, encoding="utf-8")
    response = client.post("/api/generations", json={"count": 1}, headers=_KEY)
    assert response.status_code == 400
    assert "unknown question" in response.json()["detail"]


@pytest.mark.parametrize(
    "path",
    [
        pytest.param("/api/generations/20260924-100000-missing-0001", id="unknown"),
        pytest.param("/api/generations/%2e%2e", id="traversal"),
    ],
)
def test_unknown_generation_is_404(make_app: AppFactory, path: str) -> None:
    client = make_app(FakeOpenRouter())
    assert client.get(path).status_code == 404
    assert client.post(f"{path}/cancel").status_code == 404


def test_cancel_finished_generation_reports_false(make_app: AppFactory) -> None:
    client = make_app(FakeOpenRouter())
    created = client.post("/api/generations", json={"count": 1}, headers=_KEY).json()
    generation_id = created["meta"]["id"]
    poll(client, f"/api/generations/{generation_id}", until=_done)
    result = client.post(f"/api/generations/{generation_id}/cancel").json()
    assert result == {"cancelled": False}
