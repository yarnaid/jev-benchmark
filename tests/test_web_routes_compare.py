"""Tests for jev_bench.web.routes.compare."""

from fastapi.testclient import TestClient
from tests.factories import AppFactory, FakeOpenRouter, poll, seed_generation, services_of


def _finished(client: TestClient, run_id: str) -> None:
    poll(client, f"/api/runs/{run_id}", until=lambda body: body["meta"]["status"] != "running")


def _two_runs(client: TestClient) -> tuple[str, list[str]]:
    generation_id = seed_generation(services_of(client))
    run_ids = [
        client.post("/api/runs", json={"column": column, "generation_ids": [generation_id]}).json()[
            "meta"
        ]["id"]
        for column in ("jev", "anthropic")
    ]
    for run_id in run_ids:
        _finished(client, run_id)
    return generation_id, run_ids


def test_compare_two_runs_with_reference_then_human(make_app: AppFactory) -> None:
    client = make_app(FakeOpenRouter(), api_key="sk-server")
    generation_id, run_ids = _two_runs(client)
    report = client.get("/api/compare", params={"runs": ",".join(run_ids)}).json()
    assert [rater["id"] for rater in report["raters"]] == [*run_ids, "reference"]
    assert "question_set" not in report["raters"][0]["run"]
    assert "params" not in report["raters"][0]["run"]
    category = next(question for question in report["questions"] if question["id"] == "category")
    assert len(category["pairs"]) == 3
    run_pair = next(pair for pair in category["pairs"] if {pair["a"], pair["b"]} == set(run_ids))
    assert run_pair["agreement"] == 1.0
    assert run_pair["kappa"] is None
    assert category["fleiss_kappa"] is None
    client.put(f"/api/labels/{generation_id}.0001", json={"answers": {"category": "work"}})
    with_human = client.get("/api/compare", params={"runs": ",".join(run_ids)}).json()
    assert [rater["id"] for rater in with_human["raters"]][-1] == "human"


def test_compare_errors(make_app: AppFactory) -> None:
    client = make_app(FakeOpenRouter())
    missing = client.get("/api/compare", params={"runs": "20260924-100000-missing-0001"})
    assert missing.status_code == 404
    assert client.get("/api/compare", params={"runs": " , "}).status_code == 400
    assert client.get("/api/compare").status_code == 422
