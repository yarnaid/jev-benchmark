"""Tests for jev_bench.web.routes.compare."""

from datetime import timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from tests.factories import AppFactory, FakeOpenRouter, poll, seed_generation, services_of

from jev_bench.questions import ChoiceQuestion, QuestionSet


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


def _topics(report: dict[str, Any]) -> dict[str, Any]:
    return next(question for question in report["questions"] if question["id"] == "topics")


def _label_counts(topics: dict[str, Any], rater_id: str) -> dict[str, int]:
    return next(stats for stats in topics["raters"] if stats["rater"] == rater_id)["label_counts"]


def test_compare_multi_label_threshold(make_app: AppFactory) -> None:
    client = make_app(FakeOpenRouter(), api_key="sk-server")
    _, (jev_run, chat_run) = _two_runs(client)
    runs = f"{jev_run},{chat_run}"
    default = _topics(client.get("/api/compare", params={"runs": runs}).json())
    assert (default["type"], default["threshold"]) == ("multi", 0.75)
    assert _label_counts(default, chat_run) == {"billing": 2, "meeting": 2, "travel": 0}
    assert _label_counts(default, jev_run) == {"billing": 2, "meeting": 0, "travel": 0}
    strict = _topics(client.get("/api/compare", params={"runs": runs, "threshold": 1}).json())
    assert strict["threshold"] == 1.0
    assert _label_counts(strict, chat_run) == {"billing": 2, "meeting": 0, "travel": 0}


@pytest.mark.parametrize(
    "threshold",
    [
        pytest.param("0", id="zero"),
        pytest.param("-0.5", id="negative"),
        pytest.param("1.01", id="above-one"),
        pytest.param("nan", id="nan"),
        pytest.param("inf", id="infinity"),
        pytest.param("abc", id="not-a-number"),
    ],
)
def test_compare_rejects_out_of_range_thresholds(make_app: AppFactory, threshold: str) -> None:
    client = make_app(FakeOpenRouter())
    params = {"runs": "20260924-100000-any-0001", "threshold": threshold}
    assert client.get("/api/compare", params=params).status_code == 422


def _age_with_snapshot(client: TestClient, run_id: str, snapshot: QuestionSet) -> None:
    services = services_of(client)
    older = services.runs.get(run_id)
    earlier = older.created_at - timedelta(days=1)
    services.runs.save(older.model_copy(update={"question_set": snapshot, "created_at": earlier}))


def _v1_topics(options: dict[str, str]) -> QuestionSet:
    topics = ChoiceQuestion(type="choice", id="topics", instructions="?", options=options)
    return QuestionSet(name="v1", questions=(topics,))


@pytest.mark.parametrize(
    ("options", "skipped_older"),
    [
        pytest.param(
            {"billing": "b", "meeting": "m", "travel": "t"}, False, id="v1-choice-is-reused"
        ),
        pytest.param({"billing": "b", "travel": "t"}, True, id="other-options-are-skipped"),
    ],
)
def test_compare_reads_the_newest_runs_question_set_in_any_order(
    make_app: AppFactory, options: dict[str, str], skipped_older: bool
) -> None:
    client = make_app(FakeOpenRouter(), api_key="sk-server")
    _, run_ids = _two_runs(client)
    _age_with_snapshot(client, run_ids[0], _v1_topics(options))
    for order in (run_ids, run_ids[::-1]):
        topics = _topics(client.get("/api/compare", params={"runs": ",".join(order)}).json())
        assert topics["type"] == "multi"
        assert topics["skipped"] == ([run_ids[0]] if skipped_older else [])
        assert (run_ids[0] in {stats["rater"] for stats in topics["raters"]}) is not skipped_older
