"""Tests for jev_bench.web.routes.labels."""

import pytest
from tests.factories import AppFactory, FakeOpenRouter, seed_generation, services_of


def test_put_label_sets_and_clears(make_app: AppFactory) -> None:
    client = make_app(FakeOpenRouter())
    email_id = f"{seed_generation(services_of(client))}.0001"
    first = client.put(
        f"/api/labels/{email_id}", json={"answers": {"category": "work", "needs_reply": "yes"}}
    )
    assert first.json() == {"category": "work", "needs_reply": "yes"}
    second = client.put(f"/api/labels/{email_id}", json={"answers": {"needs_reply": None}})
    assert second.json() == {"category": "work"}
    assert client.get(f"/api/emails/{email_id}").json()["human"] == {"category": "work"}


@pytest.mark.parametrize(
    ("answers", "status", "detail"),
    [
        pytest.param({"missing": "x"}, 400, "unknown question", id="unknown-question"),
        pytest.param({"category": "phishing"}, 400, "is not an option", id="unknown-option"),
        pytest.param(
            {"topics": ["billing", "billing"]}, 400, "is not an option", id="duplicate-labels"
        ),
        pytest.param({"topics": ["phishing"]}, 400, "is not an option", id="unknown-label"),
        pytest.param({"category": ["work"]}, 400, "is not an option", id="list-for-single-choice"),
        pytest.param({"category": ""}, 400, "is not an option", id="empty-string"),
    ],
)
def test_put_label_validation(
    make_app: AppFactory, answers: dict[str, object], status: int, detail: str
) -> None:
    client = make_app(FakeOpenRouter())
    email_id = f"{seed_generation(services_of(client))}.0001"
    response = client.put(f"/api/labels/{email_id}", json={"answers": answers})
    assert response.status_code == status
    assert detail in response.json()["detail"]


def test_put_label_errors(make_app: AppFactory) -> None:
    client = make_app(FakeOpenRouter())
    email_id = f"{seed_generation(services_of(client))}.0001"
    missing = client.put("/api/labels/20260924-100000-missing-0001.0001", json={"answers": {}})
    assert missing.status_code == 404
    extra = client.put(f"/api/labels/{email_id}", json={"answers": {}, "extra": 1})
    assert extra.status_code == 422


@pytest.mark.parametrize(
    ("topics", "stored"),
    [
        pytest.param(["travel", "billing"], ["travel", "billing"], id="label-list"),
        pytest.param("billing", "billing", id="single-label-id"),
    ],
)
def test_put_label_accepts_multi_answers_and_clears_with_an_empty_list(
    make_app: AppFactory, topics: object, stored: object
) -> None:
    client = make_app(FakeOpenRouter())
    email_id = f"{seed_generation(services_of(client))}.0001"
    url = f"/api/labels/{email_id}"
    first = client.put(url, json={"answers": {"topics": topics, "category": "work"}})
    assert first.json() == {"topics": stored, "category": "work"}
    assert client.get(f"/api/emails/{email_id}").json()["human"]["topics"] == stored
    cleared = client.put(url, json={"answers": {"topics": []}})
    assert cleared.json() == {"category": "work"}
