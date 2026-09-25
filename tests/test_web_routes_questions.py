"""Tests for jev_bench.web.routes.questions."""

from tests.factories import AppFactory, FakeOpenRouter, services_of


def test_questions_returns_the_current_question_set(make_app: AppFactory) -> None:
    client = make_app(FakeOpenRouter())
    body = client.get("/api/questions").json()
    assert body["name"] == "mini"
    assert [question["id"] for question in body["questions"]] == [
        "category",
        "urgency",
        "needs_reply",
        "topics",
    ]
    topics = body["questions"][3]
    assert (topics["type"], topics["threshold"]) == ("multi", 0.75)


def test_broken_question_set_is_a_readable_error(make_app: AppFactory) -> None:
    client = make_app(FakeOpenRouter())
    config_dir = services_of(client).settings.config_dir
    (config_dir / "questions.toml").write_text("name = \n", encoding="utf-8")
    response = client.get("/api/questions")
    assert response.status_code == 500
    assert "questions.toml" in response.json()["detail"]
