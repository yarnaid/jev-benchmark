"""Tests for jev_bench.classifiers.jev."""

import json
import math
from typing import Any

import httpx2
import pytest
from tests.factories import ClientFactory, EmailFactory

from jev_bench.benchmark_config import JevParams, TokenParams
from jev_bench.catalog import ModelInfo
from jev_bench.classifiers.jev import (
    JevClassifier,
    parse_decisions,
    questions_payload,
)
from jev_bench.questions import QuestionSet
from jev_bench.tokens import Budget, estimate_tokens

_ANSWERS: dict[str, Any] = {
    "category": {
        "type": "choice",
        "choice": "spam",
        "confidence": 0.75,
        "probabilities": {"spam": 0.84, "personal": 0.16, "work": 0},
    },
    "urgency": {
        "type": "score",
        "score": 1.99,
        "confidence": 0.99,
        "probabilities": {"0": 0, "1": 0.01, "2": 0.99},
    },
    "needs_reply": {"type": "noul", "noul": 0.96},
}
_EXPECTED = {
    "category": {"spam": 0.84, "personal": 0.16, "work": 0.0},
    "urgency": {"low": 0.0, "today": 0.01, "now": 0.99},
    "needs_reply": {"yes": 0.96, "no": 0.04},
}


def test_questions_payload(questions: QuestionSet) -> None:
    assert questions_payload(questions) == {
        "category": {
            "type": "choice",
            "instructions": "What kind of email?",
            "criteria": {"spam": "Junk", "personal": "From a friend", "work": "From a colleague"},
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent?",
            "criteria": ["Whenever", "Within a day", "Immediately"],
        },
        "needs_reply": {
            "type": "noul",
            "instructions": "Needs a reply?",
            "criteria": {"true": "Reply expected", "false": "No reply expected"},
        },
    }


def test_parse_decisions_full_answer(questions: QuestionSet) -> None:
    parsed, notes = parse_decisions(_ANSWERS, questions)
    assert notes == []
    assert parsed == {key: pytest.approx(value) for key, value in _EXPECTED.items()}


@pytest.mark.parametrize(
    ("override", "question", "expected", "note"),
    [
        pytest.param(
            {"type": "choice", "choice": "work"},
            "category",
            {"spam": 0.0, "personal": 0.0, "work": 1.0},
            "one-hot on choice",
            id="choice-without-probabilities",
        ),
        pytest.param(
            {"type": "choice", "choice": "unknown"},
            "category",
            None,
            "no usable choice",
            id="choice-unknown",
        ),
        pytest.param(
            {"type": "score", "score": 1.6},
            "urgency",
            {"low": 0.0, "today": 0.0, "now": 1.0},
            "round(score)",
            id="score-without-probabilities",
        ),
        pytest.param(
            {"type": "score", "score": 7.0},
            "urgency",
            {"low": 0.0, "today": 0.0, "now": 1.0},
            "round(score)",
            id="score-clamped-high",
        ),
        pytest.param(
            {"type": "score", "score": math.nan},
            "urgency",
            None,
            "no usable score",
            id="score-nan",
        ),
        pytest.param(
            {"type": "score", "probabilities": {"0": 1, "2": 3}},
            "urgency",
            {"low": 0.25, "today": 0.0, "now": 0.75},
            None,
            id="score-partial-probabilities",
        ),
        pytest.param(
            {"type": "noul", "noul": 1.3},
            "needs_reply",
            {"yes": 1.0, "no": 0.0},
            None,
            id="noul-clipped",
        ),
        pytest.param(
            {"type": "noul"}, "needs_reply", None, "noul probability missing", id="noul-missing"
        ),
        pytest.param(
            {"type": "choice", "choice": "spam"},
            "needs_reply",
            None,
            "missing or mistyped",
            id="wrong-type",
        ),
    ],
)
def test_parse_decisions_degraded_answers(
    questions: QuestionSet,
    override: dict[str, Any],
    question: str,
    expected: dict[str, float] | None,
    note: str | None,
) -> None:
    parsed, notes = parse_decisions({**_ANSWERS, question: override}, questions)
    assert parsed.get(question) == (None if expected is None else pytest.approx(expected))
    if note is None:
        assert notes == []
    else:
        assert len(notes) == 1
        assert notes[0].startswith(f"{question}: ")
        assert note in notes[0]


def test_parse_decisions_missing_question(questions: QuestionSet) -> None:
    parsed, notes = parse_decisions({"category": _ANSWERS["category"]}, questions)
    assert set(parsed) == {"category"}
    assert notes == [
        "urgency: missing or mistyped answer",
        "needs_reply: missing or mistyped answer",
    ]


def _classifier(
    client: Any, questions: QuestionSet, info: ModelInfo | None = None
) -> JevClassifier:
    return JevClassifier(
        client=client,
        api_key="sk-test",
        model="typesafe/jev-1.13",
        model_info=info,
        questions=questions,
        params=JevParams(concurrency=3),
        tokens=TokenParams(),
    )


async def test_classify_sends_state_and_parses(
    make_client: ClientFactory, questions: QuestionSet
) -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        body = {
            "id": "gen-dec-1",
            "model": "typesafe/jev-1.13-20260917",
            "answers": _ANSWERS,
            "usage": {"input_tokens": 476, "output_tokens": 70, "cost": 0.00002},
        }
        return httpx2.Response(200, json=body)

    email = EmailFactory()
    result = await _classifier(make_client(handler), questions).classify([email])
    sent = json.loads(seen[0].content)
    assert seen[0].url.path == "/api/alpha/decisions"
    assert seen[0].headers["Authorization"] == "Bearer sk-test"
    assert sent == {
        "model": "typesafe/jev-1.13",
        "state": email.to_state(),
        "questions": questions_payload(questions),
    }
    assert result.outcomes[email.id].answers == {
        key: pytest.approx(value) for key, value in _EXPECTED.items()
    }
    assert result.usage.cost == 0.00002
    assert result.usage.input_tokens == 476
    assert result.resolved_model == "typesafe/jev-1.13-20260917"
    assert result.raw is not None
    assert result.raw["id"] == "gen-dec-1"


async def test_classify_without_answers_is_an_email_error(
    make_client: ClientFactory, questions: QuestionSet
) -> None:
    client = make_client(
        lambda request: httpx2.Response(200, json={"model": "m", "answers": {}, "usage": {}})
    )
    email = EmailFactory()
    outcome = (await _classifier(client, questions).classify([email])).outcomes[email.id]
    assert outcome.answers is None
    assert outcome.error is not None
    assert "missing or mistyped" in outcome.error


async def test_classifier_shape(make_client: ClientFactory, questions: QuestionSet) -> None:
    info = ModelInfo(id="typesafe/jev-1.13", name="Jev", context_length=32_000)
    classifier = _classifier(make_client(lambda request: httpx2.Response(500)), questions, info)
    email = EmailFactory()
    prepared = await classifier.prepare([email])
    assert classifier.emails_per_request == 1
    assert classifier.concurrency == 3
    assert classifier.budget == Budget(total=32_000)
    assert classifier.sizing.output_per_email == 1000
    payload_json = json.dumps(questions_payload(questions), ensure_ascii=False)
    assert classifier.sizing.overhead == estimate_tokens(payload_json, 3.0)
    state_json = json.dumps(email.to_state(), ensure_ascii=False)
    assert classifier.input_tokens(email) == estimate_tokens(state_json, 3.0)
    assert prepared.pending == (email,)
    assert prepared.resolved == {}
