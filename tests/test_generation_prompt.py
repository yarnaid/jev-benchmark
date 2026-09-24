"""Tests for jev_bench.generation.prompt."""

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from jev_bench.generation.config import GenerationConfig
from jev_bench.generation.plan import PlanItem, resolve_traits
from jev_bench.generation.prompt import (
    count_mismatches,
    generation_schema,
    parse_generator_output,
    render_prompts,
)
from jev_bench.questions import QuestionSet, render_questions

_CONFIG = GenerationConfig.model_validate(
    {
        "models": ["m"],
        "system_prompt": "Questions:\n$questions",
        "user_prompt": ("Write a $category email ($category_prompt), $tone_prompt, sent $sent_at."),
        "traits": [
            {"name": "category", "question": "category"},
            {"name": "tone", "values": {"calm": {"prompt": "calm tone"}}},
        ],
    }
)
_ITEM = PlanItem(
    index=1,
    model="m",
    sent_at=datetime(2026, 9, 20, 9, 30, tzinfo=UTC),
    traits={"category": "spam", "tone": "calm"},
)
_OUTPUT: dict[str, Any] = {
    "email": {
        "sender": {"name": "Deals Team", "address": "deals@promo.test"},
        "to": [{"name": "Ann", "address": "ann@mail.test"}],
        "cc": [],
        "subject": "You won",
        "body": "Claim your prize now.",
    },
    "answers": {"needs_reply": "no", "category": "spam", "urgency": "now"},
}


def test_render_prompts(questions: QuestionSet) -> None:
    system, user = render_prompts(_CONFIG, resolve_traits(_CONFIG, questions), _ITEM, questions)
    assert system == "Questions:\n" + render_questions(questions)
    assert user == "Write a spam email (Junk), calm tone, sent 2026-09-20T09:30:00+00:00."


def test_generation_schema(questions: QuestionSet) -> None:
    schema = generation_schema(questions)
    email = schema["properties"]["email"]
    answers = schema["properties"]["answers"]
    assert schema["required"] == ["email", "answers"]
    assert email["required"] == ["sender", "to", "cc", "subject", "body"]
    assert email["properties"]["to"]["items"]["required"] == ["name", "address"]
    assert answers["properties"]["urgency"] == {"type": "string", "enum": ["low", "today", "now"]}
    assert answers["additionalProperties"] is False


def test_parse_generator_output_orders_answers(questions: QuestionSet) -> None:
    output = parse_generator_output(json.dumps(_OUTPUT), questions)
    assert list(output.answers) == ["category", "urgency", "needs_reply"]
    assert output.email.to[0].address == "ann@mail.test"


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda doc: doc["answers"].update(category="phishing"), id="unknown-option"),
        pytest.param(lambda doc: doc["answers"].pop("urgency"), id="missing-answer"),
        pytest.param(lambda doc: doc["email"].update(to=[]), id="no-recipients"),
        pytest.param(lambda doc: doc["email"].update(body=""), id="empty-body"),
    ],
)
def test_parse_generator_output_rejects(questions: QuestionSet, mutate: Any) -> None:
    doc = json.loads(json.dumps(_OUTPUT))
    mutate(doc)
    with pytest.raises(ValueError, match=r"validation|missing or invalid"):
        parse_generator_output(json.dumps(doc), questions)


def test_parse_generator_output_rejects_invalid_json(questions: QuestionSet) -> None:
    with pytest.raises(ValueError, match=r"JSON|Expecting"):
        parse_generator_output("{not json", questions)


@pytest.mark.parametrize(
    ("answers", "expected"),
    [
        pytest.param({"category": "spam"}, 0, id="consistent"),
        pytest.param({"category": "work"}, 1, id="contradicts"),
        pytest.param({}, 1, id="missing"),
    ],
)
def test_count_mismatches(questions: QuestionSet, answers: dict[str, str], expected: int) -> None:
    assert count_mismatches(_ITEM, resolve_traits(_CONFIG, questions), answers) == expected
