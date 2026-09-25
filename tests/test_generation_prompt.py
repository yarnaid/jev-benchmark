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
        pytest.param(
            lambda doc: doc["answers"].update(category=["spam"]), id="list-for-single-choice"
        ),
    ],
)
def test_parse_generator_output_rejects(questions: QuestionSet, mutate: Any) -> None:
    doc = json.loads(json.dumps(_OUTPUT))
    mutate(doc)
    with pytest.raises(ValueError, match=r"validation|missing or invalid"):
        parse_generator_output(json.dumps(doc), questions)


@pytest.mark.parametrize(
    "wrap",
    [
        pytest.param(lambda doc: f"From: Marcus Bell <m@bell.test>\n\n{doc}", id="preamble"),
        pytest.param(lambda doc: f"{doc}\n\nHope this helps!", id="trailing-text"),
        pytest.param(lambda doc: f"Here you go:\n{doc}\nDone.", id="preamble-and-trailing"),
    ],
)
def test_parse_generator_output_ignores_text_around_the_json(
    questions: QuestionSet, wrap: Any
) -> None:
    output = parse_generator_output(wrap(json.dumps(_OUTPUT)), questions)
    assert output.email.subject == _OUTPUT["email"]["subject"]


def test_parse_generator_output_rejects_invalid_json(questions: QuestionSet) -> None:
    with pytest.raises(ValueError, match=r"JSON|Expecting"):
        parse_generator_output("{not json", questions)


@pytest.mark.parametrize(
    ("requested", "answers", "expected"),
    [
        pytest.param(_ITEM.traits, {"category": "spam"}, 0, id="consistent"),
        pytest.param(_ITEM.traits, {"category": "work"}, 1, id="contradicts"),
        pytest.param(_ITEM.traits, {}, 1, id="missing-answer"),
        pytest.param({}, {"category": "work"}, 0, id="trait-not-requested"),
    ],
)
def test_count_mismatches(
    questions: QuestionSet, requested: dict[str, str], answers: dict[str, str], expected: int
) -> None:
    traits = resolve_traits(_CONFIG, questions)
    assert count_mismatches(requested, traits, answers) == expected


_MULTI_CONFIG = GenerationConfig.model_validate(
    {
        "models": ["m"],
        "system_prompt": "Questions:\n$questions",
        "user_prompt": "Write about $topic ($topic_prompt), sent $sent_at.",
        "traits": [{"name": "topic", "question": "topics"}],
    }
)
_MULTI_OUTPUT: dict[str, Any] = {
    **_OUTPUT,
    "answers": {**_OUTPUT["answers"], "topics": ["meeting", "billing"]},
}


def test_generation_schema_multi_is_an_array_of_options(multi_questions: QuestionSet) -> None:
    answers = generation_schema(multi_questions)["properties"]["answers"]
    assert answers["properties"]["topics"] == {
        "type": "array",
        "items": {"type": "string", "enum": ["billing", "meeting", "travel"]},
    }
    assert "topics" in answers["required"]


@pytest.mark.parametrize(
    "topics",
    [
        pytest.param(["meeting", "billing"], id="label-list-keeps-its-order"),
        pytest.param("meeting", id="single-id-is-one-label"),
    ],
)
def test_parse_generator_output_accepts_label_answers(
    multi_questions: QuestionSet, topics: object
) -> None:
    doc = {**_MULTI_OUTPUT, "answers": {**_MULTI_OUTPUT["answers"], "topics": topics}}
    output = parse_generator_output(json.dumps(doc), multi_questions)
    assert output.answers["topics"] == topics
    assert output.answers["category"] == "spam"


@pytest.mark.parametrize(
    "topics",
    [
        pytest.param([], id="empty-list"),
        pytest.param(["billing", "billing"], id="duplicate"),
        pytest.param(["billing", "phishing"], id="unknown-label"),
    ],
)
def test_parse_generator_output_rejects_bad_label_lists(
    multi_questions: QuestionSet, topics: object
) -> None:
    doc = json.loads(json.dumps(_MULTI_OUTPUT))
    doc["answers"]["topics"] = topics
    with pytest.raises(ValueError, match="missing or invalid"):
        parse_generator_output(json.dumps(doc), multi_questions)


@pytest.mark.parametrize(
    ("answers", "expected"),
    [
        pytest.param({"topics": ["meeting", "billing"]}, 0, id="requested-label-present"),
        pytest.param({"topics": ["meeting"]}, 1, id="requested-label-absent"),
        pytest.param({"topics": "billing"}, 0, id="single-id-answer"),
        pytest.param({}, 1, id="missing-answer"),
    ],
)
def test_count_mismatches_multi_membership(
    multi_questions: QuestionSet, answers: dict[str, str | list[str]], expected: int
) -> None:
    traits = resolve_traits(_MULTI_CONFIG, multi_questions)
    assert count_mismatches({"topic": "billing"}, traits, answers) == expected
