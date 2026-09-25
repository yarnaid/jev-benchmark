"""Tests for jev_bench.classifiers.llm_schema."""

import pytest

from jev_bench.classifiers.llm_schema import all_in_one_schema, answers_schema, email_refs
from jev_bench.questions import QuestionSet


def _options(description: str, *ids: str) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(ids),
        "properties": {option: {"type": "number"} for option in ids},
        "description": description,
    }


_EXPECTED_PROPERTIES = {
    "category": _options("What kind of email?", "spam", "personal", "work"),
    "urgency": _options("How urgent?", "low", "today", "now"),
    "needs_reply": {
        "type": "number",
        "description": "Probability that the answer is yes: Needs a reply?",
    },
}


def test_answers_schema(questions: QuestionSet) -> None:
    assert answers_schema(questions) == {
        "type": "object",
        "additionalProperties": False,
        "required": ["category", "urgency", "needs_reply"],
        "properties": _EXPECTED_PROPERTIES,
    }


def test_all_in_one_schema(questions: QuestionSet) -> None:
    schema = all_in_one_schema(questions, ["e001", "e002"])
    item = schema["properties"]["results"]["items"]
    assert schema["required"] == ["results"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["results"]["type"] == "array"
    assert item["required"] == ["ref", "category", "urgency", "needs_reply"]
    assert item["properties"]["ref"] == {"type": "string", "enum": ["e001", "e002"]}
    assert item["properties"]["category"] == _EXPECTED_PROPERTIES["category"]


@pytest.mark.parametrize(
    ("count", "first", "last", "length"),
    [
        pytest.param(3, "e001", "e003", 3, id="small"),
        pytest.param(1000, "e0001", "e1000", 1000, id="widens"),
    ],
)
def test_email_refs(count: int, first: str, last: str, length: int) -> None:
    refs = email_refs(count)
    assert (refs[0], refs[-1], len(refs)) == (first, last, length)


def test_email_refs_empty() -> None:
    assert email_refs(0) == []


def test_multi_schema_asks_for_one_number_per_option(multi_questions: QuestionSet) -> None:
    topics = answers_schema(multi_questions)["properties"]["topics"]
    assert topics == _options("Which topics?", "billing", "meeting", "travel")
