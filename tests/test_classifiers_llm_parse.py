"""Tests for jev_bench.classifiers.llm_parse."""

import math
from typing import Any

import pytest

from jev_bench.classifiers.llm_parse import RefCounter, parse_email_answers, split_results
from jev_bench.questions import QuestionSet

_VALID: dict[str, Any] = {
    "category": {"spam": 0.7, "personal": 0.2, "work": 0.1},
    "urgency": {"low": 1, "today": 1, "now": 2},
    "needs_reply": 0.25,
    "ref": "e001",
}


def test_parse_email_answers_valid(questions: QuestionSet) -> None:
    parsed, notes = parse_email_answers(_VALID, questions)
    assert notes == []
    assert parsed["category"] == pytest.approx({"spam": 0.7, "personal": 0.2, "work": 0.1})
    assert parsed["urgency"] == pytest.approx({"low": 0.25, "today": 0.25, "now": 0.5})
    assert parsed["needs_reply"] == pytest.approx({"yes": 0.25, "no": 0.75})


@pytest.mark.parametrize(
    ("question", "value", "expected"),
    [
        pytest.param("needs_reply", math.nan, None, id="noul-nan"),
        pytest.param("needs_reply", "0.7", None, id="noul-string"),
        pytest.param("needs_reply", 3.0, {"yes": 1.0, "no": 0.0}, id="noul-clipped"),
        pytest.param(
            "category", {"spam": 0, "personal": 0, "work": 0}, None, id="choice-zero-mass"
        ),
        pytest.param(
            "category",
            {"spam": -1, "personal": 1, "work": 0},
            {"spam": 0.0, "personal": 1.0, "work": 0.0},
            id="choice-negative-clipped",
        ),
        pytest.param("category", [0.5, 0.5], None, id="choice-not-object"),
        pytest.param("urgency", None, None, id="score-missing"),
    ],
)
def test_parse_email_answers_degraded(
    questions: QuestionSet, question: str, value: object, expected: dict[str, float] | None
) -> None:
    parsed, notes = parse_email_answers({**_VALID, question: value}, questions)
    if expected is None:
        assert question not in parsed
        assert notes == [f"{question}: missing or unusable probabilities"]
    else:
        assert parsed[question] == pytest.approx(expected)
        assert notes == []


def test_parse_email_answers_rejects_non_object(questions: QuestionSet) -> None:
    assert parse_email_answers([1, 2], questions) == ({}, ["answer is not a JSON object"])


_A = {"ref": "e001", "x": 1}
_B = {"ref": "e002", "x": 2}


@pytest.mark.parametrize(
    ("payload", "found", "notes"),
    [
        pytest.param({"results": [_A, _B]}, {"e001": _A, "e002": _B}, {}, id="complete"),
        pytest.param({"results": [_B]}, {"e002": _B}, {}, id="missing-ref"),
        pytest.param(
            {"results": [_A, {"ref": "e001", "x": 9}]},
            {"e001": _A},
            {"e001": "duplicate ref in response; first occurrence kept"},
            id="duplicate",
        ),
        pytest.param(
            {"results": [{"ref": "e999"}, "junk", 5, {"no": "ref"}]},
            {},
            {},
            id="unknown-and-junk-ignored",
        ),
    ],
)
def test_split_results(
    payload: dict[str, Any], found: dict[str, Any], notes: dict[str, str]
) -> None:
    assert split_results(payload, ["e001", "e002"]) == (found, notes)


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"items": []}, id="no-results-key"),
        pytest.param({"results": {"e001": {}}}, id="results-not-list"),
        pytest.param([_A], id="top-level-list"),
    ],
)
def test_split_results_requires_results_array(payload: object) -> None:
    with pytest.raises(ValueError, match="no results array"):
        split_results(payload, ["e001"])


def test_ref_counter_counts_tokens_split_across_chunks() -> None:
    seen: list[int] = []
    counter = RefCounter(seen.append)
    chunks = [
        '{"results":[{"',
        "re",
        'f":"e001","x":1},',
        '{"ref":"e002","x":2}]}',
        " no more ",
    ]
    for chunk in chunks:
        counter.feed(chunk)
    assert counter.count == 2
    assert seen == [1, 2]


def test_multi_answers_are_normalized_like_choice(multi_questions: QuestionSet) -> None:
    raw = {**_VALID, "topics": {"billing": 2, "meeting": 1, "travel": 1}}
    parsed, notes = parse_email_answers(raw, multi_questions)
    assert notes == []
    assert parsed["topics"] == pytest.approx({"billing": 0.5, "meeting": 0.25, "travel": 0.25})
