"""Tests for jev_bench.questions."""

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from jev_bench.questions import (
    ChoiceQuestion,
    NoulQuestion,
    QuestionSet,
    ScoreQuestion,
    compatible,
    load_question_set,
    one_hot,
    render_questions,
)

_VALID_TOML = """
name = "t"

[[questions]]
id = "category"
type = "choice"
instructions = "Kind?"

[questions.options]
spam = "Junk"
personal = "Friend"

[[questions]]
id = "needs_reply"
type = "noul"
instructions = "Reply?"

[questions.options]
yes = "Reply expected"
no = "No reply"
"""


def _doc(**question: Any) -> dict[str, Any]:
    return {"name": "t", "questions": [question]}


def _choice(qid: str = "q") -> dict[str, Any]:
    return {"id": qid, "type": "choice", "instructions": "?", "options": {"a": "A", "b": "B"}}


def test_load_question_set_keeps_order_and_types(tmp_path: Path) -> None:
    path = tmp_path / "questions.toml"
    path.write_text(_VALID_TOML, encoding="utf-8")
    loaded = load_question_set(path)
    assert loaded.ids == ("category", "needs_reply")
    assert isinstance(loaded.get("category"), ChoiceQuestion)
    assert isinstance(loaded.get("needs_reply"), NoulQuestion)
    assert loaded.get("category").option_ids == ("spam", "personal")


@pytest.mark.parametrize(
    "doc",
    [
        pytest.param({"name": "t", "questions": []}, id="no-questions"),
        pytest.param(
            _doc(id="q", type="choice", instructions="?", options={"a": "A"}),
            id="one-option",
        ),
        pytest.param(
            _doc(
                id="q",
                type="choice",
                instructions="?",
                options={"a": "A", "B": "b"},
            ),
            id="bad-option-id",
        ),
        pytest.param(
            _doc(
                id="q",
                type="choice",
                instructions="?",
                options={"a": "A", "b": " "},
            ),
            id="blank-description",
        ),
        pytest.param(
            _doc(
                id="q",
                type="noul",
                instructions="?",
                options={"no": "N", "yes": "Y"},
            ),
            id="noul-order",
        ),
        pytest.param(
            _doc(
                id="q",
                type="noul",
                instructions="?",
                options={"yes": "Y", "maybe": "M"},
            ),
            id="noul-options",
        ),
        pytest.param(
            _doc(id="Q1", type="score", instructions="?", options={"a": "A", "b": "B"}),
            id="bad-question-id",
        ),
        pytest.param(
            _doc(id="q", type="rank", instructions="?", options={"a": "A", "b": "B"}),
            id="unknown-type",
        ),
        pytest.param(
            _doc(
                id="q",
                type="choice",
                instructions="",
                options={"a": "A", "b": "B"},
            ),
            id="empty-instructions",
        ),
        pytest.param(
            {"name": "t", "questions": [_choice(), _choice()]},
            id="duplicate-ids",
        ),
    ],
)
def test_invalid_question_sets_are_rejected(doc: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        QuestionSet.model_validate(doc)


def test_get_unknown_question_raises(questions: QuestionSet) -> None:
    with pytest.raises(KeyError):
        questions.get("missing")


def test_render_questions(questions: QuestionSet) -> None:
    assert render_questions(questions) == "\n".join(
        [
            "- category (choose exactly one option): What kind of email?",
            "    - spam: Junk",
            "    - personal: From a friend",
            "    - work: From a colleague",
            "- urgency (ordered scale, lowest level first): How urgent?",
            "    - low: Whenever",
            "    - today: Within a day",
            "    - now: Immediately",
            "- needs_reply (yes/no): Needs a reply?",
            "    - yes: Reply expected",
            "    - no: No reply expected",
        ]
    )


@pytest.mark.parametrize(
    ("question_id", "option", "expected"),
    [
        pytest.param(
            "category",
            "personal",
            {"spam": 0.0, "personal": 1.0, "work": 0.0},
            id="choice",
        ),
        pytest.param(
            "urgency",
            "now",
            {"low": 0.0, "today": 0.0, "now": 1.0},
            id="score",
        ),
        pytest.param(
            "needs_reply",
            "yes",
            {"yes": 1.0, "no": 0.0},
            id="noul",
        ),
    ],
)
def test_one_hot(
    questions: QuestionSet,
    question_id: str,
    option: str,
    expected: dict[str, float],
) -> None:
    assert one_hot(questions.get(question_id), option) == expected


def test_one_hot_rejects_unknown_option(questions: QuestionSet) -> None:
    with pytest.raises(ValueError, match="not an option"):
        one_hot(questions.get("category"), "phishing")


_A_B = {"a": "A", "b": "B"}


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        pytest.param(
            ChoiceQuestion(type="choice", id="q", instructions="x", options=_A_B),
            ChoiceQuestion(
                type="choice",
                id="q",
                instructions="changed wording",
                options={"a": "1", "b": "2"},
            ),
            True,
            id="same-type-and-ids",
        ),
        pytest.param(
            ChoiceQuestion(type="choice", id="q", instructions="x", options=_A_B),
            ChoiceQuestion(
                type="choice",
                id="q",
                instructions="x",
                options={"b": "B", "a": "A"},
            ),
            False,
            id="reordered",
        ),
        pytest.param(
            ChoiceQuestion(type="choice", id="q", instructions="x", options=_A_B),
            ScoreQuestion(type="score", id="q", instructions="x", options=_A_B),
            False,
            id="different-type",
        ),
    ],
)
def test_compatible(
    left: ChoiceQuestion | ScoreQuestion,
    right: ChoiceQuestion | ScoreQuestion,
    expected: bool,
) -> None:
    assert compatible(left, right) is expected
