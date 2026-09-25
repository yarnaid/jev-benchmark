"""Tests for jev_bench.questions."""

import math
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from jev_bench.questions import (
    AnyQuestion,
    ChoiceQuestion,
    MultiQuestion,
    NoulQuestion,
    QuestionSet,
    ScoreQuestion,
    compatible,
    hard_distribution,
    load_question_set,
    one_hot,
    render_questions,
    with_threshold,
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


_TWO = {"a": "A", "b": "B"}


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
            _doc(id="q", type="multi", instructions="?", options=_TWO, threshold=0),
            id="multi-threshold-zero",
        ),
        pytest.param(
            _doc(id="q", type="multi", instructions="?", options=_TWO, threshold=1.5),
            id="multi-threshold-above-one",
        ),
        pytest.param(
            _doc(id="q", type="multi", instructions="?", options=_TWO, threshold=math.nan),
            id="multi-threshold-nan",
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
            True,
            id="choice-and-score-share-the-distribution-shape",
        ),
        pytest.param(
            ChoiceQuestion(type="choice", id="q", instructions="x", options=_A_B),
            MultiQuestion(type="multi", id="q", instructions="x", options=_A_B),
            True,
            id="choice-and-multi-share-the-distribution-shape",
        ),
        pytest.param(
            MultiQuestion(type="multi", id="q", instructions="x", options=_A_B),
            MultiQuestion(type="multi", id="q", instructions="x", options=_A_B, threshold=0.5),
            True,
            id="threshold-is-not-part-of-compatibility",
        ),
        pytest.param(
            NoulQuestion(type="noul", id="q", instructions="x", options={"yes": "Y", "no": "N"}),
            ChoiceQuestion(
                type="choice", id="q", instructions="x", options={"yes": "Y", "no": "N"}
            ),
            False,
            id="noul-only-matches-noul",
        ),
    ],
)
def test_compatible(
    left: AnyQuestion,
    right: AnyQuestion,
    expected: bool,
) -> None:
    assert compatible(left, right) is expected


_MULTI_TOML = """
name = "t"

[[questions]]
id = "topics"
type = "multi"
instructions = "Which topics?"
threshold = 0.7

[questions.options]
billing = "About money"
meeting = "About a meeting"
"""


def test_load_question_set_reads_multi_threshold(tmp_path: Path) -> None:
    path = tmp_path / "questions.toml"
    path.write_text(_MULTI_TOML, encoding="utf-8")
    topics = load_question_set(path).get("topics")
    assert isinstance(topics, MultiQuestion)
    assert topics.threshold == 0.7


def test_multi_threshold_defaults_to_80_percent() -> None:
    question = QuestionSet.model_validate(
        _doc(id="q", type="multi", instructions="?", options=_TWO)
    ).get("q")
    assert isinstance(question, MultiQuestion)
    assert question.threshold == 0.8


def test_render_questions_hints_multi_label(multi_questions: QuestionSet) -> None:
    lines = render_questions(multi_questions).splitlines()
    assert "- topics (multi-label: one or more options can apply): Which topics?" in lines
    assert "    - meeting: About a meeting" in lines


_NONE = {"billing": 0.0, "meeting": 0.0, "travel": 0.0}


@pytest.mark.parametrize(
    ("question_id", "answer", "expected"),
    [
        pytest.param(
            "category", "work", {"spam": 0.0, "personal": 0.0, "work": 1.0}, id="choice-id"
        ),
        pytest.param("category", "phishing", None, id="choice-unknown-id"),
        pytest.param("category", ["work"], None, id="choice-given-a-list"),
        pytest.param("needs_reply", "yes", {"yes": 1.0, "no": 0.0}, id="noul-id"),
        pytest.param(
            "topics",
            ["travel", "billing"],
            {**_NONE, "billing": 0.5, "travel": 0.5},
            id="multi-list",
        ),
        pytest.param("topics", ["meeting"], {**_NONE, "meeting": 1.0}, id="multi-one-label"),
        pytest.param("topics", "meeting", {**_NONE, "meeting": 1.0}, id="multi-single-id"),
        pytest.param("topics", [], None, id="multi-empty"),
        pytest.param("topics", ["billing", "billing"], None, id="multi-duplicate"),
        pytest.param("topics", ["billing", "phishing"], None, id="multi-unknown"),
        pytest.param("topics", ["billing", 3], None, id="multi-non-string"),
        pytest.param("topics", "phishing", None, id="multi-unknown-single-id"),
        pytest.param("topics", None, None, id="multi-missing"),
    ],
)
def test_hard_distribution(
    multi_questions: QuestionSet,
    question_id: str,
    answer: object,
    expected: dict[str, float] | None,
) -> None:
    assert hard_distribution(multi_questions.get(question_id), answer) == expected


def test_three_labels_share_the_mass_evenly(multi_questions: QuestionSet) -> None:
    labels = ["billing", "meeting", "travel"]
    distribution = hard_distribution(multi_questions.get("topics"), labels)
    assert distribution == pytest.approx(dict.fromkeys(labels, 1 / 3))


def test_with_threshold_replaces_only_multi_thresholds(multi_questions: QuestionSet) -> None:
    changed = with_threshold(multi_questions, 0.55)
    topics = changed.get("topics")
    assert isinstance(topics, MultiQuestion)
    assert topics.threshold == 0.55
    assert changed.get("category") == multi_questions.get("category")
    assert with_threshold(multi_questions, None) is multi_questions


@pytest.mark.parametrize(
    "threshold",
    [
        pytest.param(0.0, id="zero"),
        pytest.param(1.01, id="above-one"),
        pytest.param(-0.5, id="negative"),
        pytest.param(math.nan, id="nan"),
    ],
)
def test_with_threshold_rejects_out_of_range(
    multi_questions: QuestionSet, threshold: float
) -> None:
    with pytest.raises(ValueError, match="threshold"):
        with_threshold(multi_questions, threshold)
