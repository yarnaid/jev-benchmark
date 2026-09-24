"""Tests for jev_bench.compare.rows."""

import pytest
from tests.factories import EmailFactory

from jev_bench.compare.raters import Rater, RaterKind
from jev_bench.compare.rows import disagreement_index, email_rows
from jev_bench.questions import ChoiceQuestion, Distribution, QuestionSet

_IDS = ["g.0001", "g.0002", "g.0003", "g.0004"]
_CATEGORY_A = [
    {"spam": 0.9, "personal": 0.05, "work": 0.05},
    {"spam": 0.1, "personal": 0.8, "work": 0.1},
    {"spam": 0.2, "personal": 0.2, "work": 0.6},
    {"spam": 0.5, "personal": 0.3, "work": 0.2},
]
_URGENCY = [
    {"low": 0.1, "today": 0.2, "now": 0.7},
    {"low": 0.6, "today": 0.3, "now": 0.1},
    {"low": 0.2, "today": 0.6, "now": 0.2},
    {"low": 0.1, "today": 0.1, "now": 0.8},
]
_REPLY = [{"yes": p, "no": 1 - p} for p in (0.2, 0.9, 0.4, 0.7)]


def _answers(category: list[Distribution]) -> dict[str, dict[str, Distribution]]:
    return {
        email_id: {"category": category[i], "urgency": _URGENCY[i], "needs_reply": _REPLY[i]}
        for i, email_id in enumerate(_IDS)
    }


def _rater(
    rater_id: str,
    kind: RaterKind,
    answers: dict[str, dict[str, Distribution]],
    questions: QuestionSet,
) -> Rater:
    return Rater(
        id=rater_id, label=rater_id.upper(), kind=kind, questions=questions, answers=answers
    )


@pytest.fixture
def run_a(questions: QuestionSet) -> Rater:
    return _rater("a", "run", _answers(_CATEGORY_A), questions)


@pytest.fixture
def run_b(questions: QuestionSet) -> Rater:
    category = [*_CATEGORY_A[:2], {"spam": 0.7, "personal": 0.1, "work": 0.2}, _CATEGORY_A[3]]
    return _rater("b", "run", _answers(category), questions)


def test_email_rows_and_disagreement(questions: QuestionSet, run_a: Rater, run_b: Rater) -> None:
    emails = [EmailFactory(id=email_id) for email_id in _IDS]
    rows = email_rows(emails, [run_a, run_b], {"g.0003": {"category": "work"}}, questions)
    by_id = {row.id: row for row in rows}
    assert by_id["g.0001"].disagreement == 0.0
    assert (by_id["g.0003"].disagreement or 0.0) > 0.0
    assert by_id["g.0003"].top == {
        "a": {"category": "work", "urgency": "today", "needs_reply": "no"},
        "b": {"category": "spam", "urgency": "today", "needs_reply": "no"},
    }
    assert by_id["g.0003"].human == {"category": "work"}
    assert by_id["g.0001"].reference == emails[0].reference_answers
    assert disagreement_index("g.0001", [run_a], questions) is None


def test_incompatible_rater_excluded_from_email_rows_and_disagreement(
    questions: QuestionSet, run_a: Rater
) -> None:
    changed = QuestionSet(
        name="changed",
        questions=(
            ChoiceQuestion(
                type="choice", id="category", instructions="?", options={"spam": "s", "ham": "h"}
            ),
        ),
    )
    other = _rater("c", "run", {"g.0001": {"category": {"spam": 0.2, "ham": 0.8}}}, changed)
    email = EmailFactory(id="g.0001")
    rows = email_rows([email], [run_a, other], {}, questions)
    assert rows[0].top["c"] == {}
    assert "category" in rows[0].top["a"]
    assert disagreement_index("g.0001", [run_a, other], questions) is None
