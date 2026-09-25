"""Tests for jev_bench.compare.rows."""

import pytest
from tests.compare_data import IDS, TOPICS_A, build_run_a, build_run_b, rater, topics_answers
from tests.factories import EmailFactory

from jev_bench.compare.raters import Rater
from jev_bench.compare.rows import disagreement_index, email_rows
from jev_bench.questions import ChoiceQuestion, QuestionSet


@pytest.fixture
def run_a(questions: QuestionSet) -> Rater:
    return build_run_a(questions)


@pytest.fixture
def run_b(questions: QuestionSet) -> Rater:
    return build_run_b(questions)


def test_email_rows_and_disagreement(questions: QuestionSet, run_a: Rater, run_b: Rater) -> None:
    ids = ["g.0001", "g.0002", "g.0003", "g.0004"]
    emails = [EmailFactory(id=email_id) for email_id in ids]
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
    other = rater("c", "run", {"g.0001": {"category": {"spam": 0.2, "ham": 0.8}}}, changed)
    email = EmailFactory(id="g.0001")
    rows = email_rows([email], [run_a, other], {}, questions)
    assert rows[0].top["c"] == {}
    assert "category" in rows[0].top["a"]
    assert disagreement_index("g.0001", [run_a, other], questions) is None


def test_multi_top_answers_follow_the_relative_threshold(multi_questions: QuestionSet) -> None:
    a = rater("a", "run", topics_answers(TOPICS_A), multi_questions)
    emails = [EmailFactory(id=email_id) for email_id in IDS]

    def tops(threshold: float | None) -> dict[str, object]:
        rows = email_rows(emails, [a], {}, multi_questions, threshold=threshold)
        return {row.id: row.top["a"]["topics"] for row in rows}

    assert tops(None) == {
        "g.0001": ["billing", "meeting"],
        "g.0002": ["meeting"],
        "g.0003": ["travel"],
        "g.0004": ["billing", "meeting"],
    }
    assert tops(0.5)["g.0004"] == ["billing", "meeting", "travel"]
    assert tops(1.0)["g.0004"] == ["billing"]


def test_multi_labels_with_equal_probability_keep_option_order(
    multi_questions: QuestionSet,
) -> None:
    tie = {"g.0001": {"topics": {"billing": 0.4, "meeting": 0.2, "travel": 0.4}}}
    rows = email_rows(
        [EmailFactory(id="g.0001")], [rater("a", "run", tie, multi_questions)], {}, multi_questions
    )
    assert rows[0].top["a"]["topics"] == ["billing", "travel"]


def test_score_questions_get_0_to_100_scores(questions: QuestionSet, run_a: Rater) -> None:
    row = email_rows([EmailFactory(id="g.0001")], [run_a], {}, questions)[0]
    assert row.scores == {"a": {"urgency": pytest.approx(80.0)}}
    assert row.reference_scores == {"urgency": 50.0}
