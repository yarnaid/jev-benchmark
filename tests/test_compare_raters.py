"""Tests for jev_bench.compare.raters."""

from datetime import UTC, datetime

from tests.factories import EmailFactory

from jev_bench.benchmark_config import JevParams
from jev_bench.compare.raters import human_rater, reference_rater, run_rater
from jev_bench.compare.report import compare
from jev_bench.questions import ChoiceQuestion, QuestionSet
from jev_bench.store.runs import Prediction, RunMeta


def test_human_rater(questions: QuestionSet) -> None:
    assert human_rater({}, questions) is None
    assert human_rater({"g.0001": {"category": "not-an-option"}}, questions) is None
    rater = human_rater({"g.0001": {"category": "work"}, "g.0002": {}}, questions)
    assert rater is not None
    assert (rater.kind, rater.hard) == ("human", True)
    assert rater.answers == {"g.0001": {"category": {"spam": 0.0, "personal": 0.0, "work": 1.0}}}


def test_run_rater_skips_failed_predictions(questions: QuestionSet) -> None:
    meta = RunMeta(
        id="20260924-100000-jev-x-0001",
        column="jev",
        kind="decisions",
        model="typesafe/jev-1.13",
        generation_ids=("g",),
        mode="per_email",
        emails_per_request=1,
        question_set=questions,
        params=JevParams(),
        concurrency=1,
        created_at=datetime(2026, 9, 24, tzinfo=UTC),
        n_emails=3,
    )
    predictions = [
        Prediction(email_id="g.0001", answers={"needs_reply": {"yes": 1.0, "no": 0.0}}),
        Prediction(email_id="g.0002", error="HTTP 500"),
        Prediction(
            email_id="g.0003", answers={"needs_reply": {"yes": 0.5, "no": 0.5}}, error="partial"
        ),
    ]
    rater = run_rater(meta, predictions)
    assert (rater.id, rater.kind, rater.hard) == (meta.id, "run", False)
    assert rater.label == "jev · typesafe/jev-1.13 · per_email"
    assert list(rater.answers) == ["g.0001"]
    report = compare([rater], questions, runs={meta.id: meta}, resamples=10)
    assert report.raters[0].run == meta
    assert report.raters[0].n_items == 1


def test_reference_rater_respects_generation_snapshot(questions: QuestionSet) -> None:
    narrower = QuestionSet(
        name="narrower",
        questions=(
            ChoiceQuestion(
                type="choice",
                id="category",
                instructions="?",
                options={"spam": "s", "work": "w"},
            ),
        ),
    )
    email = EmailFactory(id="gen1.0001", reference_answers={"category": "spam"})
    incompatible = reference_rater([email], questions, {"gen1": narrower})
    compatible_snapshot = reference_rater([email], questions, {"gen1": questions})
    missing_snapshot = reference_rater([email], questions, {})
    assert incompatible.answers == {}
    assert missing_snapshot.answers == {}
    assert compatible_snapshot.answers == {
        "gen1.0001": {"category": {"spam": 1.0, "personal": 0.0, "work": 0.0}}
    }
