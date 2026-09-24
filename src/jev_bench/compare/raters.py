"""Raters: named sources of distributions over (email_id, question_id).

Types:
    RaterKind, Column
Classes:
    Rater: named source of distributions, with `.hard`, `.supports(question)` and
        `.column(question_id)`.
Functions:
    run_label: display label of a run ("column · model · mode").
    run_rater: build a rater from a run's predictions, skipping failed or answerless emails.
    reference_rater: build a rater from generator reference answers, restricted per email to
        its own generation's question-set snapshot.
    human_rater: build a rater from raw human labels (None when none are usable).
"""

from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel

from jev_bench.emails import Email
from jev_bench.questions import AnyQuestion, Distribution, QuestionSet, compatible, one_hot
from jev_bench.store.runs import Prediction, RunMeta

type RaterKind = Literal["run", "reference", "human"]
type Column = dict[str, Distribution]


class Rater(BaseModel):
    id: str
    label: str
    kind: RaterKind
    questions: QuestionSet
    answers: dict[str, dict[str, Distribution]]

    @property
    def hard(self) -> bool:
        return self.kind != "run"

    def supports(self, question: AnyQuestion) -> bool:
        try:
            return compatible(self.questions.get(question.id), question)
        except KeyError:
            return False

    def column(self, question_id: str) -> Column:
        return {
            email_id: answers[question_id]
            for email_id, answers in self.answers.items()
            if question_id in answers
        }


def run_label(meta: RunMeta) -> str:
    return f"{meta.column} · {meta.model} · {meta.mode}"


def run_rater(meta: RunMeta, predictions: Sequence[Prediction]) -> Rater:
    answers = {
        prediction.email_id: prediction.answers
        for prediction in predictions
        if prediction.answers and prediction.error is None
    }
    return Rater(
        id=meta.id, label=run_label(meta), kind="run", questions=meta.question_set, answers=answers
    )


def reference_rater(
    emails: Sequence[Email], questions: QuestionSet, snapshots: Mapping[str, QuestionSet]
) -> Rater:
    answers = {
        email.id: hard
        for email in emails
        if (snapshot := snapshots.get(email.generation_id)) is not None
        and (hard := _reference_hard(email.reference_answers, questions, snapshot))
    }
    return Rater(
        id="reference",
        label="Generator reference",
        kind="reference",
        questions=questions,
        answers=answers,
    )


def human_rater(labels: Mapping[str, Mapping[str, str]], questions: QuestionSet) -> Rater | None:
    answers = {
        email_id: hard for email_id, chosen in labels.items() if (hard := _hard(chosen, questions))
    }
    if not answers:
        return None
    return Rater(
        id="human", label="Human labels", kind="human", questions=questions, answers=answers
    )


def _hard(labels: Mapping[str, str], questions: QuestionSet) -> dict[str, Distribution]:
    return {
        question.id: one_hot(question, labels[question.id])
        for question in questions.questions
        if labels.get(question.id) in question.options
    }


def _reference_hard(
    labels: Mapping[str, str], questions: QuestionSet, snapshot: QuestionSet
) -> dict[str, Distribution]:
    return {
        question.id: one_hot(question, labels[question.id])
        for question in questions.questions
        if labels.get(question.id) in question.options and _snapshot_supports(question, snapshot)
    }


def _snapshot_supports(question: AnyQuestion, snapshot: QuestionSet) -> bool:
    try:
        return compatible(snapshot.get(question.id), question)
    except KeyError:
        return False
