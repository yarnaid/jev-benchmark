"""Per-email view: each run's top answer and 0-100 scores per question, and the cross-run
disagreement index.

Types:
    SupportedQuestions: rater id -> question ids it supports (precomputed once per call).
Classes:
    EmailRow: one email's traits, reference/human labels, per-run top answers (for a multi question
        the applied labels, p >= threshold * max(p), highest first), 0-100 scores of score questions
        per run and for the reference, and the disagreement index.
Functions:
    email_rows: EmailRow per email; `threshold` overrides every multi question's own threshold.
    disagreement_index: mean pairwise JSD across runs for one email, averaged over questions.
"""

from collections.abc import Mapping, Sequence
from itertools import combinations

import numpy as np
from pydantic import AwareDatetime, BaseModel

from jev_bench.compare.raters import Rater
from jev_bench.emails import Email
from jev_bench.metrics.distributions import js_divergence, score_0_100, to_matrix
from jev_bench.metrics.multilabel import relative_labels
from jev_bench.questions import (
    AnyQuestion,
    Distribution,
    HardAnswer,
    MultiQuestion,
    QuestionSet,
    ScoreQuestion,
    hard_distribution,
    with_threshold,
)

__all__ = [
    "EmailRow",
    "SupportedQuestions",
    "disagreement_index",
    "email_rows",
]

type SupportedQuestions = Mapping[str, frozenset[str]]


class EmailRow(BaseModel):
    id: str
    generation_id: str
    sent_at: AwareDatetime
    sender: str
    subject: str
    generator_model: str
    traits: dict[str, str]
    reference: dict[str, HardAnswer]
    human: dict[str, HardAnswer]
    top: dict[str, dict[str, HardAnswer]]
    scores: dict[str, dict[str, float]]
    reference_scores: dict[str, float]
    disagreement: float | None


def email_rows(
    emails: Sequence[Email],
    runs: Sequence[Rater],
    labels: Mapping[str, Mapping[str, HardAnswer]],
    base: QuestionSet,
    threshold: float | None = None,
) -> list[EmailRow]:
    effective = with_threshold(base, threshold)
    supported = {rater.id: _supported_questions(rater, effective) for rater in runs}
    return [
        _email_row(email, runs, labels.get(email.id, {}), effective, supported) for email in emails
    ]


def _supported_questions(rater: Rater, base: QuestionSet) -> frozenset[str]:
    return frozenset(question.id for question in base.questions if rater.supports(question))


def _email_row(
    email: Email,
    runs: Sequence[Rater],
    human: Mapping[str, HardAnswer],
    base: QuestionSet,
    supported: SupportedQuestions,
) -> EmailRow:
    answered = [rater for rater in runs if email.id in rater.answers]
    return EmailRow(
        id=email.id,
        generation_id=email.generation_id,
        sent_at=email.sent_at,
        sender=email.sender.formatted(),
        subject=email.subject,
        generator_model=email.generator_model,
        traits=dict(email.traits),
        reference=dict(email.reference_answers),
        human=dict(human),
        top={
            rater.id: _top(rater.answers[email.id], base, supported[rater.id]) for rater in answered
        },
        scores={
            rater.id: _scores(rater.answers[email.id], base, supported[rater.id])
            for rater in answered
        },
        reference_scores=_reference_scores(email.reference_answers, base),
        disagreement=_disagreement(email.id, runs, base, supported),
    )


def _top(
    answers: Mapping[str, Distribution], base: QuestionSet, supported: frozenset[str]
) -> dict[str, HardAnswer]:
    return {
        question.id: _top_answer(question, answers[question.id])
        for question in base.questions
        if question.id in answers and question.id in supported
    }


def _top_answer(question: AnyQuestion, distribution: Distribution) -> HardAnswer:
    if isinstance(question, MultiQuestion):
        return _applied(question, distribution)
    return _argmax_option(question.option_ids, distribution)


def _applied(question: MultiQuestion, distribution: Distribution) -> list[str]:
    row = to_matrix([distribution], question.option_ids)
    applied = relative_labels(row, question.threshold)[0]
    labels = [option for option, on in zip(question.option_ids, applied, strict=True) if on]
    return sorted(labels, key=lambda option: -distribution.get(option, 0.0))


def _scores(
    answers: Mapping[str, Distribution], base: QuestionSet, supported: frozenset[str]
) -> dict[str, float]:
    return {
        question.id: _score(question, answers[question.id])
        for question in base.questions
        if isinstance(question, ScoreQuestion)
        and question.id in answers
        and question.id in supported
    }


def _reference_scores(reference: Mapping[str, HardAnswer], base: QuestionSet) -> dict[str, float]:
    return {
        question.id: _score(question, hard)
        for question in base.questions
        if isinstance(question, ScoreQuestion)
        and (hard := hard_distribution(question, reference.get(question.id))) is not None
    }


def _score(question: AnyQuestion, distribution: Distribution) -> float:
    return float(score_0_100(to_matrix([distribution], question.option_ids))[0])


def _argmax_option(options: Sequence[str], distribution: Distribution) -> str:
    return options[int(np.argmax([distribution.get(option, 0.0) for option in options]))]


def disagreement_index(email_id: str, runs: Sequence[Rater], base: QuestionSet) -> float | None:
    return _disagreement(email_id, runs, base, supported=None)


def _disagreement(
    email_id: str,
    runs: Sequence[Rater],
    base: QuestionSet,
    supported: SupportedQuestions | None,
) -> float | None:
    values = [
        value
        for question in base.questions
        if (value := _question_disagreement(email_id, runs, question, supported)) is not None
    ]
    return float(np.mean(values)) if values else None


def _question_disagreement(
    email_id: str,
    runs: Sequence[Rater],
    question: AnyQuestion,
    supported: SupportedQuestions | None,
) -> float | None:
    dists = [
        rater.answers[email_id][question.id]
        for rater in runs
        if question.id in rater.answers.get(email_id, {})
        and _rater_supports(rater, question, supported)
    ]
    if len(dists) < 2:
        return None
    matrix = to_matrix(dists, question.option_ids)
    pairs = list(combinations(range(len(dists)), 2))
    left = matrix[[i for i, _ in pairs]]
    right = matrix[[j for _, j in pairs]]
    return float(js_divergence(left, right).mean())


def _rater_supports(
    rater: Rater, question: AnyQuestion, supported: SupportedQuestions | None
) -> bool:
    if supported is not None:
        return question.id in supported.get(rater.id, frozenset())
    return rater.supports(question)
