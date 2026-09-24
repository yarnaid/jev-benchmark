"""Per-email view: each run's top answer per question, and the cross-run disagreement index.

Types:
    SupportedQuestions: rater id -> question ids it supports (precomputed once per call).
Classes:
    EmailRow: one email's traits, reference/human labels, per-run top answers and disagreement.
Functions:
    email_rows: per-email top answers per run and disagreement index, for a list of emails.
    disagreement_index: mean pairwise JSD across runs for one email, averaged over questions.
"""

from collections.abc import Mapping, Sequence
from itertools import combinations

import numpy as np
from pydantic import AwareDatetime, BaseModel

from jev_bench.compare.raters import Rater
from jev_bench.emails import Email
from jev_bench.metrics.distributions import js_divergence, to_matrix
from jev_bench.questions import AnyQuestion, Distribution, QuestionSet

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
    reference: dict[str, str]
    human: dict[str, str]
    top: dict[str, dict[str, str]]
    disagreement: float | None


def email_rows(
    emails: Sequence[Email],
    runs: Sequence[Rater],
    labels: Mapping[str, Mapping[str, str]],
    base: QuestionSet,
) -> list[EmailRow]:
    supported = {rater.id: _supported_questions(rater, base) for rater in runs}
    return [_email_row(email, runs, labels.get(email.id, {}), base, supported) for email in emails]


def _supported_questions(rater: Rater, base: QuestionSet) -> frozenset[str]:
    return frozenset(question.id for question in base.questions if rater.supports(question))


def _email_row(
    email: Email,
    runs: Sequence[Rater],
    human: Mapping[str, str],
    base: QuestionSet,
    supported: SupportedQuestions,
) -> EmailRow:
    top = {
        rater.id: _top(rater, email.id, base, supported[rater.id])
        for rater in runs
        if email.id in rater.answers
    }
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
        top=top,
        disagreement=_disagreement(email.id, runs, base, supported),
    )


def _top(
    rater: Rater, email_id: str, base: QuestionSet, supported: frozenset[str]
) -> dict[str, str]:
    answers = rater.answers[email_id]
    return {
        question.id: _argmax_option(question.option_ids, answers[question.id])
        for question in base.questions
        if question.id in answers and question.id in supported
    }


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
