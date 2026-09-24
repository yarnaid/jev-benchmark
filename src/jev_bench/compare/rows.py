"""Per-email view: each run's top answer per question, and the cross-run disagreement index.

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
    return [_email_row(email, runs, labels.get(email.id, {}), base) for email in emails]


def _email_row(
    email: Email, runs: Sequence[Rater], human: Mapping[str, str], base: QuestionSet
) -> EmailRow:
    top = {rater.id: _top(rater, email.id, base) for rater in runs if email.id in rater.answers}
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
        disagreement=disagreement_index(email.id, runs, base),
    )


def _top(rater: Rater, email_id: str, base: QuestionSet) -> dict[str, str]:
    answers = rater.answers[email_id]
    return {
        question.id: _argmax_option(question.option_ids, answers[question.id])
        for question in base.questions
        if question.id in answers and rater.supports(question)
    }


def _argmax_option(options: Sequence[str], distribution: Distribution) -> str:
    return options[int(np.argmax([distribution.get(option, 0.0) for option in options]))]


def disagreement_index(email_id: str, runs: Sequence[Rater], base: QuestionSet) -> float | None:
    values = [
        value
        for question in base.questions
        if (value := _question_disagreement(email_id, runs, question)) is not None
    ]
    return float(np.mean(values)) if values else None


def _question_disagreement(
    email_id: str, runs: Sequence[Rater], question: AnyQuestion
) -> float | None:
    dists = [
        rater.answers[email_id][question.id]
        for rater in runs
        if question.id in rater.answers.get(email_id, {}) and rater.supports(question)
    ]
    if len(dists) < 2:
        return None
    matrix = to_matrix(dists, question.option_ids)
    pairs = list(combinations(range(len(dists)), 2))
    left = matrix[[i for i, _ in pairs]]
    right = matrix[[j for _, j in pairs]]
    return float(js_divergence(left, right).mean())
