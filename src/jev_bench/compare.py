"""Comparison of raters (runs, generator reference, human labels) per question and per email.

Types:
    RaterKind
Classes:
    Rater: named source of distributions over (email_id, question_id).
    RaterStats, PairStats, QuestionReport, RaterSummary, ComparisonReport, EmailRow: report models.
Functions:
    run_label: display label of a run ("column · model · mode").
    run_rater, reference_rater, human_rater: build raters from persisted data.
    compare: full per-question report (per-rater stats, pairwise metrics with CIs, Fleiss' kappa).
    email_rows: per-email top answers per run and disagreement index.
    disagreement_index: mean pairwise JSD across runs for one email, averaged over questions.
"""

from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Literal

import numpy as np
from pydantic import AwareDatetime, BaseModel

from jev_bench.emails import Email
from jev_bench.metrics.agreement import (
    brier_score,
    cohen_kappa,
    confusion_batch,
    disagreement_weights,
    fleiss_kappa,
    kappa_from_confusion,
    pearson_r,
    percent_agreement,
)
from jev_bench.metrics.bootstrap import percentile_ci, resample_index
from jev_bench.metrics.distributions import (
    FloatArray,
    IntArray,
    argmax_labels,
    entropy,
    expected_level,
    js_divergence,
    to_matrix,
)
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


class RaterStats(BaseModel):
    rater: str
    n: int
    argmax_counts: dict[str, int]
    mean: dict[str, float]
    mean_entropy: float
    mean_confidence: float
    mean_level: float | None = None


class PairStats(BaseModel):
    a: str
    b: str
    n: int
    agreement: float
    agreement_ci: tuple[float, float] | None
    kappa: float | None
    kappa_ci: tuple[float, float] | None
    jsd: float
    pearson: float | None
    brier: float | None


class QuestionReport(BaseModel):
    id: str
    type: str
    options: tuple[str, ...]
    raters: list[RaterStats]
    pairs: list[PairStats]
    fleiss_kappa: float | None
    skipped: list[str]


class RaterSummary(BaseModel):
    id: str
    label: str
    kind: RaterKind
    n_items: int
    run: RunMeta | None = None


class ComparisonReport(BaseModel):
    raters: list[RaterSummary]
    questions: list[QuestionReport]
    warnings: list[str]


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


def run_label(meta: RunMeta) -> str:
    return f"{meta.column} · {meta.model} · {meta.mode}"


def run_rater(meta: RunMeta, predictions: Sequence[Prediction]) -> Rater:
    answers = {
        prediction.email_id: prediction.answers for prediction in predictions if prediction.answers
    }
    return Rater(
        id=meta.id, label=run_label(meta), kind="run", questions=meta.question_set, answers=answers
    )


def reference_rater(emails: Sequence[Email], questions: QuestionSet) -> Rater:
    answers = {
        email.id: hard for email in emails if (hard := _hard(email.reference_answers, questions))
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


def compare(
    raters: Sequence[Rater],
    base: QuestionSet,
    *,
    resamples: int = 1000,
    seed: int = 0,
    runs: Mapping[str, RunMeta] | None = None,
) -> ComparisonReport:
    warnings: list[str] = []
    questions = [
        _question_report(question, raters, resamples, seed, warnings) for question in base.questions
    ]
    metas = runs or {}
    summaries = [
        RaterSummary(
            id=rater.id,
            label=rater.label,
            kind=rater.kind,
            n_items=len(rater.answers),
            run=metas.get(rater.id),
        )
        for rater in raters
    ]
    return ComparisonReport(raters=summaries, questions=questions, warnings=warnings)


def _question_report(
    question: AnyQuestion, raters: Sequence[Rater], resamples: int, seed: int, warnings: list[str]
) -> QuestionReport:
    usable = [rater for rater in raters if _supports(rater, question)]
    skipped = [rater.id for rater in raters if not _supports(rater, question)]
    if skipped:
        warnings.append(f"{question.id}: skipped raters with different options: {skipped}")
    columns = {rater.id: _column(rater, question.id) for rater in usable}
    pairs = [
        pair
        for left, right in combinations(usable, 2)
        if (pair := _pair_stats(left, right, columns, question, resamples, seed)) is not None
    ]
    return QuestionReport(
        id=question.id,
        type=question.type,
        options=question.option_ids,
        raters=[
            _rater_stats(rater.id, columns[rater.id], question)
            for rater in usable
            if columns[rater.id]
        ],
        pairs=pairs,
        fleiss_kappa=_fleiss([rater for rater in usable if rater.kind == "run"], columns, question),
        skipped=skipped,
    )


def _supports(rater: Rater, question: AnyQuestion) -> bool:
    try:
        return compatible(rater.questions.get(question.id), question)
    except KeyError:
        return False


def _column(rater: Rater, question_id: str) -> Column:
    return {
        email_id: answers[question_id]
        for email_id, answers in rater.answers.items()
        if question_id in answers
    }


def _rater_stats(rater_id: str, column: Column, question: AnyQuestion) -> RaterStats:
    options = question.option_ids
    matrix = to_matrix(list(column.values()), options)
    counts = np.bincount(argmax_labels(matrix), minlength=len(options))
    return RaterStats(
        rater=rater_id,
        n=len(column),
        argmax_counts={option: int(count) for option, count in zip(options, counts, strict=True)},
        mean={
            option: float(value) for option, value in zip(options, matrix.mean(axis=0), strict=True)
        },
        mean_entropy=float(entropy(matrix).mean()),
        mean_confidence=float(matrix.max(axis=1).mean()),
        mean_level=float(expected_level(matrix).mean()) if question.type == "score" else None,
    )


def _pair_stats(
    left: Rater,
    right: Rater,
    columns: Mapping[str, Column],
    question: AnyQuestion,
    resamples: int,
    seed: int,
) -> PairStats | None:
    shared = sorted(columns[left.id].keys() & columns[right.id].keys())
    if not shared:
        return None
    left_matrix = to_matrix(
        [columns[left.id][email_id] for email_id in shared], question.option_ids
    )
    right_matrix = to_matrix(
        [columns[right.id][email_id] for email_id in shared], question.option_ids
    )
    index = resample_index(len(shared), resamples=resamples, seed=seed)
    return _pair_from_matrices(left, right, left_matrix, right_matrix, question, index)


def _pair_from_matrices(
    left: Rater,
    right: Rater,
    left_matrix: FloatArray,
    right_matrix: FloatArray,
    question: AnyQuestion,
    index: IntArray,
) -> PairStats:
    left_labels, right_labels = argmax_labels(left_matrix), argmax_labels(right_matrix)
    k = len(question.option_ids)
    quadratic = question.type == "score"
    weights = disagreement_weights(k, quadratic=quadratic)
    return PairStats(
        a=left.id,
        b=right.id,
        n=int(left_labels.size),
        agreement=percent_agreement(left_labels, right_labels),
        agreement_ci=percentile_ci((left_labels[index] == right_labels[index]).mean(axis=1)),
        kappa=cohen_kappa(left_labels, right_labels, k, quadratic=quadratic),
        kappa_ci=percentile_ci(
            kappa_from_confusion(confusion_batch(left_labels, right_labels, k, index), weights)
        ),
        jsd=float(js_divergence(left_matrix, right_matrix).mean()),
        pearson=_pearson(left_matrix, right_matrix, question),
        brier=_brier(left, right, left_matrix, right_matrix),
    )


def _pearson(left: FloatArray, right: FloatArray, question: AnyQuestion) -> float | None:
    if question.type == "noul":
        return pearson_r(left[:, 0], right[:, 0])
    if question.type == "score":
        return pearson_r(expected_level(left), expected_level(right))
    return None


def _brier(
    left: Rater, right: Rater, left_matrix: FloatArray, right_matrix: FloatArray
) -> float | None:
    if left.hard == right.hard:
        return None
    probabilities, hard = (right_matrix, left_matrix) if left.hard else (left_matrix, right_matrix)
    return brier_score(probabilities, argmax_labels(hard))


def _fleiss(
    runs: Sequence[Rater], columns: Mapping[str, Column], question: AnyQuestion
) -> float | None:
    if len(runs) < 2:
        return None
    shared = sorted(set.intersection(*(set(columns[rater.id]) for rater in runs)))
    if not shared:
        return None
    labels = np.stack(
        [
            argmax_labels(
                to_matrix([columns[rater.id][email_id] for email_id in shared], question.option_ids)
            )
            for rater in runs
        ],
        axis=1,
    )
    return fleiss_kappa(labels, len(question.option_ids))


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
    top = {
        rater.id: _top(rater.answers[email.id], base) for rater in runs if email.id in rater.answers
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
        disagreement=disagreement_index(email.id, runs, base),
    )


def _top(answers: Mapping[str, Distribution], base: QuestionSet) -> dict[str, str]:
    return {
        question.id: _argmax_option(question.option_ids, answers[question.id])
        for question in base.questions
        if question.id in answers
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
        if question.id in rater.answers.get(email_id, {})
    ]
    if len(dists) < 2:
        return None
    matrix = to_matrix(dists, question.option_ids)
    pairs = list(combinations(range(len(dists)), 2))
    left = matrix[[i for i, _ in pairs]]
    right = matrix[[j for _, j in pairs]]
    return float(js_divergence(left, right).mean())
