"""Comparison of raters (runs, generator reference, human labels) per question and per email.

Types:
    RaterKind
Classes:
    Rater: named source of distributions over (email_id, question_id).
    RaterStats, PairStats, QuestionReport, RaterSummary, ComparisonReport, EmailRow: report models.
Functions:
    run_label: display label of a run ("column · model · mode").
    run_rater: build a rater from a run's predictions, skipping failed or answerless emails.
    reference_rater: build a rater from generator reference answers, restricted per email to
        its own generation's question-set snapshot.
    human_rater: build a rater from raw human labels (None when none are usable).
    compare: full per-question report (per-rater stats, pairwise metrics with CIs, Fleiss' kappa).
    email_rows: per-email top answers per run and disagreement index.
    disagreement_index: mean pairwise JSD across runs for one email, averaged over questions.
"""

from collections.abc import Mapping, Sequence
from functools import lru_cache
from itertools import combinations
from typing import Literal, NamedTuple

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
    type: Literal["choice", "score", "noul"]
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


class _RaterMatrix(NamedTuple):
    positions: dict[str, int]
    matrix: FloatArray


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
        warnings.append(_skip_warning(question, skipped))
    matrices = {
        rater.id: _rater_matrix(column, question)
        for rater in usable
        if (column := _column(rater, question.id))
    }
    pairs = [
        pair
        for left, right in combinations(usable, 2)
        if (pair := _pair_stats(left, right, matrices, question, resamples, seed)) is not None
    ]
    return QuestionReport(
        id=question.id,
        type=question.type,
        options=question.option_ids,
        raters=[
            _rater_stats(rater.id, matrices[rater.id], question)
            for rater in usable
            if rater.id in matrices
        ],
        pairs=pairs,
        fleiss_kappa=_fleiss(
            [rater for rater in usable if rater.kind == "run"], matrices, question, warnings
        ),
        skipped=skipped,
    )


def _skip_warning(question: AnyQuestion, skipped: Sequence[str]) -> str:
    return (
        f"{question.id}: incompatible snapshot (missing question, different type or options): "
        f"{', '.join(skipped)}"
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


def _rater_matrix(column: Column, question: AnyQuestion) -> _RaterMatrix:
    order = sorted(column)
    matrix = to_matrix([column[email_id] for email_id in order], question.option_ids)
    return _RaterMatrix(positions={email_id: i for i, email_id in enumerate(order)}, matrix=matrix)


def _rater_stats(rater_id: str, rm: _RaterMatrix, question: AnyQuestion) -> RaterStats:
    options = question.option_ids
    matrix = rm.matrix
    counts = np.bincount(argmax_labels(matrix), minlength=len(options))
    return RaterStats(
        rater=rater_id,
        n=len(rm.positions),
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
    matrices: Mapping[str, _RaterMatrix],
    question: AnyQuestion,
    resamples: int,
    seed: int,
) -> PairStats | None:
    left_rm, right_rm = matrices.get(left.id), matrices.get(right.id)
    if left_rm is None or right_rm is None:
        return None
    shared = sorted(left_rm.positions.keys() & right_rm.positions.keys())
    if not shared:
        return None
    left_matrix = _slice(left_rm, shared)
    right_matrix = _slice(right_rm, shared)
    index = _cached_resample_index(len(shared), resamples, seed)
    return _pair_from_matrices(left, right, left_matrix, right_matrix, question, index)


def _slice(rm: _RaterMatrix, email_ids: Sequence[str]) -> FloatArray:
    return rm.matrix[[rm.positions[email_id] for email_id in email_ids]]


@lru_cache(maxsize=128)
def _cached_resample_index(n: int, resamples: int, seed: int) -> IntArray:
    index = resample_index(n, resamples=resamples, seed=seed)
    index.flags.writeable = False
    return index


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
    confusion = confusion_batch(left_labels, right_labels, k, index)
    return PairStats(
        a=left.id,
        b=right.id,
        n=int(left_labels.size),
        agreement=percent_agreement(left_labels, right_labels),
        agreement_ci=_bootstrap_agreement_ci(confusion, left_labels.size),
        kappa=cohen_kappa(left_labels, right_labels, k, quadratic=quadratic),
        kappa_ci=percentile_ci(kappa_from_confusion(confusion, weights)),
        jsd=float(js_divergence(left_matrix, right_matrix).mean()),
        pearson=_pearson(left_matrix, right_matrix, question),
        brier=_brier(left, right, left_matrix, right_matrix),
    )


def _bootstrap_agreement_ci(confusion: FloatArray, n: int) -> tuple[float, float] | None:
    return percentile_ci(np.trace(confusion, axis1=1, axis2=2) / n)


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
    runs: Sequence[Rater],
    matrices: Mapping[str, _RaterMatrix],
    question: AnyQuestion,
    warnings: list[str],
) -> float | None:
    for rater in runs:
        if rater.id not in matrices:
            warnings.append(f"rater {rater.id} answered no emails for question {question.id}")
    answered = [rater for rater in runs if rater.id in matrices]
    if len(answered) < 2:
        return None
    shared = sorted(set.intersection(*(set(matrices[rater.id].positions) for rater in answered)))
    if not shared:
        return None
    labels = np.stack(
        [argmax_labels(_slice(matrices[rater.id], shared)) for rater in answered],
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
        if question.id in answers and _supports(rater, question)
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
        if question.id in rater.answers.get(email_id, {}) and _supports(rater, question)
    ]
    if len(dists) < 2:
        return None
    matrix = to_matrix(dists, question.option_ids)
    pairs = list(combinations(range(len(dists)), 2))
    left = matrix[[i for i, _ in pairs]]
    right = matrix[[j for _, j in pairs]]
    return float(js_divergence(left, right).mean())
