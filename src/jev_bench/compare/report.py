"""Full per-question comparison report: per-rater stats, pairwise metrics, Fleiss' kappa.

Classes:
    RaterStats: per-rater summary for one question (argmax counts, means, entropy).
    QuestionReport: one question's rater stats, pairs, Fleiss' kappa and skipped raters.
    RaterSummary: one rater's identity and item count, with its RunMeta if it is a run.
    ComparisonReport: the full report (rater summaries, per-question reports, warnings).
Functions:
    compare: full per-question report for a set of raters against a base question set.
"""

from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Literal

import numpy as np
from pydantic import BaseModel

from jev_bench.compare.pairs import PairStats, RaterMatrix, pair_stats, rater_matrix, slice_matrix
from jev_bench.compare.raters import Rater, RaterKind
from jev_bench.metrics.agreement import fleiss_kappa
from jev_bench.metrics.distributions import argmax_labels, entropy, expected_level
from jev_bench.questions import AnyQuestion, QuestionSet
from jev_bench.store.runs import RunMeta


class RaterStats(BaseModel):
    rater: str
    n: int
    argmax_counts: dict[str, int]
    mean: dict[str, float]
    mean_entropy: float
    mean_confidence: float
    mean_level: float | None = None


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
    usable = [rater for rater in raters if rater.supports(question)]
    skipped = [rater.id for rater in raters if not rater.supports(question)]
    if skipped:
        warnings.append(_skip_warning(question, skipped))
    matrices = {
        rater.id: rater_matrix(column, question)
        for rater in usable
        if (column := rater.column(question.id))
    }
    pairs = [
        pair
        for left, right in combinations(usable, 2)
        if (pair := pair_stats(left, right, matrices, question, resamples, seed)) is not None
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


def _rater_stats(rater_id: str, rm: RaterMatrix, question: AnyQuestion) -> RaterStats:
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


def _fleiss(
    runs: Sequence[Rater],
    matrices: dict[str, RaterMatrix],
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
        [argmax_labels(slice_matrix(matrices[rater.id], shared)) for rater in answered],
        axis=1,
    )
    return fleiss_kappa(labels, len(question.option_ids))
