"""Pairwise agreement between two raters, and the per-rater matrix cache pairs are built from.

Classes:
    PairStats: pairwise agreement, kappa, JSD, Pearson and Brier for one rater pair. For a multi
        question agreement is exact label-set match, kappa is the macro kappa over labels, Brier is
        against the hard rater's (uniform) distribution, Pearson is None, and jaccard / f1 are set.
    RaterMatrix: one rater's (email_id -> row) index plus its stacked distribution matrix,
        built once per question and reused across every pair that rater takes part in.
Functions:
    rater_matrix: build a RaterMatrix from a rater's column for one question.
    slice_matrix: the sub-matrix of a RaterMatrix for a given list of email ids.
    pair_stats: full PairStats for two raters on one question (None without shared emails).
    resample_index_cache: a bounded, per-call memoized resample_index(n), released with its
        caller so bootstrap indices are never retained across compare() calls.
"""

from collections.abc import Callable
from functools import lru_cache
from typing import NamedTuple

import numpy as np
from pydantic import BaseModel

from jev_bench.compare.multi import multi_pair_values
from jev_bench.compare.raters import Column, Rater
from jev_bench.metrics.agreement import (
    brier_score,
    brier_to_target,
    cohen_kappa,
    confusion_batch,
    disagreement_weights,
    kappa_from_confusion,
    pearson_r,
    percent_agreement,
)
from jev_bench.metrics.bootstrap import percentile_ci, resample_index
from jev_bench.metrics.distributions import (
    FloatArray,
    IntArray,
    argmax_labels,
    expected_level,
    js_divergence,
    to_matrix,
)
from jev_bench.questions import AnyQuestion, MultiQuestion

__all__ = [
    "PairStats",
    "RaterMatrix",
    "pair_stats",
    "rater_matrix",
    "resample_index_cache",
    "slice_matrix",
]


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
    jaccard: float | None = None
    jaccard_ci: tuple[float, float] | None = None
    f1: float | None = None


class RaterMatrix(NamedTuple):
    positions: dict[str, int]
    matrix: FloatArray


def rater_matrix(column: Column, question: AnyQuestion) -> RaterMatrix:
    order = sorted(column)
    matrix = to_matrix([column[email_id] for email_id in order], question.option_ids)
    return RaterMatrix(positions={email_id: i for i, email_id in enumerate(order)}, matrix=matrix)


def slice_matrix(rm: RaterMatrix, email_ids: list[str]) -> FloatArray:
    return rm.matrix[[rm.positions[email_id] for email_id in email_ids]]


def pair_stats(
    left: Rater,
    right: Rater,
    matrices: dict[str, RaterMatrix],
    question: AnyQuestion,
    index_for: Callable[[int], IntArray],
) -> PairStats | None:
    left_rm, right_rm = matrices.get(left.id), matrices.get(right.id)
    if left_rm is None or right_rm is None:
        return None
    shared = sorted(left_rm.positions.keys() & right_rm.positions.keys())
    if not shared:
        return None
    left_matrix = slice_matrix(left_rm, shared)
    right_matrix = slice_matrix(right_rm, shared)
    index = index_for(len(shared))
    return _pair_from_matrices(left, right, left_matrix, right_matrix, question, index)


def resample_index_cache(resamples: int, seed: int) -> Callable[[int], IntArray]:
    @lru_cache(maxsize=4)
    def index_for(n: int) -> IntArray:
        index = resample_index(n, resamples=resamples, seed=seed)
        index.flags.writeable = False
        return index

    return index_for


def _pair_from_matrices(
    left: Rater,
    right: Rater,
    left_matrix: FloatArray,
    right_matrix: FloatArray,
    question: AnyQuestion,
    index: IntArray,
) -> PairStats:
    if isinstance(question, MultiQuestion):
        return _multi_pair(left, right, left_matrix, right_matrix, question, index)
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
        brier=_brier(left, right, left_matrix, right_matrix, question),
    )


def _multi_pair(
    left: Rater,
    right: Rater,
    left_matrix: FloatArray,
    right_matrix: FloatArray,
    question: MultiQuestion,
    index: IntArray,
) -> PairStats:
    values = multi_pair_values(left_matrix, right_matrix, question.threshold, index)
    return PairStats(
        a=left.id,
        b=right.id,
        n=int(left_matrix.shape[0]),
        jsd=float(js_divergence(left_matrix, right_matrix).mean()),
        pearson=None,
        brier=_brier(left, right, left_matrix, right_matrix, question),
        **values._asdict(),
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
    left: Rater,
    right: Rater,
    left_matrix: FloatArray,
    right_matrix: FloatArray,
    question: AnyQuestion,
) -> float | None:
    if left.hard == right.hard:
        return None
    probabilities, hard = (right_matrix, left_matrix) if left.hard else (left_matrix, right_matrix)
    if isinstance(question, MultiQuestion):
        return brier_to_target(probabilities, hard)
    return brier_score(probabilities, argmax_labels(hard))
