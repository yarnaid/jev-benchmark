"""Multi-label comparison values for a MultiQuestion: label-set agreement between two raters, and
per-rater label summaries. A label is applied when p >= threshold * max(p).

Plain NamedTuples (not the report models) so `pairs.py` and `report.py` can depend on this module
without an import cycle.

Classes:
    MultiPairValues: exact-set agreement, macro kappa, Jaccard and micro-F1, with bootstrap CIs.
    MultiRaterValues: applied-label counts and labels per email.
Functions:
    multi_pair_values: MultiPairValues for two raters' distributions over shared emails.
    multi_rater_values: MultiRaterValues for one rater's distributions.
    multi_fleiss: macro Fleiss' kappa of several raters' applied labels on shared emails.
"""

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from jev_bench.metrics.bootstrap import percentile_ci
from jev_bench.metrics.distributions import FloatArray, IntArray
from jev_bench.metrics.multilabel import (
    exact_match_rows,
    jaccard_rows,
    macro_fleiss,
    macro_kappa,
    macro_kappa_batch,
    micro_f1,
    relative_labels,
)
from jev_bench.questions import MultiQuestion

__all__ = [
    "MultiPairValues",
    "MultiRaterValues",
    "multi_fleiss",
    "multi_pair_values",
    "multi_rater_values",
]


class MultiPairValues(NamedTuple):
    agreement: float
    agreement_ci: tuple[float, float] | None
    kappa: float | None
    kappa_ci: tuple[float, float] | None
    jaccard: float
    jaccard_ci: tuple[float, float] | None
    f1: float | None


def multi_pair_values(
    left: FloatArray, right: FloatArray, threshold: float, index: IntArray
) -> MultiPairValues:
    a, b = relative_labels(left, threshold), relative_labels(right, threshold)
    exact, jaccard = exact_match_rows(a, b), jaccard_rows(a, b)
    return MultiPairValues(
        agreement=float(exact.mean()),
        agreement_ci=percentile_ci(exact[index].mean(axis=1)),
        kappa=macro_kappa(a, b),
        kappa_ci=percentile_ci(macro_kappa_batch(a, b, index)),
        jaccard=float(jaccard.mean()),
        jaccard_ci=percentile_ci(jaccard[index].mean(axis=1)),
        f1=micro_f1(a, b),
    )


class MultiRaterValues(NamedTuple):
    label_counts: dict[str, int]
    mean_labels: float


def multi_rater_values(matrix: FloatArray, question: MultiQuestion) -> MultiRaterValues:
    applied = relative_labels(matrix, question.threshold)
    counts = applied.sum(axis=0)
    return MultiRaterValues(
        label_counts={
            option: int(count) for option, count in zip(question.option_ids, counts, strict=True)
        },
        mean_labels=float(applied.sum(axis=1).mean()),
    )


def multi_fleiss(matrices: Sequence[FloatArray], threshold: float) -> float | None:
    labels = np.stack([relative_labels(matrix, threshold) for matrix in matrices], axis=1)
    return macro_fleiss(labels)
