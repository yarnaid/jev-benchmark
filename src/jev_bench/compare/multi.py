"""Multi-label comparison values for a MultiQuestion: label-set agreement between two raters, and
per-rater label summaries. A label is applied when p >= threshold * max(p).

Plain NamedTuples (not the report models) so `pairs.py` and `report.py` can depend on this module
without an import cycle.

Classes:
    MultiPairValues: exact-set agreement, macro kappa, Jaccard and micro-F1, with bootstrap CIs.
Functions:
    multi_pair_values: MultiPairValues for two raters' distributions over shared emails.
"""

from typing import NamedTuple

from jev_bench.metrics.bootstrap import percentile_ci
from jev_bench.metrics.distributions import FloatArray, IntArray
from jev_bench.metrics.multilabel import (
    exact_match_rows,
    jaccard_rows,
    macro_kappa,
    macro_kappa_batch,
    micro_f1,
    relative_labels,
)

__all__ = [
    "MultiPairValues",
    "multi_pair_values",
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
