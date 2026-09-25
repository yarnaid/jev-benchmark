"""Tests for jev_bench.compare.multi."""

import numpy as np
import pytest
from tests.compare_data import TOPICS_A, TOPICS_B

from jev_bench.compare.multi import multi_pair_values
from jev_bench.metrics.bootstrap import resample_index
from jev_bench.metrics.distributions import to_matrix

_LABELS = ("billing", "meeting", "travel")
_A = to_matrix(TOPICS_A, _LABELS)
_B = to_matrix(TOPICS_B, _LABELS)


@pytest.mark.parametrize(
    ("threshold", "agreement", "jaccard", "f1", "kappa"),
    [
        pytest.param(0.8, 0.75, 0.875, 10 / 11, 5 / 6, id="default-threshold"),
        pytest.param(0.5, 0.5, 19 / 24, 5 / 6, 2 / 3, id="lower-threshold"),
    ],
)
def test_multi_pair_values(
    threshold: float, agreement: float, jaccard: float, f1: float, kappa: float
) -> None:
    values = multi_pair_values(_A, _B, threshold, resample_index(4, resamples=50, seed=0))
    assert values.agreement == pytest.approx(agreement)
    assert values.jaccard == pytest.approx(jaccard)
    assert values.f1 == pytest.approx(f1)
    assert values.kappa == pytest.approx(kappa)
    for interval in (values.agreement_ci, values.jaccard_ci):
        assert interval is not None
        assert 0.0 <= interval[0] <= interval[1] <= 1.0


def test_identical_matrices_agree_perfectly() -> None:
    values = multi_pair_values(_A, _A, 0.8, resample_index(4, resamples=20, seed=0))
    assert (values.agreement, values.jaccard, values.f1) == (1.0, 1.0, 1.0)
    assert values.kappa == pytest.approx(1.0)


def test_flat_distributions_apply_every_label() -> None:
    flat = np.full((4, 3), 1 / 3)
    values = multi_pair_values(flat, flat, 0.8, resample_index(4, resamples=20, seed=0))
    assert (values.agreement, values.jaccard, values.f1) == (1.0, 1.0, 1.0)
    assert (values.kappa, values.kappa_ci) == (None, None)
