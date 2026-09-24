"""Tests for jev_bench.metrics.bootstrap."""

import numpy as np
import pytest

from jev_bench.metrics.bootstrap import percentile_ci, resample_index


def test_resample_index_shape_range_and_determinism() -> None:
    first = resample_index(7, resamples=50, seed=3)
    assert first.shape == (50, 7)
    assert first.min() >= 0
    assert first.max() < 7
    assert (first == resample_index(7, resamples=50, seed=3)).all()


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        pytest.param(np.full(20, 0.4), (0.4, 0.4), id="constant"),
        pytest.param(np.arange(101, dtype=np.float64), (2.5, 97.5), id="uniform-grid"),
        pytest.param(np.array([np.nan, 1.0, np.nan]), (1.0, 1.0), id="nan-filtered"),
        pytest.param(np.array([np.nan, np.nan]), None, id="all-nan"),
    ],
)
def test_percentile_ci(values: np.ndarray, expected: tuple[float, float] | None) -> None:
    result = percentile_ci(values)
    assert result == (None if expected is None else pytest.approx(expected))
