"""Tests for jev_bench.metrics.multilabel."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from jev_bench.metrics.multilabel import (
    exact_match_rows,
    jaccard_rows,
    macro_fleiss,
    macro_kappa,
    macro_kappa_batch,
    micro_f1,
    relative_labels,
)

_A = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=bool)
_B = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=bool)
_EMPTY = np.zeros((4, 3), dtype=bool)
_IDENTITY = np.arange(4, dtype=np.int64)[None, :]


@pytest.mark.parametrize(
    ("matrix", "threshold", "expected"),
    [
        pytest.param([[0.5, 0.375, 0.125]], 0.75, [[True, True, False]], id="exactly-at-boundary"),
        pytest.param([[0.5, 0.37, 0.13]], 0.75, [[True, False, False]], id="just-below-boundary"),
        pytest.param([[0.7, 0.2, 0.1]], 0.8, [[True, False, False]], id="top-only"),
        pytest.param([[0.25, 0.25, 0.5]], 0.5, [[True, True, True]], id="half-of-top"),
        pytest.param([[1 / 3, 1 / 3, 1 / 3]], 1.0, [[True, True, True]], id="flat-applies-all"),
        pytest.param([[0.6, 0.4]], 1.0, [[True, False]], id="threshold-one-is-top-only"),
        pytest.param([[0.0, 0.0, 0.0]], 0.8, [[False, False, False]], id="zero-row-has-no-labels"),
        pytest.param(
            [[0.5, 0.5, 0.0]], 0.8, [[True, True, False]], id="uniform-label-set-round-trips"
        ),
    ],
)
def test_relative_labels(
    matrix: list[list[float]], threshold: float, expected: list[list[bool]]
) -> None:
    assert relative_labels(np.array(matrix), threshold).tolist() == expected


def test_set_metrics_on_the_worked_example() -> None:
    assert exact_match_rows(_A, _B).tolist() == [0.0, 1.0, 1.0, 1.0]
    assert jaccard_rows(_A, _B).tolist() == [0.5, 1.0, 1.0, 1.0]
    assert micro_f1(_A, _B) == pytest.approx(6 / 7)
    assert macro_kappa(_A, _B) == pytest.approx(5 / 6)
    assert macro_kappa_batch(_A, _B, _IDENTITY).tolist() == pytest.approx([5 / 6])


@pytest.mark.parametrize(
    ("a", "b"),
    [
        pytest.param(_EMPTY, _EMPTY, id="both-empty"),
        pytest.param(np.ones((4, 3), dtype=bool), np.ones((4, 3), dtype=bool), id="all-labels"),
    ],
)
def test_constant_labels_have_no_kappa(a: np.ndarray, b: np.ndarray) -> None:
    assert macro_kappa(a, b) is None
    assert np.isnan(macro_kappa_batch(a, b, _IDENTITY)).all()
    assert exact_match_rows(a, b).tolist() == [1.0] * 4
    assert jaccard_rows(a, b).tolist() == [1.0] * 4


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        pytest.param(_EMPTY, _EMPTY, None, id="no-positives"),
        pytest.param(_A, _EMPTY, 0.0, id="disjoint"),
        pytest.param(_A, _A, 1.0, id="identical"),
    ],
)
def test_micro_f1_edges(a: np.ndarray, b: np.ndarray, expected: float | None) -> None:
    assert micro_f1(a, b) == expected


def test_macro_kappa_skips_undefined_labels() -> None:
    varied = np.array([[1, 0], [0, 0], [1, 0], [0, 0]], dtype=bool)
    assert macro_kappa(varied, varied) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        pytest.param(np.stack([_A, _A], axis=1), 1.0, id="identical-raters"),
        pytest.param(np.stack([_A, _B], axis=1), 37 / 45, id="worked-example"),
        pytest.param(np.stack([_EMPTY, _EMPTY], axis=1), None, id="constant"),
    ],
)
def test_macro_fleiss(labels: np.ndarray, expected: float | None) -> None:
    result = macro_fleiss(labels)
    assert result == (None if expected is None else pytest.approx(expected))


_BOOLS = arrays(np.bool_, (5, 3))
_DISTRIBUTIONS = arrays(np.float64, (4, 3), elements=st.floats(0.01, 1.0)).map(
    lambda m: m / m.sum(axis=1, keepdims=True)
)


@settings(max_examples=30)
@given(a=_BOOLS, b=_BOOLS)
def test_set_metrics_are_symmetric_and_bounded(a: np.ndarray, b: np.ndarray) -> None:
    jaccard = jaccard_rows(a, b)
    assert np.array_equal(jaccard, jaccard_rows(b, a))
    assert ((jaccard >= 0.0) & (jaccard <= 1.0)).all()
    assert (exact_match_rows(a, b) <= jaccard).all()
    f1 = micro_f1(a, b)
    assert f1 == micro_f1(b, a)
    assert f1 is None or 0.0 <= f1 <= 1.0


@settings(max_examples=30)
@given(matrix=_DISTRIBUTIONS, first=st.floats(0.01, 1.0), second=st.floats(0.01, 1.0))
def test_relative_labels_keep_the_top_and_are_monotone(
    matrix: np.ndarray, first: float, second: float
) -> None:
    low, high = sorted((first, second))
    strict, loose = relative_labels(matrix, high), relative_labels(matrix, low)
    assert strict[np.arange(4), matrix.argmax(axis=1)].all()
    assert (strict <= loose).all()
