"""Tests for jev_bench.metrics.distributions."""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from jev_bench.metrics.distributions import (
    argmax_labels,
    entropy,
    expected_level,
    js_divergence,
    normalize,
    softmax,
    to_matrix,
    unit_probability,
)

_OPTIONS = ("a", "b")


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        pytest.param({"a": 2, "b": 2}, {"a": 0.5, "b": 0.5}, id="renormalizes"),
        pytest.param({"a": -1, "b": 3}, {"a": 0.0, "b": 1.0}, id="negative-clipped"),
        pytest.param({"a": math.nan, "b": 1}, {"a": 0.0, "b": 1.0}, id="nan-dropped"),
        pytest.param({"a": math.inf, "b": 1}, {"a": 0.0, "b": 1.0}, id="inf-dropped"),
        pytest.param({"a": True, "b": 1}, {"a": 0.0, "b": 1.0}, id="bool-ignored"),
        pytest.param({"a": "0.7", "b": 1}, {"a": 0.0, "b": 1.0}, id="string-ignored"),
        pytest.param({"b": 1, "extra": 5}, {"a": 0.0, "b": 1.0}, id="missing-and-extra-keys"),
        pytest.param({"a": 0, "b": 0}, None, id="all-zero"),
        pytest.param({}, None, id="empty"),
    ],
)
def test_normalize(values: dict[str, object], expected: dict[str, float] | None) -> None:
    assert normalize(values, _OPTIONS) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(0.3, 0.3, id="in-range"),
        pytest.param(1.4, 1.0, id="clipped-high"),
        pytest.param(-0.2, 0.0, id="clipped-low"),
        pytest.param(1, 1.0, id="int"),
        pytest.param(math.nan, None, id="nan"),
        pytest.param(math.inf, None, id="inf"),
        pytest.param(True, None, id="bool"),
        pytest.param("0.7", None, id="string"),
        pytest.param(None, None, id="none"),
    ],
)
def test_unit_probability(value: object, expected: float | None) -> None:
    assert unit_probability(value) == expected


def test_to_matrix_orders_columns_and_handles_empty() -> None:
    matrix = to_matrix([{"b": 0.25, "a": 0.75}], _OPTIONS)
    assert matrix.tolist() == [[0.75, 0.25]]
    assert to_matrix([], _OPTIONS).shape == (0, 2)


def test_softmax_is_stable_for_large_scores() -> None:
    result = softmax(np.array([[1000.0, 0.0]]), 1.0)
    assert result.tolist() == [[1.0, 0.0]]


@given(
    scores=arrays(np.float64, (3, 4), elements=st.integers(-100, 100).map(lambda i: i / 100)),
    temperature=st.floats(0.01, 10),
)
def test_softmax_rows_sum_to_one_and_argmax_ignores_temperature(
    scores: np.ndarray, temperature: float
) -> None:
    probabilities = softmax(scores, temperature)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert (argmax_labels(probabilities) == argmax_labels(softmax(scores, 1.0))).all()


def test_argmax_ties_resolve_to_first_option() -> None:
    assert argmax_labels(np.array([[0.5, 0.5], [0.2, 0.8]])).tolist() == [0, 1]


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        pytest.param([0.25, 0.25, 0.25, 0.25], 2.0, id="uniform-4"),
        pytest.param([1.0, 0.0, 0.0, 0.0], 0.0, id="one-hot"),
    ],
)
def test_entropy_bits(row: list[float], expected: float) -> None:
    assert entropy(np.array([row]))[0] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("p", "q", "expected"),
    [
        pytest.param([0.5, 0.5, 0.0], [0.0, 0.5, 0.5], 0.5, id="half-overlap"),
        pytest.param([1.0, 0.0], [0.0, 1.0], 1.0, id="disjoint"),
        pytest.param([0.3, 0.7], [0.3, 0.7], 0.0, id="identical"),
    ],
)
def test_js_divergence_known_values(p: list[float], q: list[float], expected: float) -> None:
    assert js_divergence(np.array([p]), np.array([q]))[0] == pytest.approx(expected)


_ROWS = arrays(np.float64, (2, 3), elements=st.floats(0.0, 1.0)).filter(
    lambda m: (m.sum(axis=1) > 0).all()
)


@given(p=_ROWS, q=_ROWS)
def test_js_divergence_is_symmetric_and_bounded(p: np.ndarray, q: np.ndarray) -> None:
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    forward, backward = js_divergence(p, q), js_divergence(q, p)
    assert np.allclose(forward, backward)
    assert ((forward >= 0.0) & (forward <= 1.0)).all()
    assert np.allclose(js_divergence(p, p), 0.0, atol=1e-12)


def test_expected_level() -> None:
    assert expected_level(np.array([[0.0, 0.0, 1.0], [0.5, 0.5, 0.0]])).tolist() == [2.0, 0.5]
