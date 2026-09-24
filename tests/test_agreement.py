"""Tests for jev_bench.metrics.agreement."""

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

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

_TEXTBOOK_A = np.array([0] * 20 + [0] * 5 + [1] * 10 + [1] * 15)
_TEXTBOOK_B = np.array([0] * 20 + [1] * 5 + [0] * 10 + [1] * 15)
_QA = np.array([0, 1, 2, 3, 4, 2, 1, 0, 3, 4])
_QB = np.array([0, 2, 2, 4, 4, 1, 1, 1, 3, 3])
_FLEISS_COUNTS = [
    [0, 0, 0, 0, 14],
    [0, 2, 6, 4, 2],
    [0, 0, 3, 5, 6],
    [0, 3, 9, 2, 0],
    [2, 2, 8, 1, 1],
    [7, 7, 0, 0, 0],
    [3, 2, 6, 3, 0],
    [2, 5, 3, 2, 2],
    [6, 5, 2, 1, 0],
    [0, 2, 2, 3, 7],
]


def _labels_from_counts(counts: list[list[int]]) -> np.ndarray:
    return np.array([np.repeat(np.arange(len(row)), row) for row in counts])


def test_percent_agreement() -> None:
    assert percent_agreement(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0])) == 0.75


def test_percent_agreement_rejects_empty() -> None:
    with pytest.raises(ValueError, match="no items"):
        percent_agreement(np.array([], dtype=np.int64), np.array([], dtype=np.int64))


@pytest.mark.parametrize(
    ("a", "b", "k", "quadratic", "expected"),
    [
        pytest.param(_TEXTBOOK_A, _TEXTBOOK_B, 2, False, 0.4, id="textbook-2x2"),
        pytest.param(_QA, _QB, 5, False, 0.375, id="nominal-5"),
        pytest.param(_QA, _QB, 5, True, 0.8648648648648649, id="quadratic-5"),
        pytest.param(np.array([0, 1, 2]), np.array([0, 1, 2]), 3, False, 1.0, id="perfect"),
        pytest.param(
            np.array([1, 1, 1]), np.array([1, 1, 1]), 3, False, None, id="constant-undefined"
        ),
    ],
)
def test_cohen_kappa(
    a: np.ndarray, b: np.ndarray, k: int, quadratic: bool, expected: float | None
) -> None:
    result = cohen_kappa(a, b, k, quadratic=quadratic)
    assert result == (None if expected is None else pytest.approx(expected))


@given(st.lists(st.integers(0, 3), min_size=2, max_size=30))
def test_kappa_of_a_rater_with_itself_is_one(labels: list[int]) -> None:
    assume(len(set(labels)) > 1)
    array = np.array(labels)
    assert cohen_kappa(array, array, 4) == pytest.approx(1.0)


def test_kappa_from_confusion_batch_matches_single() -> None:
    index = np.array([np.arange(_QA.size), np.arange(_QA.size)[::-1]])
    batch = kappa_from_confusion(
        confusion_batch(_QA, _QB, 5, index), disagreement_weights(5, quadratic=True)
    )
    assert batch == pytest.approx([0.8648648648648649, 0.8648648648648649])


def test_confusion_batch_counts() -> None:
    confusion = confusion_batch(np.array([0, 1]), np.array([1, 1]), 2, np.array([[0, 1], [1, 1]]))
    assert confusion.tolist() == [[[0.0, 1.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 2.0]]]


@pytest.mark.parametrize(
    ("k", "quadratic", "expected"),
    [
        pytest.param(3, False, [[0, 1, 1], [1, 0, 1], [1, 1, 0]], id="nominal"),
        pytest.param(3, True, [[0, 0.25, 1], [0.25, 0, 0.25], [1, 0.25, 0]], id="quadratic"),
        pytest.param(1, True, [[0]], id="single-level"),
    ],
)
def test_disagreement_weights(k: int, quadratic: bool, expected: list[list[float]]) -> None:
    assert disagreement_weights(k, quadratic=quadratic).tolist() == expected


@pytest.mark.parametrize(
    ("labels", "k", "expected"),
    [
        pytest.param(_labels_from_counts(_FLEISS_COUNTS), 5, 0.20993070442195522, id="wikipedia"),
        pytest.param(np.array([[0, 0], [1, 1], [2, 2]]), 3, 1.0, id="perfect"),
        pytest.param(np.array([[1, 1], [1, 1]]), 3, None, id="single-category-undefined"),
        pytest.param(np.array([[0], [1]]), 2, None, id="one-rater"),
        pytest.param(np.zeros((0, 3), dtype=np.int64), 2, None, id="no-items"),
    ],
)
def test_fleiss_kappa(labels: np.ndarray, k: int, expected: float | None) -> None:
    result = fleiss_kappa(labels, k)
    assert result == (None if expected is None else pytest.approx(expected))


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        pytest.param([1.0, 2.0, 3.0], [2.0, 4.0, 6.0], 1.0, id="perfect"),
        pytest.param([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], -1.0, id="anti"),
        pytest.param([1.0, 1.0, 1.0], [1.0, 2.0, 3.0], None, id="constant"),
        pytest.param([1.0], [2.0], None, id="single"),
    ],
)
def test_pearson_r(x: list[float], y: list[float], expected: float | None) -> None:
    result = pearson_r(np.array(x), np.array(y))
    assert result == (None if expected is None else pytest.approx(expected))


@pytest.mark.parametrize(
    ("probabilities", "labels", "expected"),
    [
        pytest.param([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], [0, 2], 0.8, id="example"),
        pytest.param([[1.0, 0.0], [0.0, 1.0]], [0, 1], 0.0, id="perfect"),
    ],
)
def test_brier_score(probabilities: list[list[float]], labels: list[int], expected: float) -> None:
    assert brier_score(np.array(probabilities), np.array(labels)) == pytest.approx(expected)
