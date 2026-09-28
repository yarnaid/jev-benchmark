"""Tests for jev_bench.compare.quality."""

import pytest

from jev_bench.compare.pairs import PairStats
from jev_bench.compare.quality import QualityScore, quality_scores
from jev_bench.compare.raters import RaterKind

KINDS: dict[str, RaterKind] = {"r1": "run", "r2": "run", "reference": "reference", "human": "human"}


def pair(a: str, b: str, kappa: float | None, agreement: float, n: int = 4) -> PairStats:
    return PairStats(
        a=a,
        b=b,
        n=n,
        agreement=agreement,
        agreement_ci=None,
        kappa=kappa,
        kappa_ci=None,
        jsd=0.1,
        pearson=None,
        brier=None,
    )


def score(
    rater: str, target: str, kappa: float, agreement: float, questions: int, emails: int
) -> QualityScore:
    return QualityScore(
        rater=rater,
        target=target,
        kappa=kappa,
        agreement=agreement,
        n_questions=questions,
        n_emails=emails,
    )


@pytest.mark.parametrize(
    ("pairs", "expected"),
    [
        pytest.param([], [], id="no_pairs"),
        pytest.param(
            [pair("r1", "reference", 0.5, 0.75)],
            [score("r1", "reference", 0.5, 0.75, 1, 4)],
            id="one_question",
        ),
        pytest.param(
            [pair("r1", "reference", 0.25, 0.5, n=4), pair("r1", "reference", 0.75, 1.0, n=3)],
            [score("r1", "reference", 0.5, 0.75, 2, 4)],
            id="mean_over_questions_most_emails",
        ),
        pytest.param(
            [pair("r1", "reference", None, 1.0), pair("r1", "reference", 0.5, 0.5)],
            [score("r1", "reference", 0.5, 0.5, 1, 4)],
            id="undefined_kappa_question_left_out_of_both_means",
        ),
        pytest.param([pair("r1", "reference", None, 1.0)], [], id="only_undefined_kappa"),
        pytest.param(
            [pair("reference", "r1", 0.5, 0.75)],
            [score("r1", "reference", 0.5, 0.75, 1, 4)],
            id="target_on_the_left",
        ),
        pytest.param([pair("r1", "r2", 0.5, 0.75)], [], id="run_pair_ignored"),
        pytest.param([pair("reference", "human", 0.5, 0.75)], [], id="target_pair_ignored"),
        pytest.param([pair("r1", "ghost", 0.5, 0.75)], [], id="unknown_rater_ignored"),
        pytest.param(
            [
                pair("r2", "reference", 0.25, 0.5),
                pair("r1", "human", 0.75, 1.0, n=2),
                pair("r1", "reference", 0.5, 0.75),
            ],
            [
                score("r1", "reference", 0.5, 0.75, 1, 4),
                score("r1", "human", 0.75, 1.0, 1, 2),
                score("r2", "reference", 0.25, 0.5, 1, 4),
            ],
            id="rater_order_then_target_order",
        ),
    ],
)
def test_quality_scores(pairs: list[PairStats], expected: list[QualityScore]) -> None:
    assert quality_scores(pairs, KINDS) == expected
