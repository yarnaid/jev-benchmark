"""Headline quality of each run: its agreement with the hard raters (generator reference, human
labels), averaged over questions from the pair statistics already computed per question.

Every question weighs the same. A question whose kappa is undefined is left out of both means, so
`kappa` and `agreement` always average the same `n_questions` questions.

Classes:
    QualityScore: one run against one hard rater: mean kappa (quadratic for score questions, macro
        over labels for multi-label ones), mean agreement (exact label-set match for multi-label),
        the number of questions averaged and the most emails any of them compared.
Functions:
    quality_scores: a QualityScore per (run, hard rater) sharing at least one question with a
        defined kappa, runs in rater order, then hard raters in rater order.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping
from statistics import fmean
from typing import NamedTuple

from pydantic import BaseModel

from jev_bench.compare.pairs import PairStats
from jev_bench.compare.raters import RaterKind

__all__ = [
    "QualityScore",
    "quality_scores",
]

type _Key = tuple[str, str]


class QualityScore(BaseModel):
    rater: str
    target: str
    kappa: float
    agreement: float
    n_questions: int
    n_emails: int


class _Point(NamedTuple):
    kappa: float
    agreement: float
    n: int


def quality_scores(
    pairs: Iterable[PairStats], kinds: Mapping[str, RaterKind]
) -> list[QualityScore]:
    points = _points(pairs, kinds)
    runs = [rater for rater, kind in kinds.items() if kind == "run"]
    targets = [rater for rater, kind in kinds.items() if kind != "run"]
    keys = [(run, target) for run in runs for target in targets]
    return [_score(key, points[key]) for key in keys if key in points]


def _points(pairs: Iterable[PairStats], kinds: Mapping[str, RaterKind]) -> dict[_Key, list[_Point]]:
    points: defaultdict[_Key, list[_Point]] = defaultdict(list)
    for pair in pairs:
        key = _run_and_target(pair, kinds)
        if key is not None and pair.kappa is not None:
            points[key].append(_Point(pair.kappa, pair.agreement, pair.n))
    return points


def _run_and_target(pair: PairStats, kinds: Mapping[str, RaterKind]) -> _Key | None:
    left, right = kinds.get(pair.a), kinds.get(pair.b)
    if left == "run" and _hard(right):
        return pair.a, pair.b
    if right == "run" and _hard(left):
        return pair.b, pair.a
    return None


def _hard(kind: RaterKind | None) -> bool:
    return kind is not None and kind != "run"


def _score(key: _Key, points: list[_Point]) -> QualityScore:
    return QualityScore(
        rater=key[0],
        target=key[1],
        kappa=fmean(point.kappa for point in points),
        agreement=fmean(point.agreement for point in points),
        n_questions=len(points),
        n_emails=max(point.n for point in points),
    )
