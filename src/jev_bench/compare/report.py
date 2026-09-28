"""Full per-question comparison report: per-rater stats, pairwise metrics, Fleiss' kappa.

Classes:
    RaterStats: per-rater summary for one question (argmax counts, means, entropy, confidence; the
        0-100 mean score of a score question; applied-label counts and labels per email of a multi
        question).
    QuestionReport: one question's rater stats, pairs, Fleiss' kappa (the macro Fleiss' kappa over
        applied labels for a multi question), skipped raters and the threshold used (multi only).
    RaterSummary: one rater's identity and item count, with its RunMeta if it is a run.
    ComparisonReport: the full report (rater summaries, per-question reports, the quality of every
        run against the hard raters, warnings).
Functions:
    compare: full per-question report for a set of raters against a base question set;
        `threshold` overrides every multi question's own threshold.
"""

from collections.abc import Callable, Mapping, Sequence
from itertools import combinations
from typing import Literal

import numpy as np
from pydantic import BaseModel

from jev_bench.compare.multi import multi_fleiss, multi_rater_values
from jev_bench.compare.pairs import (
    PairStats,
    RaterMatrix,
    pair_stats,
    rater_matrix,
    resample_index_cache,
    slice_matrix,
)
from jev_bench.compare.quality import QualityScore, quality_scores
from jev_bench.compare.raters import Rater, RaterKind
from jev_bench.metrics.agreement import fleiss_kappa
from jev_bench.metrics.distributions import (
    FloatArray,
    IntArray,
    argmax_labels,
    entropy,
    expected_level,
    score_0_100,
)
from jev_bench.questions import AnyQuestion, MultiQuestion, QuestionSet, with_threshold
from jev_bench.store.runs import RunMeta

__all__ = [
    "ComparisonReport",
    "QuestionReport",
    "RaterStats",
    "RaterSummary",
    "compare",
]


class RaterStats(BaseModel):
    rater: str
    n: int
    argmax_counts: dict[str, int]
    mean: dict[str, float]
    mean_entropy: float
    mean_confidence: float
    mean_level: float | None = None
    mean_score: float | None = None
    label_counts: dict[str, int] | None = None
    mean_labels: float | None = None


class QuestionReport(BaseModel):
    id: str
    type: Literal["choice", "score", "noul", "multi"]
    options: tuple[str, ...]
    raters: list[RaterStats]
    pairs: list[PairStats]
    fleiss_kappa: float | None
    skipped: list[str]
    threshold: float | None = None


class RaterSummary(BaseModel):
    id: str
    label: str
    kind: RaterKind
    n_items: int
    run: RunMeta | None = None


class ComparisonReport(BaseModel):
    raters: list[RaterSummary]
    questions: list[QuestionReport]
    quality: list[QualityScore]
    warnings: list[str]


def compare(
    raters: Sequence[Rater],
    base: QuestionSet,
    *,
    resamples: int = 1000,
    seed: int = 0,
    runs: Mapping[str, RunMeta] | None = None,
    threshold: float | None = None,
) -> ComparisonReport:
    warnings: list[str] = []
    index_for = resample_index_cache(resamples, seed)
    fleiss_gaps: dict[str, list[str]] = {}
    questions = [
        _question_report(question, raters, index_for, warnings, fleiss_gaps)
        for question in with_threshold(base, threshold).questions
    ]
    warnings.extend(_fleiss_gap_warning(rater_id, ids) for rater_id, ids in fleiss_gaps.items())
    for rater in raters:
        warnings.extend(rater.warnings)
    metas = runs or {}
    summaries = [
        RaterSummary(
            id=rater.id,
            label=rater.label,
            kind=rater.kind,
            n_items=len(rater.answers),
            run=metas.get(rater.id),
        )
        for rater in raters
    ]
    pairs = (pair for question in questions for pair in question.pairs)
    quality = quality_scores(pairs, {rater.id: rater.kind for rater in raters})
    return ComparisonReport(
        raters=summaries, questions=questions, quality=quality, warnings=warnings
    )


def _question_report(
    question: AnyQuestion,
    raters: Sequence[Rater],
    index_for: Callable[[int], IntArray],
    warnings: list[str],
    fleiss_gaps: dict[str, list[str]],
) -> QuestionReport:
    usable = [rater for rater in raters if rater.supports(question)]
    skipped = [rater.id for rater in raters if not rater.supports(question)]
    if skipped:
        warnings.append(_skip_warning(question, skipped))
    matrices = {
        rater.id: rater_matrix(column, question)
        for rater in usable
        if (column := rater.column(question.id))
    }
    pairs = [
        pair
        for left, right in combinations(usable, 2)
        if (pair := pair_stats(left, right, matrices, question, index_for)) is not None
    ]
    return QuestionReport(
        id=question.id,
        type=question.type,
        options=question.option_ids,
        raters=[
            _rater_stats(rater.id, matrices[rater.id], question)
            for rater in usable
            if rater.id in matrices
        ],
        pairs=pairs,
        fleiss_kappa=_fleiss(
            [rater for rater in usable if rater.kind == "run"], matrices, question, fleiss_gaps
        ),
        skipped=skipped,
        threshold=question.threshold if isinstance(question, MultiQuestion) else None,
    )


def _skip_warning(question: AnyQuestion, skipped: Sequence[str]) -> str:
    return (
        f"{question.id}: incompatible snapshot (missing question, different type or options): "
        f"{', '.join(skipped)}"
    )


def _fleiss_gap_warning(rater_id: str, question_ids: Sequence[str]) -> str:
    return f"rater {rater_id} answered no emails for questions: {', '.join(question_ids)}"


def _rater_stats(rater_id: str, rm: RaterMatrix, question: AnyQuestion) -> RaterStats:
    options = question.option_ids
    matrix = rm.matrix
    counts = np.bincount(argmax_labels(matrix), minlength=len(options))
    is_score = question.type == "score"
    stats = RaterStats(
        rater=rater_id,
        n=len(rm.positions),
        argmax_counts={option: int(count) for option, count in zip(options, counts, strict=True)},
        mean={
            option: float(value) for option, value in zip(options, matrix.mean(axis=0), strict=True)
        },
        mean_entropy=float(entropy(matrix).mean()),
        mean_confidence=float(matrix.max(axis=1).mean()),
        mean_level=float(expected_level(matrix).mean()) if is_score else None,
        mean_score=float(score_0_100(matrix).mean()) if is_score else None,
    )
    if isinstance(question, MultiQuestion):
        return stats.model_copy(update=multi_rater_values(matrix, question)._asdict())
    return stats


def _fleiss(
    runs: Sequence[Rater],
    matrices: dict[str, RaterMatrix],
    question: AnyQuestion,
    fleiss_gaps: dict[str, list[str]],
) -> float | None:
    for rater in runs:
        if rater.id not in matrices:
            fleiss_gaps.setdefault(rater.id, []).append(question.id)
    answered = [rater for rater in runs if rater.id in matrices]
    if len(answered) < 2:
        return None
    shared = sorted(set.intersection(*(set(matrices[rater.id].positions) for rater in answered)))
    if not shared:
        return None
    return _shared_fleiss(
        [slice_matrix(matrices[rater.id], shared) for rater in answered], question
    )


def _shared_fleiss(slices: Sequence[FloatArray], question: AnyQuestion) -> float | None:
    if isinstance(question, MultiQuestion):
        return multi_fleiss(slices, question.threshold)
    labels = np.stack([argmax_labels(matrix) for matrix in slices], axis=1)
    return fleiss_kappa(labels, len(question.option_ids))
