"""The data an analysis reads: a comparison of selected runs and its per-email view.

Classes:
    AnalysisSource: the comparison report, per-email rows, run raters and metas (in the order the
        analyst names them R1, R2, …), the emails, the runs' generations and the question set with
        the effective multi-label threshold.
Functions:
    rater_names: rater id -> the name the analyst sees (R1…, "ref", "human").
    email_refs: e001… -> email id, in email order; the analyst never sees email ids.
"""

from collections.abc import Sequence
from typing import NamedTuple

from jev_bench.compare import ComparisonReport, EmailRow, Rater
from jev_bench.emails import Email
from jev_bench.questions import QuestionSet
from jev_bench.store.generations import GenerationMeta
from jev_bench.store.runs import RunMeta

__all__ = [
    "AnalysisSource",
    "email_refs",
    "rater_names",
]


class AnalysisSource(NamedTuple):
    report: ComparisonReport
    rows: list[EmailRow]
    runs: list[Rater]
    metas: list[RunMeta]
    emails: list[Email]
    generations: list[GenerationMeta]
    questions: QuestionSet


def rater_names(source: AnalysisSource) -> dict[str, str]:
    names = {meta.id: f"R{index}" for index, meta in enumerate(source.metas, start=1)}
    return {**names, "reference": "ref", "human": "human"}


def email_refs(emails: Sequence[Email]) -> dict[str, str]:
    return {f"e{index:03d}": email.id for index, email in enumerate(emails, start=1)}
