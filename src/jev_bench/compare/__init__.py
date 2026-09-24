"""Comparison of raters (runs, generator reference, human labels) per question and per email.

Re-exports the package's public API:
    RaterKind, Rater, run_label, run_rater, reference_rater, human_rater (jev_bench.compare.raters)
    PairStats (jev_bench.compare.pairs)
    RaterStats, QuestionReport, RaterSummary, ComparisonReport, compare (jev_bench.compare.report)
    EmailRow, email_rows, disagreement_index (jev_bench.compare.rows)
"""

from jev_bench.compare.pairs import PairStats
from jev_bench.compare.raters import (
    Rater,
    RaterKind,
    human_rater,
    reference_rater,
    run_label,
    run_rater,
)
from jev_bench.compare.report import (
    ComparisonReport,
    QuestionReport,
    RaterStats,
    RaterSummary,
    compare,
)
from jev_bench.compare.rows import EmailRow, disagreement_index, email_rows

__all__ = [
    "ComparisonReport",
    "EmailRow",
    "PairStats",
    "QuestionReport",
    "Rater",
    "RaterKind",
    "RaterStats",
    "RaterSummary",
    "compare",
    "disagreement_index",
    "email_rows",
    "human_rater",
    "reference_rater",
    "run_label",
    "run_rater",
]
