"""Compact JSON of a comparison report for the analyst: rater ids become analyst names (R1…, ref,
human), floats are rounded to 3 decimals, and nulls, non-finite numbers, option lists, rater labels
and run snapshots are dropped.

Functions:
    report_json: the compact JSON text.
"""

import json
import math
from collections.abc import Mapping

from jev_bench.compare import ComparisonReport
from jev_bench.ids import is_safe_id

__all__ = [
    "report_json",
]

_DROPPED = frozenset({"options", "run", "label"})


def report_json(report: ComparisonReport, names: Mapping[str, str]) -> str:
    compact = _compact(report.model_dump(mode="json"), names)
    return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))


def _compact(value: object, names: Mapping[str, str]) -> object:
    if isinstance(value, dict):
        return {
            key: _compact(item, names)
            for key, item in value.items()
            if key not in _DROPPED and _present(item)
        }
    if isinstance(value, list):
        return [_compact(item, names) for item in value]
    if isinstance(value, float):
        return round(value, 3)
    return _renamed(value, names) if isinstance(value, str) else value


def _present(value: object) -> bool:
    return value is not None and not (isinstance(value, float) and not math.isfinite(value))


def _renamed(text: str, names: Mapping[str, str]) -> str:
    if text in names:
        return names[text]
    for rater_id, name in names.items():
        if is_safe_id(rater_id):
            text = text.replace(rater_id, name)
    return text
