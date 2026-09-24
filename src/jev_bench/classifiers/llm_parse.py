"""Parsing of chat-model answers: per-email probabilities, all-in-one results, streaming progress.

Constants:
    REF_KEY: the JSON key counted to estimate all-in-one progress.
Classes:
    RefCounter: counts completed `"ref"` keys in a text stream.
Functions:
    parse_email_answers: one email's JSON answers -> distributions and per-question notes.
    split_results: `results[]` -> answer objects by ref (first win) plus duplicate notes.
"""

from collections.abc import Sequence
from typing import Any

from jev_bench.classifiers.base import ProgressCallback
from jev_bench.metrics.distributions import normalize, unit_probability
from jev_bench.questions import AnyQuestion, Distribution, NoulQuestion, QuestionSet

__all__ = [
    "REF_KEY",
    "RefCounter",
    "parse_email_answers",
    "split_results",
]

REF_KEY = '"ref"'


def parse_email_answers(
    raw: object, questions: QuestionSet
) -> tuple[dict[str, Distribution], list[str]]:
    if not isinstance(raw, dict):
        return {}, ["answer is not a JSON object"]
    parsed: dict[str, Distribution] = {}
    notes: list[str] = []
    for question in questions.questions:
        distribution = _parse_value(question, raw.get(question.id))
        if distribution is None:
            notes.append(f"{question.id}: missing or unusable probabilities")
        else:
            parsed[question.id] = distribution
    return parsed, notes


def _parse_value(question: AnyQuestion, value: object) -> Distribution | None:
    if isinstance(question, NoulQuestion):
        probability = unit_probability(value)
        return None if probability is None else {"yes": probability, "no": 1.0 - probability}
    return normalize(value, question.option_ids) if isinstance(value, dict) else None


def split_results(
    payload: object, refs: Sequence[str]
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    items = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise ValueError("response has no results array")
    wanted = set(refs)
    found: dict[str, dict[str, Any]] = {}
    notes: dict[str, str] = {}
    for item in items:
        ref = item.get("ref") if isinstance(item, dict) else None
        if not isinstance(ref, str) or ref not in wanted:
            continue
        if ref in found:
            notes[ref] = "duplicate ref in response; first occurrence kept"
        else:
            found[ref] = item
    return found, notes


class RefCounter:
    def __init__(self, callback: ProgressCallback) -> None:
        self._callback = callback
        self._tail = ""
        self.count = 0

    def feed(self, text: str) -> None:
        window = self._tail + text
        found = window.count(REF_KEY)
        if found:
            self.count += found
            self._callback(self.count)
        self._tail = window[-(len(REF_KEY) - 1) :]
