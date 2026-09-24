### Task 9: Chat-model answer schemas and parsing

**Files:**
- Create: `src/jev_bench/json_schema.py`, `src/jev_bench/classifiers/llm_schema.py`,
  `src/jev_bench/classifiers/llm_parse.py`
- Test: `tests/test_json_schema.py`, `tests/test_classifiers_llm_schema.py`,
  `tests/test_classifiers_llm_parse.py`

**Interfaces:**
- Consumes: `QuestionSet`, `NoulQuestion`, `Distribution` (Task 2); `normalize` and `unit_probability`
  (Task 4); `JsonObject` (Task 5); `ProgressCallback` (Task 8).
- Produces:
  - `jev_bench.json_schema.strict_object(properties, description=None) -> JsonObject`: an object
    schema with `additionalProperties: false` and every property required. The generation schema in
    Task 15 reuses it.
  - `jev_bench.classifiers.llm_schema`:
    - `answers_schema(qs) -> JsonObject`;
    - `all_in_one_schema(qs, refs) -> JsonObject`;
    - `email_refs(count) -> list[str]`.
  - `jev_bench.classifiers.llm_parse`:
    - `parse_email_answers(raw, qs) -> tuple[dict[str, Distribution], list[str]]`;
    - `split_results(payload, refs) -> tuple[dict[str, dict[str, Any]], dict[str, str]]` (raises
      `ValueError` when there is no `results` array);
    - `RefCounter(callback)` with `.feed(text)` and `.count`.

Schema shape (strict mode):
- every object has `additionalProperties: false` and lists all its properties in `required`;
- noul questions are a single `number`, P(yes);
- choice and score questions are an object with one `number` per option id;
- no numeric `minimum` / `maximum`, since strict modes differ across providers; parsing clips and
  renormalizes instead.

Review Focus #4 is pinned here at the answer level. A NaN, a string, a negative value, or zero mass across
all options each becomes a per-question note, never an exception.

- [ ] **Step 1: Write the failing tests**

`tests/test_json_schema.py`:
```python
"""Tests for jev_bench.json_schema."""

import pytest

from jev_bench.json_schema import strict_object


@pytest.mark.parametrize(
    ("description", "expected_extra"),
    [
        pytest.param(None, {}, id="plain"),
        pytest.param("Kind?", {"description": "Kind?"}, id="described"),
    ],
)
def test_strict_object(description: str | None, expected_extra: dict[str, str]) -> None:
    properties = {"a": {"type": "string"}, "b": {"type": "number"}}
    assert strict_object(properties, description) == {
        "type": "object",
        "additionalProperties": False,
        "required": ["a", "b"],
        "properties": properties,
        **expected_extra,
    }
```

`tests/test_classifiers_llm_schema.py`:
```python
"""Tests for jev_bench.classifiers.llm_schema."""

import pytest

from jev_bench.classifiers.llm_schema import all_in_one_schema, answers_schema, email_refs
from jev_bench.questions import QuestionSet


def _options(description: str, *ids: str) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(ids),
        "properties": {option: {"type": "number"} for option in ids},
        "description": description,
    }


_EXPECTED_PROPERTIES = {
    "category": _options("What kind of email?", "spam", "personal", "work"),
    "urgency": _options("How urgent?", "low", "today", "now"),
    "needs_reply": {"type": "number", "description": "Probability that the answer is yes: Needs a reply?"},
}


def test_answers_schema(questions: QuestionSet) -> None:
    assert answers_schema(questions) == {
        "type": "object",
        "additionalProperties": False,
        "required": ["category", "urgency", "needs_reply"],
        "properties": _EXPECTED_PROPERTIES,
    }


def test_all_in_one_schema(questions: QuestionSet) -> None:
    schema = all_in_one_schema(questions, ["e001", "e002"])
    item = schema["properties"]["results"]["items"]
    assert schema["required"] == ["results"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["results"]["type"] == "array"
    assert item["required"] == ["ref", "category", "urgency", "needs_reply"]
    assert item["properties"]["ref"] == {"type": "string", "enum": ["e001", "e002"]}
    assert item["properties"]["category"] == _EXPECTED_PROPERTIES["category"]


@pytest.mark.parametrize(
    ("count", "first", "last", "length"),
    [
        pytest.param(3, "e001", "e003", 3, id="small"),
        pytest.param(1000, "e0001", "e1000", 1000, id="widens"),
    ],
)
def test_email_refs(count: int, first: str, last: str, length: int) -> None:
    refs = email_refs(count)
    assert (refs[0], refs[-1], len(refs)) == (first, last, length)


def test_email_refs_empty() -> None:
    assert email_refs(0) == []
```

`tests/test_classifiers_llm_parse.py`:
```python
"""Tests for jev_bench.classifiers.llm_parse."""

import math
from typing import Any

import pytest

from jev_bench.classifiers.llm_parse import RefCounter, parse_email_answers, split_results
from jev_bench.questions import QuestionSet

_VALID: dict[str, Any] = {
    "category": {"spam": 0.7, "personal": 0.2, "work": 0.1},
    "urgency": {"low": 1, "today": 1, "now": 2},
    "needs_reply": 0.25,
    "ref": "e001",
}


def test_parse_email_answers_valid(questions: QuestionSet) -> None:
    parsed, notes = parse_email_answers(_VALID, questions)
    assert notes == []
    assert parsed["category"] == pytest.approx({"spam": 0.7, "personal": 0.2, "work": 0.1})
    assert parsed["urgency"] == pytest.approx({"low": 0.25, "today": 0.25, "now": 0.5})
    assert parsed["needs_reply"] == pytest.approx({"yes": 0.25, "no": 0.75})


@pytest.mark.parametrize(
    ("question", "value", "expected"),
    [
        pytest.param("needs_reply", math.nan, None, id="noul-nan"),
        pytest.param("needs_reply", "0.7", None, id="noul-string"),
        pytest.param("needs_reply", 3.0, {"yes": 1.0, "no": 0.0}, id="noul-clipped"),
        pytest.param("category", {"spam": 0, "personal": 0, "work": 0}, None, id="choice-zero-mass"),
        pytest.param("category", {"spam": -1, "personal": 1, "work": 0}, {"spam": 0.0, "personal": 1.0, "work": 0.0}, id="choice-negative-clipped"),
        pytest.param("category", [0.5, 0.5], None, id="choice-not-object"),
        pytest.param("urgency", None, None, id="score-missing"),
    ],
)
def test_parse_email_answers_degraded(questions: QuestionSet, question: str, value: object, expected: dict[str, float] | None) -> None:
    parsed, notes = parse_email_answers({**_VALID, question: value}, questions)
    if expected is None:
        assert question not in parsed
        assert notes == [f"{question}: missing or unusable probabilities"]
    else:
        assert parsed[question] == pytest.approx(expected)
        assert notes == []


def test_parse_email_answers_rejects_non_object(questions: QuestionSet) -> None:
    assert parse_email_answers([1, 2], questions) == ({}, ["answer is not a JSON object"])


_A = {"ref": "e001", "x": 1}
_B = {"ref": "e002", "x": 2}


@pytest.mark.parametrize(
    ("payload", "found", "notes"),
    [
        pytest.param({"results": [_A, _B]}, {"e001": _A, "e002": _B}, {}, id="complete"),
        pytest.param({"results": [_B]}, {"e002": _B}, {}, id="missing-ref"),
        pytest.param({"results": [_A, {"ref": "e001", "x": 9}]}, {"e001": _A}, {"e001": "duplicate ref in response; first occurrence kept"}, id="duplicate"),
        pytest.param({"results": [{"ref": "e999"}, "junk", 5, {"no": "ref"}]}, {}, {}, id="unknown-and-junk-ignored"),
    ],
)
def test_split_results(payload: dict[str, Any], found: dict[str, Any], notes: dict[str, str]) -> None:
    assert split_results(payload, ["e001", "e002"]) == (found, notes)


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"items": []}, id="no-results-key"),
        pytest.param({"results": {"e001": {}}}, id="results-not-list"),
        pytest.param([_A], id="top-level-list"),
    ],
)
def test_split_results_requires_results_array(payload: object) -> None:
    with pytest.raises(ValueError, match="no results array"):
        split_results(payload, ["e001"])


def test_ref_counter_counts_tokens_split_across_chunks() -> None:
    seen: list[int] = []
    counter = RefCounter(seen.append)
    for chunk in ['{"results":[{"', "re", 'f":"e001","x":1},', '{"ref":"e002","x":2}]}', " no more "]:
        counter.feed(chunk)
    assert counter.count == 2
    assert seen == [1, 2]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_json_schema.py tests/test_classifiers_llm_schema.py tests/test_classifiers_llm_parse.py -v`
Expected: `ModuleNotFoundError` for `jev_bench.json_schema` / `jev_bench.classifiers.llm_schema` / `llm_parse`.

- [ ] **Step 3: Write the implementation**

`src/jev_bench/json_schema.py`:
```python
"""Builders for strict structured-output JSON schemas.

Functions:
    strict_object: object schema with `additionalProperties: false` and every property required.
"""

from typing import Any

type JsonSchema = dict[str, Any]


def strict_object(properties: dict[str, JsonSchema], description: str | None = None) -> JsonSchema:
    schema: JsonSchema = {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }
    if description:
        schema["description"] = description
    return schema
```

`src/jev_bench/classifiers/llm_schema.py`:
```python
"""JSON schemas for chat-model answers: one email, or an all-in-one `results` array.

Functions:
    answers_schema: per-email answer object (one number per option; noul questions give P(yes)).
    all_in_one_schema: `{results: [{ref, <answers>}]}` with `ref` restricted to the refs sent.
    email_refs: positional refs e001, e002, ... (zero-padded, widening past 999).
"""

from collections.abc import Sequence

from jev_bench.json_schema import JsonSchema, strict_object
from jev_bench.questions import AnyQuestion, NoulQuestion, QuestionSet


def answers_schema(questions: QuestionSet) -> JsonSchema:
    return strict_object({question.id: _question_schema(question) for question in questions.questions})


def all_in_one_schema(questions: QuestionSet, refs: Sequence[str]) -> JsonSchema:
    ref: JsonSchema = {"type": "string", "enum": list(refs)}
    answers = {question.id: _question_schema(question) for question in questions.questions}
    return strict_object({"results": {"type": "array", "items": strict_object({"ref": ref, **answers})}})


def email_refs(count: int) -> list[str]:
    width = max(3, len(str(count)))
    return [f"e{index:0{width}d}" for index in range(1, count + 1)]


def _question_schema(question: AnyQuestion) -> JsonSchema:
    if isinstance(question, NoulQuestion):
        return {"type": "number", "description": f"Probability that the answer is yes: {question.instructions}"}
    options: dict[str, JsonSchema] = {option: {"type": "number"} for option in question.option_ids}
    return strict_object(options, description=question.instructions)
```

`src/jev_bench/classifiers/llm_parse.py`:
```python
"""Parsing of chat-model answers: per-email probabilities, all-in-one results, streaming progress.

Constants:
    REF_TOKEN: the JSON key counted to estimate all-in-one progress.
Classes:
    RefCounter: counts completed `"ref"` keys in a text stream.
Functions:
    parse_email_answers: one email's JSON answers -> distributions and per-question notes.
    split_results: `results[]` -> answer objects by ref (first occurrence wins) plus duplicate notes.
"""

from collections.abc import Sequence
from typing import Any

from jev_bench.classifiers.base import ProgressCallback
from jev_bench.metrics.distributions import normalize, unit_probability
from jev_bench.questions import AnyQuestion, Distribution, NoulQuestion, QuestionSet

REF_TOKEN = '"ref"'


def parse_email_answers(raw: object, questions: QuestionSet) -> tuple[dict[str, Distribution], list[str]]:
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
        found = window.count(REF_TOKEN)
        if found:
            self.count += found
            self._callback(self.count)
        self._tail = window[-(len(REF_TOKEN) - 1) :]
```

- [ ] **Step 4: Run tests to verify they pass, lint, type-check**

Run: `uv run pytest tests/test_json_schema.py tests/test_classifiers_llm_schema.py tests/test_classifiers_llm_parse.py -v`
Expected: all PASS.

Run: `uv run ruff check --fix && uv run ruff format && uv run pyright`
Expected: `0 errors`.

- [ ] **Step 5: Commit**

```bash
git add src/jev_bench/json_schema.py src/jev_bench/classifiers/llm_schema.py src/jev_bench/classifiers/llm_parse.py tests/test_json_schema.py tests/test_classifiers_llm_schema.py tests/test_classifiers_llm_parse.py
git commit -m "feat(classifiers): strict answer schemas and tolerant chat-answer parsing

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
