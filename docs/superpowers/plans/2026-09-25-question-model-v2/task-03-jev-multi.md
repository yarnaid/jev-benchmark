### Task 3: Jev: unfold a multi question into `noul`s, fold the answers back

**Files:**
- Modify: `src/jev_bench/classifiers/base.py` (add `missing_labels_note`)
- Modify: `src/jev_bench/classifiers/jev.py`
- Test: `tests/test_classifiers_base.py`, `tests/test_classifiers_jev.py`

**Interfaces:**
- Consumes (Task 1):
  - `MultiQuestion`;
  - `independent_probabilities(values, options) -> (dict[str, float], missing: int)`;
  - the `multi_questions` fixture (topics: billing / meeting / travel).
- Produces:
  - `classifiers.base.missing_labels_note(missing: int, total: int) -> str | None`, returning
    `"<missing> of <total> labels missing or unusable, set to 0"` or `None` when `missing == 0`. Task 4 uses
    it too.
  - `classifiers.jev.label_key(question_id: str, option_id: str) -> str`, returning
    `f"{question_id}__{option_id}"`.
  - `questions_payload()` now emits one `noul` entry per option of every multi question and raises
    `ValueError` on a key collision.
  - `parse_decisions()` returns the folded `{option: p}` for a multi question. The notes are:
    - `"<qid>: <n> of <k> labels missing or unusable, set to 0"` when some labels are unusable;
    - `"<qid>: missing or mistyped answer"` when all are.

**Behavior** (spec §4.1):
- A label sub-answer counts only if it is `{"type": "noul", "noul": <finite number>}`. The value is clipped
  to [0, 1] and never renormalized.
- Unusable sub-answers become 0.0.
- The error handling of non-multi questions is unchanged. `jev_output_reserve` is unchanged; Task 12's paid
  smoke checks it.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_classifiers_base.py` (and add `missing_labels_note` to its import from
`jev_bench.classifiers.base`):

```python
@pytest.mark.parametrize(
    ("missing", "total", "expected"),
    [
        pytest.param(0, 3, None, id="none-missing"),
        pytest.param(2, 3, "2 of 3 labels missing or unusable, set to 0", id="some-missing"),
    ],
)
def test_missing_labels_note(missing: int, total: int, expected: str | None) -> None:
    assert missing_labels_note(missing, total) == expected
```

In `tests/test_classifiers_jev.py`, change the import to
`from jev_bench.questions import NoulQuestion, QuestionSet` and append:

```python
_LABEL_ANSWERS: dict[str, Any] = {
    "topics__billing": {"type": "noul", "noul": 0.9},
    "topics__meeting": {"type": "noul", "noul": 0.6},
    "topics__travel": {"type": "noul", "noul": 0.05},
}


def test_questions_payload_unfolds_multi_labels(multi_questions: QuestionSet) -> None:
    payload = questions_payload(multi_questions)
    assert list(payload) == [
        "category",
        "urgency",
        "needs_reply",
        "topics__billing",
        "topics__meeting",
        "topics__travel",
    ]
    assert payload["topics__meeting"] == {
        "type": "noul",
        "instructions": (
            "Does this label apply to the email: About a meeting? "
            "Several labels can apply to one email."
        ),
        "criteria": {"true": "About a meeting", "false": "This label does not apply to the email"},
    }


def test_questions_payload_rejects_label_key_collisions(multi_questions: QuestionSet) -> None:
    clash = NoulQuestion(
        type="noul", id="topics__billing", instructions="?", options={"yes": "y", "no": "n"}
    )
    questions = QuestionSet(name="clash", questions=(*multi_questions.questions, clash))
    with pytest.raises(ValueError, match="topics__billing"):
        questions_payload(questions)


def test_parse_decisions_folds_multi_labels(multi_questions: QuestionSet) -> None:
    parsed, notes = parse_decisions({**_ANSWERS, **_LABEL_ANSWERS}, multi_questions)
    assert notes == []
    assert parsed["topics"] == pytest.approx({"billing": 0.9, "meeting": 0.6, "travel": 0.05})
    assert parsed["category"] == pytest.approx(_EXPECTED["category"])


@pytest.mark.parametrize(
    ("labels", "expected", "note"),
    [
        pytest.param(
            {"topics__billing": {"type": "noul", "noul": 0.9}},
            {"billing": 0.9, "meeting": 0.0, "travel": 0.0},
            "topics: 2 of 3 labels missing or unusable, set to 0",
            id="two-labels-missing",
        ),
        pytest.param(
            {**_LABEL_ANSWERS, "topics__travel": {"type": "choice", "choice": "x"}},
            {"billing": 0.9, "meeting": 0.6, "travel": 0.0},
            "topics: 1 of 3 labels missing or unusable, set to 0",
            id="label-mistyped",
        ),
        pytest.param(
            {**_LABEL_ANSWERS, "topics__meeting": {"type": "noul", "noul": "high"}},
            {"billing": 0.9, "meeting": 0.0, "travel": 0.05},
            "topics: 1 of 3 labels missing or unusable, set to 0",
            id="label-not-a-number",
        ),
        pytest.param(
            {**_LABEL_ANSWERS, "topics__billing": {"type": "noul", "noul": 1.7}},
            {"billing": 1.0, "meeting": 0.6, "travel": 0.05},
            None,
            id="label-clipped",
        ),
        pytest.param({}, None, "topics: missing or mistyped answer", id="all-labels-missing"),
    ],
)
def test_parse_decisions_degraded_labels(
    multi_questions: QuestionSet,
    labels: dict[str, Any],
    expected: dict[str, float] | None,
    note: str | None,
) -> None:
    parsed, notes = parse_decisions({**_ANSWERS, **labels}, multi_questions)
    assert parsed.get("topics") == (None if expected is None else pytest.approx(expected))
    assert notes == ([] if note is None else [note])


async def test_classify_sends_unfolded_labels_and_folds_answers(
    make_client: ClientFactory, multi_questions: QuestionSet
) -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        body = {"model": "m", "answers": {**_ANSWERS, **_LABEL_ANSWERS}, "usage": {}}
        return httpx2.Response(200, json=body)

    email = EmailFactory()
    result = await _classifier(make_client(handler), multi_questions).classify([email])
    assert "topics__travel" in json.loads(seen[0].content)["questions"]
    answers = result.outcomes[email.id].answers
    assert answers is not None
    assert answers["topics"] == pytest.approx({"billing": 0.9, "meeting": 0.6, "travel": 0.05})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_classifiers_base.py tests/test_classifiers_jev.py -q`
Expected: collection ERROR (`cannot import name 'missing_labels_note'`).

- [ ] **Step 3: Add `missing_labels_note` to `src/jev_bench/classifiers/base.py`**

Add `"missing_labels_note",` to `__all__` (after `"failed_result",`), add the docstring line
`    missing_labels_note: note for a multi-label answer with unusable labels (None when none are).` after the
`outcome_from_parsed` line, and append:

```python
def missing_labels_note(missing: int, total: int) -> str | None:
    return f"{missing} of {total} labels missing or unusable, set to 0" if missing else None
```

- [ ] **Step 4: Implement the Jev changes in `src/jev_bench/classifiers/jev.py`**

Replace the module docstring with:

```python
"""Jev column: one Decisions API call per email carrying every question.

Decisions has no multi-label primitive, so a multi question is unfolded into one `noul` per option
(keyed by `label_key`) in the same call, and the answers are folded back into one vector.

Constants:
    DECISIONS_PATH
Classes:
    JevClassifier
Functions:
    questions_payload: QuestionSet -> Decisions `questions` object (raises ValueError when an
        unfolded label key collides with another question id).
    label_key: Decisions key of one option of a multi question.
    parse_decisions: Decisions `answers` -> canonical distributions and per-question notes.
"""
```

Update the imports:
- add `missing_labels_note` to the `jev_bench.classifiers.base` import;
- make the distributions import
  `from jev_bench.metrics.distributions import (independent_probabilities, normalize, unit_probability)`;
- add `MultiQuestion` to the `jev_bench.questions` import.

Add `"label_key",` to `__all__` (before `"parse_decisions",`). Below `DECISIONS_PATH`, add:

```python
_LABEL_FALSE = "This label does not apply to the email"
```

Replace `questions_payload` with the following (keep `_question_payload` as it is):

```python
def questions_payload(questions: QuestionSet) -> dict[str, JsonObject]:
    payload: dict[str, JsonObject] = {}
    for question in questions.questions:
        for key, entry in _entries(question):
            if key in payload:
                raise ValueError(f"Decisions question key {key!r} is used twice")
            payload[key] = entry
    return payload


def label_key(question_id: str, option_id: str) -> str:
    return f"{question_id}__{option_id}"


def _entries(question: AnyQuestion) -> list[tuple[str, JsonObject]]:
    if isinstance(question, MultiQuestion):
        return [
            (label_key(question.id, option), _label_payload(text))
            for option, text in question.options.items()
        ]
    return [(question.id, _question_payload(question))]


def _label_payload(description: str) -> JsonObject:
    instructions = (
        f"Does this label apply to the email: {description}? Several labels can apply to one email."
    )
    criteria = {"true": description, "false": _LABEL_FALSE}
    return {"type": "noul", "instructions": instructions, "criteria": criteria}
```

Replace the loop body of `parse_decisions` and add three helpers after it (`_parse_answer` and the rest stay
unchanged):

```python
def parse_decisions(
    answers: object, questions: QuestionSet
) -> tuple[dict[str, Distribution], list[str]]:
    parsed: dict[str, Distribution] = {}
    notes: list[str] = []
    answers_dict = answers if isinstance(answers, Mapping) else {}
    for question in questions.questions:
        distribution, note = _parse_question(question, answers_dict)
        if note:
            notes.append(f"{question.id}: {note}")
        if distribution is not None:
            parsed[question.id] = distribution
    return parsed, notes


def _parse_question(question: AnyQuestion, answers: Mapping[str, Any]) -> Parsed:
    if isinstance(question, MultiQuestion):
        return _parse_labels(question, answers)
    answer = answers.get(question.id)
    if not isinstance(answer, dict) or answer.get("type") != question.type:
        return None, "missing or mistyped answer"
    return _parse_answer(question, answer)


def _parse_labels(question: MultiQuestion, answers: Mapping[str, Any]) -> Parsed:
    options = question.option_ids
    raw = {option: _noul_value(answers.get(label_key(question.id, option))) for option in options}
    distribution, missing = independent_probabilities(raw, options)
    if missing == len(options):
        return None, "missing or mistyped answer"
    return distribution, missing_labels_note(missing, len(options))


def _noul_value(answer: object) -> object:
    if isinstance(answer, dict) and answer.get("type") == "noul":
        return answer.get("noul")
    return None
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_classifiers_base.py tests/test_classifiers_jev.py -q`
Expected: all pass (validated while planning: 46 tests). The existing exact-note test
`test_parse_decisions_missing_question` still passes, because non-multi notes are unchanged.

- [ ] **Step 6: Gates, then commit**

```bash
uv run ruff check --fix && uv run ruff format && uv run pyright && uv run pytest -q
git add src/jev_bench/classifiers/base.py src/jev_bench/classifiers/jev.py \
  tests/test_classifiers_base.py tests/test_classifiers_jev.py
git commit -m "feat(jev): send multi-label questions as one noul per label and fold the answers

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
