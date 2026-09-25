### Task 6: Label-list reference answers: generator schema/parse/mismatch, email model, raters

**Files:**
- Modify: `src/jev_bench/emails.py` (`reference_answers` type)
- Modify: `src/jev_bench/generation/prompt.py`
- Modify: `config/generation.toml` (system prompt sentence)
- Modify: `src/jev_bench/compare/raters.py` (hard answers via `hard_distribution`)
- Modify: `src/jev_bench/compare/rows.py` (type widening only: `reference`, `human`, the `labels` parameter)
- Test: `tests/test_emails.py`, `tests/test_generation_prompt.py`, `tests/test_config_files.py`,
  `tests/test_compare_raters.py`

**Why the raters and rows are in this task:** as soon as `Email.reference_answers` may hold a list, today's
`raters._reference_hard` evaluates `labels.get(q.id) in question.options`. With a list value that raises
`TypeError: unhashable type: 'list'` at runtime, and pyright also rejects `dict[str, HardAnswer]` →
`dict[str, str]` in `rows.EmailRow`. Both consumers must change together with the type (found by validating
this task while planning).

**Interfaces:**
- Consumes (Task 1): `HardAnswer`, `MultiQuestion`, `hard_distribution`; the `multi_questions` fixture.
- Produces:
  - `Email.reference_answers: dict[str, HardAnswer]`;
  - `GeneratorOutput.answers: dict[str, HardAnswer]`;
  - `generation_schema()`: a multi question is `{"type": "array", "items": {"type": "string",
    "enum": [...]}}`, with **no** `minItems`;
  - `parse_generator_output()` rejects (`ValueError("generator answers missing or invalid for: [...]")`)
    any answer for which `hard_distribution(...) is None`: an empty, duplicate or unknown label list, a bare
    string for multi, or a list for a non-multi question. It goes through the existing same-model retry;
  - `count_mismatches(requested, traits, answers: Mapping[str, HardAnswer])`: a list answer agrees iff it
    contains the requested value;
  - `human_rater(labels: Mapping[str, Mapping[str, HardAnswer]], questions)`;
  - `reference_rater` / `human_rater` build multi-hot rows for multi questions and skip invalid answers;
  - `EmailRow.reference` / `EmailRow.human` are `dict[str, HardAnswer]`, and
    `email_rows(labels: Mapping[str, Mapping[str, HardAnswer]], …)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_emails.py`: change the import to `from jev_bench.emails import Email, Party, email_id` and append:

```python
def test_reference_answers_accept_label_lists() -> None:
    answers = {"category": ["spam", "phishing"], "urgency": "today"}
    email = EmailFactory(reference_answers=answers)
    assert Email.model_validate_json(email.model_dump_json()).reference_answers == answers
```

`tests/test_generation_prompt.py`: add this row to the `test_parse_generator_output_rejects` table, after
`empty-body`:

```python
        pytest.param(
            lambda doc: doc["answers"].update(category=["spam"]), id="list-for-single-choice"
        ),
```

Then append:

```python
_MULTI_CONFIG = GenerationConfig.model_validate(
    {
        "models": ["m"],
        "system_prompt": "Questions:\n$questions",
        "user_prompt": "Write about $topic ($topic_prompt), sent $sent_at.",
        "traits": [{"name": "topic", "question": "topics"}],
    }
)
_MULTI_OUTPUT: dict[str, Any] = {
    **_OUTPUT,
    "answers": {**_OUTPUT["answers"], "topics": ["meeting", "billing"]},
}


def test_generation_schema_multi_is_an_array_of_options(multi_questions: QuestionSet) -> None:
    answers = generation_schema(multi_questions)["properties"]["answers"]
    assert answers["properties"]["topics"] == {
        "type": "array",
        "items": {"type": "string", "enum": ["billing", "meeting", "travel"]},
    }
    assert "topics" in answers["required"]


def test_parse_generator_output_keeps_label_order(multi_questions: QuestionSet) -> None:
    output = parse_generator_output(json.dumps(_MULTI_OUTPUT), multi_questions)
    assert output.answers["topics"] == ["meeting", "billing"]
    assert output.answers["category"] == "spam"


@pytest.mark.parametrize(
    "topics",
    [
        pytest.param([], id="empty-list"),
        pytest.param(["billing", "billing"], id="duplicate"),
        pytest.param(["billing", "phishing"], id="unknown-label"),
        pytest.param("billing", id="bare-string"),
    ],
)
def test_parse_generator_output_rejects_bad_label_lists(
    multi_questions: QuestionSet, topics: object
) -> None:
    doc = json.loads(json.dumps(_MULTI_OUTPUT))
    doc["answers"]["topics"] = topics
    with pytest.raises(ValueError, match="missing or invalid"):
        parse_generator_output(json.dumps(doc), multi_questions)


@pytest.mark.parametrize(
    ("answers", "expected"),
    [
        pytest.param({"topics": ["meeting", "billing"]}, 0, id="requested-label-present"),
        pytest.param({"topics": ["meeting"]}, 1, id="requested-label-absent"),
        pytest.param({}, 1, id="missing-answer"),
    ],
)
def test_count_mismatches_multi_membership(
    multi_questions: QuestionSet, answers: dict[str, list[str]], expected: int
) -> None:
    traits = resolve_traits(_MULTI_CONFIG, multi_questions)
    assert count_mismatches({"topic": "billing"}, traits, answers) == expected
```

`tests/test_config_files.py`, `test_shipped_generation_config`: add after `assert config.max_attempts == 3`:

```python
    assert "every option id that applies (at least one)" in config.system_prompt
```

`tests/test_compare_raters.py`: append (this covers Review Focus #2 on the backend side):

```python
def test_reference_and_human_raters_build_multi_hot_rows(multi_questions: QuestionSet) -> None:
    email = EmailFactory(
        id="gen1.0001", reference_answers={"topics": ["travel", "billing"], "category": "spam"}
    )
    reference = reference_rater([email], multi_questions, {"gen1": multi_questions})
    assert reference.answers["gen1.0001"]["topics"] == {
        "billing": 1.0,
        "meeting": 0.0,
        "travel": 1.0,
    }
    human = human_rater(
        {"gen1.0001": {"topics": ["meeting"]}, "gen1.0002": {"topics": "meeting"}},
        multi_questions,
    )
    assert human is not None
    only_meeting = {"billing": 0.0, "meeting": 1.0, "travel": 0.0}
    assert human.answers == {"gen1.0001": {"topics": only_meeting}}


def test_v1_string_reference_is_skipped_for_a_multi_question(multi_questions: QuestionSet) -> None:
    v1 = QuestionSet(
        name="v1",
        questions=(
            ChoiceQuestion(
                type="choice",
                id="topics",
                instructions="?",
                options={"billing": "b", "meeting": "m", "travel": "t"},
            ),
        ),
    )
    email = EmailFactory(id="gen1.0001", reference_answers={"topics": "billing"})
    reference = reference_rater([email], multi_questions, {"gen1": v1})
    assert reference.answers == {}
    assert "topics" in reference.warnings[0]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
`uv run pytest tests/test_emails.py tests/test_generation_prompt.py tests/test_config_files.py tests/test_compare_raters.py -q`
Expected: FAILs/ERRORs:
- a list reference is rejected by the `dict[str, str]` model;
- the schema is a plain string;
- `count_mismatches` treats `["meeting", "billing"] != "billing"` as a mismatch;
- `reference_rater` raises `TypeError: unhashable type: 'list'`.

- [ ] **Step 3: `src/jev_bench/emails.py`**

- Add `from jev_bench.questions import HardAnswer` below the pydantic import.
- Change the field to `reference_answers: dict[str, HardAnswer] = Field(default_factory=dict)`.
- In the docstring, make the `Email` line:

  ```
      Email: one generated email with its generation metadata and reference answers (one option id
          per question, or a list of option ids for a multi-label question).
  ```

(`questions.py` imports nothing from `emails.py`, so there is no cycle.)

- [ ] **Step 4: `src/jev_bench/generation/prompt.py`**

Replace the docstring's `GeneratorOutput`, `generation_schema`, `parse_generator_output` and
`count_mismatches` lines with:

```
    GeneratorOutput: validated generator response (email + one hard answer per question).
Functions:
    render_prompts: (system, user) prompts for one plan item.
    generation_schema: strict JSON schema for `{email, answers}` (an array of option ids for a
        multi-label question).
    parse_generator_output: JSON text -> GeneratorOutput (answers checked against the options, a
        multi-label answer must be a non-empty list of unique option ids; text around the
        outermost JSON object, e.g. a model's preamble, is ignored).
    count_mismatches: question-linked traits that the generator's own answers contradict (a
        multi-label answer agrees when it contains the trait value; traits absent from
        `requested` are skipped).
```

The questions import becomes:

```python
from jev_bench.questions import (
    AnyQuestion,
    HardAnswer,
    MultiQuestion,
    QuestionSet,
    hard_distribution,
    render_questions,
)
```

In `GeneratorOutput`, use `answers: dict[str, HardAnswer]`. In `generation_schema`, replace the
`answers_props` comprehension and add `_answer_schema`:

```python
    answers_props = {question.id: _answer_schema(question) for question in questions.questions}
    answers = strict_object(answers_props)
    return strict_object({"email": email, "answers": answers})


def _answer_schema(question: AnyQuestion) -> JsonSchema:
    option: JsonSchema = {"type": "string", "enum": list(question.option_ids)}
    return {"type": "array", "items": option} if isinstance(question, MultiQuestion) else option
```

In `parse_generator_output`, the `invalid` condition becomes:

```python
        if hard_distribution(question, output.answers.get(question.id)) is None
```

Replace `count_mismatches` and add `_agrees`:

```python
def count_mismatches(
    requested: Mapping[str, str],
    traits: Sequence[ResolvedTrait],
    answers: Mapping[str, HardAnswer],
) -> int:
    return sum(
        1
        for trait in traits
        if trait.question
        and trait.name in requested
        and not _agrees(answers.get(trait.question), requested[trait.name])
    )


def _agrees(answer: HardAnswer | None, value: str) -> bool:
    return value in answer if isinstance(answer, list) else answer == value
```

- [ ] **Step 5: `config/generation.toml`**

In `system_prompt`, replace `choosing exactly one option id per question, as a careful human reader would.`
with:

```
choosing exactly one option id per question, or every option id that applies (at least one) for a multi-label question, as a careful human reader would.
```

- [ ] **Step 6: `src/jev_bench/compare/raters.py`**

Append these docstring lines after the `human_rater` line:

```
Hard answers (reference and human) become one-hot rows, or multi-hot rows for a multi question;
an answer that is not valid for the question (for example a string for a multi question) is
skipped.
```

The questions import becomes:

```python
from jev_bench.questions import (
    AnyQuestion,
    Distribution,
    HardAnswer,
    QuestionSet,
    compatible,
    hard_distribution,
)
```

Change the signature to
`def human_rater(labels: Mapping[str, Mapping[str, HardAnswer]], questions: QuestionSet) -> Rater | None:`
(wrap it as ruff format decides) and replace `_hard` and `_reference_hard`:

```python
def _hard(labels: Mapping[str, HardAnswer], questions: QuestionSet) -> dict[str, Distribution]:
    return {
        question.id: hard
        for question in questions.questions
        if (hard := hard_distribution(question, labels.get(question.id))) is not None
    }


def _reference_hard(
    labels: Mapping[str, HardAnswer], questions: QuestionSet, snapshot: QuestionSet
) -> dict[str, Distribution]:
    return {
        question.id: hard
        for question in questions.questions
        if _supports(snapshot, question)
        and (hard := hard_distribution(question, labels.get(question.id))) is not None
    }
```

- [ ] **Step 7: `src/jev_bench/compare/rows.py`** (types only; behavior changes come in Task 8)

- Add `HardAnswer` to the `jev_bench.questions` import.
- In `EmailRow`: `reference: dict[str, HardAnswer]` and `human: dict[str, HardAnswer]`.
- `email_rows(... labels: Mapping[str, Mapping[str, HardAnswer]], ...)` and
  `_email_row(... human: Mapping[str, HardAnswer], ...)`.

- [ ] **Step 8: Run the tests to verify they pass**

Run: `uv run pytest -q`
Expected: the whole default suite passes (validated while planning: 701 tests after Tasks 1–6).

- [ ] **Step 9: Gates, then commit**

```bash
uv run ruff check --fix && uv run ruff format && uv run pyright
git add src/jev_bench/emails.py src/jev_bench/generation/prompt.py config/generation.toml \
  src/jev_bench/compare/raters.py src/jev_bench/compare/rows.py tests/test_emails.py \
  tests/test_generation_prompt.py tests/test_config_files.py tests/test_compare_raters.py
git commit -m "feat(generation): label-list reference answers for multi-label questions

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
