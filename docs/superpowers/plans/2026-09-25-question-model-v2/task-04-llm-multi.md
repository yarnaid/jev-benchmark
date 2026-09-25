### Task 4: Chat LLMs: multi-label schema, independent parsing, prompt bullet

**Files:**
- Modify: `src/jev_bench/classifiers/llm_schema.py`
- Modify: `src/jev_bench/classifiers/llm_parse.py`
- Modify: `config/benchmark.toml` (both system prompts)
- Test: `tests/test_classifiers_llm_schema.py`, `tests/test_classifiers_llm_parse.py`,
  `tests/test_config_files.py`

**Interfaces:**
- Consumes:
  - `MultiQuestion` and `independent_probabilities` (Task 1);
  - `missing_labels_note` (Task 3);
  - the `multi_questions` fixture.
- Produces:
  - `answers_schema()` / `all_in_one_schema()`: a multi question is a strict object with one `number` per
    option, and its description is `f"{instructions} {_MULTI_HINT}"`.
  - `parse_email_answers()`: a multi value that is an object keeps independent probabilities (clipped, never
    renormalized), and an all-zero vector is valid. Notes:
    - `"<qid>: <n> of <k> labels missing or unusable, set to 0"` for partially unusable values;
    - `"<qid>: missing or unusable probabilities"` for a non-object value or when every label is unusable.
  - Non-multi behavior and note texts are unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_classifiers_llm_schema.py`:

```python
def test_answers_schema_multi_describes_independent_probabilities(
    multi_questions: QuestionSet,
) -> None:
    topics = answers_schema(multi_questions)["properties"]["topics"]
    hint = (
        "Independent probability from 0 to 1 that each label applies; "
        "several labels can be likely at once; the values need not sum to 1."
    )
    assert topics == _options(f"Which topics? {hint}", "billing", "meeting", "travel")
```

Append to `tests/test_classifiers_llm_parse.py`:

```python
_TOPICS = {"billing": 0.9, "meeting": 0.6, "travel": 0.05}


@pytest.mark.parametrize(
    ("value", "expected", "notes"),
    [
        pytest.param(_TOPICS, _TOPICS, [], id="independent-not-renormalized"),
        pytest.param(
            {"billing": 0, "meeting": 0, "travel": 0},
            {"billing": 0.0, "meeting": 0.0, "travel": 0.0},
            [],
            id="all-zero-is-no-label",
        ),
        pytest.param(
            {"billing": 1.4, "meeting": -1, "travel": 0.2},
            {"billing": 1.0, "meeting": 0.0, "travel": 0.2},
            [],
            id="clipped",
        ),
        pytest.param(
            {"billing": 0.9, "meeting": "0.6"},
            {"billing": 0.9, "meeting": 0.0, "travel": 0.0},
            ["topics: 2 of 3 labels missing or unusable, set to 0"],
            id="missing-and-string",
        ),
        pytest.param({}, None, ["topics: missing or unusable probabilities"], id="all-missing"),
        pytest.param(
            [0.9, 0.6, 0.05],
            None,
            ["topics: missing or unusable probabilities"],
            id="not-an-object",
        ),
    ],
)
def test_parse_email_answers_multi(
    multi_questions: QuestionSet,
    value: object,
    expected: dict[str, float] | None,
    notes: list[str],
) -> None:
    parsed, found = parse_email_answers({**_VALID, "topics": value}, multi_questions)
    assert parsed.get("topics") == (None if expected is None else pytest.approx(expected))
    assert found == notes
```

In `tests/test_config_files.py`, `test_shipped_benchmark_config`, add after the `email_template` assertion:

```python
    multi_bullet = "for a multi-label question, give each label's probability independently"
    assert multi_bullet in config.llm.system_prompt
    assert multi_bullet in config.llm.system_prompt_all_in_one
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
`uv run pytest tests/test_classifiers_llm_schema.py tests/test_classifiers_llm_parse.py tests/test_config_files.py -q`
Expected:
- the schema test FAILS (the description lacks the hint);
- `all-zero-is-no-label` and `missing-and-string` FAIL (the choice path renormalizes, or rejects zero mass);
- the config test FAILS (no bullet yet).

- [ ] **Step 3: Implement `src/jev_bench/classifiers/llm_schema.py`**

- In the docstring, replace the `answers_schema` line with:

  ```
      answers_schema: per-email answer object (one number per option; noul questions give P(yes);
          multi questions give an independent probability per label).
  ```

- Import `MultiQuestion`:
  `from jev_bench.questions import AnyQuestion, MultiQuestion, NoulQuestion, QuestionSet`.
- Add below `__all__`:

  ```python
  _MULTI_HINT = (
      "Independent probability from 0 to 1 that each label applies; "
      "several labels can be likely at once; the values need not sum to 1."
  )
  ```

- Change the last line of `_question_schema` and add `_description`:

  ```python
      options: dict[str, JsonSchema] = {option: {"type": "number"} for option in question.option_ids}
      return strict_object(options, description=_description(question))


  def _description(question: AnyQuestion) -> str:
      if isinstance(question, MultiQuestion):
          return f"{question.instructions} {_MULTI_HINT}"
      return question.instructions
  ```

- [ ] **Step 4: Implement `src/jev_bench/classifiers/llm_parse.py`**

- In the docstring, replace the `parse_email_answers` line with:

  ```
      parse_email_answers: one email's JSON answers -> distributions and per-question notes (a multi
          question keeps independent, un-normalized probabilities; unusable labels become 0).
  ```

- Imports:

  ```python
  from jev_bench.classifiers.base import ProgressCallback, missing_labels_note
  from jev_bench.metrics.distributions import independent_probabilities, normalize, unit_probability
  from jev_bench.questions import AnyQuestion, Distribution, MultiQuestion, NoulQuestion, QuestionSet
  ```

- Below `REF_KEY = '"ref"'` add `_UNUSABLE = "missing or unusable probabilities"`.
- Replace the loop in `parse_email_answers`, and add the two helpers after the function (`_parse_value` stays
  as it is):

  ```python
      for question in questions.questions:
          distribution, note = _parse_question(question, raw.get(question.id))
          if note:
              notes.append(f"{question.id}: {note}")
          if distribution is not None:
              parsed[question.id] = distribution
      return parsed, notes


  def _parse_question(question: AnyQuestion, value: object) -> tuple[Distribution | None, str | None]:
      if isinstance(question, MultiQuestion):
          return _parse_labels(question, value)
      distribution = _parse_value(question, value)
      return distribution, (_UNUSABLE if distribution is None else None)


  def _parse_labels(question: MultiQuestion, value: object) -> tuple[Distribution | None, str | None]:
      if not isinstance(value, dict):
          return None, _UNUSABLE
      options = question.option_ids
      distribution, missing = independent_probabilities(value, options)
      if missing == len(options):
          return None, _UNUSABLE
      return distribution, missing_labels_note(missing, len(options))
  ```

- [ ] **Step 5: Add the prompt bullet to `config/benchmark.toml`**

In **both** `system_prompt` and `system_prompt_all_in_one`, insert this line between the "question with
options" bullet and the "yes/no question" bullet:

```
- for a multi-label question, give each label's probability independently; several labels can be likely at once and they need not sum to 1;
```

- [ ] **Step 6: Run the tests to verify they pass**

Run:
`uv run pytest tests/test_classifiers_llm_schema.py tests/test_classifiers_llm_parse.py tests/test_classifiers_llm.py tests/test_config_files.py -q`
Expected: all pass. The existing exact-note rows (`noul-nan`, `choice-zero-mass`, …) still expect
`"<qid>: missing or unusable probabilities"`.

- [ ] **Step 7: Gates, then commit**

```bash
uv run ruff check --fix && uv run ruff format && uv run pyright && uv run pytest -q
git add src/jev_bench/classifiers/llm_schema.py src/jev_bench/classifiers/llm_parse.py \
  config/benchmark.toml tests/test_classifiers_llm_schema.py tests/test_classifiers_llm_parse.py \
  tests/test_config_files.py
git commit -m "feat(llm): independent per-label probabilities for multi-label questions

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
