### Task 1: Question model: `multi` type, hard answers, threshold override, 0–100 score

**Files:**
- Modify: `src/jev_bench/questions.py` (whole file shown below)
- Modify: `src/jev_bench/metrics/distributions.py` (add two functions and update the docstring)
- Modify: `src/jev_bench/compare/report.py` (one line: widen `QuestionReport.type`; without it pyright fails,
  because `question.type` now includes `"multi"`)
- Modify: `tests/conftest.py` (add the `multi_questions` fixture)
- Test: `tests/test_questions.py`, `tests/test_distributions.py`

**Interfaces:**
- Consumes: nothing new.
- Produces (every later task relies on these exact names):
  - `questions.MultiQuestion(type="multi", id, instructions, options, threshold: float = 0.8)`, with
    `0 < threshold <= 1`.
  - `questions.AnyQuestion = ChoiceQuestion | ScoreQuestion | NoulQuestion | MultiQuestion`.
  - `questions.HardAnswer = str | list[str]` (a PEP 695 `type` alias).
  - `questions.multi_hot(question: AnyQuestion, option_ids: Iterable[str]) -> Distribution`: unknown ids raise
    `ValueError`; duplicates are ignored.
  - `questions.hard_distribution(question: AnyQuestion, answer: object) -> Distribution | None`: `one_hot` for
    a valid `str` on a non-multi question, `multi_hot` for a non-empty list of unique valid `str` on a multi
    question, else `None`.
  - `questions.with_threshold(questions: QuestionSet, threshold: float | None) -> QuestionSet`: `None` returns
    the same object; otherwise every `MultiQuestion.threshold` is replaced. It raises `ValueError` unless
    `0 < threshold <= 1` (NaN included).
  - `metrics.distributions.score_0_100(matrix: FloatArray) -> FloatArray`: row-wise
    `expected_level / (k − 1) × 100`.
  - `metrics.distributions.independent_probabilities(values: Mapping[str, object], options: Sequence[str]) ->
    tuple[dict[str, float], int]`: each option's `unit_probability`, with an unusable/missing value → `0.0`.
    It returns (vector, number of unusable options) and never renormalizes.
  - The pytest fixture `multi_questions: QuestionSet` (name `"mini-multi"`): the three `questions` fixture
    questions, plus `topics = MultiQuestion(id="topics", instructions="Which topics?",
    options={"billing": "About money", "meeting": "About a meeting", "travel": "About a trip"})` with the
    default threshold 0.8.

- [ ] **Step 1: Add the `multi_questions` fixture to `tests/conftest.py`**

Add `MultiQuestion` to the `jev_bench.questions` import and add this fixture right after `questions`:

```python
@pytest.fixture
def multi_questions(questions: QuestionSet) -> QuestionSet:
    topics = MultiQuestion(
        type="multi",
        id="topics",
        instructions="Which topics?",
        options={"billing": "About money", "meeting": "About a meeting", "travel": "About a trip"},
    )
    return QuestionSet(name="mini-multi", questions=(*questions.questions, topics))
```

In the module docstring's `Fixtures:` list, add the line
`multi_questions: the questions fixture plus a multi-label "topics" question (threshold 0.8).`

- [ ] **Step 2: Write the failing question tests** (append to `tests/test_questions.py`)

Extend the import to
`from jev_bench.questions import (ChoiceQuestion, MultiQuestion, NoulQuestion, QuestionSet, ScoreQuestion,
compatible, hard_distribution, load_question_set, multi_hot, one_hot, render_questions, with_threshold)` and
add `import math`.

Add three rows to the existing `test_invalid_question_sets_are_rejected` table, just before the
`duplicate-ids` row:

```python
        pytest.param(
            _doc(id="q", type="multi", instructions="?", options={"a": "A", "b": "B"}, threshold=0),
            id="multi-threshold-zero",
        ),
        pytest.param(
            _doc(id="q", type="multi", instructions="?", options={"a": "A", "b": "B"}, threshold=1.5),
            id="multi-threshold-above-one",
        ),
        pytest.param(
            _doc(
                id="q", type="multi", instructions="?", options={"a": "A", "b": "B"}, threshold=math.nan
            ),
            id="multi-threshold-nan",
        ),
```

Add a row to the existing `test_compatible` table:

```python
        pytest.param(
            MultiQuestion(type="multi", id="q", instructions="x", options=_A_B),
            ChoiceQuestion(type="choice", id="q", instructions="x", options=_A_B),
            False,
            id="multi-vs-choice",
        ),
```

Then widen that test's parameter annotations to `left: ChoiceQuestion | ScoreQuestion | MultiQuestion` and
`right: ChoiceQuestion | ScoreQuestion | MultiQuestion`.

Append these new tests:

```python
_MULTI_TOML = """
name = "t"

[[questions]]
id = "topics"
type = "multi"
instructions = "Which topics?"
threshold = 0.7

[questions.options]
billing = "About money"
meeting = "About a meeting"
"""


def test_load_question_set_reads_multi_threshold(tmp_path: Path) -> None:
    path = tmp_path / "questions.toml"
    path.write_text(_MULTI_TOML, encoding="utf-8")
    topics = load_question_set(path).get("topics")
    assert isinstance(topics, MultiQuestion)
    assert topics.threshold == 0.7


def test_multi_threshold_defaults_to_80_percent() -> None:
    doc = _doc(id="q", type="multi", instructions="?", options={"a": "A", "b": "B"})
    question = QuestionSet.model_validate(doc).get("q")
    assert isinstance(question, MultiQuestion)
    assert question.threshold == 0.8


def test_render_questions_hints_multi_label(multi_questions: QuestionSet) -> None:
    lines = render_questions(multi_questions).splitlines()
    hint = "multi-label: judge each option independently; several can apply"
    assert f"- topics ({hint}): Which topics?" in lines
    assert "    - meeting: About a meeting" in lines


_NONE = {"billing": 0.0, "meeting": 0.0, "travel": 0.0}


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        pytest.param(["meeting"], {**_NONE, "meeting": 1.0}, id="one"),
        pytest.param(["travel", "billing"], {**_NONE, "billing": 1.0, "travel": 1.0}, id="any-order"),
        pytest.param(["billing", "billing"], {**_NONE, "billing": 1.0}, id="duplicates-ignored"),
        pytest.param([], _NONE, id="none"),
    ],
)
def test_multi_hot(
    multi_questions: QuestionSet, labels: list[str], expected: dict[str, float]
) -> None:
    assert multi_hot(multi_questions.get("topics"), labels) == expected


def test_multi_hot_rejects_unknown_labels(multi_questions: QuestionSet) -> None:
    with pytest.raises(ValueError, match="not options"):
        multi_hot(multi_questions.get("topics"), ["billing", "phishing"])


@pytest.mark.parametrize(
    ("question_id", "answer", "expected"),
    [
        pytest.param("category", "work", {"spam": 0.0, "personal": 0.0, "work": 1.0}, id="choice-id"),
        pytest.param("category", "phishing", None, id="choice-unknown-id"),
        pytest.param("category", ["work"], None, id="choice-given-a-list"),
        pytest.param("needs_reply", "yes", {"yes": 1.0, "no": 0.0}, id="noul-id"),
        pytest.param(
            "topics", ["travel", "billing"], {**_NONE, "billing": 1.0, "travel": 1.0}, id="multi-list"
        ),
        pytest.param("topics", [], None, id="multi-empty"),
        pytest.param("topics", ["billing", "billing"], None, id="multi-duplicate"),
        pytest.param("topics", ["billing", "phishing"], None, id="multi-unknown"),
        pytest.param("topics", ["billing", 3], None, id="multi-non-string"),
        pytest.param("topics", "billing", None, id="multi-bare-string"),
        pytest.param("topics", None, None, id="multi-missing"),
    ],
)
def test_hard_distribution(
    multi_questions: QuestionSet,
    question_id: str,
    answer: object,
    expected: dict[str, float] | None,
) -> None:
    assert hard_distribution(multi_questions.get(question_id), answer) == expected


def test_with_threshold_replaces_only_multi_thresholds(multi_questions: QuestionSet) -> None:
    changed = with_threshold(multi_questions, 0.55)
    topics = changed.get("topics")
    assert isinstance(topics, MultiQuestion)
    assert topics.threshold == 0.55
    assert changed.get("category") == multi_questions.get("category")
    assert compatible(topics, multi_questions.get("topics"))
    assert with_threshold(multi_questions, None) is multi_questions


@pytest.mark.parametrize(
    "threshold",
    [
        pytest.param(0.0, id="zero"),
        pytest.param(1.01, id="above-one"),
        pytest.param(-0.5, id="negative"),
        pytest.param(math.nan, id="nan"),
    ],
)
def test_with_threshold_rejects_out_of_range(
    multi_questions: QuestionSet, threshold: float
) -> None:
    with pytest.raises(ValueError, match="threshold"):
        with_threshold(multi_questions, threshold)
```

- [ ] **Step 3: Write the failing distribution tests** (append to `tests/test_distributions.py`)

Add `independent_probabilities` and `score_0_100` to the import from `jev_bench.metrics.distributions`:

```python
@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        pytest.param([[1.0, 0.0, 0.0]], [0.0], id="lowest"),
        pytest.param([[0.0, 0.0, 1.0]], [100.0], id="highest"),
        pytest.param([[0.5, 0.0, 0.5], [0.0, 1.0, 0.0]], [50.0, 50.0], id="middle"),
        pytest.param([[0.0, 1.0]], [100.0], id="two-levels"),
        pytest.param([[0.25, 0.25, 0.25, 0.25]], [50.0], id="uniform-four"),
    ],
)
def test_score_0_100(rows: list[list[float]], expected: list[float]) -> None:
    assert score_0_100(np.array(rows)).tolist() == pytest.approx(expected)


@pytest.mark.parametrize(
    ("values", "expected", "missing"),
    [
        pytest.param({"a": 0.9, "b": 0.8}, {"a": 0.9, "b": 0.8}, 0, id="not-renormalized"),
        pytest.param({"a": 0.0, "b": 0.0}, {"a": 0.0, "b": 0.0}, 0, id="all-zero-is-valid"),
        pytest.param({"a": 1.7, "b": -0.2}, {"a": 1.0, "b": 0.0}, 0, id="clipped"),
        pytest.param({"a": 0.4}, {"a": 0.4, "b": 0.0}, 1, id="missing-option"),
        pytest.param({"a": "0.4", "b": math.nan}, {"a": 0.0, "b": 0.0}, 2, id="unusable-values"),
        pytest.param({"a": True, "b": 1}, {"a": 0.0, "b": 1.0}, 1, id="bool-is-unusable"),
        pytest.param({}, {"a": 0.0, "b": 0.0}, 2, id="empty"),
    ],
)
def test_independent_probabilities(
    values: dict[str, object], expected: dict[str, float], missing: int
) -> None:
    assert independent_probabilities(values, _OPTIONS) == (expected, missing)
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `uv run pytest tests/test_questions.py tests/test_distributions.py -q`
Expected: collection ERROR, `ImportError: cannot import name 'MultiQuestion'` (and later `'score_0_100'`).

- [ ] **Step 5: Implement `src/jev_bench/questions.py`** (replace the whole file)

```python
"""Typed question set: the single source of truth for what every column is asked.

Types:
    Distribution: probabilities keyed by option id. For choice, score and noul questions it
        sums to 1; for a multi question each value is an independent probability that the
        label applies.
    HardAnswer: a hard label: one option id, or a list of option ids for a multi question.
    AnyQuestion: union of the four question classes; Question: its discriminated form.
Classes:
    ChoiceQuestion, ScoreQuestion, NoulQuestion: the three Jev primitives.
    MultiQuestion: multi-label question; a label is applied when its probability >= threshold.
    QuestionSet: named, ordered, id-unique collection of questions.
Functions:
    load_question_set: parse and validate a question-set TOML file.
    render_questions: prompt-ready description of every question and option.
    one_hot: hard label -> distribution.
    multi_hot: labels -> independent 0/1 vector over the options (duplicates ignored).
    hard_distribution: a hard answer -> its distribution, or None when it is not valid for the
        question (a multi question needs a non-empty list of unique option ids; the others one
        option id).
    with_threshold: the same set with every multi question's threshold replaced (None keeps it).
    compatible: whether two questions share type and option ids (in order).
"""

import re
import tomllib
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "AnyQuestion",
    "ChoiceQuestion",
    "Distribution",
    "HardAnswer",
    "MultiQuestion",
    "NoulQuestion",
    "Question",
    "QuestionSet",
    "ScoreQuestion",
    "compatible",
    "hard_distribution",
    "load_question_set",
    "multi_hot",
    "one_hot",
    "render_questions",
    "with_threshold",
]

type Distribution = dict[str, float]
type HardAnswer = str | list[str]

_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]*")
_KIND_HINTS: dict[str, str] = {
    "choice": "choose exactly one option",
    "score": "ordered scale, lowest level first",
    "noul": "yes/no",
    "multi": "multi-label: judge each option independently; several can apply",
}


class _Question(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    instructions: str = Field(min_length=1)
    options: dict[str, str] = Field(min_length=2, max_length=255)

    @field_validator("options")
    @classmethod
    def _valid_options(cls, options: dict[str, str]) -> dict[str, str]:
        bad = [
            key
            for key, text in options.items()
            if not _IDENTIFIER.fullmatch(key) or not text.strip()
        ]
        if bad:
            raise ValueError(f"option ids must be snake_case with a non-empty description: {bad}")
        return options

    @property
    def option_ids(self) -> tuple[str, ...]:
        return tuple(self.options)


class ChoiceQuestion(_Question):
    type: Literal["choice"]


class ScoreQuestion(_Question):
    type: Literal["score"]


class NoulQuestion(_Question):
    type: Literal["noul"]

    @model_validator(mode="after")
    def _yes_then_no(self) -> NoulQuestion:
        if self.option_ids != ("yes", "no"):
            raise ValueError(f"noul question {self.id!r} must define options 'yes' then 'no'")
        return self


class MultiQuestion(_Question):
    type: Literal["multi"]
    threshold: float = Field(default=0.8, gt=0.0, le=1.0)


AnyQuestion = ChoiceQuestion | ScoreQuestion | NoulQuestion | MultiQuestion
Question = Annotated[AnyQuestion, Field(discriminator="type")]


class QuestionSet(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    questions: tuple[Question, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> QuestionSet:
        if len(set(self.ids)) != len(self.ids):
            raise ValueError(f"duplicate question ids in {self.name!r}")
        return self

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(question.id for question in self.questions)

    def get(self, question_id: str) -> AnyQuestion:
        for question in self.questions:
            if question.id == question_id:
                return question
        raise KeyError(question_id)


def load_question_set(path: Path) -> QuestionSet:
    return QuestionSet.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))


def render_questions(questions: QuestionSet) -> str:
    return "\n".join(_render_question(question) for question in questions.questions)


def _render_question(question: AnyQuestion) -> str:
    header = f"- {question.id} ({_KIND_HINTS[question.type]}): {question.instructions}"
    lines = [f"    - {key}: {text}" for key, text in question.options.items()]
    return "\n".join([header, *lines])


def one_hot(question: AnyQuestion, option_id: str) -> Distribution:
    if option_id not in question.options:
        raise ValueError(f"{option_id!r} is not an option of {question.id!r}")
    return {key: float(key == option_id) for key in question.option_ids}


def multi_hot(question: AnyQuestion, option_ids: Iterable[str]) -> Distribution:
    chosen = set(option_ids)
    unknown = sorted(chosen - set(question.options))
    if unknown:
        raise ValueError(f"{unknown} are not options of {question.id!r}")
    return {key: float(key in chosen) for key in question.option_ids}


def hard_distribution(question: AnyQuestion, answer: object) -> Distribution | None:
    if isinstance(question, MultiQuestion):
        return _labels_distribution(question, answer)
    if isinstance(answer, str) and answer in question.options:
        return one_hot(question, answer)
    return None


def _labels_distribution(question: MultiQuestion, answer: object) -> Distribution | None:
    if not isinstance(answer, list) or not answer:
        return None
    labels = [item for item in answer if isinstance(item, str) and item in question.options]
    if len(labels) != len(answer) or len(set(labels)) != len(labels):
        return None
    return multi_hot(question, labels)


def with_threshold(questions: QuestionSet, threshold: float | None) -> QuestionSet:
    if threshold is None:
        return questions
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    replaced = tuple(_with_threshold(question, threshold) for question in questions.questions)
    return questions.model_copy(update={"questions": replaced})


def _with_threshold(question: AnyQuestion, threshold: float) -> AnyQuestion:
    if isinstance(question, MultiQuestion):
        return question.model_copy(update={"threshold": threshold})
    return question


def compatible(left: AnyQuestion, right: AnyQuestion) -> bool:
    return left.type == right.type and left.option_ids == right.option_ids
```

- [ ] **Step 6: Implement the distribution helpers** in `src/jev_bench/metrics/distributions.py`

Add `"independent_probabilities"` and `"score_0_100"` to `__all__` (keep it sorted), and append these
functions at the end of the file:

```python
def independent_probabilities(
    values: Mapping[str, object], options: Sequence[str]
) -> tuple[dict[str, float], int]:
    probabilities = {option: unit_probability(values.get(option)) for option in options}
    missing = sum(value is None for value in probabilities.values())
    return {option: value or 0.0 for option, value in probabilities.items()}, missing


def score_0_100(matrix: FloatArray) -> FloatArray:
    return expected_level(matrix) * (100.0 / (matrix.shape[1] - 1))
```

In the module docstring's `Functions:` list, add:

```
    independent_probabilities: per-option unit probabilities (unusable -> 0.0) plus the number of
        unusable options; never renormalized (multi-label answers).
    score_0_100: row-wise expected level scaled to 0-100 (k >= 2 options).
```

- [ ] **Step 6b: Widen the report's question type** in `src/jev_bench/compare/report.py`

In `class QuestionReport`, replace `type: Literal["choice", "score", "noul"]` with:

```python
    type: Literal["choice", "score", "noul", "multi"]
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/test_questions.py tests/test_distributions.py -q`
Expected: all pass.

- [ ] **Step 8: Gates and the full suite**

Run:
`uv run ruff check --fix && uv run ruff format && uv run pyright && uv run pytest -q`
Expected: 0 pyright errors, and the full suite passes (validated while planning: 607 → 647 tests). Nothing else
consumes `MultiQuestion` yet; adding it to `AnyQuestion` only widens `isinstance` fall-through branches that
the existing tests do not hit.

- [ ] **Step 9: Commit**

```bash
git add src/jev_bench/questions.py src/jev_bench/metrics/distributions.py src/jev_bench/compare/report.py \
  tests/conftest.py tests/test_questions.py tests/test_distributions.py
git commit -m "feat(questions): multi-label question type, hard answers, threshold override, 0–100 score

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
