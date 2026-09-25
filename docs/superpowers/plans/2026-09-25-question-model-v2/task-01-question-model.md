### Task 1: Question model: `multi` type, hard answers, threshold override, compatibility, 0–100 score

**Files:**
- Modify: `src/jev_bench/questions.py`
- Modify: `src/jev_bench/metrics/distributions.py` (`score_0_100`)
- Modify: `src/jev_bench/compare/report.py`: one line, widening `QuestionReport.type`. Without it pyright
  fails, because `question.type` now includes `"multi"`.
- Modify: `tests/conftest.py` (the `multi_questions` fixture)
- Test: `tests/test_questions.py`, `tests/test_distributions.py`

**Interfaces:**
- Consumes: nothing new.
- Produces (every later task relies on these exact names):
  - `questions.MultiQuestion(type="multi", id, instructions, options, threshold: float = 0.8)`, with
    `0 < threshold <= 1`.
  - `questions.AnyQuestion = ChoiceQuestion | ScoreQuestion | NoulQuestion | MultiQuestion`.
  - `questions.HardAnswer = str | list[str]` (a PEP 695 `type` alias).
  - `questions.hard_distribution(question, answer: object) -> Distribution | None`:
    - `one_hot` for a valid option id (every type, including multi, where an id is a one-label set);
    - for a multi question, a non-empty list of unique valid ids gives the **uniform** distribution over
      those labels (`["billing", "travel"]` → billing 0.5, travel 0.5);
    - anything else → `None`.
  - `questions.with_threshold(questions, threshold: float | None) -> QuestionSet`: `None` returns the same
    object; otherwise every `MultiQuestion.threshold` is replaced. It raises `ValueError` unless
    `0 < threshold <= 1` (NaN included).
  - `questions.compatible(left, right)`: equal option ids (in order) **and** the same shape. Choice, score and
    multi answers are all distributions; a noul only matches a noul. So v1 `choice` → v2 `multi`/`score` with
    the same ids stays comparable.
  - `metrics.distributions.score_0_100(matrix) -> FloatArray`: row-wise `expected_level / (k − 1) × 100`.
  - The pytest fixture `multi_questions` (name `"mini-multi"`): the three `questions` fixture questions plus
    `topics = MultiQuestion(id="topics", instructions="Which topics?", options={"billing": "About money",
    "meeting": "About a meeting", "travel": "About a trip"})`, with the default threshold 0.8.

**Behavior note:** the multi reading rule (a label applies iff `p >= threshold × max(p)`) is implemented in
Task 2 (`metrics.multilabel.relative_labels`). Nothing here depends on it.

- [ ] **Step 1: Add the `multi_questions` fixture and write the failing tests**

Apply to `tests/conftest.py`:

```diff
--- a/tests/conftest.py
+++ b/tests/conftest.py
@@ -4,7 +4,8 @@
 inside a timed test.
 
 Fixtures:
-    questions: a three-question set covering every question type.
+    questions: a three-question set covering the choice, score and noul types.
+    multi_questions: the questions fixture plus a multi-label "topics" question (threshold 0.8).
     make_client: async OpenRouterClient factory.
     make_services: async Services factory on tmp dirs.
     make_app: TestClient factory with the lifespan entered.
@@ -25,7 +26,13 @@
 from tests.factories import AppFactory, ClientFactory, ServicesFactory, mini_settings
 
 from jev_bench.openrouter import OpenRouterClient
-from jev_bench.questions import ChoiceQuestion, NoulQuestion, QuestionSet, ScoreQuestion
+from jev_bench.questions import (
+    ChoiceQuestion,
+    MultiQuestion,
+    NoulQuestion,
+    QuestionSet,
+    ScoreQuestion,
+)
 from jev_bench.services import Services
 
 if TYPE_CHECKING:
@@ -79,6 +86,17 @@
             ),
         ),
     )
+
+
+@pytest.fixture
+def multi_questions(questions: QuestionSet) -> QuestionSet:
+    topics = MultiQuestion(
+        type="multi",
+        id="topics",
+        instructions="Which topics?",
+        options={"billing": "About money", "meeting": "About a meeting", "travel": "About a trip"},
+    )
+    return QuestionSet(name="mini-multi", questions=(*questions.questions, topics))
 
 
 @pytest.fixture
```

Apply to `tests/test_questions.py`:

```diff
--- a/tests/test_questions.py
+++ b/tests/test_questions.py
@@ -1,5 +1,6 @@
 """Tests for jev_bench.questions."""
 
+import math
 from pathlib import Path
 from typing import Any
 
@@ -7,14 +8,18 @@
 from pydantic import ValidationError
 
 from jev_bench.questions import (
+    AnyQuestion,
     ChoiceQuestion,
+    MultiQuestion,
     NoulQuestion,
     QuestionSet,
     ScoreQuestion,
     compatible,
+    hard_distribution,
     load_question_set,
     one_hot,
     render_questions,
+    with_threshold,
 )
 
 _VALID_TOML = """
@@ -38,6 +43,9 @@
 yes = "Reply expected"
 no = "No reply"
 """
+
+
+_TWO = {"a": "A", "b": "B"}
 
 
 def _doc(**question: Any) -> dict[str, Any]:
@@ -118,6 +126,18 @@
                 options={"a": "A", "b": "B"},
             ),
             id="empty-instructions",
+        ),
+        pytest.param(
+            _doc(id="q", type="multi", instructions="?", options=_TWO, threshold=0),
+            id="multi-threshold-zero",
+        ),
+        pytest.param(
+            _doc(id="q", type="multi", instructions="?", options=_TWO, threshold=1.5),
+            id="multi-threshold-above-one",
+        ),
+        pytest.param(
+            _doc(id="q", type="multi", instructions="?", options=_TWO, threshold=math.nan),
+            id="multi-threshold-nan",
         ),
         pytest.param(
             {"name": "t", "questions": [_choice(), _choice()]},
@@ -221,14 +241,139 @@
         pytest.param(
             ChoiceQuestion(type="choice", id="q", instructions="x", options=_A_B),
             ScoreQuestion(type="score", id="q", instructions="x", options=_A_B),
+            True,
+            id="choice-and-score-share-the-distribution-shape",
+        ),
+        pytest.param(
+            ChoiceQuestion(type="choice", id="q", instructions="x", options=_A_B),
+            MultiQuestion(type="multi", id="q", instructions="x", options=_A_B),
+            True,
+            id="choice-and-multi-share-the-distribution-shape",
+        ),
+        pytest.param(
+            MultiQuestion(type="multi", id="q", instructions="x", options=_A_B),
+            MultiQuestion(type="multi", id="q", instructions="x", options=_A_B, threshold=0.5),
+            True,
+            id="threshold-is-not-part-of-compatibility",
+        ),
+        pytest.param(
+            NoulQuestion(type="noul", id="q", instructions="x", options={"yes": "Y", "no": "N"}),
+            ChoiceQuestion(
+                type="choice", id="q", instructions="x", options={"yes": "Y", "no": "N"}
+            ),
             False,
-            id="different-type",
+            id="noul-only-matches-noul",
         ),
     ],
 )
 def test_compatible(
-    left: ChoiceQuestion | ScoreQuestion,
-    right: ChoiceQuestion | ScoreQuestion,
+    left: AnyQuestion,
+    right: AnyQuestion,
     expected: bool,
 ) -> None:
     assert compatible(left, right) is expected
+
+
+_MULTI_TOML = """
+name = "t"
+
+[[questions]]
+id = "topics"
+type = "multi"
+instructions = "Which topics?"
+threshold = 0.7
+
+[questions.options]
+billing = "About money"
+meeting = "About a meeting"
+"""
+
+
+def test_load_question_set_reads_multi_threshold(tmp_path: Path) -> None:
+    path = tmp_path / "questions.toml"
+    path.write_text(_MULTI_TOML, encoding="utf-8")
+    topics = load_question_set(path).get("topics")
+    assert isinstance(topics, MultiQuestion)
+    assert topics.threshold == 0.7
+
+
+def test_multi_threshold_defaults_to_80_percent() -> None:
+    question = QuestionSet.model_validate(
+        _doc(id="q", type="multi", instructions="?", options=_TWO)
+    ).get("q")
+    assert isinstance(question, MultiQuestion)
+    assert question.threshold == 0.8
+
+
+def test_render_questions_hints_multi_label(multi_questions: QuestionSet) -> None:
+    lines = render_questions(multi_questions).splitlines()
+    assert "- topics (multi-label: one or more options can apply): Which topics?" in lines
+    assert "    - meeting: About a meeting" in lines
+
+
+_NONE = {"billing": 0.0, "meeting": 0.0, "travel": 0.0}
+
+
+@pytest.mark.parametrize(
+    ("question_id", "answer", "expected"),
+    [
+        pytest.param(
+            "category", "work", {"spam": 0.0, "personal": 0.0, "work": 1.0}, id="choice-id"
+        ),
+        pytest.param("category", "phishing", None, id="choice-unknown-id"),
+        pytest.param("category", ["work"], None, id="choice-given-a-list"),
+        pytest.param("needs_reply", "yes", {"yes": 1.0, "no": 0.0}, id="noul-id"),
+        pytest.param(
+            "topics",
+            ["travel", "billing"],
+            {**_NONE, "billing": 0.5, "travel": 0.5},
+            id="multi-list",
+        ),
+        pytest.param("topics", ["meeting"], {**_NONE, "meeting": 1.0}, id="multi-one-label"),
+        pytest.param("topics", "meeting", {**_NONE, "meeting": 1.0}, id="multi-single-id"),
+        pytest.param("topics", [], None, id="multi-empty"),
+        pytest.param("topics", ["billing", "billing"], None, id="multi-duplicate"),
+        pytest.param("topics", ["billing", "phishing"], None, id="multi-unknown"),
+        pytest.param("topics", ["billing", 3], None, id="multi-non-string"),
+        pytest.param("topics", "phishing", None, id="multi-unknown-single-id"),
+        pytest.param("topics", None, None, id="multi-missing"),
+    ],
+)
+def test_hard_distribution(
+    multi_questions: QuestionSet,
+    question_id: str,
+    answer: object,
+    expected: dict[str, float] | None,
+) -> None:
+    assert hard_distribution(multi_questions.get(question_id), answer) == expected
+
+
+def test_three_labels_share_the_mass_evenly(multi_questions: QuestionSet) -> None:
+    labels = ["billing", "meeting", "travel"]
+    distribution = hard_distribution(multi_questions.get("topics"), labels)
+    assert distribution == pytest.approx(dict.fromkeys(labels, 1 / 3))
+
+
+def test_with_threshold_replaces_only_multi_thresholds(multi_questions: QuestionSet) -> None:
+    changed = with_threshold(multi_questions, 0.55)
+    topics = changed.get("topics")
+    assert isinstance(topics, MultiQuestion)
+    assert topics.threshold == 0.55
+    assert changed.get("category") == multi_questions.get("category")
+    assert with_threshold(multi_questions, None) is multi_questions
+
+
+@pytest.mark.parametrize(
+    "threshold",
+    [
+        pytest.param(0.0, id="zero"),
+        pytest.param(1.01, id="above-one"),
+        pytest.param(-0.5, id="negative"),
+        pytest.param(math.nan, id="nan"),
+    ],
+)
+def test_with_threshold_rejects_out_of_range(
+    multi_questions: QuestionSet, threshold: float
+) -> None:
+    with pytest.raises(ValueError, match="threshold"):
+        with_threshold(multi_questions, threshold)
```

Apply to `tests/test_distributions.py`:

```diff
--- a/tests/test_distributions.py
+++ b/tests/test_distributions.py
@@ -14,6 +14,7 @@
     expected_level,
     js_divergence,
     normalize,
+    score_0_100,
     softmax,
     to_matrix,
     unit_probability,
@@ -125,3 +126,17 @@
 
 def test_expected_level() -> None:
     assert expected_level(np.array([[0.0, 0.0, 1.0], [0.5, 0.5, 0.0]])).tolist() == [2.0, 0.5]
+
+
+@pytest.mark.parametrize(
+    ("rows", "expected"),
+    [
+        pytest.param([[1.0, 0.0, 0.0]], [0.0], id="lowest"),
+        pytest.param([[0.0, 0.0, 1.0]], [100.0], id="highest"),
+        pytest.param([[0.5, 0.0, 0.5], [0.0, 1.0, 0.0]], [50.0, 50.0], id="middle"),
+        pytest.param([[0.0, 1.0]], [100.0], id="two-levels"),
+        pytest.param([[0.25, 0.25, 0.25, 0.25]], [50.0], id="uniform-four"),
+    ],
+)
+def test_score_0_100(rows: list[list[float]], expected: list[float]) -> None:
+    assert score_0_100(np.array(rows)).tolist() == pytest.approx(expected)
```

The `compatible` table changes one expectation on purpose: `choice` vs `score` with the same ids is now
`True` (the row id is changed to `choice-and-score-share-the-distribution-shape`). Three rows are added:
choice vs multi, a different threshold, and noul vs choice.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_questions.py tests/test_distributions.py -q`
Expected: collection ERROR, `ImportError: cannot import name 'MultiQuestion'` (and `'score_0_100'`).

- [ ] **Step 3: Implement**

Apply to `src/jev_bench/questions.py`:

```diff
--- a/src/jev_bench/questions.py
+++ b/src/jev_bench/questions.py
@@ -1,20 +1,31 @@
 """Typed question set: the single source of truth for what every column is asked.
 
+Every answer is a distribution over the question's options (noul: `{yes, no}`). For a multi question
+the applied labels are the options with p >= threshold * max(p), so the top option always applies.
+
 Types:
-    Distribution: probabilities keyed by option id.
-    AnyQuestion: union of the three question classes; Question: its discriminated form.
+    Distribution: probabilities keyed by option id (summing to 1).
+    HardAnswer: a hard label: one option id, or a list of option ids for a multi question.
+    AnyQuestion: union of the four question classes; Question: its discriminated form.
 Classes:
     ChoiceQuestion, ScoreQuestion, NoulQuestion: the three Jev primitives.
+    MultiQuestion: multi-label question; its distribution is read with a relative threshold.
     QuestionSet: named, ordered, id-unique collection of questions.
 Functions:
     load_question_set: parse and validate a question-set TOML file.
     render_questions: prompt-ready description of every question and option.
     one_hot: hard label -> distribution.
-    compatible: whether two questions share type and option ids (in order).
+    hard_distribution: a hard answer -> its distribution, or None when it is not valid for the
+        question (one-hot; for a multi question, uniform over one option id or a non-empty list of
+        unique option ids).
+    with_threshold: the same set with every multi question's threshold replaced (None keeps it).
+    compatible: whether two questions' answers share shape and option ids (in order); choice, score
+        and multi answers are all distributions, a noul only matches a noul.
 """
 
 import re
 import tomllib
+from collections.abc import Sequence
 from pathlib import Path
 from typing import Annotated, Literal
 
@@ -24,23 +35,29 @@
     "AnyQuestion",
     "ChoiceQuestion",
     "Distribution",
+    "HardAnswer",
+    "MultiQuestion",
     "NoulQuestion",
     "Question",
     "QuestionSet",
     "ScoreQuestion",
     "compatible",
+    "hard_distribution",
     "load_question_set",
     "one_hot",
     "render_questions",
+    "with_threshold",
 ]
 
 type Distribution = dict[str, float]
+type HardAnswer = str | list[str]
 
 _IDENTIFIER = re.compile(r"[a-z][a-z0-9_]*")
 _KIND_HINTS: dict[str, str] = {
     "choice": "choose exactly one option",
     "score": "ordered scale, lowest level first",
     "noul": "yes/no",
+    "multi": "multi-label: one or more options can apply",
 }
 
 
@@ -86,7 +103,12 @@
         return self
 
 
-AnyQuestion = ChoiceQuestion | ScoreQuestion | NoulQuestion
+class MultiQuestion(_Question):
+    type: Literal["multi"]
+    threshold: float = Field(default=0.8, gt=0.0, le=1.0)
+
+
+AnyQuestion = ChoiceQuestion | ScoreQuestion | NoulQuestion | MultiQuestion
 Question = Annotated[AnyQuestion, Field(discriminator="type")]
 
 
@@ -133,5 +155,40 @@
     return {key: float(key == option_id) for key in question.option_ids}
 
 
+def hard_distribution(question: AnyQuestion, answer: object) -> Distribution | None:
+    if isinstance(question, MultiQuestion) and isinstance(answer, list):
+        return _uniform(question, answer)
+    if isinstance(answer, str) and answer in question.options:
+        return one_hot(question, answer)
+    return None
+
+
+def _uniform(question: MultiQuestion, answer: Sequence[object]) -> Distribution | None:
+    labels = [item for item in answer if isinstance(item, str) and item in question.options]
+    if not labels or len(labels) != len(answer) or len(set(labels)) != len(labels):
+        return None
+    share = 1.0 / len(labels)
+    return {key: share if key in labels else 0.0 for key in question.option_ids}
+
+
+def with_threshold(questions: QuestionSet, threshold: float | None) -> QuestionSet:
+    if threshold is None:
+        return questions
+    if not 0.0 < threshold <= 1.0:
+        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
+    replaced = tuple(_with_threshold(question, threshold) for question in questions.questions)
+    return questions.model_copy(update={"questions": replaced})
+
+
+def _with_threshold(question: AnyQuestion, threshold: float) -> AnyQuestion:
+    if isinstance(question, MultiQuestion):
+        return question.model_copy(update={"threshold": threshold})
+    return question
+
+
 def compatible(left: AnyQuestion, right: AnyQuestion) -> bool:
-    return left.type == right.type and left.option_ids == right.option_ids
+    return _shape(left) == _shape(right) and left.option_ids == right.option_ids
+
+
+def _shape(question: AnyQuestion) -> str:
+    return "yes/no" if isinstance(question, NoulQuestion) else "distribution"
```

Apply to `src/jev_bench/metrics/distributions.py`:

```diff
--- a/src/jev_bench/metrics/distributions.py
+++ b/src/jev_bench/metrics/distributions.py
@@ -12,6 +12,7 @@
     entropy: row-wise Shannon entropy in bits.
     js_divergence: row-wise Jensen-Shannon divergence in bits, within [0, 1].
     expected_level: row-wise expected ordinal level.
+    score_0_100: row-wise expected level scaled to 0-100 (k >= 2 options).
 """
 
 import math
@@ -28,6 +29,7 @@
     "expected_level",
     "js_divergence",
     "normalize",
+    "score_0_100",
     "softmax",
     "to_matrix",
     "unit_probability",
@@ -89,3 +91,7 @@
 
 def expected_level(matrix: FloatArray) -> FloatArray:
     return matrix @ np.arange(matrix.shape[1], dtype=np.float64)
+
+
+def score_0_100(matrix: FloatArray) -> FloatArray:
+    return expected_level(matrix) * (100.0 / (matrix.shape[1] - 1))
```

Apply to `src/jev_bench/compare/report.py`:

```diff
--- a/src/jev_bench/compare/report.py
+++ b/src/jev_bench/compare/report.py
@@ -51,7 +51,7 @@
 
 class QuestionReport(BaseModel):
     id: str
-    type: Literal["choice", "score", "noul"]
+    type: Literal["choice", "score", "noul", "multi"]
     options: tuple[str, ...]
     raters: list[RaterStats]
     pairs: list[PairStats]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_questions.py tests/test_distributions.py -q`
Expected: all pass.

- [ ] **Step 5: Gates and the full suite**

Run: `uv run ruff check --fix && uv run ruff format && uv run pyright && uv run pytest -q`
Expected: 0 pyright errors, and the whole suite passes (validated while planning: 607 → 640).

- [ ] **Step 6: Commit**

```bash
git add src/jev_bench/questions.py src/jev_bench/metrics/distributions.py src/jev_bench/compare/report.py \
  tests/conftest.py tests/test_questions.py tests/test_distributions.py
git commit -m "feat(questions): multi-label question type, hard answers, threshold override, 0–100 score

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
