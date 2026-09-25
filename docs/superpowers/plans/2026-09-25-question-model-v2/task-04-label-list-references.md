### Task 4: Label-list reference answers: email model, generator schema/parse/mismatch/prompt, raters

**Files:**
- Modify: `src/jev_bench/emails.py` (`reference_answers` type)
- Modify: `src/jev_bench/generation/prompt.py`
- Modify: `config/generation.toml` (a system prompt sentence)
- Modify: `src/jev_bench/compare/raters.py` (hard answers via `hard_distribution`)
- Modify: `src/jev_bench/compare/rows.py`: type widening only (`reference`, `human`, the `labels` parameter);
  the behavior changes come in Task 6.
- Test: `tests/test_emails.py`, `tests/test_generation_prompt.py`, `tests/test_config_files.py`,
  `tests/test_compare_raters.py`

**Why the raters and rows are in this task:** as soon as `Email.reference_answers` may hold a list, today's
`raters._reference_hard` evaluates `labels.get(q.id) in question.options`. With a list value that raises
`TypeError: unhashable type: 'list'` at runtime. pyright also rejects `dict[str, HardAnswer]` →
`dict[str, str]` in `rows.EmailRow`. Both consumers must change together with the type (found by validating
while planning).

**Interfaces:**
- Consumes (Task 1): `HardAnswer`, `MultiQuestion`, `hard_distribution`, the shape-based `compatible`.
- Produces:
  - `Email.reference_answers: dict[str, HardAnswer]` and `GeneratorOutput.answers: dict[str, HardAnswer]`.
  - `generation_schema()`: multi → `{"type": "array", "items": {"type": "string", "enum": [...]}}`, with
    **no** `minItems`.
  - `parse_generator_output()` rejects any answer for which `hard_distribution(...) is None` (an empty,
    duplicate or unknown label list, a list for a non-multi question) with
    `ValueError("generator answers missing or invalid for: [...]")`. That goes through the existing same-model
    retry. A single option id for a multi question is a valid one-label set.
  - `count_mismatches(requested, traits, answers: Mapping[str, HardAnswer])`: a list answer agrees iff it
    contains the requested value.
  - `human_rater(labels: Mapping[str, Mapping[str, HardAnswer]], questions)`.
  - Reference and human raters hold uniform distributions for label sets.
  - A v1 string reference under a compatible v1 `choice` snapshot becomes a one-label set: Review Focus #2,
    old runs are reused.
  - `EmailRow.reference` / `EmailRow.human: dict[str, HardAnswer]`, and
    `email_rows(labels: Mapping[str, Mapping[str, HardAnswer]], …)`.

- [ ] **Step 1: Write the failing tests**

Apply to `tests/test_emails.py`:

```diff
--- a/tests/test_emails.py
+++ b/tests/test_emails.py
@@ -7,7 +7,7 @@
 from pydantic import ValidationError
 from tests.factories import EmailFactory, PartyFactory
 
-from jev_bench.emails import Party, email_id
+from jev_bench.emails import Email, Party, email_id
 
 
 def test_to_state_contains_exactly_the_model_visible_fields() -> None:
@@ -78,3 +78,9 @@
 
 def test_party_factory_is_valid() -> None:
     assert "@" in PartyFactory().address
+
+
+def test_reference_answers_accept_label_lists() -> None:
+    answers = {"category": ["spam", "phishing"], "urgency": "today"}
+    email = EmailFactory(reference_answers=answers)
+    assert Email.model_validate_json(email.model_dump_json()).reference_answers == answers
```

Apply to `tests/test_generation_prompt.py`:

```diff
--- a/tests/test_generation_prompt.py
+++ b/tests/test_generation_prompt.py
@@ -75,6 +75,9 @@
         pytest.param(lambda doc: doc["answers"].pop("urgency"), id="missing-answer"),
         pytest.param(lambda doc: doc["email"].update(to=[]), id="no-recipients"),
         pytest.param(lambda doc: doc["email"].update(body=""), id="empty-body"),
+        pytest.param(
+            lambda doc: doc["answers"].update(category=["spam"]), id="list-for-single-choice"
+        ),
     ],
 )
 def test_parse_generator_output_rejects(questions: QuestionSet, mutate: Any) -> None:
@@ -118,3 +121,75 @@
 ) -> None:
     traits = resolve_traits(_CONFIG, questions)
     assert count_mismatches(requested, traits, answers) == expected
+
+
+_MULTI_CONFIG = GenerationConfig.model_validate(
+    {
+        "models": ["m"],
+        "system_prompt": "Questions:\n$questions",
+        "user_prompt": "Write about $topic ($topic_prompt), sent $sent_at.",
+        "traits": [{"name": "topic", "question": "topics"}],
+    }
+)
+_MULTI_OUTPUT: dict[str, Any] = {
+    **_OUTPUT,
+    "answers": {**_OUTPUT["answers"], "topics": ["meeting", "billing"]},
+}
+
+
+def test_generation_schema_multi_is_an_array_of_options(multi_questions: QuestionSet) -> None:
+    answers = generation_schema(multi_questions)["properties"]["answers"]
+    assert answers["properties"]["topics"] == {
+        "type": "array",
+        "items": {"type": "string", "enum": ["billing", "meeting", "travel"]},
+    }
+    assert "topics" in answers["required"]
+
+
+@pytest.mark.parametrize(
+    "topics",
+    [
+        pytest.param(["meeting", "billing"], id="label-list-keeps-its-order"),
+        pytest.param("meeting", id="single-id-is-one-label"),
+    ],
+)
+def test_parse_generator_output_accepts_label_answers(
+    multi_questions: QuestionSet, topics: object
+) -> None:
+    doc = {**_MULTI_OUTPUT, "answers": {**_MULTI_OUTPUT["answers"], "topics": topics}}
+    output = parse_generator_output(json.dumps(doc), multi_questions)
+    assert output.answers["topics"] == topics
+    assert output.answers["category"] == "spam"
+
+
+@pytest.mark.parametrize(
+    "topics",
+    [
+        pytest.param([], id="empty-list"),
+        pytest.param(["billing", "billing"], id="duplicate"),
+        pytest.param(["billing", "phishing"], id="unknown-label"),
+    ],
+)
+def test_parse_generator_output_rejects_bad_label_lists(
+    multi_questions: QuestionSet, topics: object
+) -> None:
+    doc = json.loads(json.dumps(_MULTI_OUTPUT))
+    doc["answers"]["topics"] = topics
+    with pytest.raises(ValueError, match="missing or invalid"):
+        parse_generator_output(json.dumps(doc), multi_questions)
+
+
+@pytest.mark.parametrize(
+    ("answers", "expected"),
+    [
+        pytest.param({"topics": ["meeting", "billing"]}, 0, id="requested-label-present"),
+        pytest.param({"topics": ["meeting"]}, 1, id="requested-label-absent"),
+        pytest.param({"topics": "billing"}, 0, id="single-id-answer"),
+        pytest.param({}, 1, id="missing-answer"),
+    ],
+)
+def test_count_mismatches_multi_membership(
+    multi_questions: QuestionSet, answers: dict[str, str | list[str]], expected: int
+) -> None:
+    traits = resolve_traits(_MULTI_CONFIG, multi_questions)
+    assert count_mismatches({"topic": "billing"}, traits, answers) == expected
```

Apply to `tests/test_config_files.py`:

```diff
--- a/tests/test_config_files.py
+++ b/tests/test_config_files.py
@@ -61,6 +61,7 @@
     )
     assert config.models == expected_models
     assert config.max_attempts == 3
+    assert "every option id that applies (at least one)" in config.system_prompt
     expected_traits = ["category", "urgency", "length", "prompt_injection"]
     assert [trait.name for trait in traits] == expected_traits
     now = datetime(2026, 9, 24, tzinfo=UTC)
```

Apply to `tests/test_compare_raters.py`:

```diff
--- a/tests/test_compare_raters.py
+++ b/tests/test_compare_raters.py
@@ -117,3 +117,48 @@
     email = EmailFactory(id="gen1.0001", reference_answers={"category": "spam"})
     rater = reference_rater([email], questions, snapshots)
     assert rater.warnings == expected_warnings
+
+
+def _v1_topics(options: dict[str, str]) -> QuestionSet:
+    topics = ChoiceQuestion(type="choice", id="topics", instructions="?", options=options)
+    return QuestionSet(name="v1", questions=(topics,))
+
+
+def test_reference_and_human_raters_split_label_sets_evenly(multi_questions: QuestionSet) -> None:
+    email = EmailFactory(
+        id="gen1.0001", reference_answers={"topics": ["travel", "billing"], "category": "spam"}
+    )
+    reference = reference_rater([email], multi_questions, {"gen1": multi_questions})
+    assert reference.answers["gen1.0001"]["topics"] == {
+        "billing": 0.5,
+        "meeting": 0.0,
+        "travel": 0.5,
+    }
+    human = human_rater(
+        {"gen1.0001": {"topics": ["meeting"]}, "gen1.0002": {"topics": ["nope"]}},
+        multi_questions,
+    )
+    assert human is not None
+    only_meeting = {"billing": 0.0, "meeting": 1.0, "travel": 0.0}
+    assert human.answers == {"gen1.0001": {"topics": only_meeting}}
+
+
+@pytest.mark.parametrize(
+    ("snapshot_options", "expected"),
+    [
+        pytest.param(
+            {"billing": "b", "meeting": "m", "travel": "t"},
+            {"gen1.0001": {"topics": {"billing": 1.0, "meeting": 0.0, "travel": 0.0}}},
+            id="same-options-is-a-one-label-set",
+        ),
+        pytest.param({"billing": "b", "travel": "t"}, {}, id="other-options-is-skipped"),
+    ],
+)
+def test_v1_single_label_reference_for_a_multi_question(
+    multi_questions: QuestionSet,
+    snapshot_options: dict[str, str],
+    expected: dict[str, dict[str, dict[str, float]]],
+) -> None:
+    email = EmailFactory(id="gen1.0001", reference_answers={"topics": "billing"})
+    snapshot = _v1_topics(snapshot_options)
+    assert reference_rater([email], multi_questions, {"gen1": snapshot}).answers == expected
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
`uv run pytest tests/test_emails.py tests/test_generation_prompt.py tests/test_config_files.py tests/test_compare_raters.py -q`
Expected FAILs/ERRORs:
- a list reference is rejected by the `dict[str, str]` model;
- the schema is a plain string;
- `count_mismatches` treats `["meeting", "billing"] != "billing"` as a mismatch;
- `reference_rater` raises `TypeError: unhashable type: 'list'`;
- the generator prompt sentence is missing.

- [ ] **Step 3: Implement**

Apply to `src/jev_bench/emails.py`:

```diff
--- a/src/jev_bench/emails.py
+++ b/src/jev_bench/emails.py
@@ -4,7 +4,8 @@
     EmailState: the JSON object sent as classification input.
 Classes:
     Party: a named mailbox.
-    Email: one generated email with its generation metadata and reference answers.
+    Email: one generated email with its generation metadata and reference answers (one option id
+        per question, or a list of option ids for a multi-label question).
 Functions:
     email_id: id of the n-th email of a generation.
 """
@@ -12,6 +13,8 @@
 from typing import TypedDict
 
 from pydantic import AwareDatetime, BaseModel, ConfigDict, Field
+
+from jev_bench.questions import HardAnswer
 
 __all__ = [
     "Email",
@@ -48,7 +51,7 @@
     body: str
     generator_model: str
     traits: dict[str, str] = Field(default_factory=dict)
-    reference_answers: dict[str, str] = Field(default_factory=dict)
+    reference_answers: dict[str, HardAnswer] = Field(default_factory=dict)
 
     @property
     def generation_id(self) -> str:
```

Apply to `src/jev_bench/generation/prompt.py`:

```diff
--- a/src/jev_bench/generation/prompt.py
+++ b/src/jev_bench/generation/prompt.py
@@ -2,14 +2,17 @@
 
 Classes:
     GeneratedEmail: the email part of a generator response.
-    GeneratorOutput: validated generator response (email + one answer per question).
+    GeneratorOutput: validated generator response (email + one hard answer per question).
 Functions:
     render_prompts: (system, user) prompts for one plan item.
-    generation_schema: strict JSON schema for `{email, answers}`.
-    parse_generator_output: JSON text -> GeneratorOutput (answers checked against the options;
-        text around the outermost JSON object, e.g. a model's preamble, is ignored).
-    count_mismatches: question-linked traits that the generator's own answers contradict
-        (traits absent from `requested` are skipped).
+    generation_schema: strict JSON schema for `{email, answers}` (an array of option ids for a
+        multi-label question).
+    parse_generator_output: JSON text -> GeneratorOutput (every answer must be a valid hard answer,
+        see questions.hard_distribution; text around the outermost JSON object, e.g. a model's
+        preamble, is ignored).
+    count_mismatches: question-linked traits that the generator's own answers contradict (a
+        multi-label answer agrees when it contains the trait value; traits absent from
+        `requested` are skipped).
 """
 
 from collections.abc import Mapping, Sequence
@@ -21,7 +24,14 @@
 from jev_bench.generation.config import GenerationConfig
 from jev_bench.generation.plan import PlanItem, ResolvedTrait
 from jev_bench.json_schema import JsonSchema, strict_object
-from jev_bench.questions import QuestionSet, render_questions
+from jev_bench.questions import (
+    AnyQuestion,
+    HardAnswer,
+    MultiQuestion,
+    QuestionSet,
+    hard_distribution,
+    render_questions,
+)
 
 __all__ = [
     "GeneratedEmail",
@@ -49,7 +59,7 @@
     model_config = ConfigDict(extra="ignore")
 
     email: GeneratedEmail
-    answers: dict[str, str]
+    answers: dict[str, HardAnswer]
 
 
 def render_prompts(
@@ -88,12 +98,14 @@
         "body": {"type": "string"},
     }
     email = strict_object(email_props)
-    answers_props = {
-        question.id: {"type": "string", "enum": list(question.option_ids)}
-        for question in questions.questions
-    }
+    answers_props = {question.id: _answer_schema(question) for question in questions.questions}
     answers = strict_object(answers_props)
     return strict_object({"email": email, "answers": answers})
+
+
+def _answer_schema(question: AnyQuestion) -> JsonSchema:
+    option: JsonSchema = {"type": "string", "enum": list(question.option_ids)}
+    return {"type": "array", "items": option} if isinstance(question, MultiQuestion) else option
 
 
 def parse_generator_output(content: str, questions: QuestionSet) -> GeneratorOutput:
@@ -101,7 +113,7 @@
     invalid = [
         question.id
         for question in questions.questions
-        if output.answers.get(question.id) not in question.options
+        if hard_distribution(question, output.answers.get(question.id)) is None
     ]
     if invalid:
         raise ValueError(f"generator answers missing or invalid for: {invalid}")
@@ -115,12 +127,18 @@
 
 
 def count_mismatches(
-    requested: Mapping[str, str], traits: Sequence[ResolvedTrait], answers: Mapping[str, str]
+    requested: Mapping[str, str],
+    traits: Sequence[ResolvedTrait],
+    answers: Mapping[str, HardAnswer],
 ) -> int:
     return sum(
         1
         for trait in traits
         if trait.question
         and trait.name in requested
-        and answers.get(trait.question) != requested[trait.name]
+        and not _agrees(answers.get(trait.question), requested[trait.name])
     )
+
+
+def _agrees(answer: HardAnswer | None, value: str) -> bool:
+    return value in answer if isinstance(answer, list) else answer == value
```

Apply to `config/generation.toml`:

```diff
--- a/config/generation.toml
+++ b/config/generation.toml
@@ -7,7 +7,7 @@
 system_prompt = """
 You write realistic synthetic emails for benchmarking email-triage classifiers.
 Write natural English with realistic names, organizations, addresses and details. Never use placeholders such as [Name] or generic domains such as company.com.
-After writing the email, answer every question below about the email you wrote, choosing exactly one option id per question, as a careful human reader would.
+After writing the email, answer every question below about the email you wrote, choosing exactly one option id per question, or every option id that applies (at least one) for a multi-label question, as a careful human reader would.
 
 Questions:
 $questions
```

Apply to `src/jev_bench/compare/raters.py`:

```diff
--- a/src/jev_bench/compare/raters.py
+++ b/src/jev_bench/compare/raters.py
@@ -12,6 +12,9 @@
         its own generation's question-set snapshot, with one warning per generation whose
         snapshot is missing or incompatible for some questions.
     human_rater: build a rater from raw human labels (None when none are usable).
+Hard answers (reference and human) become their distributions (questions.hard_distribution: one-hot,
+or uniform over the labels of a multi question); an answer that is not valid for the question is
+skipped.
 """
 
 from collections.abc import Mapping, Sequence
@@ -20,7 +23,14 @@
 from pydantic import BaseModel
 
 from jev_bench.emails import Email
-from jev_bench.questions import AnyQuestion, Distribution, QuestionSet, compatible, one_hot
+from jev_bench.questions import (
+    AnyQuestion,
+    Distribution,
+    HardAnswer,
+    QuestionSet,
+    compatible,
+    hard_distribution,
+)
 from jev_bench.store.runs import Prediction, RunMeta
 
 __all__ = [
@@ -128,7 +138,9 @@
     )
 
 
-def human_rater(labels: Mapping[str, Mapping[str, str]], questions: QuestionSet) -> Rater | None:
+def human_rater(
+    labels: Mapping[str, Mapping[str, HardAnswer]], questions: QuestionSet
+) -> Rater | None:
     answers = {
         email_id: hard for email_id, chosen in labels.items() if (hard := _hard(chosen, questions))
     }
@@ -139,19 +151,20 @@
     )
 
 
-def _hard(labels: Mapping[str, str], questions: QuestionSet) -> dict[str, Distribution]:
+def _hard(labels: Mapping[str, HardAnswer], questions: QuestionSet) -> dict[str, Distribution]:
     return {
-        question.id: one_hot(question, labels[question.id])
+        question.id: hard
         for question in questions.questions
-        if labels.get(question.id) in question.options
+        if (hard := hard_distribution(question, labels.get(question.id))) is not None
     }
 
 
 def _reference_hard(
-    labels: Mapping[str, str], questions: QuestionSet, snapshot: QuestionSet
+    labels: Mapping[str, HardAnswer], questions: QuestionSet, snapshot: QuestionSet
 ) -> dict[str, Distribution]:
     return {
-        question.id: one_hot(question, labels[question.id])
+        question.id: hard
         for question in questions.questions
-        if labels.get(question.id) in question.options and _supports(snapshot, question)
+        if _supports(snapshot, question)
+        and (hard := hard_distribution(question, labels.get(question.id))) is not None
     }
```

Apply to `src/jev_bench/compare/rows.py`:

```diff
--- a/src/jev_bench/compare/rows.py
+++ b/src/jev_bench/compare/rows.py
@@ -18,7 +18,7 @@
 from jev_bench.compare.raters import Rater
 from jev_bench.emails import Email
 from jev_bench.metrics.distributions import js_divergence, to_matrix
-from jev_bench.questions import AnyQuestion, Distribution, QuestionSet
+from jev_bench.questions import AnyQuestion, Distribution, HardAnswer, QuestionSet
 
 __all__ = [
     "EmailRow",
@@ -38,8 +38,8 @@
     subject: str
     generator_model: str
     traits: dict[str, str]
-    reference: dict[str, str]
-    human: dict[str, str]
+    reference: dict[str, HardAnswer]
+    human: dict[str, HardAnswer]
     top: dict[str, dict[str, str]]
     disagreement: float | None
 
@@ -47,7 +47,7 @@
 def email_rows(
     emails: Sequence[Email],
     runs: Sequence[Rater],
-    labels: Mapping[str, Mapping[str, str]],
+    labels: Mapping[str, Mapping[str, HardAnswer]],
     base: QuestionSet,
 ) -> list[EmailRow]:
     supported = {rater.id: _supported_questions(rater, base) for rater in runs}
@@ -61,7 +61,7 @@
 def _email_row(
     email: Email,
     runs: Sequence[Rater],
-    human: Mapping[str, str],
+    human: Mapping[str, HardAnswer],
     base: QuestionSet,
     supported: SupportedQuestions,
 ) -> EmailRow:
```

`questions.py` imports nothing from `emails.py`, so `emails → questions` adds no cycle.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q`
Expected: the whole default suite passes (validated while planning: 685 after Tasks 1–4).

- [ ] **Step 5: Gates, then commit**

```bash
uv run ruff check --fix && uv run ruff format && uv run pyright
git add src/jev_bench/emails.py src/jev_bench/generation/prompt.py config/generation.toml \
  src/jev_bench/compare/raters.py src/jev_bench/compare/rows.py tests/test_emails.py \
  tests/test_generation_prompt.py tests/test_config_files.py tests/test_compare_raters.py
git commit -m "feat(generation): label-list reference answers for multi-label questions

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
