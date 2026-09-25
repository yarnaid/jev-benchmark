### Task 3: Columns: Jev sends `multi` as its native `choice`; LLM and embeddings already treat it as a distribution

**Files:**
- Modify: `src/jev_bench/classifiers/jev.py`
- Test: `tests/test_classifiers_jev.py`, `tests/test_classifiers_llm_parse.py`,
  `tests/test_classifiers_llm_schema.py`, `tests/test_classifiers_embeddings.py`

**Interfaces:**
- Consumes: `MultiQuestion` and the `multi_questions` fixture (Task 1).
- Produces:
  - `questions_payload()` sends a multi question as `{"type": "choice", "instructions", "criteria": {option:
    description}}`;
  - `parse_decisions()` expects `answer["type"] == "choice"` for it and parses it with the existing choice
    path (probabilities, else one-hot on `choice`).
  - No other column changes:
    - the LLM schema already asks for one number per option;
    - `parse_email_answers` already normalizes;
    - `benchmark.toml` already says one question's probabilities sum to 1;
    - embeddings already softmax.

  The tests for those three are **characterization tests**: they pass immediately and pin the behavior
  the multi reading rule relies on.

**Why:** Decisions has no multi-label primitive. Sending `choice` keeps Jev's recommended primitive, keeps one
call per email, and makes Jev return the same shape as every other column. The label set is read later
(Task 2's `relative_labels`).

- [ ] **Step 1: Write the tests**

Apply to `tests/test_classifiers_jev.py`:

```diff
--- a/tests/test_classifiers_jev.py
+++ b/tests/test_classifiers_jev.py
@@ -272,3 +272,49 @@
     assert outcome.error is not None
     assert "missing or mistyped" in outcome.error
     assert result.usage.cost_estimated is True
+
+
+def test_multi_questions_travel_as_a_decisions_choice(multi_questions: QuestionSet) -> None:
+    assert questions_payload(multi_questions)["topics"] == {
+        "type": "choice",
+        "instructions": "Which topics?",
+        "criteria": {
+            "billing": "About money",
+            "meeting": "About a meeting",
+            "travel": "About a trip",
+        },
+    }
+
+
+@pytest.mark.parametrize(
+    ("answer", "expected", "notes"),
+    [
+        pytest.param(
+            {"type": "choice", "choice": "billing", "probabilities": {"billing": 3, "meeting": 1}},
+            {"billing": 0.75, "meeting": 0.25, "travel": 0.0},
+            [],
+            id="choice-probabilities",
+        ),
+        pytest.param(
+            {"type": "choice", "choice": "meeting"},
+            {"billing": 0.0, "meeting": 1.0, "travel": 0.0},
+            ["topics: probabilities missing, one-hot on choice"],
+            id="choice-without-probabilities",
+        ),
+        pytest.param(
+            {"type": "multi", "probabilities": {"billing": 1}},
+            None,
+            ["topics: missing or mistyped answer"],
+            id="multi-is-not-a-decisions-type",
+        ),
+    ],
+)
+def test_parse_decisions_reads_multi_as_choice(
+    multi_questions: QuestionSet,
+    answer: dict[str, Any],
+    expected: dict[str, float] | None,
+    notes: list[str],
+) -> None:
+    parsed, found = parse_decisions({**_ANSWERS, "topics": answer}, multi_questions)
+    assert parsed.get("topics") == (None if expected is None else pytest.approx(expected))
+    assert found == notes
```

Apply to `tests/test_classifiers_llm_parse.py`:

```diff
--- a/tests/test_classifiers_llm_parse.py
+++ b/tests/test_classifiers_llm_parse.py
@@ -115,3 +115,10 @@
         counter.feed(chunk)
     assert counter.count == 2
     assert seen == [1, 2]
+
+
+def test_multi_answers_are_normalized_like_choice(multi_questions: QuestionSet) -> None:
+    raw = {**_VALID, "topics": {"billing": 2, "meeting": 1, "travel": 1}}
+    parsed, notes = parse_email_answers(raw, multi_questions)
+    assert notes == []
+    assert parsed["topics"] == pytest.approx({"billing": 0.5, "meeting": 0.25, "travel": 0.25})
```

Apply to `tests/test_classifiers_llm_schema.py`:

```diff
--- a/tests/test_classifiers_llm_schema.py
+++ b/tests/test_classifiers_llm_schema.py
@@ -60,3 +60,8 @@
 
 def test_email_refs_empty() -> None:
     assert email_refs(0) == []
+
+
+def test_multi_schema_asks_for_one_number_per_option(multi_questions: QuestionSet) -> None:
+    topics = answers_schema(multi_questions)["properties"]["topics"]
+    assert topics == _options("Which topics?", "billing", "meeting", "travel")
```

Apply to `tests/test_classifiers_embeddings.py`:

```diff
--- a/tests/test_classifiers_embeddings.py
+++ b/tests/test_classifiers_embeddings.py
@@ -2,6 +2,7 @@
 
 import asyncio
 import json
+import math
 from pathlib import Path
 from typing import Any
 
@@ -28,6 +29,9 @@
     "now": [0.0, 0.0, 1.0],
     "yes": [1.0, 0.0, 0.0],
     "no": [0.0, 1.0, 0.0],
+    "billing": [1.0, 0.0, 0.0],
+    "meeting": [1.0, 0.1, 0.0],
+    "travel": [0.0, 1.0, 0.0],
     "junk mail": [2.0, 0.0, 0.0],
     "hi friend": [0.0, 3.0, 0.0],
     "blank": [0.0, 0.0, 0.0],
@@ -264,3 +268,20 @@
     assert classifier.budget == Budget(total=100_000, item=8192)
     assert classifier.sizing.overhead == 0
     assert classifier.input_tokens(_email("junk mail")) == 3
+
+
+async def test_multi_questions_get_a_softmax_distribution(
+    make_client: ClientFactory, multi_questions: QuestionSet, tmp_path: Path
+) -> None:
+    classifier = _classifier(
+        make_client(EmbeddingServer()), multi_questions, EmbeddingCache(tmp_path / "v.jsonl")
+    )
+    junk = _email("junk mail")
+    await classifier.prepare([junk])
+    answers = (await classifier.classify([junk])).outcomes[junk.id].answers
+    assert answers is not None
+    topics = answers["topics"]
+    assert sum(topics.values()) == pytest.approx(1.0)
+    assert topics["meeting"] / topics["billing"] == pytest.approx(
+        math.exp((1 / math.sqrt(1.01) - 1) / 0.05)
+    )
```

The email "junk mail" embeds to `[2, 0, 0]`. Its cosines are billing 1 and meeting 1/√1.01, so the softmax
ratio is `exp((1/√1.01 − 1)/0.05)` ≈ 0.906. At t = 0.8 that applies meeting next to billing, which is the
geometric meaning of the relative rule for embeddings (spec §4).

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/test_classifiers_jev.py tests/test_classifiers_llm_parse.py tests/test_classifiers_llm_schema.py tests/test_classifiers_embeddings.py -q`
Expected: the two new Jev tests FAIL (the payload says `"type": "multi"`, and a `choice` answer is reported
as `missing or mistyped`). The LLM, schema and embeddings characterization tests PASS.

- [ ] **Step 3: Implement**

Apply to `src/jev_bench/classifiers/jev.py`:

```diff
--- a/src/jev_bench/classifiers/jev.py
+++ b/src/jev_bench/classifiers/jev.py
@@ -1,4 +1,7 @@
 """Jev column: one Decisions API call per email carrying every question.
+
+Decisions has no multi-label primitive; a multi question is sent and parsed as a `choice`, and its
+label set is read from that distribution later (`p >= threshold * max(p)`).
 
 Constants:
     DECISIONS_PATH
@@ -31,6 +34,7 @@
 from jev_bench.questions import (
     AnyQuestion,
     Distribution,
+    MultiQuestion,
     NoulQuestion,
     QuestionSet,
     ScoreQuestion,
@@ -64,7 +68,15 @@
         criteria = list(question.options.values())
     else:
         criteria = dict(question.options)
-    return {"type": question.type, "instructions": question.instructions, "criteria": criteria}
+    return {
+        "type": _decision_type(question),
+        "instructions": question.instructions,
+        "criteria": criteria,
+    }
+
+
+def _decision_type(question: AnyQuestion) -> str:
+    return "choice" if isinstance(question, MultiQuestion) else question.type
 
 
 def parse_decisions(
@@ -75,7 +87,7 @@
     answers_dict = answers if isinstance(answers, Mapping) else {}
     for question in questions.questions:
         answer = answers_dict.get(question.id)
-        if not isinstance(answer, dict) or answer.get("type") != question.type:
+        if not isinstance(answer, dict) or answer.get("type") != _decision_type(question):
             notes.append(f"{question.id}: missing or mistyped answer")
             continue
         distribution, note = _parse_answer(question, answer)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run the same command as Step 2. Expected: all pass.

- [ ] **Step 5: Gates, then commit**

```bash
uv run ruff check --fix && uv run ruff format && uv run pyright && uv run pytest -q
git add src/jev_bench/classifiers/jev.py tests/test_classifiers_jev.py tests/test_classifiers_llm_parse.py \
  tests/test_classifiers_llm_schema.py tests/test_classifiers_embeddings.py
git commit -m "feat(jev): send multi-label questions as a native Decisions choice

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

Expected after this task: 670 tests.
