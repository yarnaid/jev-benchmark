### Task 5: Embeddings: softmax ÷ max for multi-label questions

**Files:**
- Modify: `src/jev_bench/classifiers/embeddings.py`
- Test: `tests/test_classifiers_embeddings.py`

**Interfaces:**
- Consumes: `MultiQuestion` and the `multi_questions` fixture (Task 1).
- Produces: for a multi question, `EmailOutcome.answers[qid][option] = softmax(cos/τ) / max(softmax(cos/τ))`,
  which equals `exp((cos − cos_max) / τ)`. The best label is exactly 1.0, and the values do not sum to 1. The
  `similarities` (raw cosines) are unchanged. Other question types are unchanged.

**Why (spec §4.3):** embeddings can't produce calibrated independent probabilities without tuning, and tuning
on references is forbidden. Dividing by the maximum makes the threshold geometric: at threshold t a label is
applied iff `cos_best − cos_label ≤ τ·ln(1/t)` (≈ 0.011 at τ = 0.05, t = 0.8).

- [ ] **Step 1: Write the failing test** (in `tests/test_classifiers_embeddings.py`)

Add `import math` to the imports, and add three entries to `_VECTORS`, right after `"no": [0.0, 1.0, 0.0],`:

```python
    "billing": [1.0, 0.0, 0.0],
    "meeting": [1.0, 0.1, 0.0],
    "travel": [0.0, 1.0, 0.0],
```

Append:

```python
async def test_multi_labels_are_relative_to_the_best_match(
    make_client: ClientFactory, multi_questions: QuestionSet, tmp_path: Path
) -> None:
    classifier = _classifier(
        make_client(EmbeddingServer()), multi_questions, EmbeddingCache(tmp_path / "v.jsonl")
    )
    junk = _email("junk mail")
    await classifier.prepare([junk])
    answers = (await classifier.classify([junk])).outcomes[junk.id].answers
    assert answers is not None
    topics = answers["topics"]
    assert topics["billing"] == pytest.approx(1.0)
    assert topics["meeting"] == pytest.approx(math.exp((1 / math.sqrt(1.01) - 1) / 0.05))
    assert topics["travel"] < 1e-6
    assert sum(answers["category"].values()) == pytest.approx(1.0)
```

The email "junk mail" embeds to `[2, 0, 0]`, whose unit vector is `[1, 0, 0]`. Its cosines are:
- billing = 1;
- meeting = 1/√1.01 ≈ 0.99504, which gives p ≈ 0.9055 at τ = 0.05, so meeting is applied at an 80%
  threshold;
- travel = 0.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_classifiers_embeddings.py -k multi_labels -q`
Expected: FAIL. `billing` is ≈ 0.525, because the plain softmax is shared between billing and meeting.

- [ ] **Step 3: Implement** in `src/jev_bench/classifiers/embeddings.py`

Replace the first docstring line with the following paragraph (the rest of the docstring stays):

```python
"""Embedding column: cosine similarity between email and option texts, softmaxed per question.

For a multi question the softmax is divided by its maximum, so the best-matching label is 1.0 and
another label reaches threshold t iff its cosine is within temperature * ln(1 / t) of the best one
(a documented heuristic: embeddings give no calibrated independent probabilities, and nothing is
tuned on references).
```

Import `MultiQuestion`: `from jev_bench.questions import AnyQuestion, MultiQuestion, QuestionSet`. In
`EmbeddingClassifier._score`, after the `softmax` line add:

```python
        if isinstance(question, MultiQuestion):
            probabilities = probabilities / probabilities.max()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_classifiers_embeddings.py -q`
Expected: all pass (11 tests).

- [ ] **Step 5: Gates, then commit**

```bash
uv run ruff check --fix && uv run ruff format && uv run pyright && uv run pytest -q
git add src/jev_bench/classifiers/embeddings.py tests/test_classifiers_embeddings.py
git commit -m "feat(embeddings): best-match-relative probabilities for multi-label questions

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
