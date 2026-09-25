### Task 9: API: mini config multi wiring, `?threshold=`, newest-run base, list labels

**Files:**
- Modify: `tests/factories.py`: the mini config gains a multi question, and the fakes answer it.
- Modify: `src/jev_bench/web/deps.py` (`ThresholdQuery`)
- Modify: `src/jev_bench/web/routes/compare.py`
- Modify: `src/jev_bench/web/routes/emails.py`
- Modify: `src/jev_bench/web/routes/labels.py`
- Modify: `src/jev_bench/store/labels.py` (types)
- Test: `tests/test_web_routes_compare.py`, `tests/test_web_routes_emails.py`,
  `tests/test_web_routes_labels.py`, `tests/test_store_labels.py`

**Interfaces:**
- Consumes:
  - `compare(..., threshold=)` and `email_rows(..., threshold=)` (Task 8);
  - `hard_distribution`, `HardAnswer`, `AnyQuestion` (Task 1).
- Produces (used by the UI in Tasks 10–11):
  - `web.deps.ThresholdQuery = Annotated[float | None, Query(gt=0.0, le=1.0, allow_inf_nan=False)]`.
    Verified while planning on FastAPI 0.141.1: `0`, `1.01`, `nan`, `inf` and `abc` give 422; `1` is
    accepted.
  - `GET /api/compare?runs=…&threshold=`. The base question set is the snapshot of the **newest** selected run
    (`max` by `(created_at, id)`), no longer `metas[0]`.
  - `GET /api/emails?generations=…&runs=…&threshold=`. Rows carry `top` (a label list for multi), `scores`
    and `reference_scores`.
  - `GET /api/emails/{id}`: `human` is `dict[str, HardAnswer]`.
  - `PUT /api/labels/{id}` with `{"answers": {qid: "option" | ["a", "b"] | null}}`. `[]` clears like `null`.
    An answer for which `hard_distribution(...) is None` → 400 `"<answer> is not an option of '<qid>'"`.
    The empty string stays a 400, as before.
  - `LabelStore` types: `Labels = dict[str, dict[str, HardAnswer]]`,
    `update(email_id, changes: Mapping[str, HardAnswer | None]) -> dict[str, HardAnswer]`.
- Mini-config fakes (used by every service-level test from now on):
  - question `topics` (multi; billing / meeting / travel; `threshold = 0.7`);
  - fake chat answer `topics = {billing 0.9, meeting 0.7, travel 0.05}`. Meeting sits exactly **at** the
    threshold (Review Focus #3);
  - fake Jev sub-answers billing 0.9 / meeting 0.4 / travel 0.1;
  - generator answer `topics = ["billing"]`;
  - `EmailFactory` default reference `topics = ["billing", "meeting"]`.

- [ ] **Step 1: Wire the multi question into `tests/factories.py`**

- Append to `MINI_QUESTIONS_TOML`, after the `needs_reply` options:

  ```toml

  [[questions]]
  id = "topics"
  type = "multi"
  instructions = "Which topics?"
  threshold = 0.7

  [questions.options]
  billing = "About money"
  meeting = "About a meeting"
  travel = "About a trip"
  ```

- `EmailFactory.reference_answers`:

  ```python
      reference_answers = LazyFunction(
          lambda: {
              "category": "spam",
              "urgency": "today",
              "needs_reply": "yes",
              "topics": ["billing", "meeting"],
          }
      )
  ```

- In `generator_output`, the answers become
  `{"category": category, "urgency": "now", "needs_reply": "no", "topics": ["billing"]}`.
- Add `"topics": {"billing": 0.9, "meeting": 0.7, "travel": 0.05},` to `CHAT_ANSWER`.
- Add to `JEV_ANSWERS`:

  ```python
      "topics__billing": {"type": "noul", "noul": 0.9},
      "topics__meeting": {"type": "noul", "noul": 0.4},
      "topics__travel": {"type": "noul", "noul": 0.1},
  ```

- In the docstring, the `MINI_*` constants line becomes:

  ```
      MINI_QUESTIONS_TOML, MINI_BENCHMARK_TOML, MINI_GENERATION_TOML: mini config file contents
          (the mini question set has a multi-label "topics" question with threshold 0.7; the fake
          chat answer puts "meeting" exactly at 0.7, the fake Jev answer puts it at 0.4).
  ```

Run: `uv run pytest -q`
Expected: the whole suite still passes (validated while planning: 722). Tasks 1–8 already made every column,
the generator and the compare paths multi-aware. The extra keys are ignored by tests that use the three-question
`questions` fixture.

- [ ] **Step 2: Write the failing route and store tests**

`tests/test_web_routes_compare.py`: add these imports at the top:

```python
from datetime import timedelta
from typing import Any

import pytest
```

and, after the `tests.factories` import, `from jev_bench.questions import ChoiceQuestion, QuestionSet`. Then
append (the last test covers Review Focus #1):

```python
def _topics(report: dict[str, Any]) -> dict[str, Any]:
    return next(question for question in report["questions"] if question["id"] == "topics")


def _label_counts(topics: dict[str, Any], rater_id: str) -> dict[str, int]:
    return next(stats for stats in topics["raters"] if stats["rater"] == rater_id)["label_counts"]


def test_compare_multi_label_threshold(make_app: AppFactory) -> None:
    client = make_app(FakeOpenRouter(), api_key="sk-server")
    _, (jev_run, chat_run) = _two_runs(client)
    runs = f"{jev_run},{chat_run}"
    default = _topics(client.get("/api/compare", params={"runs": runs}).json())
    assert (default["type"], default["threshold"]) == ("multi", 0.7)
    assert _label_counts(default, chat_run) == {"billing": 2, "meeting": 2, "travel": 0}
    assert _label_counts(default, jev_run) == {"billing": 2, "meeting": 0, "travel": 0}
    strict = _topics(client.get("/api/compare", params={"runs": runs, "threshold": 1}).json())
    assert strict["threshold"] == 1.0
    assert _label_counts(strict, chat_run) == {"billing": 0, "meeting": 0, "travel": 0}


@pytest.mark.parametrize(
    "threshold",
    [
        pytest.param("0", id="zero"),
        pytest.param("-0.5", id="negative"),
        pytest.param("1.01", id="above-one"),
        pytest.param("nan", id="nan"),
        pytest.param("inf", id="infinity"),
        pytest.param("abc", id="not-a-number"),
    ],
)
def test_compare_rejects_out_of_range_thresholds(make_app: AppFactory, threshold: str) -> None:
    client = make_app(FakeOpenRouter())
    params = {"runs": "20260924-100000-any-0001", "threshold": threshold}
    assert client.get("/api/compare", params=params).status_code == 422


def test_compare_uses_the_newest_runs_question_set_in_any_order(make_app: AppFactory) -> None:
    client = make_app(FakeOpenRouter(), api_key="sk-server")
    _, run_ids = _two_runs(client)
    services = services_of(client)
    older = services.runs.get(run_ids[0])
    v1_topics = ChoiceQuestion(
        type="choice",
        id="topics",
        instructions="?",
        options={"billing": "b", "meeting": "m", "travel": "t"},
    )
    v1 = QuestionSet(name="v1", questions=(v1_topics,))
    earlier = older.created_at - timedelta(days=1)
    services.runs.save(older.model_copy(update={"question_set": v1, "created_at": earlier}))
    for order in (run_ids, run_ids[::-1]):
        report = client.get("/api/compare", params={"runs": ",".join(order)}).json()
        topics = _topics(report)
        assert topics["type"] == "multi"
        assert topics["skipped"] == [run_ids[0]]
        assert any("incompatible snapshot" in warning for warning in report["warnings"])
```

`tests/test_web_routes_emails.py`: add `import pytest` at the top and append:

```python
def test_list_emails_multi_labels_scores_and_threshold(make_app: AppFactory) -> None:
    client = make_app(FakeOpenRouter(), api_key="sk-server")
    generation_id = seed_generation(services_of(client))
    run_id = client.post(
        "/api/runs", json={"column": "anthropic", "generation_ids": [generation_id]}
    ).json()["meta"]["id"]
    poll(client, f"/api/runs/{run_id}", until=lambda body: body["meta"]["status"] != "running")
    params = {"generations": generation_id, "runs": run_id}
    row = client.get("/api/emails", params=params).json()["rows"][0]
    assert row["top"][run_id]["topics"] == ["billing", "meeting"]
    assert row["reference"]["topics"] == ["billing", "meeting"]
    assert row["scores"][run_id]["urgency"] == pytest.approx(75.0)
    assert row["reference_scores"] == {"urgency": 50.0}
    strict = client.get("/api/emails", params={**params, "threshold": 0.8}).json()["rows"][0]
    assert strict["top"][run_id]["topics"] == ["billing"]
    assert client.get("/api/emails", params={**params, "threshold": 0}).status_code == 422
```

(The fake chat urgency {low 0.1, today 0.3, now 0.6} is level 1.5 of 2, so its score is 75.)

`tests/test_web_routes_labels.py` covers Review Focus #5. Add these rows to the `test_put_label_validation`
table, after `unknown-option`:

```python
        pytest.param(
            {"topics": ["billing", "billing"]}, 400, "is not an option", id="duplicate-labels"
        ),
        pytest.param({"topics": ["phishing"]}, 400, "is not an option", id="unknown-label"),
        pytest.param({"topics": "billing"}, 400, "is not an option", id="bare-string-for-multi"),
        pytest.param({"category": ["work"]}, 400, "is not an option", id="list-for-single-choice"),
        pytest.param({"category": ""}, 400, "is not an option", id="empty-string"),
```

Change that test's `answers` annotation to `dict[str, object]`, and append:

```python
def test_put_label_accepts_label_lists_and_clears_with_an_empty_list(make_app: AppFactory) -> None:
    client = make_app(FakeOpenRouter())
    email_id = f"{seed_generation(services_of(client))}.0001"
    url = f"/api/labels/{email_id}"
    first = client.put(url, json={"answers": {"topics": ["travel", "billing"], "category": "work"}})
    assert first.json() == {"topics": ["travel", "billing"], "category": "work"}
    assert client.get(f"/api/emails/{email_id}").json()["human"]["topics"] == ["travel", "billing"]
    cleared = client.put(url, json={"answers": {"topics": []}})
    assert cleared.json() == {"category": "work"}
```

`tests/test_store_labels.py`: append:

```python
async def test_update_stores_label_lists(tmp_path: Path) -> None:
    store = LabelStore(tmp_path)
    labels = {"topics": ["billing", "meeting"], "category": "spam"}
    assert await store.update(_E1, labels) == labels
    assert LabelStore(tmp_path).for_email(_E1) == labels
```

- [ ] **Step 3: Run the tests to verify they fail**

Run:
`uv run pytest tests/test_web_routes_compare.py tests/test_web_routes_emails.py tests/test_web_routes_labels.py tests/test_store_labels.py -q`
Expected FAILs:
- the threshold params are ignored, so the counts at `threshold=1` are wrong and `0` is not a 422;
- the newest-base test fails for the `run_ids` order, because `topics` is a `choice` there;
- list labels get 422 from `dict[str, str | None]`.

- [ ] **Step 4: `src/jev_bench/web/deps.py`**

- Import `Query` from `fastapi`.
- Add `"ThresholdQuery",` to `__all__`.
- Add the docstring `Types:` line
  `    ThresholdQuery: optional `?threshold=` in (0, 1] (422 otherwise, NaN and infinity included).`
- Add below `ApiKeyDep`:

  ```python
  ThresholdQuery = Annotated[float | None, Query(gt=0.0, le=1.0, allow_inf_nan=False)]
  ```

- [ ] **Step 5: `src/jev_bench/web/routes/compare.py`**

The docstring's `Functions:` entry becomes:

```
    compare_runs: GET /compare?runs=a,b,c[&threshold=0.8]. The base question set is the snapshot of
        the most recent selected run (by created_at, then id), so the URL order of `runs` never
        changes the report.
```

Add `from collections.abc import Sequence` and `from jev_bench.store.runs import RunMeta`, and import
`ThresholdQuery` from `jev_bench.web.deps`. Then:

```python
@router.get("/compare", response_model_exclude=LIGHT)
def compare_runs(
    services: ServicesDep, runs: str, threshold: ThresholdQuery = None
) -> ComparisonReport:
    metas = load_runs(services, split_ids(runs))
    if not metas:
        raise HTTPException(status_code=400, detail="no run ids given")
    base = _newest(metas).question_set
    generation_ids = generations_of(metas)
    reference = reference_rater(
        load_emails(services, generation_ids),
        base,
        generation_snapshots(services, generation_ids),
    )
    raters = [*run_raters(services, metas), reference]
    human = human_rater(services.labels.for_generations(generation_ids), base)
    if human is not None:
        raters.append(human)
    return compare(raters, base, runs={meta.id: meta for meta in metas}, threshold=threshold)


def _newest(metas: Sequence[RunMeta]) -> RunMeta:
    return max(metas, key=lambda meta: (meta.created_at, meta.id))
```

- [ ] **Step 6: `src/jev_bench/web/routes/emails.py`**

- The docstring line becomes `    list_emails: GET /emails?generations=…&runs=…[&threshold=0.8]`.
- Import `HardAnswer` (`from jev_bench.questions import HardAnswer, QuestionSet`) and `ThresholdQuery`.
- In `EmailDetail`, use `human: dict[str, HardAnswer]`.
- Replace `list_emails`:

```python
@router.get("/emails")
def list_emails(
    services: ServicesDep,
    generations: str,
    runs: str | None = None,
    threshold: ThresholdQuery = None,
) -> EmailList:
    generation_ids = split_ids(generations)
    questions = services.question_set()
    emails = load_emails(services, generation_ids)
    raters = run_raters(services, load_runs(services, split_ids(runs)))
    labels = services.labels.for_generations(generation_ids)
    rows = email_rows(emails, raters, labels, questions, threshold=threshold)
    return EmailList(questions=questions, rows=rows)
```

- [ ] **Step 7: `src/jev_bench/web/routes/labels.py`**

The docstring's `LabelUpdate` line becomes:

```
    LabelUpdate: `{answers: {question_id: option_id | [option_id, ...] | null}}`; a list is for a
        multi-label question, and `null` or `[]` clears the answer.
```

The questions import becomes
`from jev_bench.questions import AnyQuestion, HardAnswer, QuestionSet, hard_distribution`. Replace
everything below `class LabelUpdate(BaseModel):` / `model_config` with:

```python
    answers: dict[str, HardAnswer | None]


@router.put("/labels/{email_id}")
async def put_label(
    email_id: str, update: LabelUpdate, services: ServicesDep
) -> dict[str, HardAnswer]:
    load_email(services, email_id)
    changes = _validated(update.answers, services.question_set())
    return await services.labels.update(email_id, changes)


def _validated(
    answers: Mapping[str, HardAnswer | None], questions: QuestionSet
) -> dict[str, HardAnswer | None]:
    changes = {
        question_id: None if answer == [] else answer for question_id, answer in answers.items()
    }
    for question_id, answer in changes.items():
        question = _question(questions, question_id)
        if answer is not None and hard_distribution(question, answer) is None:
            detail = f"{answer!r} is not an option of {question_id!r}"
            raise HTTPException(status_code=400, detail=detail)
    return changes


def _question(questions: QuestionSet, question_id: str) -> AnyQuestion:
    try:
        return questions.get(question_id)
    except KeyError as exc:
        detail = f"unknown question {question_id!r}"
        raise HTTPException(status_code=400, detail=detail) from exc
```

- [ ] **Step 8: `src/jev_bench/store/labels.py`** (types only)

- The first docstring line becomes:

  ```
  """Human labels per generation: `{email_id: {question_id: option_id | [option_id, ...]}}`, edited
  atomically (a list is the answer to a multi-label question).
  ```

- Add `from jev_bench.questions import HardAnswer`.
- `type Labels = dict[str, dict[str, HardAnswer]]`.
- `for_email(...) -> dict[str, HardAnswer]`.
- `update(self, email_id: str, changes: Mapping[str, HardAnswer | None]) -> dict[str, HardAnswer]`.
- `_apply(current: Mapping[str, HardAnswer], changes: Mapping[str, HardAnswer | None]) -> dict[str, HardAnswer]`.

`LabelStore` never validates. Validation stays at the route boundary.

- [ ] **Step 9: Run the tests to verify they pass**

Run: `uv run pytest -q --durations=5`
Expected: the whole suite passes (validated while planning: 738 tests). None of the new tests is among the
slowest five.

- [ ] **Step 10: Gates, then commit**

```bash
uv run ruff check --fix && uv run ruff format && uv run pyright
git add tests/factories.py src/jev_bench/web/deps.py src/jev_bench/web/routes/compare.py \
  src/jev_bench/web/routes/emails.py src/jev_bench/web/routes/labels.py src/jev_bench/store/labels.py \
  tests/test_web_routes_compare.py tests/test_web_routes_emails.py tests/test_web_routes_labels.py \
  tests/test_store_labels.py
git commit -m "feat(web): label threshold query, newest-run base question set, multi-label human labels

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
