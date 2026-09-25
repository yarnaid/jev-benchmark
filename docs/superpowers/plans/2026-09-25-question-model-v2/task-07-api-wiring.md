### Task 7: API: mini config multi wiring, `?threshold=`, newest-run base, list labels

**Files:**
- Modify: `tests/factories.py`: the mini config gains a multi question, and the fakes answer it.
- Modify: `src/jev_bench/web/deps.py` (`ThresholdQuery`)
- Modify: `src/jev_bench/web/routes/compare.py`, `routes/emails.py`, `routes/labels.py`
- Modify: `src/jev_bench/store/labels.py` (types)
- Test: `tests/test_web_routes_compare.py`, `tests/test_web_routes_emails.py`,
  `tests/test_web_routes_labels.py`, `tests/test_store_labels.py`

**Interfaces:**
- Consumes:
  - `compare(..., threshold=)` and `email_rows(..., threshold=)` (Task 6);
  - `hard_distribution`, `HardAnswer`, `AnyQuestion` (Task 1).
- Produces (used by the UI in Tasks 8–9):
  - `web.deps.ThresholdQuery = Annotated[float | None, Query(gt=0.0, le=1.0, allow_inf_nan=False)]`.
    Verified while planning on FastAPI 0.141.1: `0`, `1.01`, `nan`, `inf` and `abc` give 422; `1` is
    accepted.
  - `GET /api/compare?runs=…&threshold=`. The base question set is the snapshot of the **newest** selected run
    (`max` by `(created_at, id)`), no longer `metas[0]`. With the shape-based compatibility of Task 1 this
    decides whether a mix of v1 (choice) and v2 (multi) runs is read as single-label or multi-label.
  - `GET /api/emails?generations=…&runs=…&threshold=`. Rows carry `top` (a label list for multi), `scores` and
    `reference_scores`.
  - `GET /api/emails/{id}`: `human` is `dict[str, HardAnswer]`.
  - `PUT /api/labels/{id}` with `{"answers": {qid: "option" | ["a", "b"] | null}}`:
    - `[]` clears the label like `null`;
    - an invalid answer (`hard_distribution(...) is None`) → 400 `"<answer> is not an option of '<qid>'"`;
    - the empty string stays a 400.
  - `LabelStore` types: `dict[str, dict[str, HardAnswer]]`.
- Mini-config fakes (from now on every service-level test exercises multi end to end):
  - the question `topics` (multi; billing / meeting / travel; `threshold = 0.75`);
  - the fake chat answer `topics = {billing 0.5, meeting 0.375, travel 0.125}`. `meeting` is **exactly** at
    `0.75 × 0.5`, with binary-exact values (Review Focus #3);
  - the fake Jev answer: a native `choice` with probabilities {billing 0.7, meeting 0.2, travel 0.1}, so only
    billing is applied;
  - the generator answer `topics = ["billing"]`;
  - the `EmailFactory` default reference `topics = ["billing", "meeting"]`.

- [ ] **Step 1: Wire the multi question into `tests/factories.py`**

Apply to `tests/factories.py`:

```diff
--- a/tests/factories.py
+++ b/tests/factories.py
@@ -15,7 +15,9 @@
     services_of: typed access to a TestClient's app.state.services.
     poll: GET a path repeatedly until a predicate on its JSON body holds, or fail.
 Constants:
-    MINI_QUESTIONS_TOML, MINI_BENCHMARK_TOML, MINI_GENERATION_TOML: mini config file contents.
+    MINI_QUESTIONS_TOML, MINI_BENCHMARK_TOML, MINI_GENERATION_TOML: mini config file contents
+        (the mini question set has a multi-label "topics" question with threshold 0.75; the fake
+        chat answer puts "meeting" exactly at 0.75 * max, the fake Jev answer below it).
     CHAT_PARAMETERS: the fake catalog's supported_parameters for chat models (like the real
         Claude Sonnet 5 / GPT-5.6 Terra entries: no `temperature`).
 Types:
@@ -69,7 +71,12 @@
     generator_model = "google/gemini-3.8-flash"
     traits = LazyFunction(dict)
     reference_answers = LazyFunction(
-        lambda: {"category": "spam", "urgency": "today", "needs_reply": "yes"}
+        lambda: {
+            "category": "spam",
+            "urgency": "today",
+            "needs_reply": "yes",
+            "topics": ["billing", "meeting"],
+        }
     )
 
 
@@ -148,6 +155,17 @@
 [questions.options]
 yes = "Reply expected"
 no = "No reply expected"
+
+[[questions]]
+id = "topics"
+type = "multi"
+instructions = "Which topics?"
+threshold = 0.75
+
+[questions.options]
+billing = "About money"
+meeting = "About a meeting"
+travel = "About a trip"
 """
 
 MINI_BENCHMARK_TOML = """
@@ -232,7 +250,12 @@
                 "subject": "You won a prize",
                 "body": "Claim it now.",
             },
-            "answers": {"category": category, "urgency": "now", "needs_reply": "no"},
+            "answers": {
+                "category": category,
+                "urgency": "now",
+                "needs_reply": "no",
+                "topics": ["billing"],
+            },
         }
     )
 
@@ -241,6 +264,7 @@
     "category": {"spam": 0.7, "personal": 0.2, "work": 0.1},
     "urgency": {"low": 0.1, "today": 0.3, "now": 0.6},
     "needs_reply": 0.2,
+    "topics": {"billing": 0.5, "meeting": 0.375, "travel": 0.125},
 }
 JEV_ANSWERS: dict[str, Any] = {
     "category": {
@@ -254,6 +278,11 @@
         "probabilities": {"0": 0.05, "1": 0.1, "2": 0.85},
     },
     "needs_reply": {"type": "noul", "noul": 0.15},
+    "topics": {
+        "type": "choice",
+        "choice": "billing",
+        "probabilities": {"billing": 0.7, "meeting": 0.2, "travel": 0.1},
+    },
 }
 CHAT_PARAMETERS = ("max_tokens", "reasoning", "response_format", "structured_outputs")
 _CATALOG: dict[str, list[dict[str, Any]]] = {
```

Run: `uv run pytest -q`
Expected: the whole suite still passes (validated while planning: 706). Tasks 1–6 already made every column,
the generator and the compare paths multi-aware.

- [ ] **Step 2: Write the failing route and store tests**

Apply to `tests/test_web_routes_compare.py`:

```diff
--- a/tests/test_web_routes_compare.py
+++ b/tests/test_web_routes_compare.py
@@ -1,7 +1,13 @@
 """Tests for jev_bench.web.routes.compare."""
 
+from datetime import timedelta
+from typing import Any
+
+import pytest
 from fastapi.testclient import TestClient
 from tests.factories import AppFactory, FakeOpenRouter, poll, seed_generation, services_of
+
+from jev_bench.questions import ChoiceQuestion, QuestionSet
 
 
 def _finished(client: TestClient, run_id: str) -> None:
@@ -45,3 +51,75 @@
     assert missing.status_code == 404
     assert client.get("/api/compare", params={"runs": " , "}).status_code == 400
     assert client.get("/api/compare").status_code == 422
+
+
+def _topics(report: dict[str, Any]) -> dict[str, Any]:
+    return next(question for question in report["questions"] if question["id"] == "topics")
+
+
+def _label_counts(topics: dict[str, Any], rater_id: str) -> dict[str, int]:
+    return next(stats for stats in topics["raters"] if stats["rater"] == rater_id)["label_counts"]
+
+
+def test_compare_multi_label_threshold(make_app: AppFactory) -> None:
+    client = make_app(FakeOpenRouter(), api_key="sk-server")
+    _, (jev_run, chat_run) = _two_runs(client)
+    runs = f"{jev_run},{chat_run}"
+    default = _topics(client.get("/api/compare", params={"runs": runs}).json())
+    assert (default["type"], default["threshold"]) == ("multi", 0.75)
+    assert _label_counts(default, chat_run) == {"billing": 2, "meeting": 2, "travel": 0}
+    assert _label_counts(default, jev_run) == {"billing": 2, "meeting": 0, "travel": 0}
+    strict = _topics(client.get("/api/compare", params={"runs": runs, "threshold": 1}).json())
+    assert strict["threshold"] == 1.0
+    assert _label_counts(strict, chat_run) == {"billing": 2, "meeting": 0, "travel": 0}
+
+
+@pytest.mark.parametrize(
+    "threshold",
+    [
+        pytest.param("0", id="zero"),
+        pytest.param("-0.5", id="negative"),
+        pytest.param("1.01", id="above-one"),
+        pytest.param("nan", id="nan"),
+        pytest.param("inf", id="infinity"),
+        pytest.param("abc", id="not-a-number"),
+    ],
+)
+def test_compare_rejects_out_of_range_thresholds(make_app: AppFactory, threshold: str) -> None:
+    client = make_app(FakeOpenRouter())
+    params = {"runs": "20260924-100000-any-0001", "threshold": threshold}
+    assert client.get("/api/compare", params=params).status_code == 422
+
+
+def _age_with_snapshot(client: TestClient, run_id: str, snapshot: QuestionSet) -> None:
+    services = services_of(client)
+    older = services.runs.get(run_id)
+    earlier = older.created_at - timedelta(days=1)
+    services.runs.save(older.model_copy(update={"question_set": snapshot, "created_at": earlier}))
+
+
+def _v1_topics(options: dict[str, str]) -> QuestionSet:
+    topics = ChoiceQuestion(type="choice", id="topics", instructions="?", options=options)
+    return QuestionSet(name="v1", questions=(topics,))
+
+
+@pytest.mark.parametrize(
+    ("options", "skipped_older"),
+    [
+        pytest.param(
+            {"billing": "b", "meeting": "m", "travel": "t"}, False, id="v1-choice-is-reused"
+        ),
+        pytest.param({"billing": "b", "travel": "t"}, True, id="other-options-are-skipped"),
+    ],
+)
+def test_compare_reads_the_newest_runs_question_set_in_any_order(
+    make_app: AppFactory, options: dict[str, str], skipped_older: bool
+) -> None:
+    client = make_app(FakeOpenRouter(), api_key="sk-server")
+    _, run_ids = _two_runs(client)
+    _age_with_snapshot(client, run_ids[0], _v1_topics(options))
+    for order in (run_ids, run_ids[::-1]):
+        topics = _topics(client.get("/api/compare", params={"runs": ",".join(order)}).json())
+        assert topics["type"] == "multi"
+        assert topics["skipped"] == ([run_ids[0]] if skipped_older else [])
+        assert (run_ids[0] in {stats["rater"] for stats in topics["raters"]}) is not skipped_older
```

Apply to `tests/test_web_routes_emails.py`:

```diff
--- a/tests/test_web_routes_emails.py
+++ b/tests/test_web_routes_emails.py
@@ -1,5 +1,6 @@
 """Tests for jev_bench.web.routes.emails."""
 
+import pytest
 from tests.factories import (
     AppFactory,
     EmailFactory,
@@ -68,3 +69,21 @@
     assert missing.status_code == 404
     assert client.get("/api/emails/20260924-100000-missing-0001.0001").status_code == 404
     assert client.get("/api/emails/not-an-id").status_code == 404
+
+
+def test_list_emails_multi_labels_scores_and_threshold(make_app: AppFactory) -> None:
+    client = make_app(FakeOpenRouter(), api_key="sk-server")
+    generation_id = seed_generation(services_of(client))
+    run_id = client.post(
+        "/api/runs", json={"column": "anthropic", "generation_ids": [generation_id]}
+    ).json()["meta"]["id"]
+    poll(client, f"/api/runs/{run_id}", until=lambda body: body["meta"]["status"] != "running")
+    params = {"generations": generation_id, "runs": run_id}
+    row = client.get("/api/emails", params=params).json()["rows"][0]
+    assert row["top"][run_id]["topics"] == ["billing", "meeting"]
+    assert row["reference"]["topics"] == ["billing", "meeting"]
+    assert row["scores"][run_id]["urgency"] == pytest.approx(75.0)
+    assert row["reference_scores"] == {"urgency": 50.0}
+    strict = client.get("/api/emails", params={**params, "threshold": 0.8}).json()["rows"][0]
+    assert strict["top"][run_id]["topics"] == ["billing"]
+    assert client.get("/api/emails", params={**params, "threshold": 0}).status_code == 422
```

Apply to `tests/test_web_routes_labels.py`:

```diff
--- a/tests/test_web_routes_labels.py
+++ b/tests/test_web_routes_labels.py
@@ -21,10 +21,16 @@
     [
         pytest.param({"missing": "x"}, 400, "unknown question", id="unknown-question"),
         pytest.param({"category": "phishing"}, 400, "is not an option", id="unknown-option"),
+        pytest.param(
+            {"topics": ["billing", "billing"]}, 400, "is not an option", id="duplicate-labels"
+        ),
+        pytest.param({"topics": ["phishing"]}, 400, "is not an option", id="unknown-label"),
+        pytest.param({"category": ["work"]}, 400, "is not an option", id="list-for-single-choice"),
+        pytest.param({"category": ""}, 400, "is not an option", id="empty-string"),
     ],
 )
 def test_put_label_validation(
-    make_app: AppFactory, answers: dict[str, str], status: int, detail: str
+    make_app: AppFactory, answers: dict[str, object], status: int, detail: str
 ) -> None:
     client = make_app(FakeOpenRouter())
     email_id = f"{seed_generation(services_of(client))}.0001"
@@ -40,3 +46,23 @@
     assert missing.status_code == 404
     extra = client.put(f"/api/labels/{email_id}", json={"answers": {}, "extra": 1})
     assert extra.status_code == 422
+
+
+@pytest.mark.parametrize(
+    ("topics", "stored"),
+    [
+        pytest.param(["travel", "billing"], ["travel", "billing"], id="label-list"),
+        pytest.param("billing", "billing", id="single-label-id"),
+    ],
+)
+def test_put_label_accepts_multi_answers_and_clears_with_an_empty_list(
+    make_app: AppFactory, topics: object, stored: object
+) -> None:
+    client = make_app(FakeOpenRouter())
+    email_id = f"{seed_generation(services_of(client))}.0001"
+    url = f"/api/labels/{email_id}"
+    first = client.put(url, json={"answers": {"topics": topics, "category": "work"}})
+    assert first.json() == {"topics": stored, "category": "work"}
+    assert client.get(f"/api/emails/{email_id}").json()["human"]["topics"] == stored
+    cleared = client.put(url, json={"answers": {"topics": []}})
+    assert cleared.json() == {"category": "work"}
```

Apply to `tests/test_store_labels.py`:

```diff
--- a/tests/test_store_labels.py
+++ b/tests/test_store_labels.py
@@ -53,3 +53,10 @@
 async def test_invalid_email_ids_raise_key_error(tmp_path: Path, email_id: str) -> None:
     with pytest.raises(KeyError):
         await LabelStore(tmp_path).update(email_id, {"category": "spam"})
+
+
+async def test_update_stores_label_lists(tmp_path: Path) -> None:
+    store = LabelStore(tmp_path)
+    labels = {"topics": ["billing", "meeting"], "category": "spam"}
+    assert await store.update(_E1, labels) == labels
+    assert LabelStore(tmp_path).for_email(_E1) == labels
```

- [ ] **Step 3: Run the tests to verify they fail**

Run:
`uv run pytest tests/test_web_routes_compare.py tests/test_web_routes_emails.py tests/test_web_routes_labels.py tests/test_store_labels.py -q`
Expected FAILs:
- the threshold params are ignored, so the counts at `threshold=1` are wrong and `0` is not a 422;
- the newest-base test fails for the `run_ids` order, because base = the older v1 snapshot reads `topics` as
  `choice`;
- list labels get 422 from `dict[str, str | None]`.

- [ ] **Step 4: Implement**

Apply to `src/jev_bench/web/deps.py`:

```diff
--- a/src/jev_bench/web/deps.py
+++ b/src/jev_bench/web/deps.py
@@ -4,6 +4,7 @@
     NO_KEY_DETAIL
 Types:
     ServicesDep, ApiKeyDep
+    ThresholdQuery: optional `?threshold=` in (0, 1] (422 otherwise, NaN and infinity included).
 Functions:
     get_services: Services stored on the app by the lifespan.
     require_api_key: server key, else the X-OpenRouter-Key header, else HTTP 400.
@@ -12,7 +13,7 @@
 
 from typing import Annotated
 
-from fastapi import Depends, Header, HTTPException, Request
+from fastapi import Depends, Header, HTTPException, Query, Request
 
 from jev_bench.services import Services
 
@@ -20,6 +21,7 @@
     "NO_KEY_DETAIL",
     "ApiKeyDep",
     "ServicesDep",
+    "ThresholdQuery",
     "get_services",
     "require_api_key",
     "split_ids",
@@ -48,6 +50,7 @@
 
 
 ApiKeyDep = Annotated[str, Depends(require_api_key)]
+ThresholdQuery = Annotated[float | None, Query(gt=0.0, le=1.0, allow_inf_nan=False)]
 
 
 def split_ids(value: str | None) -> list[str]:
```

Apply to `src/jev_bench/web/routes/compare.py`:

```diff
--- a/src/jev_bench/web/routes/compare.py
+++ b/src/jev_bench/web/routes/compare.py
@@ -4,13 +4,18 @@
 Constants:
     LIGHT: response fields excluded from embedded run metas.
 Functions:
-    compare_runs: GET /compare?runs=a,b,c
+    compare_runs: GET /compare?runs=a,b,c[&threshold=0.8]. The base question set is the snapshot of
+        the most recent selected run (by created_at, then id), so the URL order of `runs` never
+        changes the report.
 """
+
+from collections.abc import Sequence
 
 from fastapi import APIRouter, HTTPException
 
 from jev_bench.compare import ComparisonReport, compare, human_rater, reference_rater
-from jev_bench.web.deps import ServicesDep, split_ids
+from jev_bench.store.runs import RunMeta
+from jev_bench.web.deps import ServicesDep, ThresholdQuery, split_ids
 from jev_bench.web.loaders import (
     generation_snapshots,
     generations_of,
@@ -30,11 +35,13 @@
 
 
 @router.get("/compare", response_model_exclude=LIGHT)
-def compare_runs(services: ServicesDep, runs: str) -> ComparisonReport:
+def compare_runs(
+    services: ServicesDep, runs: str, threshold: ThresholdQuery = None
+) -> ComparisonReport:
     metas = load_runs(services, split_ids(runs))
     if not metas:
         raise HTTPException(status_code=400, detail="no run ids given")
-    base = metas[0].question_set
+    base = _newest(metas).question_set
     generation_ids = generations_of(metas)
     reference = reference_rater(
         load_emails(services, generation_ids),
@@ -45,4 +52,8 @@
     human = human_rater(services.labels.for_generations(generation_ids), base)
     if human is not None:
         raters.append(human)
-    return compare(raters, base, runs={meta.id: meta for meta in metas})
+    return compare(raters, base, runs={meta.id: meta for meta in metas}, threshold=threshold)
+
+
+def _newest(metas: Sequence[RunMeta]) -> RunMeta:
+    return max(metas, key=lambda meta: (meta.created_at, meta.id))
```

Apply to `src/jev_bench/web/routes/emails.py`:

```diff
--- a/src/jev_bench/web/routes/emails.py
+++ b/src/jev_bench/web/routes/emails.py
@@ -4,7 +4,7 @@
     EmailList: current question set + rows.
     EmailDetail: email, human labels, predictions per run, run labels, current question set.
 Functions:
-    list_emails: GET /emails?generations=…&runs=…
+    list_emails: GET /emails?generations=…&runs=…[&threshold=0.8]
     email_detail: GET /emails/{email_id}?runs=…
 """
 
@@ -13,10 +13,10 @@
 
 from jev_bench.compare import EmailRow, email_rows, run_label
 from jev_bench.emails import Email
-from jev_bench.questions import QuestionSet
+from jev_bench.questions import HardAnswer, QuestionSet
 from jev_bench.services import Services
 from jev_bench.store.runs import Prediction
-from jev_bench.web.deps import ServicesDep, split_ids
+from jev_bench.web.deps import ServicesDep, ThresholdQuery, split_ids
 from jev_bench.web.loaders import load_email, load_emails, load_runs, run_raters
 
 __all__ = [
@@ -37,20 +37,26 @@
 
 class EmailDetail(BaseModel):
     email: Email
-    human: dict[str, str]
+    human: dict[str, HardAnswer]
     predictions: dict[str, Prediction]
     run_labels: dict[str, str]
     questions: QuestionSet
 
 
 @router.get("/emails")
-def list_emails(services: ServicesDep, generations: str, runs: str | None = None) -> EmailList:
+def list_emails(
+    services: ServicesDep,
+    generations: str,
+    runs: str | None = None,
+    threshold: ThresholdQuery = None,
+) -> EmailList:
     generation_ids = split_ids(generations)
     questions = services.question_set()
     emails = load_emails(services, generation_ids)
     raters = run_raters(services, load_runs(services, split_ids(runs)))
     labels = services.labels.for_generations(generation_ids)
-    return EmailList(questions=questions, rows=email_rows(emails, raters, labels, questions))
+    rows = email_rows(emails, raters, labels, questions, threshold=threshold)
+    return EmailList(questions=questions, rows=rows)
 
 
 @router.get("/emails/{email_id}")
```

Apply to `src/jev_bench/web/routes/labels.py`:

```diff
--- a/src/jev_bench/web/routes/labels.py
+++ b/src/jev_bench/web/routes/labels.py
@@ -1,7 +1,8 @@
 """Human labelling route: set or clear answers for one email, validated against the question set.
 
 Classes:
-    LabelUpdate: `{answers: {question_id: option_id | null}}`.
+    LabelUpdate: `{answers: {question_id: option_id | [option_id, ...] | null}}`; a list is for a
+        multi-label question, and `null` or `[]` clears the answer.
 Functions:
     put_label: PUT /labels/{email_id}
 """
@@ -11,7 +12,7 @@
 from fastapi import APIRouter, HTTPException
 from pydantic import BaseModel, ConfigDict
 
-from jev_bench.questions import QuestionSet
+from jev_bench.questions import AnyQuestion, HardAnswer, QuestionSet, hard_distribution
 from jev_bench.web.deps import ServicesDep
 from jev_bench.web.loaders import load_email
 
@@ -27,23 +28,35 @@
 class LabelUpdate(BaseModel):
     model_config = ConfigDict(extra="forbid")
 
-    answers: dict[str, str | None]
+    answers: dict[str, HardAnswer | None]
 
 
 @router.put("/labels/{email_id}")
-async def put_label(email_id: str, update: LabelUpdate, services: ServicesDep) -> dict[str, str]:
+async def put_label(
+    email_id: str, update: LabelUpdate, services: ServicesDep
+) -> dict[str, HardAnswer]:
     load_email(services, email_id)
-    _validate(update.answers, services.question_set())
-    return await services.labels.update(email_id, update.answers)
+    changes = _validated(update.answers, services.question_set())
+    return await services.labels.update(email_id, changes)
 
 
-def _validate(answers: Mapping[str, str | None], questions: QuestionSet) -> None:
-    for question_id, option in answers.items():
-        try:
-            question = questions.get(question_id)
-        except KeyError as exc:
-            detail = f"unknown question {question_id!r}"
-            raise HTTPException(status_code=400, detail=detail) from exc
-        if option is not None and option not in question.options:
-            detail = f"{option!r} is not an option of {question_id!r}"
+def _validated(
+    answers: Mapping[str, HardAnswer | None], questions: QuestionSet
+) -> dict[str, HardAnswer | None]:
+    changes = {
+        question_id: None if answer == [] else answer for question_id, answer in answers.items()
+    }
+    for question_id, answer in changes.items():
+        question = _question(questions, question_id)
+        if answer is not None and hard_distribution(question, answer) is None:
+            detail = f"{answer!r} is not an option of {question_id!r}"
             raise HTTPException(status_code=400, detail=detail)
+    return changes
+
+
+def _question(questions: QuestionSet, question_id: str) -> AnyQuestion:
+    try:
+        return questions.get(question_id)
+    except KeyError as exc:
+        detail = f"unknown question {question_id!r}"
+        raise HTTPException(status_code=400, detail=detail) from exc
```

Apply to `src/jev_bench/store/labels.py`:

```diff
--- a/src/jev_bench/store/labels.py
+++ b/src/jev_bench/store/labels.py
@@ -1,4 +1,5 @@
-"""Human labels per generation: `{email_id: {question_id: option_id}}`, edited atomically.
+"""Human labels per generation: `{email_id: {question_id: option_id | [option_id, ...]}}`, edited
+atomically (a list is the answer to a multi-label question).
 
 Classes:
     LabelStore: read labels per email or generation; apply edits under a lock.
@@ -9,6 +10,7 @@
 from pathlib import Path
 
 from jev_bench.ids import is_safe_id, split_email_id
+from jev_bench.questions import HardAnswer
 from jev_bench.store.jsonfiles import read_json, write_json_atomic
 
 __all__ = [
@@ -16,7 +18,7 @@
     "Labels",
 ]
 
-type Labels = dict[str, dict[str, str]]
+type Labels = dict[str, dict[str, HardAnswer]]
 
 
 class LabelStore:
@@ -30,11 +32,13 @@
             merged.update(self._read(generation_id))
         return merged
 
-    def for_email(self, email_id: str) -> dict[str, str]:
+    def for_email(self, email_id: str) -> dict[str, HardAnswer]:
         generation_id, _ = split_email_id(email_id)
         return self._read(generation_id).get(email_id, {})
 
-    async def update(self, email_id: str, changes: Mapping[str, str | None]) -> dict[str, str]:
+    async def update(
+        self, email_id: str, changes: Mapping[str, HardAnswer | None]
+    ) -> dict[str, HardAnswer]:
         generation_id, _ = split_email_id(email_id)
         async with self._lock:
             labels = self._read(generation_id)
@@ -56,7 +60,9 @@
         return self._root / f"{generation_id}.json"
 
 
-def _apply(current: Mapping[str, str], changes: Mapping[str, str | None]) -> dict[str, str]:
+def _apply(
+    current: Mapping[str, HardAnswer], changes: Mapping[str, HardAnswer | None]
+) -> dict[str, HardAnswer]:
     updated = dict(current)
     for question_id, option in changes.items():
         if option is None:
```

`LabelStore` never validates; validation stays at the route boundary.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -q --durations=5`
Expected: the whole suite passes (validated while planning: 723). None of the new tests is among the slowest
five, which are the pre-existing Hypothesis tests.

- [ ] **Step 6: Gates, then commit**

```bash
uv run ruff check --fix && uv run ruff format && uv run pyright
git add tests/factories.py src/jev_bench/web/deps.py src/jev_bench/web/routes/compare.py \
  src/jev_bench/web/routes/emails.py src/jev_bench/web/routes/labels.py src/jev_bench/store/labels.py \
  tests/test_web_routes_compare.py tests/test_web_routes_emails.py tests/test_web_routes_labels.py \
  tests/test_store_labels.py
git commit -m "feat(web): label threshold query, newest-run base question set, multi-label human labels

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
