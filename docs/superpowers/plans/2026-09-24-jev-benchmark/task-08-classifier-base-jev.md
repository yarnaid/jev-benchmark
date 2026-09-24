### Task 8: Classifier contract and the Jev classifier

**Files:**
- Create: `src/jev_bench/classifiers/__init__.py`, `src/jev_bench/classifiers/base.py`,
  `src/jev_bench/classifiers/jev.py`
- Test: `tests/test_classifiers_base.py`, `tests/test_classifiers_jev.py`

**Interfaces:**
- Consumes:
  - `Email` and `EmailState` (Task 2);
  - `QuestionSet`, `NoulQuestion`, `ScoreQuestion`, `Distribution` and `one_hot` (Task 2);
  - `normalize` and `unit_probability` (Task 4);
  - `OpenRouterClient` and `JsonObject` (Task 5);
  - `ModelInfo` and `JevParams`/`TokenParams` (Task 6);
  - `Budget`, `jev_budget`, `estimate_tokens`, `Sizing` (Task 7).
- Produces:
  - `jev_bench.classifiers.base`:
    - `Usage(input_tokens, output_tokens, cost, cost_estimated)` with `.plus(other)`;
    - `EmailOutcome(answers, error, notes, similarities, cached, cached_cost)`;
    - `PrepareResult(usage, cached_cost, resolved, pending)`;
    - `RequestResult(outcomes, usage, latency_ms, resolved_model, raw, error)`;
    - `ProgressCallback`;
    - `Classifier` (a `Protocol` with `emails_per_request`, `concurrency`, `budget`, `sizing`,
      `input_tokens(email)`, `await prepare(emails)` and `await classify(emails, on_progress=None)`);
    - `usage_from_body(usage, pricing) -> Usage`;
    - `failed_result(emails, message) -> RequestResult`;
    - `outcome_from_parsed(answers, notes) -> EmailOutcome`.
  - `jev_bench.classifiers.jev`:
    - `DECISIONS_PATH = "/alpha/decisions"`;
    - `questions_payload(qs) -> dict[str, JsonObject]`;
    - `parse_decisions(answers, qs) -> tuple[dict[str, Distribution], list[str]]`;
    - `JevClassifier(*, client, api_key, model, model_info, questions, params: JevParams,
      tokens: TokenParams)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_classifiers_base.py`:
```python
"""Tests for jev_bench.classifiers.base."""

from typing import Any

import pytest

from jev_bench.catalog import ModelInfo
from jev_bench.classifiers.base import EmailOutcome, Usage, failed_result, outcome_from_parsed, usage_from_body
from tests.factories import EmailFactory

_PRICED = ModelInfo(id="m", name="m", prompt_price=0.001, completion_price=0.002)


@pytest.mark.parametrize(
    ("usage", "pricing", "expected"),
    [
        pytest.param({"input_tokens": 476, "output_tokens": 70, "cost": 0.00002}, None, Usage(input_tokens=476, output_tokens=70, cost=0.00002), id="decisions-style"),
        pytest.param({"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.5}, None, Usage(input_tokens=10, output_tokens=5, cost=0.5), id="chat-style"),
        pytest.param({"prompt_tokens": 10, "total_tokens": 10}, _PRICED, Usage(input_tokens=10, cost=0.01, cost_estimated=True), id="missing-cost-estimated"),
        pytest.param(None, None, Usage(cost_estimated=True), id="no-usage-no-pricing"),
        pytest.param({"prompt_tokens": 3, "cost": True}, _PRICED, Usage(input_tokens=3, cost=0.003, cost_estimated=True), id="bool-cost-ignored"),
    ],
)
def test_usage_from_body(usage: dict[str, Any] | None, pricing: ModelInfo | None, expected: Usage) -> None:
    result = usage_from_body(usage, pricing)
    assert result.model_copy(update={"cost": round(result.cost, 9)}) == expected


def test_usage_plus() -> None:
    total = Usage(input_tokens=1, output_tokens=2, cost=0.1).plus(Usage(input_tokens=3, cost=0.2, cost_estimated=True))
    assert (total.input_tokens, total.output_tokens, total.cost_estimated) == (4, 2, True)
    assert total.cost == pytest.approx(0.3)


def test_failed_result_marks_every_email() -> None:
    emails = [EmailFactory(), EmailFactory()]
    result = failed_result(emails, "HTTP 500: boom")
    assert result.error == "HTTP 500: boom"
    assert {email_id: outcome.error for email_id, outcome in result.outcomes.items()} == {
        emails[0].id: "HTTP 500: boom",
        emails[1].id: "HTTP 500: boom",
    }


@pytest.mark.parametrize(
    ("answers", "notes", "expected"),
    [
        pytest.param({"q": {"a": 1.0}}, ["x: missing"], EmailOutcome(answers={"q": {"a": 1.0}}, notes=("x: missing",)), id="partial"),
        pytest.param({}, ["q: missing", "x: missing"], EmailOutcome(error="q: missing; x: missing", notes=("q: missing", "x: missing")), id="nothing-parsed"),
        pytest.param({}, [], EmailOutcome(error="no answers"), id="nothing-at-all"),
    ],
)
def test_outcome_from_parsed(answers: dict[str, dict[str, float]], notes: list[str], expected: EmailOutcome) -> None:
    assert outcome_from_parsed(answers, notes) == expected
```

`tests/test_classifiers_jev.py`:
```python
"""Tests for jev_bench.classifiers.jev."""

import json
import math
from typing import Any

import httpx2
import pytest

from jev_bench.benchmark_config import JevParams, TokenParams
from jev_bench.catalog import ModelInfo
from jev_bench.classifiers.jev import JevClassifier, parse_decisions, questions_payload
from jev_bench.questions import QuestionSet
from jev_bench.tokens import Budget, estimate_tokens
from tests.factories import ClientFactory, EmailFactory

_ANSWERS: dict[str, Any] = {
    "category": {"type": "choice", "choice": "spam", "confidence": 0.75, "probabilities": {"spam": 0.84, "personal": 0.16, "work": 0}},
    "urgency": {"type": "score", "score": 1.99, "confidence": 0.99, "probabilities": {"0": 0, "1": 0.01, "2": 0.99}},
    "needs_reply": {"type": "noul", "noul": 0.96},
}
_EXPECTED = {
    "category": {"spam": 0.84, "personal": 0.16, "work": 0.0},
    "urgency": {"low": 0.0, "today": 0.01, "now": 0.99},
    "needs_reply": {"yes": 0.96, "no": 0.04},
}


def test_questions_payload(questions: QuestionSet) -> None:
    assert questions_payload(questions) == {
        "category": {"type": "choice", "instructions": "What kind of email?", "criteria": {"spam": "Junk", "personal": "From a friend", "work": "From a colleague"}},
        "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["Whenever", "Within a day", "Immediately"]},
        "needs_reply": {"type": "noul", "instructions": "Needs a reply?", "criteria": {"true": "Reply expected", "false": "No reply expected"}},
    }


def test_parse_decisions_full_answer(questions: QuestionSet) -> None:
    parsed, notes = parse_decisions(_ANSWERS, questions)
    assert notes == []
    assert parsed == {key: pytest.approx(value) for key, value in _EXPECTED.items()}


@pytest.mark.parametrize(
    ("override", "question", "expected", "note"),
    [
        pytest.param({"type": "choice", "choice": "work"}, "category", {"spam": 0.0, "personal": 0.0, "work": 1.0}, "one-hot on choice", id="choice-without-probabilities"),
        pytest.param({"type": "choice", "choice": "unknown"}, "category", None, "no usable choice", id="choice-unknown"),
        pytest.param({"type": "score", "score": 1.6}, "urgency", {"low": 0.0, "today": 0.0, "now": 1.0}, "round(score)", id="score-without-probabilities"),
        pytest.param({"type": "score", "score": 7.0}, "urgency", {"low": 0.0, "today": 0.0, "now": 1.0}, "round(score)", id="score-clamped-high"),
        pytest.param({"type": "score", "score": math.nan}, "urgency", None, "no usable score", id="score-nan"),
        pytest.param({"type": "score", "probabilities": {"0": 1, "2": 3}}, "urgency", {"low": 0.25, "today": 0.0, "now": 0.75}, None, id="score-partial-probabilities"),
        pytest.param({"type": "noul", "noul": 1.3}, "needs_reply", {"yes": 1.0, "no": 0.0}, None, id="noul-clipped"),
        pytest.param({"type": "noul"}, "needs_reply", None, "noul probability missing", id="noul-missing"),
        pytest.param({"type": "choice", "choice": "spam"}, "needs_reply", None, "missing or mistyped", id="wrong-type"),
    ],
)
def test_parse_decisions_degraded_answers(
    questions: QuestionSet, override: dict[str, Any], question: str, expected: dict[str, float] | None, note: str | None
) -> None:
    parsed, notes = parse_decisions({**_ANSWERS, question: override}, questions)
    assert parsed.get(question) == (None if expected is None else pytest.approx(expected))
    if note is None:
        assert notes == []
    else:
        assert len(notes) == 1
        assert notes[0].startswith(f"{question}: ")
        assert note in notes[0]


def test_parse_decisions_missing_question(questions: QuestionSet) -> None:
    parsed, notes = parse_decisions({"category": _ANSWERS["category"]}, questions)
    assert set(parsed) == {"category"}
    assert notes == ["urgency: missing or mistyped answer", "needs_reply: missing or mistyped answer"]


def _classifier(client: Any, questions: QuestionSet, info: ModelInfo | None = None) -> JevClassifier:
    return JevClassifier(
        client=client,
        api_key="sk-test",
        model="typesafe/jev-1.13",
        model_info=info,
        questions=questions,
        params=JevParams(concurrency=3),
        tokens=TokenParams(),
    )


async def test_classify_sends_state_and_parses(make_client: ClientFactory, questions: QuestionSet) -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        body = {"id": "gen-dec-1", "model": "typesafe/jev-1.13-20260917", "answers": _ANSWERS, "usage": {"input_tokens": 476, "output_tokens": 70, "cost": 0.00002}}
        return httpx2.Response(200, json=body)

    email = EmailFactory()
    result = await _classifier(make_client(handler), questions).classify([email])
    sent = json.loads(seen[0].content)
    assert seen[0].url.path == "/api/alpha/decisions"
    assert seen[0].headers["Authorization"] == "Bearer sk-test"
    assert sent == {"model": "typesafe/jev-1.13", "state": email.to_state(), "questions": questions_payload(questions)}
    assert result.outcomes[email.id].answers == {key: pytest.approx(value) for key, value in _EXPECTED.items()}
    assert result.usage.cost == 0.00002
    assert result.usage.input_tokens == 476
    assert result.resolved_model == "typesafe/jev-1.13-20260917"
    assert result.raw is not None
    assert result.raw["id"] == "gen-dec-1"


async def test_classify_without_answers_is_an_email_error(make_client: ClientFactory, questions: QuestionSet) -> None:
    client = make_client(lambda request: httpx2.Response(200, json={"model": "m", "answers": {}, "usage": {}}))
    email = EmailFactory()
    outcome = (await _classifier(client, questions).classify([email])).outcomes[email.id]
    assert outcome.answers is None
    assert outcome.error is not None
    assert "missing or mistyped" in outcome.error


async def test_classifier_shape(make_client: ClientFactory, questions: QuestionSet) -> None:
    info = ModelInfo(id="typesafe/jev-1.13", name="Jev", context_length=32_000)
    classifier = _classifier(make_client(lambda request: httpx2.Response(500)), questions, info)
    email = EmailFactory()
    prepared = await classifier.prepare([email])
    assert classifier.emails_per_request == 1
    assert classifier.concurrency == 3
    assert classifier.budget == Budget(total=32_000)
    assert classifier.sizing.output_per_email == 1000
    assert classifier.sizing.overhead == estimate_tokens(json.dumps(questions_payload(questions), ensure_ascii=False), 3.0)
    assert classifier.input_tokens(email) == estimate_tokens(json.dumps(email.to_state(), ensure_ascii=False), 3.0)
    assert prepared.pending == (email,)
    assert prepared.resolved == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_classifiers_base.py tests/test_classifiers_jev.py -v`
Expected: `ModuleNotFoundError: No module named 'jev_bench.classifiers'`.

- [ ] **Step 3: Write the implementation**

`src/jev_bench/classifiers/__init__.py`:
```python
"""Column classifiers: Jev decisions, chat LLMs, embedding similarity."""
```

`src/jev_bench/classifiers/base.py`:
```python
"""Classifier contract shared by the Jev, chat-LLM and embedding columns.

Types:
    ProgressCallback
Classes:
    Usage: tokens and cost of one request (or a sum of requests).
    EmailOutcome: answers (or an error) for one email.
    PrepareResult: one-off setup outcome: paid usage, cache-resolved emails, emails still to request.
    RequestResult: outcomes of one request plus its usage, latency and raw body.
    Classifier: protocol implemented by every column kind.
Functions:
    usage_from_body: OpenRouter `usage` (any endpoint's key style) -> Usage, estimating a missing cost.
    failed_result: RequestResult in which every email carries the same error.
    outcome_from_parsed: parsed answers + notes -> EmailOutcome (error when nothing parsed).
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from pydantic import BaseModel, Field

from jev_bench.catalog import ModelInfo
from jev_bench.emails import Email
from jev_bench.questions import Distribution
from jev_bench.request_plan import Sizing
from jev_bench.tokens import Budget

type ProgressCallback = Callable[[int], None]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    cost_estimated: bool = False

    def plus(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost=self.cost + other.cost,
            cost_estimated=self.cost_estimated or other.cost_estimated,
        )


class EmailOutcome(BaseModel):
    answers: dict[str, Distribution] | None = None
    error: str | None = None
    notes: tuple[str, ...] = ()
    similarities: dict[str, dict[str, float]] | None = None
    cached: bool = False
    cached_cost: float = 0.0


class PrepareResult(BaseModel):
    usage: Usage = Field(default_factory=Usage)
    cached_cost: float = 0.0
    resolved: dict[str, EmailOutcome] = Field(default_factory=dict)
    pending: tuple[Email, ...] = ()


class RequestResult(BaseModel):
    outcomes: dict[str, EmailOutcome]
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float | None = None
    resolved_model: str | None = None
    raw: dict[str, Any] | None = None
    error: str | None = None


class Classifier(Protocol):
    @property
    def emails_per_request(self) -> int | None: ...

    @property
    def concurrency(self) -> int: ...

    @property
    def budget(self) -> Budget: ...

    @property
    def sizing(self) -> Sizing: ...

    def input_tokens(self, email: Email) -> int: ...

    async def prepare(self, emails: Sequence[Email]) -> PrepareResult: ...

    async def classify(
        self, emails: Sequence[Email], on_progress: ProgressCallback | None = None
    ) -> RequestResult: ...


def usage_from_body(usage: Mapping[str, Any] | None, pricing: ModelInfo | None) -> Usage:
    raw = usage or {}
    input_tokens = _count(raw, "input_tokens", "prompt_tokens")
    output_tokens = _count(raw, "output_tokens", "completion_tokens")
    cost = raw.get("cost")
    if isinstance(cost, int | float) and not isinstance(cost, bool):
        return Usage(input_tokens=input_tokens, output_tokens=output_tokens, cost=float(cost))
    estimate = pricing.estimate_cost(input_tokens, output_tokens) if pricing else 0.0
    return Usage(input_tokens=input_tokens, output_tokens=output_tokens, cost=estimate, cost_estimated=True)


def _count(raw: Mapping[str, Any], *keys: str) -> int:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return 0


def failed_result(emails: Sequence[Email], message: str) -> RequestResult:
    outcomes = {email.id: EmailOutcome(error=message) for email in emails}
    return RequestResult(outcomes=outcomes, error=message)


def outcome_from_parsed(answers: dict[str, Distribution], notes: Sequence[str]) -> EmailOutcome:
    if answers:
        return EmailOutcome(answers=answers, notes=tuple(notes))
    return EmailOutcome(error="; ".join(notes) or "no answers", notes=tuple(notes))
```

`src/jev_bench/classifiers/jev.py`:
```python
"""Jev column: one Decisions API call per email carrying every question.

Constants:
    DECISIONS_PATH
Classes:
    JevClassifier
Functions:
    questions_payload: QuestionSet -> Decisions `questions` object.
    parse_decisions: Decisions `answers` -> canonical distributions and per-question notes.
"""

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from jev_bench.benchmark_config import JevParams, TokenParams
from jev_bench.catalog import ModelInfo
from jev_bench.classifiers.base import (
    PrepareResult,
    ProgressCallback,
    RequestResult,
    outcome_from_parsed,
    usage_from_body,
)
from jev_bench.emails import Email
from jev_bench.metrics.distributions import normalize, unit_probability
from jev_bench.openrouter import JsonObject, OpenRouterClient
from jev_bench.questions import AnyQuestion, Distribution, NoulQuestion, QuestionSet, ScoreQuestion, one_hot
from jev_bench.request_plan import Sizing
from jev_bench.tokens import estimate_tokens, jev_budget

DECISIONS_PATH = "/alpha/decisions"

type Parsed = tuple[Distribution | None, str | None]


def questions_payload(questions: QuestionSet) -> dict[str, JsonObject]:
    return {question.id: _question_payload(question) for question in questions.questions}


def _question_payload(question: AnyQuestion) -> JsonObject:
    criteria: object
    if isinstance(question, NoulQuestion):
        criteria = {"true": question.options["yes"], "false": question.options["no"]}
    elif isinstance(question, ScoreQuestion):
        criteria = list(question.options.values())
    else:
        criteria = dict(question.options)
    return {"type": question.type, "instructions": question.instructions, "criteria": criteria}


def parse_decisions(
    answers: Mapping[str, Any], questions: QuestionSet
) -> tuple[dict[str, Distribution], list[str]]:
    parsed: dict[str, Distribution] = {}
    notes: list[str] = []
    for question in questions.questions:
        answer = answers.get(question.id)
        if not isinstance(answer, dict) or answer.get("type") != question.type:
            notes.append(f"{question.id}: missing or mistyped answer")
            continue
        distribution, note = _parse_answer(question, answer)
        if note:
            notes.append(f"{question.id}: {note}")
        if distribution is not None:
            parsed[question.id] = distribution
    return parsed, notes


def _parse_answer(question: AnyQuestion, answer: Mapping[str, Any]) -> Parsed:
    if isinstance(question, NoulQuestion):
        return _parse_noul(answer)
    if isinstance(question, ScoreQuestion):
        return _parse_score(question, answer)
    return _parse_choice(question, answer)


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return float(value)


def _parse_noul(answer: Mapping[str, Any]) -> Parsed:
    probability = unit_probability(answer.get("noul"))
    if probability is None:
        return None, "noul probability missing"
    return {"yes": probability, "no": 1.0 - probability}, None


def _parse_choice(question: AnyQuestion, answer: Mapping[str, Any]) -> Parsed:
    probabilities = answer.get("probabilities")
    if isinstance(probabilities, dict) and (dist := normalize(probabilities, question.option_ids)):
        return dist, None
    choice = answer.get("choice")
    if isinstance(choice, str) and choice in question.options:
        return one_hot(question, choice), "probabilities missing, one-hot on choice"
    return None, "no usable choice"


def _parse_score(question: ScoreQuestion, answer: Mapping[str, Any]) -> Parsed:
    probabilities = answer.get("probabilities")
    if isinstance(probabilities, dict):
        by_level = {level: probabilities.get(str(index)) for index, level in enumerate(question.option_ids)}
        if dist := normalize(by_level, question.option_ids):
            return dist, None
    score = _finite(answer.get("score"))
    if score is None:
        return None, "no usable score"
    index = min(len(question.option_ids) - 1, max(0, round(score)))
    return one_hot(question, question.option_ids[index]), "probabilities missing, one-hot at round(score)"


class JevClassifier:
    def __init__(
        self,
        *,
        client: OpenRouterClient,
        api_key: str,
        model: str,
        model_info: ModelInfo | None,
        questions: QuestionSet,
        params: JevParams,
        tokens: TokenParams,
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._model = model
        self._info = model_info
        self._questions = questions
        self._payload = questions_payload(questions)
        self._bytes_per_token = tokens.bytes_per_token
        self.emails_per_request: int | None = 1
        self.concurrency = params.concurrency
        self.budget = jev_budget(model_info, tokens)
        overhead = estimate_tokens(json.dumps(self._payload, ensure_ascii=False), tokens.bytes_per_token)
        self.sizing = Sizing(overhead=overhead, output_per_email=tokens.jev_output_reserve)

    def input_tokens(self, email: Email) -> int:
        return estimate_tokens(json.dumps(email.to_state(), ensure_ascii=False), self._bytes_per_token)

    async def prepare(self, emails: Sequence[Email]) -> PrepareResult:
        return PrepareResult(pending=tuple(emails))

    async def classify(
        self, emails: Sequence[Email], on_progress: ProgressCallback | None = None
    ) -> RequestResult:
        (email,) = emails
        body = {"model": self._model, "state": email.to_state(), "questions": self._payload}
        response = await self._client.post_json(DECISIONS_PATH, body, api_key=self._api_key)
        answers, notes = parse_decisions(response.body.get("answers") or {}, self._questions)
        return RequestResult(
            outcomes={email.id: outcome_from_parsed(answers, notes)},
            usage=usage_from_body(response.body.get("usage"), self._info),
            latency_ms=response.latency_ms,
            resolved_model=response.body.get("model"),
            raw=response.body,
        )
```

- [ ] **Step 4: Run tests to verify they pass, lint, type-check**

Run: `uv run pytest tests/test_classifiers_base.py tests/test_classifiers_jev.py -v`
Expected: all PASS.

Run: `uv run ruff check --fix && uv run ruff format && uv run pyright`
Expected: `0 errors`.

- [ ] **Step 5: Commit**

```bash
git add src/jev_bench/classifiers tests/test_classifiers_base.py tests/test_classifiers_jev.py
git commit -m "feat(classifiers): classifier contract and Jev decisions classifier

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
