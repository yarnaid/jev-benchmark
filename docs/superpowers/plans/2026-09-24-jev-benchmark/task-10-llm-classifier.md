### Task 10: Chat-model classifier (per email / all in one)

**Files:**
- Create: `src/jev_bench/classifiers/llm.py`
- Modify: `tests/factories.py` (add `chat_body` and `sse_body` response builders)
- Test: `tests/test_classifiers_llm.py`

**Interfaces:**
- Consumes:
  - Task 2: `Email`, `QuestionSet`, `render_questions`;
  - Task 5: `OpenRouterClient`, `OpenRouterError`, `ApiResponse`, `ChatContentError`, `CHAT_PATH`,
    `chat_content`, `json_schema_format`, `JsonObject`;
  - Task 6: `ModelInfo`, `LlmParams`, `TokenParams`, `ChatMode`;
  - Task 7: `chat_budget`, `estimate_tokens`, `Sizing`;
  - Task 8: `EmailOutcome`, `PrepareResult`, `RequestResult`, `ProgressCallback`,
    `outcome_from_parsed`, `usage_from_body`;
  - Task 9: `answers_schema`, `all_in_one_schema`, `email_refs`, `parse_email_answers`,
    `split_results`, `RefCounter`.
- Produces:
  - `jev_bench.classifiers.llm.LlmClassifier(*, client, api_key, model, model_info, questions,
    params: LlmParams, tokens: TokenParams, mode: ChatMode, cache_system_prompt: bool)`;
  - it implements the `Classifier` protocol: `emails_per_request` is `1` in per_email mode and `None`
    in all_in_one mode;
  - `tests.factories.chat_body(content, *, finish_reason="stop", model="test/model", cost=0.001) ->
    dict`;
  - `tests.factories.sse_body(parts, *, finish_reason="stop", model="test/model", cost=0.002) -> bytes`.

Request body (both modes):
- `model`, `temperature`, `reasoning: {enabled}`, `response_format` (strict json_schema),
  `provider: {require_parameters: true}`;
- `max_tokens = max(1, min(budget.output, budget.total − estimated_input))`;
- `messages`: a system message and a user message:
  - per_email: the user content is the email state JSON;
  - all_in_one: the user content is a JSON array of `{ref, ...state}`;
  - the system message is a plain string, or, when `cache_system_prompt`, a single text part with
    `cache_control: {type: "ephemeral"}`.

All-in-one requests use `post_stream` inside `asyncio.timeout(all_in_one_timeout_s)`. A timeout becomes
`OpenRouterError` (not retryable), which the runner turns into a per-email error for the whole request.

- [ ] **Step 1: Add response builders to `tests/factories.py`**

Add `import json`, `from collections.abc import Sequence` and `from typing import Any` to the imports,
list both functions in the docstring, and append:
```python
def chat_body(content: str, *, finish_reason: str = "stop", model: str = "test/model", cost: float = 0.001) -> dict[str, Any]:
    return {
        "id": "gen-test",
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": cost},
    }


def sse_body(parts: Sequence[str], *, finish_reason: str = "stop", model: str = "test/model", cost: float = 0.002) -> bytes:
    chunks: list[dict[str, Any]] = [
        {"id": "gen-test", "model": model, "choices": [{"delta": {"content": part}}]} for part in parts
    ]
    chunks.append(
        {
            "choices": [{"delta": {"content": ""}, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": cost},
        }
    )
    return "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks).encode() + b"data: [DONE]\n\n"
```

- [ ] **Step 2: Write the failing tests**

`tests/test_classifiers_llm.py`:
````python
"""Tests for jev_bench.classifiers.llm."""

import asyncio
import json
from typing import Any

import httpx2
import pytest

from jev_bench.benchmark_config import ChatMode, LlmParams, TokenParams
from jev_bench.catalog import ModelInfo
from jev_bench.classifiers.llm import LlmClassifier
from jev_bench.classifiers.llm_schema import all_in_one_schema, answers_schema
from jev_bench.emails import Email
from jev_bench.openrouter import OpenRouterClient, OpenRouterError
from jev_bench.questions import QuestionSet, render_questions
from tests.factories import ClientFactory, EmailFactory, chat_body, sse_body

_INFO = ModelInfo(id="anthropic/claude-sonnet-5", name="Sonnet", context_length=1_000_000, max_completion_tokens=128_000)
_ANSWER = {"category": {"spam": 0.6, "personal": 0.3, "work": 0.1}, "urgency": {"low": 0.2, "today": 0.5, "now": 0.3}, "needs_reply": 0.8}


def _params(**overrides: Any) -> LlmParams:
    return LlmParams(system_prompt="Classify.\n$questions", system_prompt_all_in_one="Batch.\n$questions", **overrides)


def _classifier(
    client: OpenRouterClient, questions: QuestionSet, mode: ChatMode = "per_email", *, cache: bool = False, **params: Any
) -> LlmClassifier:
    return LlmClassifier(
        client=client,
        api_key="sk-test",
        model="anthropic/claude-sonnet-5",
        model_info=_INFO,
        questions=questions,
        params=_params(**params),
        tokens=TokenParams(),
        mode=mode,
        cache_system_prompt=cache,
    )


class Recorder:
    def __init__(self, response: httpx2.Response) -> None:
        self.response = response
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.bodies.append(json.loads(request.content))
        return self.response


def _json_response(content: str, **kwargs: Any) -> httpx2.Response:
    return httpx2.Response(200, json=chat_body(content, **kwargs))


def _sse_response(parts: list[str], **kwargs: Any) -> httpx2.Response:
    return httpx2.Response(200, content=sse_body(parts, **kwargs), headers={"content-type": "text/event-stream"})


async def test_per_email_request_body(make_client: ClientFactory, questions: QuestionSet) -> None:
    recorder = Recorder(_json_response(json.dumps(_ANSWER)))
    classifier = _classifier(make_client(recorder), questions)
    email = EmailFactory()
    await classifier.classify([email])
    body = recorder.bodies[0]
    expected_input = classifier.sizing.overhead + classifier.input_tokens(email)
    assert body["model"] == "anthropic/claude-sonnet-5"
    assert body["messages"] == [
        {"role": "system", "content": "Classify.\n" + render_questions(questions)},
        {"role": "user", "content": json.dumps(email.to_state(), ensure_ascii=False)},
    ]
    assert body["temperature"] == 0.0
    assert body["reasoning"] == {"enabled": False}
    assert body["provider"] == {"require_parameters": True}
    assert body["response_format"]["json_schema"] == {"name": "email_triage", "strict": True, "schema": answers_schema(questions)}
    assert body["max_tokens"] == min(128_000, 1_000_000 - expected_input)
    assert "stream" not in body


async def test_cached_system_prompt_uses_a_cache_control_part(make_client: ClientFactory, questions: QuestionSet) -> None:
    recorder = Recorder(_json_response(json.dumps(_ANSWER)))
    await _classifier(make_client(recorder), questions, cache=True).classify([EmailFactory()])
    system = recorder.bodies[0]["messages"][0]
    assert system["content"] == [{"type": "text", "text": "Classify.\n" + render_questions(questions), "cache_control": {"type": "ephemeral"}}]


@pytest.mark.parametrize(
    ("content", "finish_reason", "error"),
    [
        pytest.param(json.dumps(_ANSWER), "stop", None, id="valid"),
        pytest.param("```json\n" + json.dumps(_ANSWER) + "\n```", "stop", None, id="code-fence"),
        pytest.param("{not json", "stop", "response is not valid JSON", id="invalid-json"),
        pytest.param('{"category": {"spam"', "length", "truncated", id="truncated"),
        pytest.param("   ", "stop", "empty response", id="blank"),
    ],
)
async def test_per_email_outcomes(
    make_client: ClientFactory, questions: QuestionSet, content: str, finish_reason: str, error: str | None
) -> None:
    email = EmailFactory()
    result = await _classifier(make_client(Recorder(_json_response(content, finish_reason=finish_reason))), questions).classify([email])
    outcome = result.outcomes[email.id]
    assert result.usage.cost == 0.001
    assert result.resolved_model == "test/model"
    if error is None:
        assert outcome.answers is not None
        assert outcome.answers["needs_reply"] == pytest.approx({"yes": 0.8, "no": 0.2})
    else:
        assert outcome.answers is None
        assert outcome.error is not None
        assert error in outcome.error


async def test_all_in_one_streams_refs_and_parses(make_client: ClientFactory, questions: QuestionSet) -> None:
    emails = [EmailFactory(), EmailFactory(), EmailFactory()]
    results = {"results": [{"ref": "e001", **_ANSWER}, {"ref": "e003", **_ANSWER}, {"ref": "e001", **_ANSWER}]}
    text = json.dumps(results)
    recorder = Recorder(_sse_response([text[:30], text[30:]]))
    progress: list[int] = []
    classifier = _classifier(make_client(recorder), questions, "all_in_one")
    result = await classifier.classify(emails, progress.append)
    body = recorder.bodies[0]
    user = body["messages"][1]["content"]
    assert body["stream"] is True
    assert body["messages"][0]["content"] == "Batch.\n" + render_questions(questions)
    assert json.loads(user) == [{"ref": f"e00{i + 1}", **email.to_state()} for i, email in enumerate(emails)]
    assert all(email.id not in user for email in emails)
    assert body["response_format"]["json_schema"]["schema"] == all_in_one_schema(questions, ["e001", "e002", "e003"])
    assert result.outcomes[emails[0].id].notes == ("duplicate ref in response; first occurrence kept",)
    assert result.outcomes[emails[0].id].answers is not None
    assert result.outcomes[emails[1].id].error == "missing from all-in-one response"
    assert result.outcomes[emails[2].id].answers is not None
    assert result.usage.cost == 0.002
    assert progress[-1] == 3


async def test_all_in_one_truncated_fails_every_email(make_client: ClientFactory, questions: QuestionSet) -> None:
    emails = [EmailFactory(), EmailFactory()]
    recorder = Recorder(_sse_response(['{"results": [{"ref": "e001"'], finish_reason="length"))
    result = await _classifier(make_client(recorder), questions, "all_in_one").classify(emails)
    for email in emails:
        error = result.outcomes[email.id].error
        assert error is not None
        assert error.startswith("unusable all-in-one response: response truncated")


async def test_all_in_one_timeout_raises(make_client: ClientFactory, questions: QuestionSet) -> None:
    async def slow(request: httpx2.Request) -> httpx2.Response:
        await asyncio.sleep(0.05)
        return _sse_response(["{}"])

    classifier = _classifier(make_client(slow), questions, "all_in_one", all_in_one_timeout_s=0.01)
    with pytest.raises(OpenRouterError, match="exceeded"):
        await classifier.classify([EmailFactory()])


def _unused(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(500)


@pytest.mark.parametrize(
    ("mode", "per_request"),
    [
        pytest.param("per_email", 1, id="per-email"),
        pytest.param("all_in_one", None, id="all-in-one"),
    ],
)
async def test_classifier_shape(
    make_client: ClientFactory, questions: QuestionSet, mode: ChatMode, per_request: int | None
) -> None:
    classifier = _classifier(make_client(_unused), questions, mode)
    email: Email = EmailFactory()
    assert classifier.emails_per_request == per_request
    assert classifier.concurrency == 8
    assert classifier.sizing.output_per_email == 400
    assert classifier.sizing.overhead > 0
    assert (await classifier.prepare([email])).pending == (email,)


async def test_all_in_one_counts_ref_overhead(make_client: ClientFactory, questions: QuestionSet) -> None:
    client = make_client(_unused)
    email = EmailFactory()
    batched = _classifier(client, questions, "all_in_one").input_tokens(email)
    assert batched > _classifier(client, questions).input_tokens(email)
````

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_classifiers_llm.py -v`
Expected: `ModuleNotFoundError: No module named 'jev_bench.classifiers.llm'`.

- [ ] **Step 4: Write the implementation**

`src/jev_bench/classifiers/llm.py`:
```python
"""Chat-model column: verbalized probabilities through a strict JSON schema, per email or all in one.

Classes:
    LlmClassifier: implements the Classifier protocol for `per_email` and `all_in_one` modes.
"""

import asyncio
import json
import math
from collections.abc import Sequence
from string import Template
from typing import Any

from jev_bench.benchmark_config import ChatMode, LlmParams, TokenParams
from jev_bench.catalog import ModelInfo
from jev_bench.classifiers.base import (
    EmailOutcome,
    PrepareResult,
    ProgressCallback,
    RequestResult,
    outcome_from_parsed,
    usage_from_body,
)
from jev_bench.classifiers.llm_parse import RefCounter, parse_email_answers, split_results
from jev_bench.classifiers.llm_schema import all_in_one_schema, answers_schema, email_refs
from jev_bench.emails import Email
from jev_bench.openrouter import (
    CHAT_PATH,
    ApiResponse,
    ChatContentError,
    JsonObject,
    OpenRouterClient,
    OpenRouterError,
    TextCallback,
    chat_content,
    json_schema_format,
)
from jev_bench.questions import QuestionSet, render_questions
from jev_bench.request_plan import Sizing
from jev_bench.tokens import chat_budget, estimate_tokens

_REF_BYTES = len('{"ref":"e0000",},') + len('"e0000",')


class LlmClassifier:
    def __init__(
        self,
        *,
        client: OpenRouterClient,
        api_key: str,
        model: str,
        model_info: ModelInfo | None,
        questions: QuestionSet,
        params: LlmParams,
        tokens: TokenParams,
        mode: ChatMode,
        cache_system_prompt: bool,
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._model = model
        self._info = model_info
        self._questions = questions
        self._params = params
        self._mode = mode
        self._cache = cache_system_prompt
        self._bytes_per_token = tokens.bytes_per_token
        template = params.system_prompt if mode == "per_email" else params.system_prompt_all_in_one
        self._system = Template(template).substitute(questions=render_questions(questions))
        self.emails_per_request: int | None = 1 if mode == "per_email" else None
        self.concurrency = params.concurrency
        self.budget = chat_budget(model_info, tokens)
        self.sizing = Sizing(overhead=self._overhead(), output_per_email=params.est_output_tokens_per_email)

    def _overhead(self) -> int:
        if self._mode == "per_email":
            schema = answers_schema(self._questions)
        else:
            schema = all_in_one_schema(self._questions, [])
        return estimate_tokens(self._system + json.dumps(schema), self._bytes_per_token)

    def input_tokens(self, email: Email) -> int:
        size = len(json.dumps(email.to_state(), ensure_ascii=False).encode("utf-8"))
        extra = _REF_BYTES if self._mode == "all_in_one" else 0
        return math.ceil((size + extra) / self._bytes_per_token)

    async def prepare(self, emails: Sequence[Email]) -> PrepareResult:
        return PrepareResult(pending=tuple(emails))

    async def classify(
        self, emails: Sequence[Email], on_progress: ProgressCallback | None = None
    ) -> RequestResult:
        if self._mode == "per_email":
            return await self._classify_one(emails[0])
        return await self._classify_all(emails, on_progress)

    async def _classify_one(self, email: Email) -> RequestResult:
        user = json.dumps(email.to_state(), ensure_ascii=False)
        response_format = json_schema_format("email_triage", answers_schema(self._questions))
        body = self._body(user, response_format, [email])
        response = await self._client.post_json(CHAT_PATH, body, api_key=self._api_key)
        return self._result(response, {email.id: self._single_outcome(response.body)})

    async def _classify_all(
        self, emails: Sequence[Email], on_progress: ProgressCallback | None
    ) -> RequestResult:
        refs = email_refs(len(emails))
        states = [{"ref": ref, **email.to_state()} for ref, email in zip(refs, emails, strict=True)]
        response_format = json_schema_format("email_triage_batch", all_in_one_schema(self._questions, refs))
        body = self._body(json.dumps(states, ensure_ascii=False), response_format, emails)
        on_text = RefCounter(on_progress).feed if on_progress is not None else None
        response = await self._stream(body, on_text)
        return self._result(response, self._batch_outcomes(response.body, refs, emails))

    async def _stream(self, body: JsonObject, on_text: TextCallback | None) -> ApiResponse:
        try:
            async with asyncio.timeout(self._params.all_in_one_timeout_s):
                return await self._client.post_stream(CHAT_PATH, body, api_key=self._api_key, on_text=on_text)
        except TimeoutError as exc:
            raise OpenRouterError(f"all-in-one request exceeded {self._params.all_in_one_timeout_s:g} s") from exc

    def _body(self, user: str, response_format: JsonObject, emails: Sequence[Email]) -> JsonObject:
        estimated_input = self.sizing.overhead + sum(self.input_tokens(email) for email in emails)
        max_output = self.budget.output or 1
        return {
            "model": self._model,
            "messages": [self._system_message(), {"role": "user", "content": user}],
            "temperature": self._params.temperature,
            "reasoning": {"enabled": self._params.reasoning_enabled},
            "response_format": response_format,
            "provider": {"require_parameters": True},
            "max_tokens": max(1, min(max_output, self.budget.total - estimated_input)),
        }

    def _system_message(self) -> JsonObject:
        if not self._cache:
            return {"role": "system", "content": self._system}
        part = {"type": "text", "text": self._system, "cache_control": {"type": "ephemeral"}}
        return {"role": "system", "content": [part]}

    def _single_outcome(self, body: JsonObject) -> EmailOutcome:
        try:
            raw = json.loads(chat_content(body))
        except ChatContentError as exc:
            return EmailOutcome(error=str(exc))
        except json.JSONDecodeError:
            return EmailOutcome(error="response is not valid JSON")
        return outcome_from_parsed(*parse_email_answers(raw, self._questions))

    def _batch_outcomes(
        self, body: JsonObject, refs: Sequence[str], emails: Sequence[Email]
    ) -> dict[str, EmailOutcome]:
        try:
            found, notes = split_results(json.loads(chat_content(body)), refs)
        except ValueError as exc:
            error = f"unusable all-in-one response: {exc}"
            return {email.id: EmailOutcome(error=error) for email in emails}
        pairs = zip(refs, emails, strict=True)
        return {email.id: self._ref_outcome(found.get(ref), notes.get(ref)) for ref, email in pairs}

    def _ref_outcome(self, raw: dict[str, Any] | None, note: str | None) -> EmailOutcome:
        if raw is None:
            return EmailOutcome(error="missing from all-in-one response")
        answers, notes = parse_email_answers(raw, self._questions)
        return outcome_from_parsed(answers, [*notes, note] if note else notes)

    def _result(self, response: ApiResponse, outcomes: dict[str, EmailOutcome]) -> RequestResult:
        return RequestResult(
            outcomes=outcomes,
            usage=usage_from_body(response.body.get("usage"), self._info),
            latency_ms=response.latency_ms,
            resolved_model=response.body.get("model"),
            raw=response.body,
        )
```

- [ ] **Step 5: Run tests to verify they pass, lint, type-check**

Run: `uv run pytest tests/test_classifiers_llm.py -v`
Expected: all PASS; the timeout test takes about 10 ms.

Run: `uv run ruff check --fix && uv run ruff format && uv run pyright`
Expected: `0 errors`.

- [ ] **Step 6: Commit**

```bash
git add src/jev_bench/classifiers/llm.py tests/factories.py tests/test_classifiers_llm.py
git commit -m "feat(classifiers): chat classifier with per-email and streamed all-in-one modes

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
