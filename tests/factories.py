"""Factory Boy factories for domain models shared across tests.

Classes:
    PartyFactory: named mailbox.
    EmailFactory: generated email with deterministic ids and a fixed sent_at.
    FakeOpenRouter: MockTransport handler serving /v1/models, /alpha/decisions,
        /v1/chat/completions and /v1/embeddings for the mini question set. A chat request without
        `response_format` is an analysis: it streams `analysis_parts` with `analysis_finish`.
Functions:
    chat_body: OpenRouter chat completion response.
    sse_body: OpenRouter streaming chat completion response.
    write_mini_config: write a mini questions/benchmark/generation TOML config to a directory.
    mini_settings: isolated Settings over a mini config, reading no env or `.env`.
    generator_output: JSON text of one generator response.
    seed_generation: save a completed GenerationMeta plus its emails to services.generations.
    services_of: typed access to a TestClient's app.state.services.
    poll: GET a path repeatedly until a predicate on its JSON body holds, or fail.
Constants:
    MINI_QUESTIONS_TOML, MINI_BENCHMARK_TOML, MINI_GENERATION_TOML, MINI_ANALYSIS_TOML: mini
        config file contents
        (the mini question set has a multi-label "topics" question with threshold 0.75; the fake
        chat answer puts "meeting" exactly at 0.75 * max, the fake Jev answer below it).
    CHAT_PARAMETERS: the fake catalog's supported_parameters for chat models (like the real
        Claude Sonnet 5 / GPT-5.6 Terra entries: no `temperature`).
    ANALYSIS_PARTS: the fake analyst's streamed Markdown.
Types:
    ClientFactory: type of the make_client fixture.
    ServicesFactory: type of the make_services fixture.
    AppFactory: type of the make_app fixture.
"""

import json
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import httpx2
from factory.base import Factory
from factory.declarations import LazyFunction, SubFactory
from factory.declarations import Sequence as FactorySequence

from jev_bench.emails import Email, Party
from jev_bench.openrouter import OpenRouterClient
from jev_bench.services import Services
from jev_bench.settings import Settings
from jev_bench.store.generations import GenerationMeta

if TYPE_CHECKING:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient


class PartyFactory(Factory[Party]):
    class Meta:
        model = Party

    name = FactorySequence(lambda n: f"Person {n}")
    address = FactorySequence(lambda n: f"person{n}@mail.test")


class EmailFactory(Factory[Email]):
    class Meta:
        model = Email

    id = FactorySequence(lambda n: f"20260924-100000-gen-abcd.{n + 1:04d}")
    sent_at = datetime(2026, 9, 20, 9, 30, tzinfo=UTC)
    sender = SubFactory(PartyFactory)
    to = LazyFunction(lambda: (PartyFactory(),))
    cc = ()
    subject = FactorySequence(lambda n: f"Subject {n}")
    body = "Hello, please confirm the meeting."
    generator_model = "google/gemini-3.8-flash"
    traits = LazyFunction(dict)
    reference_answers = LazyFunction(
        lambda: {
            "category": "spam",
            "urgency": "today",
            "needs_reply": "yes",
            "topics": ["billing", "meeting"],
        }
    )


def chat_body(
    content: str, *, finish_reason: str = "stop", model: str = "test/model", cost: float = 0.001
) -> dict[str, Any]:
    return {
        "id": "gen-test",
        "model": model,
        "choices": [
            {"message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": cost},
    }


def sse_body(
    parts: Sequence[str],
    *,
    finish_reason: str = "stop",
    model: str = "test/model",
    cost: float = 0.002,
) -> bytes:
    chunks: list[dict[str, Any]] = [
        {"id": "gen-test", "model": model, "choices": [{"delta": {"content": part}}]}
        for part in parts
    ]
    chunks.append(
        {
            "choices": [{"delta": {"content": ""}, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": cost},
        }
    )
    return (
        "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks).encode() + b"data: [DONE]\n\n"
    )


type ClientFactory = Callable[..., OpenRouterClient]
type ServicesFactory = Callable[..., Services]
type AppFactory = Callable[..., TestClient]


def services_of(client: TestClient) -> Services:
    return cast("FastAPI", client.app).state.services


MINI_QUESTIONS_TOML = """
name = "mini"

[[questions]]
id = "category"
type = "choice"
instructions = "What kind of email?"

[questions.options]
spam = "Junk"
personal = "From a friend"
work = "From a colleague"

[[questions]]
id = "urgency"
type = "score"
instructions = "How urgent?"

[questions.options]
low = "Whenever"
today = "Within a day"
now = "Immediately"

[[questions]]
id = "needs_reply"
type = "noul"
instructions = "Needs a reply?"

[questions.options]
yes = "Reply expected"
no = "No reply expected"

[[questions]]
id = "topics"
type = "multi"
instructions = "Which topics?"
threshold = 0.75

[questions.options]
billing = "About money"
meeting = "About a meeting"
travel = "About a trip"
"""

MINI_BENCHMARK_TOML = """
[jev]
concurrency = 2

[llm]
system_prompt = "Classify.\\n$questions"
system_prompt_all_in_one = "Batch.\\n$questions"
concurrency = 2

[embeddings]
email_template = "$subject"
option_template = "$option"
emails_per_request = 2
concurrency = 2

[[columns]]
id = "jev"
title = "Jev"
kind = "decisions"
modality = "decisions"
default_model = "typesafe/jev-1.13"

[[columns]]
id = "anthropic"
title = "Anthropic"
kind = "chat"
modality = "text"
prefix = "anthropic/"
default_model = "anthropic/claude-sonnet-5"
cache_system_prompt = true

[[columns]]
id = "embeddings"
title = "Embeddings"
kind = "embeddings"
modality = "embeddings"
default_model = "openai/text-embedding-3-large"
"""

MINI_GENERATION_TOML = """
models = ["gen/a", "gen/b"]
concurrency = 2
system_prompt = "Write.\\n$questions"
user_prompt = "Category $category ($category_prompt) at $sent_at."

[[traits]]
name = "category"
question = "category"
stratify = true
"""


MINI_ANALYSIS_TOML = """
default_model = "anthropic/claude-sonnet-5"
max_disputed_emails = 2
max_output_tokens = 500
system_prompt = "Analyse.\\n$questions"
user_prompt = "$generations\\n$runs\\n$report\\n$emails\\n$disputed"
"""


def write_mini_config(config_dir: Path) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "questions.toml").write_text(MINI_QUESTIONS_TOML, encoding="utf-8")
    (config_dir / "benchmark.toml").write_text(MINI_BENCHMARK_TOML, encoding="utf-8")
    (config_dir / "generation.toml").write_text(MINI_GENERATION_TOML, encoding="utf-8")
    (config_dir / "analysis.toml").write_text(MINI_ANALYSIS_TOML, encoding="utf-8")


def mini_settings(root: Path, api_key: str | None = None) -> Settings:
    write_mini_config(root / "config")
    return Settings.model_validate(
        {
            "data_dir": root / "data",
            "config_dir": root / "config",
            "openrouter_api_key": api_key,
            "max_retries": 0,
            "retry_base_delay_s": 0.0,
        }
    )


def generator_output(category: str = "spam") -> str:
    return json.dumps(
        {
            "email": {
                "sender": {"name": "Deals Team", "address": "deals@promo.test"},
                "to": [{"name": "Ann", "address": "ann@mail.test"}],
                "cc": [],
                "subject": "You won a prize",
                "body": "Claim it now.",
            },
            "answers": {
                "category": category,
                "urgency": "now",
                "needs_reply": "no",
                "topics": ["billing"],
            },
        }
    )


CHAT_ANSWER: dict[str, Any] = {
    "category": {"spam": 0.7, "personal": 0.2, "work": 0.1},
    "urgency": {"low": 0.1, "today": 0.3, "now": 0.6},
    "needs_reply": 0.2,
    "topics": {"billing": 0.5, "meeting": 0.375, "travel": 0.125},
}
JEV_ANSWERS: dict[str, Any] = {
    "category": {
        "type": "choice",
        "choice": "spam",
        "probabilities": {"spam": 0.8, "personal": 0.1, "work": 0.1},
    },
    "urgency": {
        "type": "score",
        "score": 1.8,
        "probabilities": {"0": 0.05, "1": 0.1, "2": 0.85},
    },
    "needs_reply": {"type": "noul", "noul": 0.15},
    "topics": {
        "type": "choice",
        "choice": "billing",
        "probabilities": {"billing": 0.7, "meeting": 0.2, "travel": 0.1},
    },
}
CHAT_PARAMETERS = ("max_tokens", "reasoning", "response_format", "structured_outputs")
ANALYSIS_PARTS = ("## Executive summary\n", "- Jev agrees with the reference.\n")
_CATALOG: dict[str, list[dict[str, Any]]] = {
    "text": [
        {
            "id": "anthropic/claude-sonnet-5",
            "name": "Claude Sonnet 5",
            "pricing": {"prompt": "0.000002", "completion": "0.00001"},
            "context_length": 1000000,
            "top_provider": {"max_completion_tokens": 128000},
        },
        {
            "id": "openai/gpt-5.6-terra",
            "name": "GPT-5.6 Terra",
            "pricing": {"prompt": "0.000002", "completion": "0.000012"},
            "context_length": 1050000,
            "top_provider": {"max_completion_tokens": 128000},
        },
    ],
    "decisions": [
        {
            "id": "typesafe/jev-1.13",
            "name": "Jev 1.13",
            "pricing": {"prompt": "0.000000042"},
            "context_length": 32000,
        }
    ],
    "embeddings": [
        {
            "id": "openai/text-embedding-3-large",
            "name": "Embedding 3 Large",
            "pricing": {"prompt": "0.00000013"},
            "context_length": 8192,
        }
    ],
}


def _vector(text: str) -> list[float]:
    return [float(len(text)), 1.0, float(sum(map(ord, text)) % 7)]


class FakeOpenRouter:
    def __init__(
        self,
        *,
        models_status: int = 200,
        chat_parameters: tuple[str, ...] = CHAT_PARAMETERS,
        analysis_parts: Sequence[str] = ANALYSIS_PARTS,
        analysis_finish: str = "stop",
    ) -> None:
        self.models_status = models_status
        self.chat_parameters = chat_parameters
        self.analysis_parts = analysis_parts
        self.analysis_finish = analysis_finish
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        path = request.url.path
        if path.endswith("/v1/models"):
            return self._models(request)
        if (unauthorized := self._unauthorized(request)) is not None:
            return unauthorized
        body = json.loads(request.content)
        if path.endswith("/alpha/decisions"):
            usage = {"input_tokens": 100, "output_tokens": 10, "cost": 0.00001}
            return httpx2.Response(
                200, json={"model": "typesafe/jev-1.13", "answers": JEV_ANSWERS, "usage": usage}
            )
        if path.endswith("/v1/embeddings"):
            data = [
                {"index": i, "embedding": _vector(text)} for i, text in enumerate(body["input"])
            ]
            usage = {"prompt_tokens": 5 * len(data), "cost": 0.00001 * len(data)}
            return httpx2.Response(200, json={"model": body["model"], "data": data, "usage": usage})
        return self._chat(body)

    def _unauthorized(self, request: httpx2.Request) -> httpx2.Response | None:
        token = request.headers.get("authorization", "")
        if token.startswith("Bearer ") and token.removeprefix("Bearer ").strip():
            return None
        return httpx2.Response(
            401, json={"error": {"message": "missing or invalid API key", "code": 401}}
        )

    def _models(self, request: httpx2.Request) -> httpx2.Response:
        if self.models_status != 200:
            return httpx2.Response(self.models_status, json={"error": {"message": "catalog down"}})
        modality = request.url.params.get("output_modalities") or "text"
        models = _CATALOG[modality]
        if modality == "text":
            models = [{**model, "supported_parameters": self.chat_parameters} for model in models]
        return httpx2.Response(200, json={"data": models})

    def _chat(self, body: dict[str, Any]) -> httpx2.Response:
        if "response_format" not in body:
            return self._analysis(body)
        if body["response_format"]["json_schema"]["name"] == "email_generation":
            return httpx2.Response(200, json=chat_body(generator_output(), model=body["model"]))
        if not body.get("stream"):
            content = json.dumps(CHAT_ANSWER)
            return httpx2.Response(200, json=chat_body(content, model=body["model"]))
        items = body["response_format"]["json_schema"]["schema"]["properties"]["results"]["items"]
        refs = items["properties"]["ref"]["enum"]
        content = json.dumps({"results": [{"ref": ref, **CHAT_ANSWER} for ref in refs]})
        headers = {"content-type": "text/event-stream"}
        return httpx2.Response(
            200, content=sse_body([content], model=body["model"]), headers=headers
        )

    def _analysis(self, body: dict[str, Any]) -> httpx2.Response:
        stream = sse_body(
            self.analysis_parts, finish_reason=self.analysis_finish, model=body["model"]
        )
        return httpx2.Response(200, content=stream, headers={"content-type": "text/event-stream"})


def seed_generation(
    services: Services, *, emails: int = 2, generation_id: str = "20260924-100000-seed-abcd"
) -> str:
    meta = GenerationMeta(
        id=generation_id,
        name="seed",
        created_at=datetime(2026, 9, 24, tzinfo=UTC),
        status="completed",
        requested=emails,
        done=emails,
        seed=1,
        models=("gen/a",),
        question_set=services.question_set(),
        config=services.generation_config(),
    )
    services.generations.save(meta)
    for index in range(1, emails + 1):
        email = EmailFactory(id=f"{generation_id}.{index:04d}")
        services.generations.append_email(generation_id, email)
    return generation_id


def poll(
    client: TestClient,
    path: str,
    *,
    until: Callable[[dict[str, Any]], bool],
    attempts: int = 100,
    delay_s: float = 0.002,
) -> dict[str, Any]:
    for _ in range(attempts):
        body = client.get(path).json()
        if until(body):
            return body
        time.sleep(delay_s)
    raise AssertionError(f"{path} never satisfied the condition")
