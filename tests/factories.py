"""Factory Boy factories for domain models shared across tests.

Classes:
    PartyFactory: named mailbox.
    EmailFactory: generated email with deterministic ids and a fixed sent_at.
Functions:
    chat_body: OpenRouter chat completion response.
    sse_body: OpenRouter streaming chat completion response.
    write_mini_config: write a mini questions/benchmark/generation TOML config to a directory.
    mini_settings: isolated Settings over a mini config, reading no env or `.env`.
    generator_output: JSON text of one generator response.
Constants:
    MINI_QUESTIONS_TOML, MINI_BENCHMARK_TOML, MINI_GENERATION_TOML: mini config file contents.
Types:
    ClientFactory: type of the make_client fixture.
    ServicesFactory: type of the make_services fixture.
"""

import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from factory.base import Factory
from factory.declarations import LazyFunction, SubFactory
from factory.declarations import Sequence as FactorySequence

from jev_bench.emails import Email, Party
from jev_bench.openrouter import OpenRouterClient
from jev_bench.services import Services
from jev_bench.settings import Settings


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
        lambda: {"category": "spam", "urgency": "today", "needs_reply": "yes"}
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


def write_mini_config(config_dir: Path) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "questions.toml").write_text(MINI_QUESTIONS_TOML, encoding="utf-8")
    (config_dir / "benchmark.toml").write_text(MINI_BENCHMARK_TOML, encoding="utf-8")
    (config_dir / "generation.toml").write_text(MINI_GENERATION_TOML, encoding="utf-8")


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
            "answers": {"category": category, "urgency": "now", "needs_reply": "no"},
        }
    )
