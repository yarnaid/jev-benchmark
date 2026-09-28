"""Shared pytest configuration and fixtures.

Pre-warms numpy.random at collection time so Hypothesis's lazy import never lands
inside a timed test.

Fixtures:
    questions: a three-question set covering the choice, score and noul types.
    multi_questions: the questions fixture plus a multi-label "topics" question (threshold 0.8).
    make_client: async OpenRouterClient factory.
    make_services: async Services factory on tmp dirs.
    make_app: TestClient factory with the lifespan entered.
    log_records: worst-case loguru sink (diagnose=True, backtrace=True) for leak regression tests.
    _no_hf_token_in_env (autouse): HF users often export HF_TOKEN; only integration tests see it.
Hooks:
    pytest_configure: pre-warm numpy.random module.
    pytest_collection_modifyitems: mark tests listed in tests/slow_tests.txt as `slow`.
"""

import importlib
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import httpx2
import pytest
from loguru import logger
from tests.factories import AppFactory, ClientFactory, ServicesFactory, mini_settings

from jev_bench.openrouter import OpenRouterClient
from jev_bench.questions import (
    ChoiceQuestion,
    MultiQuestion,
    NoulQuestion,
    QuestionSet,
    ScoreQuestion,
)
from jev_bench.services import Services

if TYPE_CHECKING:
    from fastapi.testclient import TestClient
    from loguru import Message

_SLOW_LIST = Path(__file__).with_name("slow_tests.txt")
type Handler = Callable[[httpx2.Request], httpx2.Response]


def _slow_node_ids() -> set[str]:
    if not _SLOW_LIST.exists():
        return set()
    lines = (line.strip() for line in _SLOW_LIST.read_text(encoding="utf-8").splitlines())
    return {line for line in lines if line and not line.startswith("#")}


def pytest_configure(config: pytest.Config) -> None:
    importlib.import_module("numpy.random")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    slow = _slow_node_ids()
    for item in items:
        if item.nodeid in slow:
            item.add_marker(pytest.mark.slow)


@pytest.fixture(autouse=True)
def _no_hf_token_in_env(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("integration") is None:
        monkeypatch.delenv("HF_TOKEN", raising=False)


@pytest.fixture
def questions() -> QuestionSet:
    return QuestionSet(
        name="mini",
        questions=(
            ChoiceQuestion(
                type="choice",
                id="category",
                instructions="What kind of email?",
                options={"spam": "Junk", "personal": "From a friend", "work": "From a colleague"},
            ),
            ScoreQuestion(
                type="score",
                id="urgency",
                instructions="How urgent?",
                options={"low": "Whenever", "today": "Within a day", "now": "Immediately"},
            ),
            NoulQuestion(
                type="noul",
                id="needs_reply",
                instructions="Needs a reply?",
                options={"yes": "Reply expected", "no": "No reply expected"},
            ),
        ),
    )


@pytest.fixture
def multi_questions(questions: QuestionSet) -> QuestionSet:
    topics = MultiQuestion(
        type="multi",
        id="topics",
        instructions="Which topics?",
        options={"billing": "About money", "meeting": "About a meeting", "travel": "About a trip"},
    )
    return QuestionSet(name="mini-multi", questions=(*questions.questions, topics))


@pytest.fixture
async def make_client() -> AsyncIterator[ClientFactory]:
    opened: list[httpx2.AsyncClient] = []

    def build(handler: Handler, *, max_retries: int = 0) -> OpenRouterClient:
        http = httpx2.AsyncClient(
            base_url="https://openrouter.test/api", transport=httpx2.MockTransport(handler)
        )
        opened.append(http)
        return OpenRouterClient(http, max_retries=max_retries, retry_base_delay_s=0.0)

    yield build
    for http in opened:
        await http.aclose()


@pytest.fixture
def log_records() -> Iterator[list[Message]]:
    records: list[Message] = []
    handler_id = logger.add(records.append, level="DEBUG", diagnose=True, backtrace=True)
    try:
        yield records
    finally:
        logger.remove(handler_id)


@pytest.fixture
async def make_services(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[ServicesFactory]:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    opened: list[httpx2.AsyncClient] = []

    def build(
        handler: Handler, *, api_key: str | None = None, hf_token: str | None = None
    ) -> Services:
        http = httpx2.AsyncClient(
            base_url="https://openrouter.test/api", transport=httpx2.MockTransport(handler)
        )
        opened.append(http)
        return Services(mini_settings(tmp_path, api_key, hf_token), http)

    yield build
    for http in opened:
        await http.aclose()


@pytest.fixture
def make_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[AppFactory]:
    from fastapi.testclient import TestClient

    from jev_bench.web.app import create_app

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    clients: list[TestClient] = []

    def build(
        handler: Handler, *, api_key: str | None = None, hf_token: str | None = None
    ) -> TestClient:
        http = httpx2.AsyncClient(
            base_url="https://openrouter.test/api", transport=httpx2.MockTransport(handler)
        )
        client = TestClient(create_app(mini_settings(tmp_path, api_key, hf_token), http=http))
        client.__enter__()
        clients.append(client)
        return client

    yield build
    for client in clients:
        client.__exit__(None, None, None)
