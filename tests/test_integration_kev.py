"""Opt-in smoke test against Kev's real Hugging Face Space: one hand-written email, every question.

Run with: uv run pytest -m integration tests/test_integration_kev.py
Free, but it uses ZeroGPU quota; HF_TOKEN from the environment or .env raises it.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from jev_bench.benchmark_config import load_benchmark_config
from jev_bench.classifiers.kev import KevClassifier
from jev_bench.emails import Email, Party
from jev_bench.kev_space import KevSpaceClient
from jev_bench.openrouter import build_http_client
from jev_bench.questions import load_question_set
from jev_bench.settings import load_settings

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]
CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
EMAIL = Email(
    id="20260928-000000-smoke-0001.0001",
    sent_at=datetime(2026, 9, 28, 9, 0, tzinfo=UTC),
    sender=Party(name="Dana Ruiz", address="dana.ruiz@example.com"),
    to=(Party(name="Sam Lee", address="sam.lee@example.com"),),
    subject="Invoice 4471 is overdue",
    body=(
        "Hi Sam, invoice 4471 was due last Friday. "
        "Could you confirm payment by Wednesday? Thanks, Dana"
    ),
    generator_model="hand/written",
)


async def test_one_email_answers_every_shipped_question() -> None:
    settings = load_settings()
    config = load_benchmark_config(CONFIG_DIR / "benchmark.toml")
    questions = load_question_set(CONFIG_DIR / "questions.toml")
    http = build_http_client(settings)
    try:
        kev = KevClassifier(
            client=KevSpaceClient(http, max_retries=1, retry_base_delay_s=1.0),
            hf_token=settings.server_hf_token(),
            model=config.column("kev").default_model,
            model_info=None,
            questions=questions,
            params=config.kev,
            tokens=config.tokens,
        )
        await kev.prepare([EMAIL])
        result = await kev.classify([EMAIL])
    finally:
        await http.aclose()
    outcome = result.outcomes[EMAIL.id]
    assert outcome.error is None, outcome.notes
    assert outcome.answers is not None
    assert set(outcome.answers) == set(questions.ids)
    assert result.resolved_model == "jaredpalmer/kev-4b"
