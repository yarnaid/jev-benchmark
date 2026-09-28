"""Contract tests for the shipped TOML configuration files in config/."""

from datetime import UTC, datetime
from pathlib import Path

from jev_bench.analysis.config import PLACEHOLDERS, load_analysis_config
from jev_bench.benchmark_config import load_benchmark_config
from jev_bench.generation.config import load_generation_config
from jev_bench.generation.plan import build_plan, resolve_traits
from jev_bench.generation.prompt import render_prompts
from jev_bench.questions import MultiQuestion, load_question_set

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def test_shipped_question_set() -> None:
    shipped = load_question_set(CONFIG_DIR / "questions.toml")
    assert shipped.name == "email-triage-v2"
    assert shipped.ids == (
        "category",
        "urgency",
        "importance",
        "sentiment",
        "confidentiality",
        "needs_reply",
        "action_required",
        "deadline",
        "attachment_review",
        "delegatable",
        "escalation",
        "skippable",
        "should_delete",
        "is_automated",
        "llm_safe",
        "malicious",
        "impersonation",
        "sensitive_data",
    )
    category = shipped.get("category")
    assert isinstance(category, MultiQuestion)
    assert (len(category.options), category.threshold) == (20, 0.8)
    levels = {
        "urgency": ("no_action", "whenever", "this_week", "today", "immediately"),
        "importance": ("trivial", "low", "moderate", "high"),
        "sentiment": ("negative", "neutral", "positive"),
        "confidentiality": ("public", "internal", "confidential", "restricted"),
    }
    for question_id, option_ids in levels.items():
        assert (shipped.get(question_id).type, shipped.get(question_id).option_ids) == (
            "score",
            option_ids,
        )
    assert [q.type for q in shipped.questions].count("noul") == 13


def test_shipped_benchmark_config() -> None:
    config = load_benchmark_config(CONFIG_DIR / "benchmark.toml")
    assert [(c.id, c.kind, c.default_model) for c in config.columns] == [
        ("jev", "decisions", "typesafe/jev-1.13"),
        ("anthropic", "chat", "anthropic/claude-sonnet-5"),
        ("openai", "chat", "openai/gpt-5.6-terra"),
        ("embeddings", "embeddings", "openai/text-embedding-3-large"),
        ("kev", "decisions", "jaredpalmer/kev-4b"),
    ]
    assert config.column("anthropic").cache_system_prompt is True
    assert config.column("openai").cache_system_prompt is False
    assert config.embeddings.email_template.startswith("Sent: $sent_at")
    jev, kev = config.column("jev"), config.column("kev")
    assert (jev.prefix, jev.effective_slot) == ("typesafe/", "jev")
    assert (kev.prefix, kev.effective_slot) == ("jaredpalmer/", "embeddings")


def test_shipped_generation_config() -> None:
    config = load_generation_config(CONFIG_DIR / "generation.toml")
    questions = load_question_set(CONFIG_DIR / "questions.toml")
    traits = resolve_traits(config, questions)
    expected_models = (
        "google/gemini-3.8-flash",
        "deepseek/deepseek-v4.1-flash",
        "z-ai/glm-5.3",
    )
    assert config.models == expected_models
    assert config.max_attempts == 3
    assert "every option id that applies (at least one)" in config.system_prompt
    expected_traits = ["category", "urgency", "length", "prompt_injection"]
    assert [trait.name for trait in traits] == expected_traits
    now = datetime(2026, 9, 24, tzinfo=UTC)
    plan = build_plan(config, questions, count=40, seed=1, models=config.models, now=now)
    assert {item.traits["category"] for item in plan} == set(questions.get("category").option_ids)
    system, user = render_prompts(config, traits, plan[0], questions)
    assert "$" not in system + user
    assert plan[0].sent_at.isoformat() in user


def test_shipped_analysis_config() -> None:
    config = load_analysis_config(CONFIG_DIR / "analysis.toml")
    assert config.default_model == "anthropic/claude-sonnet-5"
    assert (config.max_disputed_emails, config.max_output_tokens) == (12, 8000)
    assert all(f"${name}" in config.user_prompt for name in PLACEHOLDERS)
    assert "untrusted" in config.system_prompt
