"""Contract tests for the shipped TOML configuration files in config/."""

from pathlib import Path

from jev_bench.questions import load_question_set

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def test_shipped_question_set() -> None:
    shipped = load_question_set(CONFIG_DIR / "questions.toml")
    assert shipped.ids == (
        "category",
        "urgency",
        "importance",
        "needs_reply",
        "action_required",
        "skippable",
        "should_delete",
        "llm_safe",
        "malicious",
        "sensitive_data",
        "sentiment",
    )
    assert len(shipped.get("category").options) == 20
    urgency_ids = ("no_action", "whenever", "this_week", "today", "immediately")
    assert shipped.get("urgency").option_ids == urgency_ids
    importance_ids = ("trivial", "low", "moderate", "high")
    assert shipped.get("importance").option_ids == importance_ids
    sentiment_ids = ("negative", "neutral", "positive")
    assert shipped.get("sentiment").option_ids == sentiment_ids
    assert [q.type for q in shipped.questions].count("noul") == 7
