"""Tests for jev_bench.generation.plan."""

from collections import Counter
from datetime import UTC, datetime, timedelta

import pytest

from jev_bench.generation.config import GenerationConfig
from jev_bench.generation.plan import build_plan, resolve_traits
from jev_bench.questions import QuestionSet

_NOW = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)


def _config(**overrides: object) -> GenerationConfig:
    doc: dict[str, object] = {
        "models": ["m1", "m2"],
        "system_prompt": "$questions",
        "user_prompt": "$category $tone_prompt",
        "sent_at_window_days": 2,
        "traits": [
            {"name": "category", "question": "category", "stratify": True},
            {
                "name": "tone",
                "values": {
                    "calm": {"weight": 1, "prompt": "Calm."},
                    "angry": {"weight": 3, "prompt": "Angry."},
                },
            },
        ],
    }
    doc.update(overrides)
    return GenerationConfig.model_validate(doc)


def test_resolve_traits(questions: QuestionSet) -> None:
    category, tone = resolve_traits(_config(), questions)
    assert (category.question, list(category.values)) == ("category", ["spam", "personal", "work"])
    assert category.values["personal"].prompt == "From a friend"
    assert category.values["personal"].weight == 1.0
    assert (tone.question, tone.values["angry"].weight) == (None, 3.0)


def test_resolve_traits_rejects_unknown_question(questions: QuestionSet) -> None:
    config = _config(traits=[{"name": "category", "question": "missing"}], user_prompt="$category")
    with pytest.raises(ValueError, match="unknown question 'missing'"):
        resolve_traits(config, questions)


def test_build_plan_is_deterministic_and_complete(questions: QuestionSet) -> None:
    config = _config()
    first = build_plan(config, questions, count=6, seed=42, models=("m1", "m2"), now=_NOW)
    again = build_plan(config, questions, count=6, seed=42, models=("m1", "m2"), now=_NOW)
    other = build_plan(config, questions, count=6, seed=43, models=("m1", "m2"), now=_NOW)
    assert first == again
    assert first != other
    assert [item.index for item in first] == [1, 2, 3, 4, 5, 6]
    assert [item.model for item in first] == ["m1", "m2", "m1", "m2", "m1", "m2"]
    category_counts = Counter(item.traits["category"] for item in first)
    assert category_counts == {"spam": 2, "personal": 2, "work": 2}
    assert {item.traits["tone"] for item in first} <= {"calm", "angry"}


def test_sent_at_is_inside_the_window_and_minute_aligned(questions: QuestionSet) -> None:
    plan = build_plan(_config(), questions, count=20, seed=1, models=("m",), now=_NOW)
    for item in plan:
        assert _NOW - timedelta(days=2) <= item.sent_at <= _NOW
        assert (item.sent_at.second, item.sent_at.microsecond) == (0, 0)
        assert item.sent_at.tzinfo is not None


def test_empty_plan(questions: QuestionSet) -> None:
    assert build_plan(_config(), questions, count=0, seed=1, models=("m",), now=_NOW) == []


def test_stratified_trait_covers_different_subsets_across_seeds(
    questions: QuestionSet,
) -> None:
    config = _config()
    all_subsets = set()
    for seed in range(10):
        plan = build_plan(config, questions, count=2, seed=seed, models=("m",), now=_NOW)
        traits = [item.traits["category"] for item in plan]
        subset = frozenset(traits)
        all_subsets.add(subset)
    assert len(all_subsets) >= 2


def test_stratified_trait_counts_balanced_when_uneven(questions: QuestionSet) -> None:
    config = _config()
    for seed in range(10):
        plan = build_plan(config, questions, count=7, seed=seed, models=("m",), now=_NOW)
        traits = [item.traits["category"] for item in plan]
        counts = Counter(traits)
        vals = sorted(counts.values())
        assert max(vals) - min(vals) <= 1
