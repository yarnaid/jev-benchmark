"""Tests for jev_bench.generation.config."""

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from jev_bench.generation.config import GenerationConfig, load_generation_config


def _doc(**overrides: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "models": ["google/gemini-3.8-flash"],
        "system_prompt": "Write emails.\n$questions",
        "user_prompt": (
            "Category $category: $category_prompt. Length: $length_prompt. Sent $sent_at."
        ),
        "traits": [
            {"name": "category", "question": "category", "stratify": True},
            {
                "name": "length",
                "values": {
                    "short": {"weight": 3, "prompt": "Short."},
                    "long": {"prompt": "Long."},
                },
            },
        ],
    }
    doc.update(overrides)
    return doc


def test_valid_config() -> None:
    config = GenerationConfig.model_validate(_doc())
    assert config.temperature == 1.0
    assert config.concurrency == 8
    assert config.sent_at_window_days == 30
    assert config.traits[0].question == "category"
    assert config.traits[1].values["long"].weight == 1.0


@pytest.mark.parametrize(
    "doc",
    [
        pytest.param(_doc(models=[]), id="no-models"),
        pytest.param(_doc(user_prompt="$missing"), id="unknown-placeholder"),
        pytest.param(
            _doc(
                traits=[
                    {
                        "name": "x",
                        "question": "q",
                        "values": {"a": {"prompt": "A"}},
                    }
                ]
            ),
            id="question-and-values",
        ),
        pytest.param(_doc(traits=[{"name": "x"}]), id="neither-source"),
        pytest.param(
            _doc(
                traits=[
                    {"name": "x", "question": "q"},
                    {"name": "x", "question": "q"},
                ],
                user_prompt="$x",
            ),
            id="duplicate-trait",
        ),
        pytest.param(
            _doc(
                traits=[{"name": "x", "values": {"a": {"weight": 0, "prompt": "A"}}}],
                user_prompt="$x",
            ),
            id="zero-weight",
        ),
        pytest.param(
            _doc(traits=[{"name": "Bad-Name", "question": "q"}], user_prompt="x"),
            id="bad-trait-name",
        ),
    ],
)
def test_invalid_configs(doc: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        GenerationConfig.model_validate(doc)


def test_load_generation_config(tmp_path: Path) -> None:
    path = tmp_path / "generation.toml"
    path.write_text(
        'models = ["m"]\nsystem_prompt = "$questions"\nuser_prompt = "$tone_prompt"\n'
        '[[traits]]\nname = "tone"\n[traits.values.calm]\nprompt = "Calm."\n',
        encoding="utf-8",
    )
    assert load_generation_config(path).traits[0].values["calm"].prompt == "Calm."
