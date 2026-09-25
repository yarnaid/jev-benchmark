"""Tests for jev_bench.analysis.config."""

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from jev_bench.analysis.config import (
    PLACEHOLDERS,
    AnalysisConfig,
    check_prompt,
    load_analysis_config,
)


def _doc(**overrides: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "default_model": "anthropic/claude-sonnet-5",
        "system_prompt": "You analyse benchmarks.",
        "user_prompt": "$generations $runs $questions $report $emails $disputed",
    }
    doc.update(overrides)
    return doc


def test_defaults() -> None:
    config = AnalysisConfig.model_validate(_doc())
    assert (config.max_disputed_emails, config.max_output_tokens) == (12, 8000)
    assert (config.temperature, config.timeout_s) == (0.2, 900.0)


@pytest.mark.parametrize(
    "doc",
    [
        pytest.param(_doc(default_model=""), id="blank-model"),
        pytest.param(_doc(user_prompt="$unknown"), id="unknown-placeholder"),
        pytest.param(_doc(system_prompt="costs $5"), id="bare-dollar"),
        pytest.param(_doc(max_disputed_emails=-1), id="negative-disputed"),
        pytest.param(_doc(max_output_tokens=0), id="zero-output"),
        pytest.param(_doc(timeout_s=0), id="zero-timeout"),
        pytest.param(_doc(extra=1), id="unknown-key"),
    ],
)
def test_invalid_config(doc: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        AnalysisConfig.model_validate(doc)


@pytest.mark.parametrize(
    ("template", "valid"),
    [
        pytest.param("$report and $$5", True, id="placeholder-and-escaped-dollar"),
        pytest.param("", True, id="empty"),
        pytest.param("${emails}", True, id="braced"),
        pytest.param("$email", False, id="unknown"),
        pytest.param("$", False, id="dangling-dollar"),
    ],
)
def test_check_prompt(template: str, valid: bool) -> None:
    if valid:
        assert check_prompt(template) == template
    else:
        with pytest.raises(ValueError, match="placeholders"):
            check_prompt(template)


def test_load_analysis_config(tmp_path: Path) -> None:
    path = tmp_path / "analysis.toml"
    path.write_text(
        'default_model = "m/x"\nsystem_prompt = "S"\nuser_prompt = "$report"\n', encoding="utf-8"
    )
    assert load_analysis_config(path).default_model == "m/x"
    assert PLACEHOLDERS == ("generations", "runs", "questions", "report", "emails", "disputed")
