"""Tests for jev_bench.benchmark_config."""

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from jev_bench.benchmark_config import BenchmarkConfig, ColumnConfig, load_benchmark_config


def _doc(**overrides: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "llm": {
            "system_prompt": "Q:\n$questions",
            "system_prompt_all_in_one": "ALL:\n$questions",
        },
        "embeddings": {
            "email_template": "$subject\n$body",
            "option_template": "$instructions $description",
        },
        "columns": [
            {
                "id": "jev",
                "title": "Jev",
                "kind": "decisions",
                "modality": "decisions",
                "default_model": "typesafe/jev-1.13",
            },
            {
                "id": "anthropic",
                "title": "A",
                "kind": "chat",
                "modality": "text",
                "prefix": "anthropic/",
                "default_model": "anthropic/claude-sonnet-5",
            },
        ],
    }
    doc.update(overrides)
    return doc


_JEV_COLUMN: dict[str, Any] = _doc()["columns"][0]


@pytest.mark.parametrize(
    ("column", "slot"),
    [
        pytest.param(_JEV_COLUMN, "jev", id="own-id-by-default"),
        pytest.param({**_JEV_COLUMN, "slot": "embeddings"}, "embeddings", id="shared-slot"),
        pytest.param(
            {**_JEV_COLUMN, "prefix": None, "cache_system_prompt": False},
            "jev",
            id="legacy-snapshot-without-slot",
        ),
    ],
)
def test_effective_slot(column: dict[str, Any], slot: str) -> None:
    assert ColumnConfig.model_validate(column).effective_slot == slot


def test_minimal_config_uses_defaults() -> None:
    config = BenchmarkConfig.model_validate(_doc())
    assert config.jev.concurrency == 8
    assert config.tokens.bytes_per_token == 3.0
    assert config.llm.est_output_tokens_per_email == 400
    assert config.embeddings.temperature == 0.05
    assert config.column("anthropic").prefix == "anthropic/"


def test_unknown_column_raises_key_error() -> None:
    with pytest.raises(KeyError):
        BenchmarkConfig.model_validate(_doc()).column("missing")


@pytest.mark.parametrize(
    "doc",
    [
        pytest.param(_doc(columns=[]), id="no-columns"),
        pytest.param(
            _doc(columns=_doc()["columns"] + [_doc()["columns"][0]]),
            id="duplicate-columns",
        ),
        pytest.param(
            _doc(
                llm={
                    "system_prompt": "$questions $x",
                    "system_prompt_all_in_one": "$questions",
                }
            ),
            id="llm-unknown-placeholder",
        ),
        pytest.param(
            _doc(
                embeddings={
                    "email_template": "$secret",
                    "option_template": "$description",
                }
            ),
            id="email-unknown-placeholder",
        ),
        pytest.param(
            _doc(embeddings={"email_template": "$body", "option_template": "$email"}),
            id="option-unknown-placeholder",
        ),
        pytest.param(
            _doc(
                embeddings={
                    "email_template": "$body",
                    "option_template": "$description",
                    "temperature": 0,
                }
            ),
            id="zero-temperature",
        ),
        pytest.param(_doc(jev={"concurrency": 0}), id="zero-concurrency"),
        pytest.param(_doc(columns=[{**_JEV_COLUMN, "slot": "Bad Slot"}]), id="bad-slot-name"),
    ],
)
def test_invalid_configs_are_rejected(doc: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        BenchmarkConfig.model_validate(doc)


def test_load_benchmark_config(tmp_path: Path) -> None:
    path = tmp_path / "benchmark.toml"
    toml_content = (
        '[llm]\nsystem_prompt = "$questions"\nsystem_prompt_all_in_one = "$questions"\n'
        '[embeddings]\nemail_template = "$body"\noption_template = "$description"\n'
        '[[columns]]\nid = "jev"\ntitle = "Jev"\nkind = "decisions"\n'
        'modality = "decisions"\ndefault_model = "typesafe/jev-1.13"\n'
    )
    path.write_text(toml_content, encoding="utf-8")
    assert load_benchmark_config(path).column("jev").kind == "decisions"
