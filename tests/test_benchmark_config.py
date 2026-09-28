"""Tests for jev_bench.benchmark_config."""

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from jev_bench.benchmark_config import (
    BenchmarkConfig,
    ColumnConfig,
    KevParams,
    load_benchmark_config,
)


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


_KEV: dict[str, Any] = {
    "id": "kev",
    "title": "Kev",
    "kind": "kev",
    "models": ["Kev-4B", "Kev-0.8B"],
    "default_model": "Kev-4B",
    "slot": "embeddings",
}
_JEV_COLUMN: dict[str, Any] = _doc()["columns"][0]


def test_kev_column_and_params_defaults() -> None:
    config = BenchmarkConfig.model_validate(_doc(columns=[_JEV_COLUMN, _KEV]))
    kev = config.column("kev")
    assert (kev.models, kev.modality, kev.effective_slot) == (
        ("Kev-4B", "Kev-0.8B"),
        None,
        "embeddings",
    )
    assert config.column("jev").effective_slot == "jev"
    assert config.kev == KevParams()
    assert (config.kev.space_url, config.kev.api_name, config.kev.calibrated) == (
        "https://jaredpalmer-kev.hf.space",
        "decide",
        True,
    )
    assert (config.kev.concurrency, config.kev.timeout_s) == (2, 300.0)


def test_legacy_column_snapshot_still_validates() -> None:
    column = ColumnConfig.model_validate(
        {**_JEV_COLUMN, "prefix": None, "cache_system_prompt": False}
    )
    assert (column.models, column.slot, column.effective_slot) == ((), None, "jev")


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
        pytest.param(_doc(columns=[{**_KEV, "models": []}]), id="kev-without-models"),
        pytest.param(
            _doc(columns=[{**_KEV, "default_model": "Kev-9B"}]), id="kev-default-not-in-models"
        ),
        pytest.param(_doc(columns=[{**_KEV, "modality": "decisions"}]), id="kev-with-modality"),
        pytest.param(_doc(columns=[{**_KEV, "prefix": "kev/"}]), id="kev-with-prefix"),
        pytest.param(
            _doc(columns=[{k: v for k, v in _JEV_COLUMN.items() if k != "modality"}]),
            id="decisions-without-modality",
        ),
        pytest.param(
            _doc(columns=[{**_JEV_COLUMN, "models": ["x"]}]), id="static-models-on-decisions"
        ),
        pytest.param(_doc(columns=[{**_KEV, "slot": "Bad Slot"}]), id="bad-slot-name"),
        pytest.param(_doc(kev={"space_url": "https://kev.test/"}), id="space-url-trailing-slash"),
        pytest.param(_doc(kev={"space_url": "kev.test"}), id="space-url-without-scheme"),
        pytest.param(_doc(kev={"timeout_s": 0}), id="zero-kev-timeout"),
        pytest.param(_doc(kev={"concurrency": 0}), id="zero-kev-concurrency"),
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
