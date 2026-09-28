"""Tests for jev_bench.web.routes.catalog."""

from pathlib import Path

import pytest
from tests.factories import AppFactory, FakeOpenRouter


def test_catalog_lists_columns_with_models_and_prices(make_app: AppFactory) -> None:
    columns = make_app(FakeOpenRouter()).get("/api/catalog").json()
    by_id = {column["id"]: column for column in columns}
    assert list(by_id) == ["jev", "anthropic", "embeddings", "kev"]
    anthropic = by_id["anthropic"]
    assert anthropic["default_model"] == "anthropic/claude-sonnet-5"
    assert [model["id"] for model in anthropic["models"]] == ["anthropic/claude-sonnet-5"]
    assert anthropic["models"][0]["prompt_price_per_m"] == pytest.approx(2.0)
    assert anthropic["models"][0]["max_completion_tokens"] == 128000
    assert anthropic["error"] is None
    embeddings = by_id["embeddings"]
    assert (embeddings["embedding_temperature"], embeddings["emails_per_request"]) == (0.05, 2)
    assert by_id["jev"]["embedding_temperature"] is None
    kev = by_id["kev"]
    assert [model["id"] for model in kev["models"]] == ["Kev-4B", "Kev-0.8B"]
    assert (kev["models"][0]["prompt_price_per_m"], kev["error"]) == (0.0, None)
    assert (kev["slot"], kev["calibrated"]) == ("embeddings", True)
    assert (by_id["embeddings"]["slot"], by_id["jev"]["slot"]) == ("embeddings", "jev")
    assert by_id["jev"]["calibrated"] is None


def test_catalog_outage_keeps_the_default_model(make_app: AppFactory) -> None:
    columns = make_app(FakeOpenRouter(models_status=503)).get("/api/catalog").json()
    jev = columns[0]
    assert "503" in jev["error"]
    assert [model["id"] for model in jev["models"]] == ["typesafe/jev-1.13"]
    assert jev["models"][0]["name"] == "typesafe/jev-1.13 (default)"
    assert columns[-1]["error"] is None


def test_catalog_with_invalid_config_returns_400(tmp_path: Path, make_app: AppFactory) -> None:
    client = make_app(FakeOpenRouter())
    (tmp_path / "config" / "benchmark.toml").write_text("not [valid", encoding="utf-8")
    response = client.get("/api/catalog")
    assert response.status_code == 400
    assert "invalid config" in response.json()["detail"]
