"""Tests for jev_bench.site.static_copy."""

from pathlib import Path

import pytest

from jev_bench.site.errors import ExportError
from jev_bench.site.static_copy import STATIC_MODULE, copy_static
from jev_bench.web.app import STATIC_DIR


def _source(root: Path, head: str = "<head>") -> Path:
    source = root / "ui"
    (source / "js").mkdir(parents=True)
    page = f"<!doctype html><html>{head}<title>x</title></head></html>"
    (source / "index.html").write_text(page, encoding="utf-8")
    (source / "js" / "deployment.js").write_text("export const STATIC = false;\n", encoding="utf-8")
    return source


def test_copy_static_switches_to_static_mode_and_adds_the_policy(tmp_path: Path) -> None:
    out = tmp_path / "out"
    copy_static(_source(tmp_path), out)
    assert (out / "js" / "deployment.js").read_text(encoding="utf-8") == STATIC_MODULE
    html = (out / "index.html").read_text(encoding="utf-8")
    assert html.count('http-equiv="Content-Security-Policy"') == 1
    assert '<meta name="referrer" content="no-referrer">' in html
    assert "default-src 'self'" in html
    assert "frame-ancestors" not in html


@pytest.mark.parametrize(
    "head",
    [
        pytest.param("", id="no-head"),
        pytest.param("<head></head><head>", id="two-heads"),
    ],
)
def test_copy_static_refuses_a_page_without_exactly_one_head(tmp_path: Path, head: str) -> None:
    with pytest.raises(ExportError, match="exactly one <head>"):
        copy_static(_source(tmp_path, head), tmp_path / "out")


@pytest.mark.parametrize("page", sorted(STATIC_DIR.glob("*.html")), ids=lambda path: path.name)
def test_every_real_page_has_exactly_one_head(page: Path) -> None:
    assert page.read_text(encoding="utf-8").count("<head>") == 1
