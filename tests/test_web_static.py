"""Contract tests for the static UI: no HTML-injection sinks, no inline scripts, pinned CDN
assets with SRI, and every local reference (script, stylesheet, ES-module import) resolves to an
existing file."""

import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

from jev_bench.web.app import STATIC_DIR

FORBIDDEN_SINKS = re.compile(r"\binnerHTML\b|\bouterHTML\b|insertAdjacentHTML|document\.write")
IMPORT = re.compile(r"""from\s+["'](\./[^"']+)["']""")
SHELL_MODULES = ("dom.js", "format.js", "storage.js", "key.js", "api.js", "layout.js", "widgets.js")
PAGE_MODULES = (
    "benchmark.js",
    "report.js",
    "charts.js",
    "generations.js",
    "explorer.js",
    "runs.js",
    "distribution.js",
    "answers.js",
    "selection.js",
    "email-detail.js",
    "glossary.js",
    "help.js",
    "palette.js",
    "column-card.js",
    "report-summary.js",
    "option-order.js",
    "analyze.js",
    "analysis-result.js",
    "analysis-setup.js",
    "analysis-links.js",
    "prompt-editor.js",
    "markdown.js",
    "markdown-render.js",
)


class TagCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[tuple[str, dict[str, str | None]]] = []
        self.inline_script = False
        self._in_script = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, dict(attrs)))
        self._in_script = tag == "script"

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self._in_script = False

    def handle_data(self, data: str) -> None:
        if self._in_script and data.strip():
            self.inline_script = True


def _pages() -> list[Path]:
    return sorted(STATIC_DIR.glob("*.html"))


def _parse(page: Path) -> TagCollector:
    collector = TagCollector()
    collector.feed(page.read_text(encoding="utf-8"))
    return collector


def test_shell_modules_exist() -> None:
    assert [name for name in SHELL_MODULES if not (STATIC_DIR / "js" / name).exists()] == []
    assert (STATIC_DIR / "css" / "app.css").exists()


def test_page_modules_exist() -> None:
    assert [name for name in PAGE_MODULES if not (STATIC_DIR / "js" / name).exists()] == []


def test_javascript_has_no_html_injection_sinks() -> None:
    offenders = [
        path.name
        for path in STATIC_DIR.glob("js/*.js")
        if FORBIDDEN_SINKS.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_module_imports_resolve() -> None:
    missing = [
        f"{path.name} -> {target}"
        for path in STATIC_DIR.glob("js/*.js")
        for target in IMPORT.findall(path.read_text(encoding="utf-8"))
        if not (path.parent / target).exists()
    ]
    assert missing == []


@pytest.mark.parametrize("page", _pages(), ids=lambda path: path.name)
def test_pages_follow_the_csp_contract(page: Path) -> None:
    collector = _parse(page)
    assert collector.inline_script is False
    for tag, attrs in collector.tags:
        assert not any(name.startswith("on") for name in attrs), f"inline handler in <{tag}>"
        url = attrs.get("src") or attrs.get("href") or ""
        if tag == "script":
            assert url, "inline <script> without src"
        if "cdn.jsdelivr.net" in url:
            assert re.search(r"@\d+\.\d+\.\d+/", url), f"unpinned CDN asset {url}"
            assert (attrs.get("integrity") or "").startswith("sha384-"), f"missing SRI on {url}"
            assert attrs.get("crossorigin") == "anonymous"
        if url.startswith("/") and not url.startswith("//"):
            assert (STATIC_DIR / url.lstrip("/")).exists(), f"missing local asset {url}"
