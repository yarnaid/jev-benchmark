"""Contract tests for the static UI: no HTML-injection sinks, no inline scripts, pinned CDN
assets with SRI, every local reference (script, stylesheet, ES-module import) resolves to an
existing file, and every configured column has the same chart color in palette.js and app.css."""

import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

from jev_bench.benchmark_config import load_benchmark_config
from jev_bench.web.app import STATIC_DIR

FORBIDDEN_SINKS = re.compile(r"\binnerHTML\b|\bouterHTML\b|insertAdjacentHTML|document\.write")
ROOT_ABSOLUTE = re.compile(r"""["'`]/(?:api\b|[\w-]+\.html)|href:\s*["'`]/["'`]|go\(\s*["'`]/""")
IMPORT = re.compile(r"""from\s+["'](\./[^"']+)["']""")
SERIES_BLOCK = re.compile(r"^const SERIES = \{(.*?)^\};", re.MULTILINE | re.DOTALL)
SERIES_ENTRY = re.compile(r'(\w+): \["(#[0-9a-f]{6})", "(#[0-9a-f]{6})"\]')
CSS_BLOCKS = re.compile(r'^(:root|\[data-bs-theme="dark"\]) \{(.*?)^\}', re.MULTILINE | re.DOTALL)
COLUMN_IDS = [
    column.id
    for column in load_benchmark_config(
        Path(__file__).resolve().parents[1] / "config" / "benchmark.toml"
    ).columns
]
SHELL_MODULES = (
    "dom.js",
    "format.js",
    "storage.js",
    "key.js",
    "api.js",
    "layout.js",
    "widgets.js",
    "deployment.js",
)
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
    "slots.js",
    "report-summary.js",
    "option-order.js",
    "analyze.js",
    "analysis-result.js",
    "analysis-setup.js",
    "analysis-links.js",
    "prompt-editor.js",
    "markdown.js",
    "markdown-render.js",
    "static-api.js",
    "snapshot.js",
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


def test_javascript_uses_relative_urls() -> None:
    offenders = [
        f"{path.name}: {match.group(0)}"
        for path in sorted(STATIC_DIR.glob("js/*.js"))
        for match in ROOT_ABSOLUTE.finditer(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


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
        if url and "://" not in url and not url.startswith("data:"):
            assert not url.startswith("/"), f"root-absolute URL {url}"
            assert (STATIC_DIR / url).exists(), f"missing local asset {url}"


def _read(relative: str) -> str:
    return (STATIC_DIR / relative).read_text(encoding="utf-8")


def _css_color(block: str, column_id: str) -> str | None:
    match = re.search(rf"--jb-col-{column_id}: (#[0-9a-f]{{6}});", block)
    return match.group(1) if match else None


@pytest.mark.parametrize("column_id", COLUMN_IDS)
def test_every_column_has_one_chart_color_in_palette_and_css(column_id: str) -> None:
    block = SERIES_BLOCK.search(_read("js/palette.js"))
    assert block is not None
    series = {key: steps for key, *steps in SERIES_ENTRY.findall(block.group(1))}
    css = _read("css/app.css")
    blocks = dict(CSS_BLOCKS.findall(css))
    themes = (":root", '[data-bs-theme="dark"]')
    assert series.get(column_id) == [_css_color(blocks[theme], column_id) for theme in themes]
    assert f".accent-{column_id} {{ --jb-accent: var(--jb-col-{column_id}); }}" in css
