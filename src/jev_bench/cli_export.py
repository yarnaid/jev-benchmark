"""Async body of `jev-bench export-site`: export the static snapshot and report the outcome.

Functions:
    export_and_report: run export_site with the environment's Settings; print a green summary or a
        red reason on stderr; return the exit code.
"""

from pathlib import Path

from rich.console import Console
from rich.text import Text

from jev_bench.settings import load_settings
from jev_bench.site.errors import ExportError
from jev_bench.site.export import export_site

__all__ = [
    "console",
    "export_and_report",
]

console = Console(stderr=True)


async def export_and_report(out: Path, commit: str | None) -> int:
    try:
        summary = await export_site(load_settings(), out, commit=commit)
    except ExportError as exc:
        console.print(Text.assemble(("Export failed: ", "red"), str(exc)))
        return 1
    size = f"{summary.files} files · {summary.bytes / 1_000_000:.1f} MB · {summary.seconds:.1f} s"
    console.print(Text.assemble(("Exported ", "green"), f"{size} → {out}"))
    return 0
