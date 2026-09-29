"""Tests for jev_bench.cli_export."""

from pathlib import Path

import pytest

from jev_bench import cli_export
from jev_bench.site.errors import ExportError
from jev_bench.site.export import ExportSummary


@pytest.mark.parametrize(
    ("outcome", "code", "text"),
    [
        pytest.param(
            ExportSummary(files=3, bytes=2_500_000, seconds=1.3),
            0,
            "Exported 3 files · 2.5 MB · 1.3 s",
            id="success",
        ),
        pytest.param(
            ExportError("jobs still running: r1"),
            1,
            "Export failed: jobs still running: r1",
            id="failure",
        ),
    ],
)
async def test_export_and_report(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    outcome: ExportSummary | ExportError,
    code: int,
    text: str,
) -> None:
    seen: list[tuple[Path, str | None]] = []

    async def fake(settings: object, out: Path, *, commit: str | None = None) -> ExportSummary:
        seen.append((out, commit))
        if isinstance(outcome, ExportError):
            raise outcome
        return outcome

    monkeypatch.setattr(cli_export, "export_site", fake)
    assert await cli_export.export_and_report(tmp_path / "site", "abc") == code
    assert seen == [(tmp_path / "site", "abc")]
    assert text in capsys.readouterr().err
