"""Static half of the snapshot: the UI files, switched to static mode, with the CSP as meta tags.

GitHub Pages cannot send response headers, so the policy of `web.security.CSP` (still the one
source) goes into a <meta> tag, minus `frame-ancestors`, which browsers ignore there.

Constants:
    STATIC_MODULE: the `js/deployment.js` written into the snapshot.
Functions:
    meta_policy: the CSP for a <meta> tag.
    head_tags: the CSP and referrer meta tags inserted right after <head>.
    copy_static: copy the UI into `out`, write STATIC_MODULE, insert head_tags into every page.
"""

import shutil
from pathlib import Path

from jev_bench.site.errors import ExportError
from jev_bench.web.security import CSP

__all__ = [
    "STATIC_MODULE",
    "copy_static",
    "head_tags",
    "meta_policy",
]

STATIC_MODULE = (
    "/** Static snapshot flag, written by `jev-bench export-site`. Exports: STATIC. */\n"
    "export const STATIC = true;\n"
)
HEAD = "<head>"


def meta_policy(policy: str = CSP) -> str:
    parts = policy.split("; ")
    return "; ".join(part for part in parts if not part.startswith("frame-ancestors"))


def head_tags() -> str:
    return (
        f'\n  <meta http-equiv="Content-Security-Policy" content="{meta_policy()}">'
        '\n  <meta name="referrer" content="no-referrer">'
    )


def copy_static(source: Path, out: Path) -> None:
    shutil.copytree(source, out, dirs_exist_ok=True)
    (out / "js" / "deployment.js").write_text(STATIC_MODULE, encoding="utf-8")
    for page in sorted(out.glob("*.html")):
        _insert_head_tags(page)


def _insert_head_tags(page: Path) -> None:
    html = page.read_text(encoding="utf-8")
    if html.count(HEAD) != 1:
        raise ExportError(f"{page.name}: expected exactly one {HEAD}")
    page.write_text(html.replace(HEAD, HEAD + head_tags(), 1), encoding="utf-8")
