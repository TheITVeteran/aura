"""Deterministic, always-openable artifact builder.

Bryan's concern: when Aura wants to *show* something ("something like this?"),
she should not depend on Excel or any specific app being installed. This
builds real, openable artifacts in portable formats that work on any machine:

- **table** → CSV (opens in Excel/Numbers/Sheets/any editor) plus a styled
  HTML view (opens in any browser) — no spreadsheet app required.
- **doc** → HTML (rich, universal) with an optional Markdown source.
- **program** → a self-contained runnable file (Python or a single-file HTML
  app) she can point the user at.

All writes go through the governed file-write gateway. The output is a real
path on disk, ready to open — the deterministic backbone under the
``demonstrate_artifact`` affordance (the autonomous task engine remains the
richer, creative path; this is the floor that always succeeds).
"""
from __future__ import annotations

import html
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger("Aura.ArtifactBuilder")


@dataclass(frozen=True)
class BuiltArtifact:
    ok: bool
    kind: str
    paths: list[str]
    primary: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "kind": self.kind,
            "paths": self.paths,
            "primary": self.primary,
            "detail": self.detail,
        }


def _output_dir() -> Path:
    try:
        from core.config import config

        base = Path(config.paths.data_dir)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        base = Path.home() / ".aura" / "data"
    out = base / "generated_artifacts"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write(path: Path, text: str) -> None:
    """Governed write — falls back to a direct atomic write off the live runtime."""
    try:
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        with local_internal_governed_scope("artifact_builder.write", domain="file_write"):
            get_file_write_gateway().write_text(path, text, source="artifact_builder", durable=True)
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
        logger.debug("Governed write unavailable, using atomic fallback: %s", exc)
        from core.runtime.atomic_writer import atomic_write_text

        atomic_write_text(path, text, encoding="utf-8", durable=True)


def _csv_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    if any(c in text for c in (",", '"', "\n")):
        return '"' + text.replace('"', '""') + '"'
    return text


def build_table(
    rows: Sequence[Sequence[Any]],
    *,
    headers: Sequence[str] | None = None,
    title: str = "Table",
    stem: str | None = None,
) -> BuiltArtifact:
    """Build a table as CSV (universal) + styled HTML (any browser)."""
    stem = stem or f"table_{int(time.time())}"
    out = _output_dir()
    all_rows: list[Sequence[Any]] = []
    if headers:
        all_rows.append(list(headers))
    all_rows.extend(rows)
    if not all_rows:
        return BuiltArtifact(False, "table", [], "", "no rows provided")

    csv_text = "\n".join(",".join(_csv_cell(c) for c in row) for row in all_rows) + "\n"
    csv_path = out / f"{stem}.csv"
    _write(csv_path, csv_text)

    def _row_html(cells: Sequence[Any], tag: str) -> str:
        return "<tr>" + "".join(f"<{tag}>{html.escape(str(c))}</{tag}>" for c in cells) + "</tr>"

    body_rows = ""
    data_rows = rows if headers else all_rows
    header_html = _row_html(headers, "th") if headers else ""
    for row in data_rows:
        body_rows += _row_html(row, "td")
    html_text = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
 body{{font-family:-apple-system,system-ui,sans-serif;margin:2rem;color:#1a1a1a}}
 h1{{font-size:1.2rem}}
 table{{border-collapse:collapse;width:100%}}
 th,td{{border:1px solid #d0d0d0;padding:.5rem .75rem;text-align:left}}
 th{{background:#f3f4f6;font-weight:600}}
 tr:nth-child(even) td{{background:#fafafa}}
</style></head><body>
<h1>{html.escape(title)}</h1>
<table><thead>{header_html}</thead><tbody>{body_rows}</tbody></table>
</body></html>"""
    html_path = out / f"{stem}.html"
    _write(html_path, html_text)

    return BuiltArtifact(
        True, "table",
        [str(csv_path), str(html_path)],
        str(csv_path),
        f"{len(data_rows)} data row(s); CSV opens in any spreadsheet, HTML in any browser",
    )


def build_doc(body_markdown: str, *, title: str = "Document", stem: str | None = None) -> BuiltArtifact:
    """Build a portable document as HTML (with the Markdown source alongside)."""
    stem = stem or f"doc_{int(time.time())}"
    out = _output_dir()
    md_path = out / f"{stem}.md"
    _write(md_path, f"# {title}\n\n{body_markdown}\n")

    # Minimal, dependency-free markdown → HTML (headings, bold, paragraphs).
    lines_html = []
    for line in body_markdown.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("### "):
            lines_html.append(f"<h3>{html.escape(stripped[4:])}</h3>")
        elif stripped.startswith("## "):
            lines_html.append(f"<h2>{html.escape(stripped[3:])}</h2>")
        elif stripped.startswith("# "):
            lines_html.append(f"<h1>{html.escape(stripped[2:])}</h1>")
        elif stripped.startswith(("- ", "* ")):
            lines_html.append(f"<li>{html.escape(stripped[2:])}</li>")
        else:
            lines_html.append(f"<p>{html.escape(stripped)}</p>")
    html_text = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>body{{font-family:-apple-system,system-ui,sans-serif;max-width:44rem;margin:2rem auto;line-height:1.6;color:#1a1a1a}}</style>
</head><body><h1>{html.escape(title)}</h1>{''.join(lines_html)}</body></html>"""
    html_path = out / f"{stem}.html"
    _write(html_path, html_text)
    return BuiltArtifact(True, "doc", [str(html_path), str(md_path)], str(html_path),
                         "HTML opens in any browser; Markdown source alongside")


def build_program(source: str, *, language: str = "python", stem: str | None = None) -> BuiltArtifact:
    """Write a self-contained runnable program she can point the user at."""
    stem = stem or f"program_{int(time.time())}"
    out = _output_dir()
    ext = {"python": "py", "html": "html", "javascript": "js", "shell": "sh"}.get(language.lower(), "txt")
    path = out / f"{stem}.{ext}"
    _write(path, source if source.endswith("\n") else source + "\n")
    return BuiltArtifact(True, "program", [str(path)], str(path), f"self-contained {language} file")


async def open_artifact(path: str) -> bool:
    """Open an artifact with the OS default handler (governed subprocess, best-effort)."""
    try:
        from core.runtime.subprocess_gateway import get_subprocess_gateway

        result = get_subprocess_gateway().run(
            ["open", str(path)],
            timeout=10.0,
            offline_tooling=True,
            check=False,
            source="maintenance_tooling:artifact_builder_open",
        )
        return getattr(result, "returncode", 1) == 0
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
        logger.debug("Artifact open skipped: %s", exc)
        return False
