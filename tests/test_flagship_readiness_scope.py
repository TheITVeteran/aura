from __future__ import annotations

from pathlib import Path

from core.runtime.flagship_readiness import scan_codebase


def test_flagship_readiness_ignores_scratch_workspace(tmp_path: Path) -> None:
    scratch_file = tmp_path / "scratch" / "audit_bundler.py"
    scratch_file.parent.mkdir(parents=True)
    scratch_file.write_text('from pathlib import Path\nPath("x").write_text("scratch")\n', encoding="utf-8")

    report = scan_codebase(tmp_path)

    assert not any(issue.path.startswith("scratch/") for issue in report.issues)


def test_flagship_readiness_still_flags_production_write_text(tmp_path: Path) -> None:
    prod_file = tmp_path / "core" / "bad_writer.py"
    prod_file.parent.mkdir(parents=True)
    prod_file.write_text('from pathlib import Path\nout = Path("x")\nout.write_text("prod")\n', encoding="utf-8")

    report = scan_codebase(tmp_path)

    assert any(issue.code == "DIRECT_WRITE_TEXT" and issue.path == "core/bad_writer.py" for issue in report.issues)
