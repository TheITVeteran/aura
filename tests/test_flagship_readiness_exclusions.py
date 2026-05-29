from pathlib import Path

from core.runtime.flagship_readiness import scan_codebase


def test_flagship_readiness_excludes_generated_artifacts(tmp_path: Path):
    artifact = tmp_path / "artifacts/current/generated_candidate.py"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("broken = 'unterminated\n", encoding="utf-8")

    report = scan_codebase(tmp_path)

    assert not any(issue.path.startswith("artifacts/") for issue in report.issues)


def test_flagship_readiness_still_scans_source_after_artifact_exclusion(tmp_path: Path):
    source = tmp_path / "core/runtime/bad.py"
    source.parent.mkdir(parents=True)
    source.write_text("broken = 'unterminated\n", encoding="utf-8")

    report = scan_codebase(tmp_path)

    assert any(issue.code == "SYNTAX_ERROR" and issue.path == "core/runtime/bad.py" for issue in report.issues)
