import asyncio
import json
import subprocess
import sys
from pathlib import Path

from core.architecture_quality.gate import ArchitectureQualityGate, ArchitectureQualityPolicy
from core.architecture_quality.scorer import score_codebase


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_scorer_detects_project_local_import_cycle(tmp_path: Path):
    _write(tmp_path / "core" / "__init__.py", "")
    _write(tmp_path / "core" / "a.py", "import core.b\nVALUE = 1\n")
    _write(tmp_path / "core" / "b.py", "from core import a\nVALUE = a.VALUE\n")

    report = score_codebase(tmp_path, include_roots=("core",))

    assert report.metrics.module_count == 3
    assert report.metrics.cycle_count == 1
    assert ("core.a", "core.b") in report.cycles
    assert any(finding.code == "import_cycle" for finding in report.findings)


def test_gate_rejects_overlay_that_introduces_cycle(tmp_path: Path):
    _write(tmp_path / "core" / "__init__.py", "")
    _write(tmp_path / "core" / "a.py", "VALUE = 1\n")
    _write(tmp_path / "core" / "b.py", "import core.a\nVALUE = core.a.VALUE\n")

    gate = ArchitectureQualityGate(tmp_path, include_roots=("core",))
    result = gate.evaluate_overlay(
        {"core/a.py": "import core.b\nVALUE = core.b.VALUE\n"},
        changed_paths=("core/a.py",),
    )

    assert not result.passed
    assert any("new import cycle" in reason for reason in result.reasons)


def test_gate_allows_cycle_shrinkage_even_when_membership_changes(tmp_path: Path):
    _write(tmp_path / "core" / "__init__.py", "")
    _write(tmp_path / "core" / "a.py", "import core.b\nVALUE = 1\n")
    _write(tmp_path / "core" / "b.py", "import core.c\nVALUE = 2\n")
    _write(tmp_path / "core" / "c.py", "import core.a\nVALUE = 3\n")

    gate = ArchitectureQualityGate(tmp_path, include_roots=("core",))
    result = gate.evaluate_overlay(
        {"core/c.py": "import core.b\nVALUE = 3\n"},
        changed_paths=("core/c.py",),
    )

    assert result.passed
    assert result.after.metrics.largest_cycle_size < result.before.metrics.largest_cycle_size


def test_gate_allows_large_cycle_decomposition_into_smaller_cycles(tmp_path: Path):
    _write(tmp_path / "core" / "__init__.py", "")
    for idx in range(12):
        current = chr(ord("a") + idx)
        nxt = chr(ord("a") + ((idx + 1) % 12))
        _write(tmp_path / "core" / f"{current}.py", f"import core.{nxt}\n")

    overlay = {}
    for idx in range(12):
        current = chr(ord("a") + idx)
        if idx < 3:
            nxt = chr(ord("a") + ((idx + 1) % 3))
        elif idx < 6:
            nxt = chr(ord("a") + 3 + ((idx - 3 + 1) % 3))
        else:
            nxt = None
        overlay[f"core/{current}.py"] = f"import core.{nxt}\n" if nxt else "VALUE = 1\n"

    gate = ArchitectureQualityGate(tmp_path, include_roots=("core",))
    result = gate.evaluate_overlay(overlay, changed_paths=overlay.keys())

    assert result.passed
    assert result.before.metrics.cycle_count == 1
    assert result.after.metrics.cycle_count == 2
    assert result.after.metrics.largest_cycle_size == 3


def test_gate_rejects_large_file_growth(tmp_path: Path):
    _write(tmp_path / "core" / "__init__.py", "")
    base_content = "\n".join(f"BASE_{idx} = {idx}" for idx in range(25)) + "\n"
    grown_content = "\n".join(f"NEXT_{idx} = {idx}" for idx in range(60)) + "\n"
    _write(tmp_path / "core" / "big.py", base_content)

    gate = ArchitectureQualityGate(
        tmp_path,
        include_roots=("core",),
        god_file_threshold=20,
        policy=ArchitectureQualityPolicy(max_line_growth_for_large_file=10),
    )
    result = gate.evaluate_overlay(
        {"core/big.py": grown_content},
        changed_paths=("core/big.py",),
    )

    assert not result.passed
    assert any("grew to 60 lines" in reason for reason in result.reasons)


def test_gate_allows_neutral_overlay(tmp_path: Path):
    _write(tmp_path / "core" / "__init__.py", "")
    _write(tmp_path / "core" / "a.py", "VALUE = 1\n")

    gate = ArchitectureQualityGate(tmp_path, include_roots=("core",))
    result = gate.evaluate_overlay(
        {"core/a.py": "VALUE = 2\n"},
        changed_paths=("core/a.py",),
    )

    assert result.passed
    assert "architecture quality passed" in result.summary()


def test_cli_compares_against_written_baseline(tmp_path: Path):
    repo = tmp_path / "repo"
    _write(repo / "core" / "__init__.py", "")
    _write(repo / "core" / "a.py", "VALUE = 1\n")
    baseline = tmp_path / "baseline.json"
    tool = Path("tools/closeout/architecture_quality_gate.py").resolve()

    subprocess.run(
        [
            sys.executable,
            str(tool),
            "--root",
            str(repo),
            "--include",
            "core",
            "--write-baseline",
            str(baseline),
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(baseline.read_text(encoding="utf-8"))
    assert payload["metrics"]["module_count"] == 2

    result = subprocess.run(
        [
            sys.executable,
            str(tool),
            "--root",
            str(repo),
            "--include",
            "core",
            "--baseline",
            str(baseline),
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout)["passed"] is True


def test_self_modification_hook_rejects_architecture_regression(tmp_path: Path):
    from core.self_modification.safe_modification import SafeSelfModification

    _write(tmp_path / "core" / "__init__.py", "")
    _write(tmp_path / "core" / "a.py", "VALUE = 1\n")
    _write(tmp_path / "core" / "b.py", "import core.a\nVALUE = core.a.VALUE\n")

    system = object.__new__(SafeSelfModification)
    system.code_base = tmp_path

    ok, message = asyncio.run(
        SafeSelfModification._run_architecture_quality_gate(
            system,
            "core/a.py",
            "import core.b\nVALUE = core.b.VALUE\n",
        )
    )

    assert not ok
    assert "new import cycle" in message
