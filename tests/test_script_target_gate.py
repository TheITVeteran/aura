"""The shell-script path gate, and proof it fails when it should.

`scripts/run_audit_suite.sh quick` named `crucible_test.py` for months after
494cb0a4b deleted it. Under `set -euo pipefail` pytest exits 4 on an
unrecognised path, so the documented validation entrypoint aborted before its
first test. tools/check_script_targets.py is the standing check.

A gate is only worth its runtime if it can fail, so the negative control here
rebuilds the exact defect in a temporary repo and asserts the gate rejects it.
Without that, a gate that always prints a tick is indistinguishable from one
whose matcher stopped matching.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GATE = PROJECT_ROOT / "tools" / "check_script_targets.py"


def _run_gate(cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(GATE)],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _git_repo(root: Path, scripts: dict[str, str]) -> Path:
    """A throwaway git repo holding `scripts`, because the gate reads git."""
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    for rel, body in scripts.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    return root


def test_repo_shell_scripts_name_only_existing_paths() -> None:
    result = _run_gate(PROJECT_ROOT)
    assert result.returncode == 0, result.stdout + result.stderr


def test_gate_rejects_a_deleted_pytest_target(tmp_path: Path) -> None:
    """The negative control: the CP defect itself, rebuilt."""
    root = _git_repo(
        tmp_path,
        {
            "scripts/run_audit_suite.sh": (
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "python -m pytest tests/test_audit_contracts.py crucible_test.py -q\n"
            ),
            "tests/test_audit_contracts.py": "",
        },
    )
    # The gate lives in this repo but reads the repo it is run from.
    gate_copy = root / "tools" / "check_script_targets.py"
    gate_copy.parent.mkdir(parents=True, exist_ok=True)
    gate_copy.write_text(GATE.read_text(encoding="utf-8"), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(gate_copy)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, result.stdout
    assert "crucible_test.py" in result.stderr


def test_gate_ignores_paths_inside_printed_text(tmp_path: Path) -> None:
    """An echoed upstream filename is documentation, not an operand."""
    root = _git_repo(
        tmp_path,
        {
            "scripts/template.sh": (
                "#!/usr/bin/env bash\n"
                'echo "accelerate launch run_clm.py --do_train"\n'
                "cat <<EOF\n"
                "python vendor_only/train.py\n"
                "EOF\n"
            ),
        },
    )
    gate_copy = root / "tools" / "check_script_targets.py"
    gate_copy.parent.mkdir(parents=True, exist_ok=True)
    gate_copy.write_text(GATE.read_text(encoding="utf-8"), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(gate_copy)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_gate_ignores_interpolated_prefixes(tmp_path: Path) -> None:
    """`${ROOT}/scripts/x.py` says nothing about this repo's scripts/x.py."""
    root = _git_repo(
        tmp_path,
        {
            "scripts/deploy.sh": (
                "#!/usr/bin/env bash\n"
                'REMOTE=/srv/app\n'
                'ssh host "python ${REMOTE}/scripts/absent.py"\n'
            ),
        },
    )
    gate_copy = root / "tools" / "check_script_targets.py"
    gate_copy.parent.mkdir(parents=True, exist_ok=True)
    gate_copy.write_text(GATE.read_text(encoding="utf-8"), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(gate_copy)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("mode", ["quick"])
def test_audit_suite_targets_exist(mode: str) -> None:
    """Every target `run_audit_suite.sh quick` names is present."""
    script = (PROJECT_ROOT / "scripts" / "run_audit_suite.sh").read_text(
        encoding="utf-8"
    )
    block = script.split("QUICK_TARGETS=(", 1)[1].split(")", 1)[0]
    targets = [line.strip() for line in block.splitlines() if line.strip()]
    assert targets, "quick mode names no test targets"
    # The comment above QUICK_TARGETS records why crucible_test.py went; the
    # contract is that no target *line* names it.
    assert "crucible_test.py" not in targets
    for target in targets:
        assert (PROJECT_ROOT / target).exists(), f"{mode}: missing {target}"
