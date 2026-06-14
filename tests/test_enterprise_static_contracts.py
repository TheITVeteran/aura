from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

from core.runtime.subprocess_gateway import get_subprocess_gateway

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "config" / "aura_enterprise_gate_baseline.json"
GATE = ROOT / "tools" / "aura_enterprise_gate.py"


def _load_gate_module():
    spec = importlib.util.spec_from_file_location("aura_enterprise_gate_under_test", GATE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_static_gate(tmp_path: Path) -> dict:
    report_path = tmp_path / "enterprise_gate.json"
    env = os.environ.copy()
    env.setdefault("AURA_TEST_MODE", "1")
    env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    proc = get_subprocess_gateway().run(
        [
            sys.executable,
            str(GATE),
            "--root",
            str(ROOT),
            "--skip-compile",
            "--skip-pytest-collect",
            "--baseline",
            str(BASELINE),
            "--fail-on-regression",
            "--out",
            str(report_path),
        ],
        cwd=ROOT,
        env=env,
        timeout=60,
        offline_tooling=True,
        source="certification_tooling:test_enterprise_static_contracts",
    )
    assert proc.returncode == 0, proc.stdout[-4000:]
    return json.loads(report_path.read_text(encoding="utf-8"))


def test_enterprise_gate_baseline_blocks_static_regressions(tmp_path: Path):
    report = _run_static_gate(tmp_path)
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))

    assert report["python_files"] >= 2000
    assert (
        report["counts"].get("broad_exception_review", 0)
        <= baseline["max_counts"].get("broad_exception_review", 0)
    )
    assert not [
        finding
        for finding in report["findings"]
        if finding["kind"] == "baseline_regression"
    ]


def test_enterprise_gate_has_zero_secret_literals_and_shell_true(tmp_path: Path):
    report = _run_static_gate(tmp_path)
    counts = report["counts"]

    assert counts.get("potential_secret", 0) == 0
    assert counts.get("subprocess_shell_true", 0) == 0


def test_enterprise_gate_todo_markers_are_comment_markers_not_training_words(tmp_path: Path):
    gate = _load_gate_module()
    sample = tmp_path / "sample.py"
    sample.write_text(
        "\n".join(
            [
                'PROMPT = "Can you hack it? This is training data, not code debt."',
                "# This comment mentions no TODO string as a policy example.",
                "# TODO: replace temporary fixture with production path",
                "",
            ]
        ),
        encoding="utf-8",
    )
    report = gate.GateReport(root=str(tmp_path), generated_at_unix=0.0)

    gate.scan_file(sample, tmp_path, report)

    todo_findings = [finding for finding in report.findings if finding.kind == "todo_fixme_hack"]
    assert [(finding.file, finding.line) for finding in todo_findings] == [("sample.py", 3)]
