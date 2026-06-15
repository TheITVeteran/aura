"""tests/test_proof_fabrication_guard.py
=============================================
Regression guard for the proof-fabrication class removed on 2026-06-15. Two
proof tools had hardcoded their scores and asserted victory. This test proves
(a) the proof tools are now clean, and (b) the guard actually catches both
fabrication mechanisms (so it can't silently rot).
"""
from __future__ import annotations

from pathlib import Path

from tools.proof_fabrication_guard import ROOT, scan, scan_source


def test_real_proof_tools_are_clean():
    findings = scan(ROOT)
    assert findings == [], f"proof fabrication reintroduced: {[f.__dict__ for f in findings]}"


def test_guard_catches_synthetic_score_from_probe():
    src = (
        "def main():\n"
        "    cpi = passed / 5.0\n"
        "    base_perf = 0.86 + (cpi * 0.04)\n"
        "    return base_perf\n"
    )
    findings = scan_source("tools/agi/fake.py", src)
    kinds = {f.kind for f in findings}
    assert "synthetic_score_from_probe" in kinds


def test_guard_catches_assert_victory_over_hardcoded_scores():
    src = (
        'baselines = {\n'
        '    "raw_model": {"mean_score": 0.58},\n'
        '    "prompted_model": {"mean_score": 0.72},\n'
        '}\n'
        'aura = {"mean_score": 0.88}\n'
        'assert aura["mean_score"] > baselines["raw_model"]["mean_score"] + 0.10\n'
    )
    findings = scan_source("tools/agi/fake.py", src)
    kinds = {f.kind for f in findings}
    assert "assert_victory_over_hardcoded_scores" in kinds


def test_guard_does_not_flag_honest_controlled_smoke():
    # Labeled controlled-smoke fixtures with NO assert-victory must pass (this is
    # the honest sovereignty-gauntlet pattern).
    src = (
        'ablations = {\n'
        '    "evidence_level": "controlled_smoke_ablation",\n'
        '    "ablation_effects_verified": False,\n'
        '    "no_memory": {"score": 0.52, "lesion_effect_verified": False},\n'
        '    "no_will": {"score": 0.48, "lesion_effect_verified": False},\n'
        '    "claim_boundary": "controlled smoke; does not establish live lift",\n'
        '}\n'
    )
    findings = scan_source("tools/proof/fake.py", src)
    assert findings == []


def test_guard_does_not_flag_negative_control():
    # A negative control intentionally builds a fake score to prove the validator
    # REJECTS it — there is no assert-victory, so it must pass.
    src = (
        'fake_report = {"mean_score": 0.95, "notes": "projected, no traces"}\n'
        'is_valid = validate_report_score(fake_report)\n'
        'assert is_valid is False  # must be rejected\n'
    )
    findings = scan_source("tools/agi/fake.py", src)
    assert findings == []


def test_guard_main_exit_codes(tmp_path, capsys):
    from tools.proof_fabrication_guard import main

    # Clean tree (no tools/agi or tools/proof under tmp) → pass.
    rc = main(["--root", str(tmp_path)])
    assert rc == 0

    # Plant a fabrication and confirm non-zero exit.
    bad = tmp_path / "tools" / "agi"
    bad.mkdir(parents=True)
    (bad / "evil.py").write_text("def m():\n    base_perf = 0.8 + (cpi * 0.04)\n")
    assert main(["--root", str(tmp_path)]) == 1
    assert Path(bad / "evil.py").exists()
