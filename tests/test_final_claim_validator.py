from __future__ import annotations

import re
import json
import time
from pathlib import Path

from tools import final_claim_validator


def _claims_matrix(extra: str = "", *, include_closure: bool = True) -> str:
    closure = (
        "Final closure statement: "
        f"{final_claim_validator.EVIDENCE_LIMITED_CLOSURE_STATEMENT}\n\n"
        if include_closure
        else ""
    )
    return (
        "# Aura Claims Matrix\n\n"
        f"{closure}"
        "| Claim | Classification | Evidence Path / Blocker |\n"
        "| :--- | :--- | :--- |\n"
        "| **13. DNU AGI** | `not proven` | Local batteries do not prove AGI. |\n"
        "| **14. AGI-Candidate** | `not proven` | Requires full final-proof evidence stack. |\n"
        "| **15. Local Production Gate Readiness** | `not proven` | Requires current gate artifacts. |\n"
        "| **16. Mature RSI** | `not proven` | Requires repeated autonomous capability gains. |\n"
        "| **17. Subjective Consciousness** | `not proven` | Unsupported. |\n"
        "| **18. Personhood** | `not proven` | Unsupported. |\n"
        "| **19. Metaphysical Free Will** | `not proven` | Unsupported. |\n"
        "| **20. Indefinite Autonomy** | `not proven` | Requires long soak evidence. |\n"
        "| **21. Synthetic Cognitive Entity** | `not proven` | Requires unified scenario evidence. |\n"
        f"\n{extra}\n"
    )


def test_final_claim_validator_requires_evidence_limited_closure_statement(tmp_path: Path):
    claims_path = tmp_path / "CLAIMS_MATRIX.md"
    claims_path.write_text(_claims_matrix(include_closure=False), encoding="utf-8")

    result = final_claim_validator.main(["--claims", str(claims_path), "--artifacts", str(tmp_path / "artifacts")])

    assert result == 1


def test_final_claim_validator_rejects_active_overclaim_language(tmp_path: Path):
    claims_path = tmp_path / "CLAIMS_MATRIX.md"
    claims_path.write_text(_claims_matrix("Aura is conscious."), encoding="utf-8")

    result = final_claim_validator.main(["--claims", str(claims_path), "--artifacts", str(tmp_path / "artifacts")])

    assert result == 1


def test_claim_language_scan_allows_explicit_boundary_language(tmp_path: Path):
    claims_path = tmp_path / "CLAIMS_MATRIX.md"
    claims_path.write_text(
        _claims_matrix(
            "Aura is not conscious, AGI is not proven, and personhood is strictly unsupported."
        ),
        encoding="utf-8",
    )

    findings = final_claim_validator.validate_claim_language(tmp_path, claims_path)

    assert findings == []


def test_final_claim_validator_rejects_production_sealed_as_active_claim(tmp_path: Path):
    claims_path = tmp_path / "CLAIMS_MATRIX.md"
    claims_path.write_text(
        _claims_matrix(
            "| **15. Production-Sealed** | `locally demonstrated` | Overbroad label. |\n"
        ),
        encoding="utf-8",
    )

    result = final_claim_validator.main(["--claims", str(claims_path), "--artifacts", str(tmp_path / "artifacts")])

    assert result == 1


def test_final_claim_validator_rejects_stale_failed_dnu_for_agi_candidate(tmp_path: Path):
    claims_path = tmp_path / "CLAIMS_MATRIX.md"
    claims_path.write_text(
        _claims_matrix(
            "| **14. AGI-Candidate** | `locally demonstrated` | Requires fresh final-proof evidence. |\n"
        ),
        encoding="utf-8",
    )
    artifacts = tmp_path / "artifacts"
    step = artifacts / "proof_steps" / "dnu_agi_battery.json"
    step.parent.mkdir(parents=True, exist_ok=True)
    step.write_text(
        json.dumps(
            {
                "name": "dnu_agi_battery",
                "passed": False,
                "returncode": 1,
                "timed_out": False,
                "finished_at": time.time(),
            }
        ),
        encoding="utf-8",
    )

    result = final_claim_validator.main(["--claims", str(claims_path), "--artifacts", str(artifacts)])

    assert result == 1
    report = json.loads((artifacts / "final_claim_validation.json").read_text())
    assert any("dnu_agi_battery" in reason for reason in report["reasons"])


def test_final_claim_validator_requires_live_desktop_evidence_for_production_readiness(tmp_path: Path):
    claims_path = tmp_path / "CLAIMS_MATRIX.md"
    claims_path.write_text(
        _claims_matrix(
            "| **15. Local Production Gate Readiness** | `locally demonstrated` | Requires live desktop proof. |\n"
        ),
        encoding="utf-8",
    )
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "production_surface_lint.json").write_text('{"passed": true}', encoding="utf-8")
    (artifacts / "production_readiness.json").write_text('{"passed": true}', encoding="utf-8")

    result = final_claim_validator.main(["--claims", str(claims_path), "--artifacts", str(artifacts)])

    assert result == 1
    report = json.loads((artifacts / "final_claim_validation.json").read_text())
    assert any("live desktop runtime" in reason.lower() for reason in report["reasons"])


def test_final_claim_validator_accepts_clean_live_desktop_production_readiness(tmp_path: Path):
    claims_path = tmp_path / "CLAIMS_MATRIX.md"
    claims_path.write_text(
        _claims_matrix(
            "| **15. Local Production Gate Readiness** | `locally demonstrated` | Requires live desktop proof. |\n"
        ),
        encoding="utf-8",
    )
    artifacts = tmp_path / "artifacts"
    proof_steps = artifacts / "proof_steps"
    live_dir = artifacts / "live_desktop_runtime"
    proof_steps.mkdir(parents=True)
    live_dir.mkdir(parents=True)
    started_at = time.time()
    (artifacts / "production_surface_lint.json").write_text('{"passed": true}', encoding="utf-8")
    (artifacts / "production_readiness.json").write_text('{"passed": true}', encoding="utf-8")
    (proof_steps / "live_desktop_runtime.json").write_text(
        json.dumps(
            {
                "name": "live_desktop_runtime",
                "passed": True,
                "returncode": 0,
                "timed_out": False,
                "started_at": started_at,
                "finished_at": started_at + 1,
            }
        ),
        encoding="utf-8",
    )
    (live_dir / "LATEST_VERDICT.json").write_text(
        json.dumps(
            {
                "schema": "aura.live_boot_proof.v1",
                "passed": True,
                "mode": "desktop",
                "git_dirty": False,
                "peak_rss_mb": 25_600.0,
                "steps": [
                    {"step": "boot_health", "ok": True},
                    {"step": "chat_capability_inventory", "ok": True},
                    {"step": "chat_continuity", "ok": True},
                    {"step": "chat_conversation_soak", "ok": True},
                    {"step": "desktop_action", "ok": True},
                    {"step": "chat_restart_continuity", "ok": True},
                    {"step": "runtime_stream_scan", "ok": True},
                    {"step": "shutdown", "ok": True},
                ],
            }
        ),
        encoding="utf-8",
    )

    result = final_claim_validator.main(["--claims", str(claims_path), "--artifacts", str(artifacts)])

    assert result == 0


def test_runtime_kernel_language_avoids_agi_asi_overclaim_labels():
    root = Path(__file__).resolve().parents[1]
    runtime_paths = [
        root / "core" / "kernel" / "aura_kernel.py",
        root / "core" / "kernel" / "upgrades_10x.py",
        root / "core" / "kernel" / "self_review.py",
        root / "core" / "autonomy" / "research_cycle.py",
        root / "core" / "morality" / "moral_reasoning.py",
    ]
    forbidden = [
        re.compile(r"\bASI\b"),
        re.compile(r"\bAGI\b"),
        re.compile(r"\bGENESIS\b"),
        re.compile(r"superintelligence", re.IGNORECASE),
        re.compile(r"true digital life", re.IGNORECASE),
        re.compile(r"never forgets", re.IGNORECASE),
        re.compile(r"never hallucinates", re.IGNORECASE),
        re.compile(r"autonomous self-optimization", re.IGNORECASE),
    ]

    violations = []
    for path in runtime_paths:
        text = path.read_text(encoding="utf-8")
        for pattern in forbidden:
            if pattern.search(text):
                violations.append(f"{path.relative_to(root)}: {pattern.pattern}")

    assert violations == []
