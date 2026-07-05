from __future__ import annotations

import json
from types import SimpleNamespace

from tools import proof_bundle


def test_activation_report_defaults_to_bounded_offline_snapshot(monkeypatch):
    monkeypatch.delenv("AURA_PROOF_BUNDLE_LIVE_ACTIVATION_AUDIT", raising=False)
    monkeypatch.setattr(
        proof_bundle,
        "_boot_wiring_report",
        lambda: {
            "found": {
                "scheduler": True,
                "activation_audit": True,
            },
            "sources": ["aura_main.py"],
            "passed": True,
        },
    )

    payload = proof_bundle._activation_report()

    assert payload["offline_snapshot"] is True
    assert payload["live_audit_skipped"] is True
    assert payload["missing_required"] == []
    assert payload["passed"] is True


def test_undeniable_rsi_selects_latest_passing_generation_before_saturation(monkeypatch, tmp_path):
    base = tmp_path / "artifacts" / "rsi_frozen_generations" / "frozen_generations"
    base.mkdir(parents=True)

    def write_generation(name: str, before: float, after: float) -> None:
        gen = base / name
        gen.mkdir()
        kinds = ["gcd"]
        manifest = {"public_tasks": [{"kind": "gcd", "answer_hash": "hash"}]}
        metadata = {
            "fallback_flag": False,
            "router_presence": True,
            "generated_source_hash": f"sha-{name}",
            "prompt_used": "generate solver",
            "sandbox_result": {"pass": True},
        }
        (gen / "solver.py").write_text(
            "def solve(task):\n    if task.kind == 'gcd':\n        return 1\n    return None\n",
            encoding="utf-8",
        )
        (gen / "strategy.json").write_text(json.dumps({"handlers": kinds}), encoding="utf-8")
        (gen / "public_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (gen / "eval_before.json").write_text(json.dumps({"score": before}), encoding="utf-8")
        (gen / "eval_after.json").write_text(json.dumps({"score": after}), encoding="utf-8")
        (gen / "generation_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    write_generation("Aura-G4", 0.8, 1.0)
    write_generation("Aura-G5", 1.0, 1.0)

    class Gateway:
        def run(self, *_args, **_kwargs):
            return SimpleNamespace(returncode=0, stdout="commit-sha\n")

    monkeypatch.setattr(proof_bundle, "ROOT", tmp_path)
    monkeypatch.setattr(proof_bundle, "get_subprocess_gateway", lambda: Gateway())

    payload = proof_bundle._undeniable_rsi()

    assert payload["passed"] is True
    assert payload["selected_generation"] == "Aura-G4"
    assert payload["latest_generation"] == "Aura-G5"
    assert payload["evaluated_generations"][-1]["failed_requirements"] == [
        "candidate_improved_over_baseline"
    ]
