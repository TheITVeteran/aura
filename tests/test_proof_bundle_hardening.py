from __future__ import annotations

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
