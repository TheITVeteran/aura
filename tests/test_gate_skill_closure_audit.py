from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from tools.closeout.audit_gate_skill_closure import audit_gate_skill_closure


@dataclass
class _Completed:
    returncode: int
    stdout: str
    stderr: str = ""


def _reports() -> dict[str, dict[str, Any]]:
    return {
        "audit_cognitive_candidate_gates.py": {
            "declared_count": 4,
            "discovered_count": 4,
            "passed": True,
        },
        "audit_skill_catalog.py": {
            "accepted_count": 3,
            "live_count": 3,
            "ok": True,
            "parity_status": "matched",
            "preflight_complete": True,
            "preflight_proves_skill_body_not_invoked": True,
        },
        "audit_skill_runtime_route.py": {
            "authority_closure": {"closed": True},
            "execution": {"ok": True},
            "ok": True,
        },
    }


def test_composite_accepts_only_complete_component_receipts() -> None:
    reports = _reports()

    def runner(argv):
        name = argv[0].rsplit("/", 1)[-1]
        payload = json.dumps(reports[name], sort_keys=True)
        if name == "audit_skill_runtime_route.py":
            payload = "AURA_SKILL_RUNTIME_ROUTE_AUDIT=" + payload
        return _Completed(0, payload)

    report = audit_gate_skill_closure(runner=runner)

    assert report["ok"] is True
    assert all(report["invariants"].values())


def test_composite_fails_closed_on_malformed_or_negative_child() -> None:
    reports = _reports()

    def runner(argv):
        name = argv[0].rsplit("/", 1)[-1]
        if name == "audit_skill_catalog.py":
            return _Completed(0, "not-json")
        payload = json.dumps(reports[name], sort_keys=True)
        if name == "audit_skill_runtime_route.py":
            payload = "AURA_SKILL_RUNTIME_ROUTE_AUDIT=" + payload
            return _Completed(3, payload, "route failed")
        return _Completed(0, payload)

    report = audit_gate_skill_closure(runner=runner)

    assert report["ok"] is False
    assert any(item.startswith("skill_catalog:invalid_receipt") for item in report["failures"])
    assert "skill_runtime_route:exit_3" in report["failures"]
