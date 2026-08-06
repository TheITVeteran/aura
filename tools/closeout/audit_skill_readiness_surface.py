#!/usr/bin/env python3
"""Prove canonical skill readiness reaches Aura's production API and UI bootstrap."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.capability_engine import CapabilityEngine  # noqa: E402
from core.container import ServiceContainer  # noqa: E402
from interface.routes.subsystems import api_skills  # noqa: E402
from interface.routes.system import api_tools_catalog, api_ui_bootstrap  # noqa: E402


def _read_response(response: Any) -> dict[str, Any]:
    return json.loads(response.body)


def _health_identity(health: dict[str, Any]) -> dict[str, Any]:
    preflight = dict(health.get("execution_preflight") or {})
    return {
        "backend": health.get("backend"),
        "digest": health.get("digest"),
        "expected_live_count": health.get("expected_live_count"),
        "live_count": health.get("live_count"),
        "missing_live": health.get("missing_live"),
        "parity_status": health.get("parity_status"),
        "preflight_complete": preflight.get("complete"),
        "preflight_failed": preflight.get("failed"),
        "preflight_ok": preflight.get("ok"),
        "quarantined": health.get("quarantined"),
        "quarantined_count": health.get("quarantined_count"),
        "ready": health.get("ready"),
        "reason": health.get("reason"),
    }


async def audit_skill_readiness_surface() -> dict[str, Any]:
    engine = CapabilityEngine()
    ServiceContainer.register_instance(
        "capability_engine",
        engine,
        owner="audit_skill_readiness_surface",
        registered_by=__name__,
    )
    failures: list[str] = []
    try:
        preflight = engine.dry_run_catalog()
        tools_payload = _read_response(await api_tools_catalog())
        skills_payload = _read_response(await api_skills())
        bootstrap_payload = _read_response(await api_ui_bootstrap())

        direct_health = _health_identity(engine.get_catalog_health())
        route_health = {
            "skills": _health_identity(dict(skills_payload.get("health") or {})),
            "tools": _health_identity(dict(tools_payload.get("health") or {})),
            "ui_bootstrap": _health_identity(
                dict(bootstrap_payload.get("skill_catalog") or {})
            ),
        }
        for surface, health in route_health.items():
            if health != direct_health:
                failures.append(f"health_identity_diverged:{surface}")

        direct_names = sorted(
            item["name"] for item in engine.iter_tool_catalog(include_inactive=True)
        )
        tool_names = sorted(
            str(item.get("name") or "") for item in tools_payload.get("tools") or ()
        )
        skill_names = sorted(
            str(item.get("name") or "") for item in skills_payload.get("catalog") or ()
        )
        bootstrap_names = sorted(
            str(item.get("name") or "") for item in bootstrap_payload.get("tools") or ()
        )
        if not direct_names:
            failures.append("live_catalog_empty")
        if tool_names != direct_names:
            failures.append("tools_route_catalog_diverged")
        if skill_names != direct_names:
            failures.append("skills_route_catalog_diverged")
        if bootstrap_names != direct_names:
            failures.append("bootstrap_catalog_diverged")

        if direct_health.get("ready") is not True:
            failures.append(f"catalog_not_ready:{direct_health.get('reason')}")
        if direct_health.get("missing_live"):
            failures.append("catalog_has_missing_live_skills")
        if direct_health.get("quarantined_count"):
            failures.append("catalog_has_quarantined_skills")
        if preflight.get("complete") is not True or preflight.get("ok") is not True:
            failures.append("catalog_preflight_not_verified")
        if preflight.get("failed"):
            failures.append("catalog_preflight_has_failures")

        ui_flags = list((bootstrap_payload.get("ui") or {}).get("status_flags") or ())
        blocked_flags = sorted(
            set(ui_flags)
            & {"skill_catalog_blocked", "skill_missing_live", "skill_quarantined"}
        )
        if blocked_flags:
            failures.append(f"healthy_catalog_reported_blocked:{blocked_flags}")

        return {
            "catalog_count": len(direct_names),
            "catalog_names_sha256": __import__("hashlib").sha256(
                "\n".join(direct_names).encode("utf-8")
            ).hexdigest(),
            "failures": failures,
            "health": direct_health,
            "ok": not failures,
            "preflight_entry_count": len(preflight.get("entries") or ()),
            "routes": {
                "api_skills": len(skill_names),
                "api_tools_catalog": len(tool_names),
                "api_ui_bootstrap": len(bootstrap_names),
            },
            "schema": "aura.skill_readiness_surface_audit.v1",
            "ui_status_flags": ui_flags,
        }
    finally:
        await engine.on_stop_async()


def main() -> int:
    report = asyncio.run(audit_skill_readiness_surface())
    print(
        "AURA_SKILL_READINESS_SURFACE_AUDIT="
        + json.dumps(report, separators=(",", ":"), sort_keys=True)
    )
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
