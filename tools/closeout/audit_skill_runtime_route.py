#!/usr/bin/env python3
"""Exercise one real discovered skill through Aura's production API route."""

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
from core.cognitive.router import IntentRouter  # noqa: E402
from core.container import ServiceContainer  # noqa: E402
from interface.routes.subsystems import api_skill_execute  # noqa: E402


async def audit_skill_runtime_route() -> dict[str, Any]:
    """Return exact route, preflight, execution, and authority evidence."""

    engine = CapabilityEngine()
    router = IntentRouter()
    ServiceContainer.register_instance(
        "intent_router",
        router,
        owner="audit_skill_runtime_route",
        registered_by=__name__,
    )
    ServiceContainer.register_instance(
        "capability_engine",
        engine,
        owner="audit_skill_runtime_route",
        registered_by=__name__,
    )

    failures: list[str] = []
    try:
        metadata = engine.skills.get("clock")
        if metadata is None:
            return {
                "failures": ["clock_missing_from_discovered_catalog"],
                "ok": False,
                "route": "api.skill.execute -> intent_router.route_execution -> capability_engine.execute",
            }

        response = await api_skill_execute("clock", {}, None, None)
        payload = json.loads(response.body)
        receipt = engine.preflight_skill("clock")
        instance = engine.instances.get("clock")
        closure = dict(payload.get("authority_closure") or {})

        if response.status_code != 200:
            failures.append(f"unexpected_http_status:{response.status_code}")
        if payload.get("ok") is not True:
            failures.append("skill_execution_failed")
        if payload.get("skill") != "clock":
            failures.append("wrong_skill_executed")
        if not payload.get("time") or not payload.get("readable"):
            failures.append("clock_body_result_missing")
        if receipt.get("ok") is not True or receipt.get("stage") != "ready":
            failures.append("execution_preflight_not_ready")
        if receipt.get("skill_body_invoked") is not False:
            failures.append("preflight_invoked_skill_body")
        if instance is None or type(instance).__name__ != metadata.class_name:
            failures.append("prepared_instance_identity_mismatch")
        if closure.get("closed") is not True or closure.get("success") is not True:
            failures.append("authority_closure_failed")
        if closure.get("token_revoked") is not True:
            failures.append("authority_token_not_revoked")

        health = engine.get_catalog_health()
        return {
            "authority_closure": closure,
            "catalog": {
                "backend": health.get("backend"),
                "digest": health.get("digest"),
                "live_count": health.get("live_count"),
                "parity_status": health.get("parity_status"),
                "ready": health.get("ready"),
            },
            "execution": {
                "duration_ms": payload.get("duration_ms"),
                "http_status": response.status_code,
                "instance_class": type(instance).__name__ if instance is not None else None,
                "ok": payload.get("ok"),
                "result_fields": sorted(payload),
                "skill": payload.get("skill"),
            },
            "failures": failures,
            "metadata": {
                "authority_class": metadata.authority_class,
                "catalog_id": metadata.catalog_id,
                "class_name": metadata.class_name,
                "effect_scope": metadata.effect_scope,
                "module_path": metadata.module_path,
                "validation_state": metadata.validation_state,
            },
            "ok": not failures,
            "preflight": receipt,
            "route": "api.skill.execute -> intent_router.route_execution -> capability_engine.execute",
        }
    finally:
        await engine.on_stop_async()


def main() -> int:
    report = asyncio.run(audit_skill_runtime_route())
    print(
        "AURA_SKILL_RUNTIME_ROUTE_AUDIT="
        + json.dumps(report, separators=(",", ":"), sort_keys=True)
    )
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
