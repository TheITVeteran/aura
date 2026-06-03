import asyncio
import json

import pytest


def test_conversation_lane_resilient_helper_contains_legacy_override_failure(monkeypatch):
    from interface.routes import system as system_routes

    def broken_legacy_override():
        failure = RuntimeError("legacy lane collector exploded")
        raise failure

    monkeypatch.setattr(system_routes, "_collect_conversation_lane_status", broken_legacy_override)

    lane = system_routes._collect_conversation_lane_status_resilient()

    assert lane["state"] == "degraded"
    assert lane["conversation_ready"] is False
    assert "legacy lane collector exploded" in lane["last_failure_reason"]


def test_stability_details_mark_missing_guardian_unhealthy(monkeypatch):
    from interface.routes import system as system_routes

    monkeypatch.setattr(
        system_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )
    monkeypatch.setattr(
        system_routes,
        "_collect_conversation_lane_status_resilient",
        lambda: {"conversation_ready": True, "state": "ready", "runtime_identity_ok": True},
    )

    details = system_routes._collect_stability_details()

    assert details["healthy"] is False
    assert details["status"] == "unavailable"
    assert details["active_issues"][0]["name"] == "stability_guardian"


def test_stability_details_do_not_default_missing_report_field_to_healthy(monkeypatch):
    from interface.routes import system as system_routes

    class Guardian:
        def get_latest_report(self):
            return {
                "checks": [{"name": "probe", "message": "missing boolean"}],
                "memory_pct": 12.0,
                "cpu_pct": 3.0,
            }

    monkeypatch.setattr(
        system_routes.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: Guardian() if name == "stability_guardian" else default),
    )
    monkeypatch.setattr(
        system_routes,
        "_collect_conversation_lane_status_resilient",
        lambda: {"conversation_ready": True, "state": "ready", "runtime_identity_ok": True},
    )

    details = system_routes._collect_stability_details()

    assert details["healthy"] is False
    assert details["status"] == "degraded"
    assert details["active_issues"][0]["name"] == "probe"


@pytest.mark.asyncio
async def test_telemetry_stream_emits_idle_heartbeat_and_unsubscribes(monkeypatch):
    from interface.routes import system as system_routes

    queue: asyncio.Queue = asyncio.Queue()
    unsubscribed = []

    class _Request:
        def __init__(self):
            self.checks = 0

        async def is_disconnected(self):
            self.checks += 1
            return self.checks > 2

    class _Bus:
        async def subscribe(self):
            return queue

        async def unsubscribe(self, subscribed_queue):
            unsubscribed.append(subscribed_queue)

    monkeypatch.setattr(system_routes.config.security, "internal_only_mode", False)
    monkeypatch.setattr(system_routes, "_SSE_IDLE_HEARTBEAT_S", 0.001)
    monkeypatch.setattr(system_routes, "broadcast_bus", _Bus())

    response = await system_routes.telemetry_stream(_Request())
    iterator = response.body_iterator
    first_event = await anext(iterator)
    heartbeat_event = await anext(iterator)
    await iterator.aclose()

    assert "event: telemetry" in first_event
    assert "event: heartbeat" in heartbeat_event
    assert unsubscribed == [queue]


@pytest.mark.asyncio
async def test_runtime_heartbeat_fails_closed_when_required_probes_fail(monkeypatch):
    from interface.routes import system as system_routes

    monkeypatch.setattr(system_routes, "_get_runtime_state_safe", lambda: {"state": {}})
    monkeypatch.setattr(
        system_routes,
        "_collect_conversation_lane_status_resilient",
        lambda: {"conversation_ready": False, "state": "failed"},
    )
    monkeypatch.setattr(
        system_routes,
        "build_boot_health_snapshot",
        lambda orch, rt, is_gui_proxy=False, conversation_lane=None: (
            {
                "ready": False,
                "system_ready": False,
                "conversation_ready": False,
                "boot_phase": "kernel_ready",
                "blockers": ["runtime_required_probes", "probe:scheduler"],
                "required_probes": {
                    "scheduler": {
                        "ok": False,
                        "components": {"scheduler": False},
                    },
                    "all_passed": False,
                },
            },
            503,
        ),
    )

    response = await system_routes.api_heartbeat()
    payload = json.loads(response.body)

    assert response.status_code == 503
    assert payload["healthy"] is False
    assert payload["status"] == "unhealthy"
    assert payload["required_probes"]["scheduler"]["ok"] is False
    assert "runtime_required_probes" in payload["blockers"]


@pytest.mark.asyncio
async def test_runtime_heartbeat_refuses_success_code_when_probe_group_missing(monkeypatch):
    from interface.routes import system as system_routes

    monkeypatch.setattr(system_routes, "_get_runtime_state_safe", lambda: {"state": {}})
    monkeypatch.setattr(
        system_routes,
        "_collect_conversation_lane_status_resilient",
        lambda: {"conversation_ready": True, "state": "ready"},
    )
    monkeypatch.setattr(
        system_routes,
        "build_boot_health_snapshot",
        lambda orch, rt, is_gui_proxy=False, conversation_lane=None: (
            {
                "ready": True,
                "system_ready": True,
                "conversation_ready": True,
                "boot_phase": "kernel_ready",
                "blockers": [],
                "required_probes": {
                    "all_passed": True,
                    "kernel": {"ok": True, "components": {"kernel_interface": True}},
                    "memory": {"ok": True, "components": {"state_repository": True}},
                    "scheduler": {"ok": True, "components": {"scheduler": True}},
                    "tool_governance": {"ok": True, "components": {"unified_will": True}},
                },
            },
            200,
        ),
    )

    response = await system_routes.api_heartbeat()
    payload = json.loads(response.body)

    assert response.status_code == 503
    assert payload["healthy"] is False
    assert payload["status"] == "unhealthy"
    assert "probe:inference" in payload["blockers"]


@pytest.mark.asyncio
async def test_runtime_heartbeat_refuses_partial_probe_components(monkeypatch):
    from interface.routes import system as system_routes

    monkeypatch.setattr(system_routes, "_get_runtime_state_safe", lambda: {"state": {}})
    monkeypatch.setattr(
        system_routes,
        "_collect_conversation_lane_status_resilient",
        lambda: {"conversation_ready": True, "state": "ready"},
    )
    monkeypatch.setattr(
        system_routes,
        "build_boot_health_snapshot",
        lambda orch, rt, is_gui_proxy=False, conversation_lane=None: (
            {
                "ready": True,
                "system_ready": True,
                "conversation_ready": True,
                "boot_phase": "kernel_ready",
                "blockers": [],
                "required_probes": {
                    "all_passed": True,
                    "kernel": {"ok": True, "components": {"kernel_interface": True}},
                    "inference": {
                        "ok": True,
                        "components": {"inference_gate": True, "llm_router": True},
                    },
                    "memory": {"ok": True, "components": {"state_repository": True}},
                    "scheduler": {"ok": True, "components": {"scheduler": True}},
                    "tool_governance": {
                        "ok": True,
                        "components": {
                            "unified_will": True,
                            "authority_gateway": True,
                            "capability_engine": True,
                        },
                    },
                },
            },
            200,
        ),
    )

    response = await system_routes.api_heartbeat()
    payload = json.loads(response.body)

    assert response.status_code == 503
    assert payload["healthy"] is False
    assert "probe:memory" in payload["blockers"]
