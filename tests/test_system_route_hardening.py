import asyncio
import json
import time

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
    monkeypatch.setattr(
        system_routes,
        "runtime_heartbeat_payload",
        lambda kind="heartbeat": {
            "type": kind,
            "healthy": False,
            "runtime_probe_healthy": False,
            "transport_only": False,
            "required_probes": {"all_passed": False},
            "blockers": ["runtime_required_probes"],
        },
    )

    response = await system_routes.telemetry_stream(_Request())
    iterator = response.body_iterator
    first_event = await anext(iterator)
    heartbeat_event = await anext(iterator)
    await iterator.aclose()

    assert "event: telemetry" in first_event
    assert "event: heartbeat" in heartbeat_event
    heartbeat_payload = json.loads(heartbeat_event.split("data: ", 1)[1])
    assert heartbeat_payload["type"] == "heartbeat"
    assert heartbeat_payload["healthy"] is False
    assert heartbeat_payload["runtime_probe_healthy"] is False
    assert heartbeat_payload["transport_only"] is False
    assert heartbeat_payload["required_probes"]["all_passed"] is False
    assert "runtime_required_probes" in heartbeat_payload["blockers"]
    assert unsubscribed == [queue]


def test_websocket_runtime_heartbeat_requires_conversation_lane(monkeypatch):
    from core.runtime.health_contract import REQUIRED_HEALTH_PROBE_GROUPS
    from interface import websocket_manager
    from interface.routes import chat as chat_routes

    required_probes = {
        group: {"ok": True, "components": {key: True for key in keys}}
        for group, keys in REQUIRED_HEALTH_PROBE_GROUPS.items()
    }
    required_probes["all_passed"] = True

    monkeypatch.setattr(
        "core.runtime.health_contract.runtime_health_report",
        lambda: {
            "healthy": True,
            "status": "healthy",
            "required_probes": required_probes,
            "failures": {"critical": [], "important": [], "optional": []},
            "services": [
                {
                    "container_key": component,
                    "present": True,
                    "liveness": "ok",
                }
                for group, components in REQUIRED_HEALTH_PROBE_GROUPS.items()
                for component in components
            ],
        },
    )
    monkeypatch.setattr(
        chat_routes,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": False,
            "state": "failed",
            "last_failure_reason": "desktop_cognitive_engine_required_no_reply",
        },
    )
    monkeypatch.setattr(chat_routes, "_conversation_lane_is_standby", lambda _lane: False)

    payload = websocket_manager.runtime_heartbeat_payload("heartbeat")

    assert payload["healthy"] is False
    assert payload["status"] == "unhealthy"
    assert payload["runtime_probe_healthy"] is True
    assert payload["conversation_ready"] is False
    assert "conversation_ready" in payload["blockers"]
    assert "conversation_lane:failed" in payload["blockers"]


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


@pytest.mark.asyncio
async def test_runtime_heartbeat_refuses_boot_blockers_even_when_required_probes_pass(monkeypatch):
    from interface.routes import system as system_routes
    from core.runtime.health_contract import REQUIRED_HEALTH_PROBE_GROUPS

    required_probes = {
        group: {"ok": True, "components": {key: True for key in keys}}
        for group, keys in REQUIRED_HEALTH_PROBE_GROUPS.items()
    }
    required_probes["all_passed"] = True
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
                "ready": True,
                "system_ready": True,
                "conversation_ready": False,
                "boot_phase": "conversation_failed",
                "blockers": ["conversation_ready", "conversation_failed"],
                "required_probes": required_probes,
            },
            200,
        ),
    )

    response = await system_routes.api_heartbeat()
    payload = json.loads(response.body)

    assert response.status_code == 503
    assert payload["healthy"] is False
    assert payload["runtime_probe_healthy"] is True
    assert payload["status"] == "unhealthy"
    assert payload["required_probes"]["all_passed"] is True
    assert "conversation_failed" in payload["blockers"]


@pytest.mark.asyncio
async def test_boot_health_probe_times_out_instead_of_hanging_http_loop(monkeypatch):
    from interface.routes import system as system_routes

    def slow_health_snapshot(*, is_gui_proxy: bool):
        time.sleep(0.2)
        return ({"ready": True, "required_probes": {"all_passed": True}}, 200)

    monkeypatch.setattr(system_routes, "_HEALTH_PROBE_TIMEOUT_S", 0.01)
    monkeypatch.setattr(system_routes, "_build_boot_health_payload_sync", slow_health_snapshot)

    started_at = time.perf_counter()
    payload, status_code = await system_routes._build_boot_health_payload_bounded(is_gui_proxy=False)
    elapsed = time.perf_counter() - started_at

    assert status_code == 503
    assert elapsed < 0.15
    assert payload["ready"] is False
    assert payload["required_probes"]["all_passed"] is False
    assert payload["blockers"] == ["health_probe_timeout"]
    await asyncio.sleep(0.25)


def test_boot_health_probe_single_flight_fails_closed_instead_of_stacking():
    from interface.routes import system as system_routes

    assert system_routes._HEALTH_PROBE_LOCK.acquire(False)
    try:
        with pytest.raises(TimeoutError, match="health_probe_already_running"):
            system_routes._build_boot_health_payload_sync(is_gui_proxy=False)
    finally:
        system_routes._HEALTH_PROBE_LOCK.release()
