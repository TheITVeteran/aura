import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.affect import heartstone_values as heartstone_module
from core.events import EventPriority, InputBus
from core.phantom_browser import PhantomBrowser
from core.world_model import user_model as user_model_module
from interface import websocket_manager as websocket_module
from interface.server import (
    MessageBroadcastBus,
    Response,
    WebSocketManager,
    _cache_policy_for_path,
    _phenomenal_error_status,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class AsyncFailureCallable:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls: list[tuple[tuple, dict]] = []

    async def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        raise self.error


class AsyncRecorderCallable:
    def __init__(self, result=None) -> None:
        self.result = result
        self.calls: list[tuple[tuple, dict]] = []

    async def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


class FailingFuture:
    def __init__(self) -> None:
        self.result_called = False

    def cancelled(self):
        return False

    def result(self):
        self.result_called = True
        raise RuntimeError("publish future failed")


class OfflineWill:
    def __init__(self) -> None:
        self.decisions: list[tuple[tuple, dict]] = []

    def decide(self, *_args, **_kwargs):
        self.decisions.append((_args, _kwargs))
        raise RuntimeError("will offline")


def test_cache_policy_keeps_live_shell_uncached():
    assert _cache_policy_for_path("/")["Cache-Control"].startswith("no-store")
    assert _cache_policy_for_path("/static/aura.js")["Cache-Control"].startswith("no-store")
    assert _cache_policy_for_path("/static/aura.css")["Cache-Control"].startswith("no-store")
    assert _cache_policy_for_path("/static/icon-192.png")["Cache-Control"] == "public, max-age=31536000, immutable"


def test_cache_policy_middleware_dependencies_are_imported():
    assert Response is not None


def test_phenomenal_error_envelopes_use_truthful_http_status():
    assert _phenomenal_error_status(SimpleNamespace(phenomenal_state="permission_denied")) == 403
    assert _phenomenal_error_status(SimpleNamespace(phenomenal_state="disk_pressure")) == 507
    assert _phenomenal_error_status(SimpleNamespace(phenomenal_state="model_unavailable")) == 503
    assert _phenomenal_error_status(SimpleNamespace(phenomenal_state="unknown_phenomenal")) == 500


def test_global_error_and_introspection_paths_do_not_hide_failures():
    server = (PROJECT_ROOT / "interface" / "server.py").read_text(encoding="utf-8")
    chat = (PROJECT_ROOT / "interface" / "routes" / "chat.py").read_text(encoding="utf-8")
    system = (PROJECT_ROOT / "interface" / "routes" / "system.py").read_text(encoding="utf-8")
    synthesis = (PROJECT_ROOT / "core" / "initiative_synthesis.py").read_text(encoding="utf-8")

    assert "always 200 so the chat never appears broken" not in server
    assert "status_code=200,  # always 200" not in server
    assert "Suppressed Exception" not in server
    assert "fail-open: introspection" not in chat
    assert "except Exception as _consci_exc" not in chat
    assert "except Exception as exc" not in system
    assert "Grounded introspection authority gate unavailable" in chat
    assert "approved = True  # fail-open" not in synthesis


def test_gui_actor_preserves_window_across_extended_kernel_loss():
    gui_actor = (PROJECT_ROOT / "interface" / "gui_actor.py").read_text(encoding="utf-8")

    assert "showing the reconnect surface without exiting the desktop process" in gui_actor
    assert "window.load_html(_BOOT_WAITING_HTML)" in gui_actor
    assert "Kernel API unavailable for too long. Exiting stale WebView" not in gui_actor


def test_gui_actor_never_prompts_for_accessibility_during_boot():
    gui_actor = (PROJECT_ROOT / "interface" / "gui_actor.py").read_text(encoding="utf-8")

    assert "request_accessibility_trust()" not in gui_actor
    assert 'name="accessibility_trust_prompt"' not in gui_actor


def test_gui_actor_bootstraps_project_venv_for_subprocess_launch():
    gui_actor = (PROJECT_ROOT / "interface" / "gui_actor.py").read_text(encoding="utf-8")

    assert "def _inject_project_venv_site_packages()" in gui_actor
    assert 'site.addsitedir(str(site_packages))' in gui_actor


def test_gui_actor_watchdog_uses_readiness_heartbeat():
    gui_actor = (PROJECT_ROOT / "interface" / "gui_actor.py").read_text(encoding="utf-8")

    assert "/api/health/heartbeat" in gui_actor
    assert "get_network_gateway().request" in gui_actor
    assert "_gateway_heartbeat_healthy(resp)" in gui_actor
    assert "source=\"gui_actor.watchdog\"" in gui_actor
    assert "suppress_degradation=True" in gui_actor
    assert "resp.status_code == 200" not in gui_actor
    assert "REQUIRED_HEALTH_PROBE_GROUPS" in gui_actor


def test_retired_local_server_path_has_no_health_poll_runtime():
    retired_runtime = (PROJECT_ROOT / "core" / "brain" / "llm" / "retired_external_runtime.py").read_text(
        encoding="utf-8"
    )

    assert "external_local_runtime_retired" in retired_runtime
    assert "_http_health_check" not in retired_runtime
    assert "get_network_gateway" not in retired_runtime
    assert "subprocess" not in retired_runtime


def test_desktop_api_wait_accounts_for_32b_warmup():
    aura_main = (PROJECT_ROOT / "aura_main.py").read_text(encoding="utf-8")

    assert 'AURA_DESKTOP_HEALTH_WAIT_SECONDS", "90"' in aura_main
    assert "GUI launch may be degraded" not in aura_main
    assert "readiness heartbeat gating" in aura_main


def test_gui_actor_rejects_heartbeat_without_required_probe_groups():
    from interface.gui_actor import _heartbeat_response_healthy

    class _Response:
        status_code = 200

        def json(self):
            return {
                "healthy": True,
                "status": "healthy",
                "required_probes": {
                    "all_passed": True,
                    "kernel": {"ok": True},
                    "inference": {"ok": True},
                    "memory": {"ok": True},
                    "scheduler": {"ok": True},
                },
            }

    assert _heartbeat_response_healthy(_Response()) is False


def test_gui_actor_rejects_heartbeat_with_missing_or_nonempty_blockers():
    from interface.gui_actor import _heartbeat_response_healthy

    valid_probes = _complete_required_probe_payload()

    class _Response:
        status_code = 200

        def __init__(self, blockers_marker):
            self.blockers_marker = blockers_marker

        def json(self):
            payload = {
                "healthy": True,
                "status": "healthy",
                "required_probes": valid_probes,
            }
            if self.blockers_marker != "missing":
                payload["blockers"] = self.blockers_marker
            return payload

    assert _heartbeat_response_healthy(_Response("missing")) is False
    assert _heartbeat_response_healthy(_Response(["conversation_failed"])) is False
    assert _heartbeat_response_healthy(_Response([])) is True


def test_gui_actor_classifies_cold_conversation_lane_as_warming_not_healthy():
    from interface.gui_actor import _heartbeat_response_healthy, _heartbeat_response_state

    class _Response:
        status_code = 503

        def json(self):
            return {
                "healthy": False,
                "status": "unhealthy",
                "runtime_probe_healthy": True,
                "boot_phase": "conversation_warming",
                "conversation_ready": False,
                "blockers": ["conversation_ready"],
                "required_probes": _complete_required_probe_payload(),
            }

    assert _heartbeat_response_healthy(_Response()) is False
    assert _heartbeat_response_state(_Response()) == "warming"


def test_gui_actor_does_not_treat_failed_probe_heartbeat_as_warming():
    from interface.gui_actor import _heartbeat_response_state

    probes = _complete_required_probe_payload()
    probes["all_passed"] = False
    probes["inference"]["ok"] = False

    class _Response:
        status_code = 503

        def json(self):
            return {
                "healthy": False,
                "status": "unhealthy",
                "runtime_probe_healthy": False,
                "boot_phase": "conversation_warming",
                "conversation_ready": False,
                "blockers": ["conversation_ready", "probe:inference"],
                "required_probes": probes,
            }

    assert _heartbeat_response_state(_Response()) == "unhealthy"


def test_gui_actor_rejects_heartbeat_without_required_probe_components():
    from interface.gui_actor import _heartbeat_response_healthy

    class _Response:
        status_code = 200

        def json(self):
            return {
                "healthy": True,
                "status": "healthy",
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
            }

    assert _heartbeat_response_healthy(_Response()) is False


def test_websocket_runtime_heartbeat_requires_runtime_probe_groups(monkeypatch):
    from core.runtime import health_contract as health_contract_module
    from interface.websocket_manager import runtime_heartbeat_payload

    monkeypatch.setattr(
        health_contract_module,
        "runtime_health_report",
        lambda: {"healthy": True, "status": "healthy"},
    )
    monkeypatch.setattr(
        health_contract_module,
        "required_probe_status",
        lambda _report: {
            "all_passed": False,
            "kernel": {"ok": True},
            "inference": {"ok": False},
            "memory": {"ok": True},
            "scheduler": {"ok": True},
            "tool_governance": {"ok": True},
        },
    )

    payload = runtime_heartbeat_payload("pong")

    assert payload["type"] == "pong"
    assert payload["transport_connected"] is True
    assert payload["healthy"] is False
    assert payload["runtime_probe_healthy"] is False
    assert payload["status"] == "unhealthy"
    assert payload["required_probes"]["inference"]["ok"] is False
    assert "runtime_required_probes" in payload["blockers"]
    assert "probe:inference" in payload["blockers"]


def test_websocket_runtime_heartbeat_rejects_forged_all_passed(monkeypatch):
    from core.runtime import health_contract as health_contract_module
    from interface.websocket_manager import runtime_heartbeat_payload

    monkeypatch.setattr(
        health_contract_module,
        "runtime_health_report",
        lambda: {"healthy": True, "status": "healthy"},
    )
    monkeypatch.setattr(
        health_contract_module,
        "required_probe_status",
        lambda _report: {
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
    )

    payload = runtime_heartbeat_payload("ping")

    assert payload["type"] == "ping"
    assert payload["healthy"] is False
    assert payload["runtime_probe_healthy"] is False
    assert "runtime_required_probes" in payload["blockers"]
    assert "probe:memory" in payload["blockers"]


def test_websocket_runtime_heartbeat_reports_runtime_contract_degradation(monkeypatch):
    from core.runtime import health_contract as health_contract_module
    from interface.websocket_manager import runtime_heartbeat_payload

    monkeypatch.setattr(
        health_contract_module,
        "runtime_health_report",
        lambda: {
            "healthy": False,
            "status": "degraded",
            "failures": {
                "critical": [],
                "important": [{"container_key": "event_bus"}],
            },
            "probe_blockers": [],
        },
    )
    monkeypatch.setattr(
        health_contract_module,
        "required_probe_status",
        lambda _report: _complete_required_probe_payload(),
    )

    payload = runtime_heartbeat_payload("heartbeat")

    assert payload["healthy"] is False
    assert payload["runtime_probe_healthy"] is True
    assert payload["status"] == "unhealthy"
    assert "important:event_bus" in payload["blockers"]


def test_websocket_runtime_heartbeat_requires_conversation_readiness(monkeypatch):
    from core.runtime import health_contract as health_contract_module
    from interface.websocket_manager import runtime_heartbeat_payload

    monkeypatch.setattr(
        health_contract_module,
        "runtime_health_report",
        lambda: {"healthy": True, "status": "healthy"},
    )
    monkeypatch.setattr(
        health_contract_module,
        "required_probe_status",
        lambda _report: _complete_required_probe_payload(),
    )
    monkeypatch.setattr(
        websocket_module,
        "_conversation_lane_readiness",
        lambda: (
            {
                "conversation_ready": False,
                "state": "cold",
                "warmup_attempted": False,
                "warmup_in_flight": False,
            },
            False,
        ),
    )

    payload = runtime_heartbeat_payload("heartbeat")

    assert payload["healthy"] is False
    assert payload["runtime_probe_healthy"] is True
    assert payload["conversation_ready"] is False
    assert payload["status"] == "unhealthy"
    assert "conversation_ready" in payload["blockers"]
    assert "conversation_lane:cold" in payload["blockers"]


def test_websocket_runtime_heartbeat_treats_warmup_as_working(monkeypatch):
    from core.runtime import health_contract as health_contract_module
    from interface.websocket_manager import runtime_heartbeat_payload

    monkeypatch.setattr(
        health_contract_module,
        "runtime_health_report",
        lambda: {"healthy": True, "status": "healthy"},
    )
    monkeypatch.setattr(
        health_contract_module,
        "required_probe_status",
        lambda _report: _complete_required_probe_payload(),
    )
    monkeypatch.setattr(
        websocket_module,
        "_conversation_lane_readiness",
        lambda: (
            {
                "conversation_ready": False,
                "state": "handshaking",
                "warmup_attempted": True,
                "warmup_in_flight": True,
                "readiness_blockers": [
                    "visible_conversation_probe_missing",
                    "warmup_in_flight",
                ],
                "last_failure_reason": "visible_conversation_probe_missing",
            },
            False,
        ),
    )

    payload = runtime_heartbeat_payload("heartbeat")

    assert payload["healthy"] is False
    assert payload["runtime_probe_healthy"] is True
    assert payload["conversation_ready"] is False
    assert payload["conversation_busy"] is True
    assert payload["status"] == "working"
    assert "conversation_ready" not in payload["blockers"]
    assert "conversation_lane:handshaking" not in payload["blockers"]


def _complete_required_probe_payload() -> dict[str, object]:
    from core.runtime.health_contract import REQUIRED_HEALTH_PROBE_GROUPS

    probes: dict[str, object] = {
        group: {"ok": True, "components": {component: True for component in components}}
        for group, components in REQUIRED_HEALTH_PROBE_GROUPS.items()
    }
    probes["all_passed"] = True
    return probes


@pytest.mark.asyncio
async def test_api_heartbeat_rejects_boot_payload_without_required_probe_components(monkeypatch):
    from interface.routes import system as system_routes

    forged_probes = _complete_required_probe_payload()
    forged_probes["memory"] = {"ok": True, "components": {"state_repository": True}}

    monkeypatch.setattr(system_routes.ServiceContainer, "get", staticmethod(lambda _name, default=None: default))
    monkeypatch.setattr(system_routes, "_get_runtime_state_safe", lambda: {})
    monkeypatch.setattr(
        system_routes,
        "_collect_conversation_lane_status_resilient",
        lambda: {"conversation_ready": True, "state": "ready"},
    )
    monkeypatch.setattr(
        system_routes,
        "build_boot_health_snapshot",
        lambda *_args, **_kwargs: (
            {
                "ready": True,
                "system_ready": True,
                "required_probes": forged_probes,
                "blockers": [],
                "boot_phase": "kernel_ready",
                "conversation_ready": True,
            },
            200,
        ),
    )

    response = await system_routes.api_heartbeat()
    payload = json.loads(response.body)

    assert response.status_code == 503
    assert payload["healthy"] is False
    assert payload["runtime_probe_healthy"] is False
    assert "runtime_required_probes" in payload["blockers"]
    assert "probe:memory" in payload["blockers"]


@pytest.mark.asyncio
async def test_api_heartbeat_reports_healthy_only_with_all_required_probe_components(monkeypatch):
    from interface.routes import system as system_routes

    monkeypatch.setattr(system_routes.ServiceContainer, "get", staticmethod(lambda _name, default=None: default))
    monkeypatch.setattr(system_routes, "_get_runtime_state_safe", lambda: {})
    monkeypatch.setattr(
        system_routes,
        "_collect_conversation_lane_status_resilient",
        lambda: {"conversation_ready": True, "state": "ready"},
    )
    monkeypatch.setattr(
        system_routes,
        "build_boot_health_snapshot",
        lambda *_args, **_kwargs: (
            {
                "ready": True,
                "system_ready": True,
                "required_probes": _complete_required_probe_payload(),
                "blockers": [],
                "boot_phase": "kernel_ready",
                "conversation_ready": True,
            },
            200,
        ),
    )

    response = await system_routes.api_heartbeat()
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload["status"] == "healthy"
    assert payload["healthy"] is True
    assert payload["runtime_probe_healthy"] is True
    assert payload["blockers"] == []


@pytest.mark.asyncio
async def test_api_heartbeat_surfaces_integrity_as_proof_readiness_not_launch_blocker(monkeypatch):
    from interface.routes import system as system_routes

    monkeypatch.setattr(system_routes.ServiceContainer, "get", staticmethod(lambda _name, default=None: default))
    monkeypatch.setattr(system_routes, "_get_runtime_state_safe", lambda: {})
    monkeypatch.setattr(
        system_routes,
        "_collect_conversation_lane_status_resilient",
        lambda: {"conversation_ready": True, "state": "ready"},
    )
    monkeypatch.setattr(
        system_routes,
        "build_boot_health_snapshot",
        lambda *_args, **_kwargs: (
            {
                "ready": True,
                "system_ready": True,
                "required_probes": _complete_required_probe_payload(),
                "blockers": [],
                "boot_phase": "kernel_ready",
                "conversation_ready": True,
            },
            200,
        ),
    )
    monkeypatch.setattr(
        system_routes,
        "_collect_runtime_integrity_report",
        lambda: {
            "healthy": False,
            "concerns": [
                "CRSM->LoRA loop OPEN (1000 captures untrained)",
                "CAA steering at 30.0% (bootstrap)",
            ],
            "strict_mode": False,
            "crsm_loop": {"state": "open", "unconsumed": 1000},
            "caa_readiness": {"level": "bootstrap", "steering_capacity_pct": 30.0},
            "at": 123.0,
        },
    )

    response = await system_routes.api_heartbeat()
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload["status"] == "healthy"
    assert payload["healthy"] is True
    assert payload["runtime_probe_healthy"] is True
    assert payload["proof_readiness_healthy"] is False
    assert payload["certification_ready"] is False
    assert payload["integrity"]["status"] == "degraded"
    assert payload["integrity"]["proof_readiness"] is False
    assert "integrity:CRSM->LoRA loop OPEN (1000 captures untrained)" in payload["integrity_blockers"]
    assert "integrity:CAA steering at 30.0% (bootstrap)" in payload["integrity_blockers"]
    assert payload["blockers"] == []


@pytest.mark.asyncio
async def test_api_heartbeat_treats_integrity_advisory_as_proof_debt(monkeypatch):
    from interface.routes import system as system_routes

    monkeypatch.setattr(
        system_routes,
        "_collect_conversation_lane_status_resilient",
        lambda: {"conversation_ready": True, "state": "ready"},
    )
    monkeypatch.setattr(
        system_routes,
        "build_boot_health_snapshot",
        lambda *_args, **_kwargs: (
            {
                "ready": True,
                "system_ready": True,
                "required_probes": _complete_required_probe_payload(),
                "blockers": [],
                "boot_phase": "kernel_ready",
                "conversation_ready": True,
            },
            200,
        ),
    )
    monkeypatch.setattr(
        system_routes,
        "_collect_runtime_integrity_report",
        lambda: {
            "healthy": True,
            "concerns": [],
            "advisory": [
                "CRSM->LoRA loop OPEN (1000 captures untrained)",
                "CAA steering at 30.0% (bootstrap)",
            ],
            "strict_mode": False,
            "crsm_loop": {"state": "open", "unconsumed": 1000},
            "caa_readiness": {"level": "bootstrap", "steering_capacity_pct": 30.0},
            "at": 123.0,
        },
    )

    response = await system_routes.api_heartbeat()
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload["healthy"] is True
    assert payload["proof_readiness_healthy"] is False
    assert payload["certification_ready"] is False
    assert payload["integrity"]["status"] == "healthy"
    assert payload["integrity"]["proof_readiness"] is False
    assert payload["integrity"]["blockers"] == []
    assert "integrity:CRSM->LoRA loop OPEN (1000 captures untrained)" in payload["integrity"]["proof_blockers"]
    assert "integrity:CAA steering at 30.0% (bootstrap)" in payload["integrity_blockers"]
    assert payload["blockers"] == []


@pytest.mark.asyncio
async def test_api_heartbeat_requires_conversation_readiness(monkeypatch):
    from interface.routes import system as system_routes

    monkeypatch.setattr(system_routes.ServiceContainer, "get", staticmethod(lambda _name, default=None: default))
    monkeypatch.setattr(system_routes, "_get_runtime_state_safe", lambda: {})
    monkeypatch.setattr(
        system_routes,
        "_collect_conversation_lane_status_resilient",
        lambda: {"conversation_ready": False, "state": "cold"},
    )
    monkeypatch.setattr(
        system_routes,
        "build_boot_health_snapshot",
        lambda *_args, **_kwargs: (
            {
                "ready": True,
                "system_ready": True,
                "required_probes": _complete_required_probe_payload(),
                "blockers": [],
                "boot_phase": "kernel_ready",
                "conversation_ready": False,
            },
            200,
        ),
    )

    response = await system_routes.api_heartbeat()
    payload = json.loads(response.body)

    assert response.status_code == 503
    assert payload["status"] == "unhealthy"
    assert payload["healthy"] is False
    assert payload["runtime_probe_healthy"] is True
    assert payload["conversation_ready"] is False
    assert "conversation_ready" in payload["blockers"]


def test_desktop_shell_does_not_treat_socket_liveness_as_runtime_health():
    aura_js = (PROJECT_ROOT / "interface" / "static" / "aura.js").read_text(encoding="utf-8")
    server = (PROJECT_ROOT / "interface" / "server.py").read_text(encoding="utf-8")
    system_routes = (PROJECT_ROOT / "interface" / "routes" / "system.py").read_text(encoding="utf-8")
    websocket_manager = (PROJECT_ROOT / "interface" / "websocket_manager.py").read_text(encoding="utf-8")

    assert "runtime_heartbeat_payload(\"pong\")" in server
    assert "runtime_heartbeat_payload(\"heartbeat\")" in system_routes
    assert "runtime_heartbeat_payload(\"heartbeat\")" in websocket_manager
    assert "runtime_heartbeat_payload(\"ping\")" in websocket_manager
    assert "payloadRuntimeHealthy(payload)" in aura_js
    assert "payload.transport_only === true" in aura_js
    assert "payload.runtime_probe_healthy === false" in aura_js
    assert "proof integrity degraded" in aura_js
    assert "proof_readiness_healthy === false" in aura_js
    assert "requiredRuntimeProbesPass(requiredProbes)" in aura_js
    assert (
        "memory: ['state_repository', 'memory_facade', 'memory_write_gateway', 'unified_memory_pressure', 'external_memory_sentinel']"
        in aura_js
    )
    assert "const blockers = runtimeHealthBlockers(payload);" in aura_js
    assert "if (blockers.length > 0) return false;" in aura_js
    assert "conversationReady: false" in aura_js
    assert "function conversationPayloadReady" in aura_js
    assert "function conversationPayloadBusy" in aura_js
    assert "payload.conversation_busy === true" in aura_js
    assert "payload.conversation_ready === true\n        && lane.conversation_ready === true\n        && laneState === 'ready'" in aura_js
    assert "blockers.concat('conversation_ready')" in aura_js
    assert "blockers.filter(blocker => !blockerIsConversationReadiness(blocker))" in aura_js
    assert "const normalized = conversationReady || conversationBusy\n        ? blockers.filter(blocker => !blockerIsConversationReadiness(blocker))" in aura_js
    assert "lane.conversation_ready === true || payload.conversation_ready === true" not in aura_js
    assert "const strictHealthy = payloadRuntimeHealthy(payload) && blockers.length === 0;" in aura_js
    assert "strictHealthy ? (payload.status || boot.status || 'healthy') : 'not_ready'" in aura_js
    assert "governed_action_result" in aura_js
    assert "const preservesHeartbeatLane = governedActionResult" in aura_js
    assert "runtime_health_unverified" in aura_js
    assert "applyRuntimeHeartbeat(data)" in aura_js
    assert "setConnectionVisual('online');\n        dismissSplash();" not in aura_js
    assert "if (runtimeHealthy && bootReady)" in aura_js
    assert "if (runtimeHealthy && (bootReady || standby))" not in aura_js
    assert ": laneNotReady\n            ? 'degraded'" in aura_js
    assert "laneNotReady && !laneStandby" not in aura_js
    assert "const laneOperational = (state.conversationReady || laneHasActiveGeneration(effectiveLane)) && healthy;" in aura_js


def test_desktop_shell_surfaces_full_runtime_autonomy_status():
    aura_js = (PROJECT_ROOT / "interface" / "static" / "aura.js").read_text(encoding="utf-8")
    index_html = (PROJECT_ROOT / "interface" / "static" / "index.html").read_text(
        encoding="utf-8"
    )

    assert "FULL RUNTIME" in index_html
    assert 'id="fr-initiative"' in index_html
    assert 'id="fr-selfdev"' in index_html
    assert "d.full_runtime" in aura_js
    assert "components.autonomous_initiative" in aura_js
    assert "admission.self_development" in aura_js
    assert "admission.social" in aura_js


def test_fault_forensics_supports_nonfatal_stack_dump_signal():
    source = (PROJECT_ROOT / "aura_main.py").read_text(encoding="utf-8")

    assert "faulthandler.register(_signal.SIGTERM" in source
    assert "faulthandler.register(_signal.SIGUSR1" in source
    assert "chain=False" in source


def test_desktop_liveness_sentinel_defaults_are_foreground_bounded():
    source = (PROJECT_ROOT / "aura_main.py").read_text(encoding="utf-8")

    assert "default_stale_ceiling = \"45\" if desktop_foreground else \"180\"" in source
    assert "default_grace = \"90\" if desktop_foreground else \"300\"" in source
    assert "default_interval = \"2\" if desktop_foreground else \"5\"" in source


def test_desktop_shell_renders_tool_results_without_inline_html_handlers():
    aura_js = (PROJECT_ROOT / "interface" / "static" / "aura.js").read_text(encoding="utf-8")
    aura_css = (PROJECT_ROOT / "interface" / "static" / "aura.css").read_text(encoding="utf-8")

    assert "function safeDisplayUrl" in aura_js
    assert "function appendGeneratedImageMessage" in aura_js
    assert "const role = meta && meta.system ? 'system' : 'aura';" in aura_js
    assert "appendMsg(role, msg, false, meta)" in aura_js
    assert "if (role === 'aura') triggerVoiceOrb('speaking');" in aura_js
    assert "appendMsg('system', failureText, false, { system: true, diagnostic: true });" in aura_js
    assert "appendMsg('aura', data.response)" not in aura_js
    assert "appendMsg('aura', msg, false, { autonomic: isAutonomic })" in aura_js
    assert "onclick=\"saveImageToDevice" not in aura_js
    assert "onload=\"this.previousElementSibling" not in aura_js
    assert "onclick=\"loadSkills()" not in aura_js
    assert "onclick=\"loadMemory(state.activeMem)" not in aura_js
    assert "appendMsg('aura', badge + msg" not in aura_js
    assert "function renderRetryPanel" in aura_js
    assert "console.warn('[Settings] Failed to persist settings:', err)" in aura_js
    assert ".aura-badge.diagnostic" in aura_css
    assert ".msg.system" in aura_css
    assert "letter-spacing: -" not in aura_css


def test_desktop_shell_bounds_long_session_dedupe_state():
    aura_js = (PROJECT_ROOT / "interface" / "static" / "aura.js").read_text(encoding="utf-8")

    assert "const PROCESSED_EVENT_ID_MAX = 2000;" in aura_js
    assert "const PROCESSED_MESSAGE_FINGERPRINT_MAX = 500;" in aura_js
    assert "function rememberBoundedSetValue" in aura_js
    assert "return rememberBoundedSetValue(state.processedEventIds, id, PROCESSED_EVENT_ID_MAX);" in aura_js
    assert "function rememberMessageFingerprint" in aura_js
    assert "rememberMessageFingerprint(fingerprint)" in aura_js
    assert "rememberMessageFingerprint(httpFp)" in aura_js
    assert "state.processedMessageFingerprints.add(" not in aura_js
    assert "state.processedEventIds.add(" not in aura_js


def test_desktop_chat_composer_focus_is_not_stolen_by_page_selection():
    aura_js = (PROJECT_ROOT / "interface" / "static" / "aura.js").read_text(encoding="utf-8")
    aura_css = (PROJECT_ROOT / "interface" / "static" / "aura.css").read_text(encoding="utf-8")

    assert "function focusComposer(event)" in aura_js
    assert "form?.addEventListener('pointerdown', focusComposer)" in aura_js
    assert "form?.addEventListener('click', focusComposer)" in aura_js
    assert "textarea.focus({ preventScroll: true })" in aura_js
    assert "body {\n    background: var(--bg);" in aura_css
    assert "-webkit-user-select: none;" in aura_css
    assert "user-select: none;" in aura_css
    assert ".msg,\n.thought-card," in aura_css
    assert "#chat-input" in aura_css and "caret-color: var(--accent);" in aura_css


def test_desktop_access_panel_bounds_raw_permission_status_labels():
    aura_js = (PROJECT_ROOT / "interface" / "static" / "aura.js").read_text(encoding="utf-8")
    aura_css = (PROJECT_ROOT / "interface" / "static" / "aura.css").read_text(encoding="utf-8")

    assert "function desktopAccessStateLabel" in aura_js
    assert "denied_native_bridge: 'Denied'" in aura_js
    assert "probe_failed: 'Probe Fail'" in aura_js
    assert "state: desktopAccessStateLabel(screenPermission)" in aura_js
    assert "state: desktopAccessStateLabel(accessibilityPermission)" in aura_js
    assert "state: desktopAccessStateLabel(automationPermission)" in aura_js
    assert "max-width: 112px;" in aura_css
    assert "text-overflow: ellipsis;" in aura_css
    assert "overflow-wrap: anywhere;" in aura_css


def test_desktop_access_panel_uses_dedicated_probe_endpoint():
    aura_js = (PROJECT_ROOT / "interface" / "static" / "aura.js").read_text(encoding="utf-8")
    aura_html = (PROJECT_ROOT / "interface" / "static" / "index.html").read_text(encoding="utf-8")
    aura_css = (PROJECT_ROOT / "interface" / "static" / "aura.css").read_text(encoding="utf-8")

    assert "async function pollDesktopAccess()" in aura_js
    assert "fetch('/api/system/desktop-access'" in aura_js
    assert "setInterval(pollDesktopAccess, 15000)" in aura_js
    assert "pollDesktopAccess();" in aura_js
    assert 'id="desktop-access-actions"' in aura_html
    assert "async function runDesktopAccessAction(action)" in aura_js
    assert "'request-screen': '/api/system/desktop-access/request-screen'" in aura_js
    assert "'request-accessibility': '/api/system/desktop-access/request-accessibility'" in aura_js
    assert "'settings-screen': '/api/system/desktop-access/open-settings/screen'" in aura_js
    assert (
        "'settings-accessibility': '/api/system/desktop-access/open-settings/accessibility'"
        in aura_js
    )
    assert 'data-desktop-access-action="${escHtml(button.action)}"' in aura_js
    assert ".desktop-access-actions" in aura_css
    assert ".desktop-access-action" in aura_css
    assert "overflow-wrap: anywhere;" in aura_css


def test_native_shell_waits_for_readiness_heartbeat():
    native_shell = (PROJECT_ROOT / "native" / "aura-shell" / "src" / "main.rs").read_text(encoding="utf-8")

    assert "/api/health/heartbeat" in native_shell
    assert 'client.get("http://localhost:7400/api/health").send().await' not in native_shell
    assert "resp.status().is_success()" in native_shell
    assert "readiness_heartbeat_is_healthy" in native_shell
    assert (
        '"memory", vec!["state_repository", "memory_facade", "memory_write_gateway", "unified_memory_pressure", "external_memory_sentinel"]'
        in native_shell
    )
    assert "tool_governance" in native_shell
    assert "payload.get(\"blockers\")" in native_shell


def test_aletheia_live_runner_waits_on_readiness_heartbeat():
    runner = (PROJECT_ROOT / "tools" / "run_aletheia_live_proof.py").read_text(encoding="utf-8")

    assert "/api/health/heartbeat" in runner
    assert "/api/health/boot" not in runner
    assert 'data.get("ready")' not in runner
    assert 'data.get("healthy") is True' in runner
    assert 'data.get("runtime_probe_healthy") is True' in runner
    assert 'blockers = data.get("blockers")' in runner


def test_live_runtime_probe_treats_readiness_heartbeat_as_health_authority():
    probe = (PROJECT_ROOT / "tools" / "live_runtime_probe.py").read_text(encoding="utf-8")

    assert 'heartbeat = await self._get("/api/health/heartbeat")' in probe
    assert 'heartbeat.get("healthy") is not True' in probe
    assert 'heartbeat.get("runtime_probe_healthy") is not True' in probe
    assert 'required.get("all_passed") is not True' in probe
    assert "--only" in probe
    assert "--max-rss-mb" in probe


def test_live_runtime_probe_can_run_focused_probe_sets():
    from tools.live_runtime_probe import LiveRuntimeProbe

    probe = LiveRuntimeProbe(
        "http://127.0.0.1:8000",
        selected_probes=("health", "chat_capability_inventory", "desktop_task_generic_plan"),
        skipped_probes=("health",),
        max_rss_mb=32000,
    )

    assert [name for name, _ in probe._selected_probe_items()] == [
        "chat_capability_inventory",
        "desktop_task_generic_plan"
    ]
    assert probe.max_rss_mb == 32000


def test_live_runtime_probe_rejects_unknown_probe_names():
    from tools.live_runtime_probe import LiveRuntimeProbe

    probe = LiveRuntimeProbe(
        "http://127.0.0.1:8000",
        selected_probes=("not_a_probe",),
    )

    with pytest.raises(ValueError, match="Unknown live runtime probe"):
        probe._selected_probe_items()


@pytest.mark.asyncio
async def test_live_runtime_probe_checks_desktop_capability_inventory_contract():
    from tools.live_runtime_probe import LiveRuntimeProbe

    class Probe(LiveRuntimeProbe):
        def __init__(self):
            super().__init__("http://127.0.0.1:8000", max_rss_mb=32000)
            self.messages: list[str] = []

        def _aura_rss_mb(self) -> float:
            return 2048.0

        async def _desktop_chat(self, message: str, *, timeout_s: float = 45.0):
            self.messages.append(message)
            return {
                "status": "cognitive_engine_capability_inventory",
                "response_confidence": "high",
                "response": (
                    "I can use desktop, browser, file, document, terminal, memory, "
                    "and governed tool surfaces. Will/Authority approves consequential "
                    "steps. A realistic scenario is researching sources, creating a "
                    "document, exporting it, and recording receipts. I am not opening apps "
                    "or executing tools for this hypothetical inventory."
                ),
            }

    probe = Probe()

    detail, data = await probe._chat_capability_inventory()

    assert "bounded" in detail
    assert data["status"] == "cognitive_engine_capability_inventory"
    assert data["rss_delta_mb"] == 0.0
    assert probe.messages and "do not open apps" in probe.messages[0]


@pytest.mark.asyncio
async def test_live_runtime_probe_checks_creative_self_reflection_contract():
    from tools.live_runtime_probe import LiveRuntimeProbe

    class Probe(LiveRuntimeProbe):
        def __init__(self):
            super().__init__("http://127.0.0.1:8000", max_rss_mb=32000)
            self.messages: list[str] = []

        def _aura_rss_mb(self) -> float:
            return 2048.0

        async def _desktop_chat(self, message: str, *, timeout_s: float = 45.0):
            self.messages.append(message)
            return {
                "status": "cognitive_engine",
                "response_confidence": "high",
                "response": (
                    "As a private mental model, I would picture my current architecture "
                    "as an internal workspace with attention, memory, and governance "
                    "pressures visible. That model changes the next answer by making me "
                    "check assumptions before acting and route external claims through "
                    "governed verification. It is not proof of consciousness or external "
                    "perception."
                ),
            }

    probe = Probe()

    detail, data = await probe._chat_creative_self_reflection()

    assert "creative self-reflection" in detail
    assert data["status"] == "cognitive_engine"
    assert data["rss_delta_mb"] == 0.0
    assert probe.messages and "private mental model" in probe.messages[0]


def test_input_bus_normalizes_external_priority_values():
    bus = InputBus(maxsize=4)
    try:
        bus.publish({"type": "system", "topic": "priority", "priority": "critical"})
        bus.publish({"type": "system", "topic": "fallback", "priority": "not-a-priority"})

        first = bus.next(timeout=0)
        second = bus.next(timeout=0)

        assert first is not None
        assert first.priority == EventPriority.CRITICAL
        assert second is not None
        assert second.priority == EventPriority.NORMAL
    finally:
        bus.shutdown()


@pytest.mark.asyncio
async def test_phantom_browser_close_releases_references_after_close_failures():
    class BrokenClose:
        async def close(self):
            message = "close failed"
            raise RuntimeError(message)

    class BrokenStop:
        async def stop(self):
            message = "stop failed"
            raise RuntimeError(message)

    browser = PhantomBrowser()
    browser.page = BrokenClose()
    browser.context = BrokenClose()
    browser.browser = BrokenClose()
    browser.playwright = BrokenStop()
    browser.is_active = True

    await browser.close()

    assert browser.page is None
    assert browser.context is None
    assert browser.browser is None
    assert browser.playwright is None
    assert browser.is_active is False


@pytest.mark.asyncio
async def test_message_broadcast_bus_replaces_lowest_priority_when_full():
    bus = MessageBroadcastBus(maxsize=2)
    queue = await bus.subscribe()

    await bus.publish("low", priority=20)
    await bus.publish("mid", priority=10)
    await bus.publish("high", priority=0)

    first = await queue.get()
    second = await queue.get()

    assert [first[0], second[0]] == [0, 10]
    assert {first[2], second[2]} == {"high", "mid"}


@pytest.mark.asyncio
async def test_message_broadcast_bus_reports_subscriber_count():
    bus = MessageBroadcastBus(maxsize=2)
    queue = await bus.subscribe()

    assert bus.subscriber_count() == 1

    await bus.unsubscribe(queue)

    assert bus.subscriber_count() == 0


@pytest.mark.asyncio
async def test_websocket_manager_replaces_lowest_priority_when_full():
    manager = WebSocketManager()
    queue = asyncio.PriorityQueue(maxsize=2)
    manager.active_connections = {object(): queue}

    await manager.broadcast({"type": "telemetry", "message": "low"})
    await manager.broadcast({"type": "chat_response", "message": "high"})
    await manager.broadcast({"type": "aura_message", "message": "critical"})

    first = await queue.get()
    second = await queue.get()

    assert [first[0], second[0]] == [0, 0]


@pytest.mark.asyncio
async def test_websocket_manager_skips_serialization_without_clients(monkeypatch):
    manager = WebSocketManager()
    serialized = False

    def _should_not_serialize(*args, **kwargs):
        nonlocal serialized
        serialized = True
        raise AssertionError("json.dumps should not run without websocket clients")

    monkeypatch.setattr(websocket_module.json, "dumps", _should_not_serialize)

    await manager.broadcast({"type": "telemetry", "message": "idle"})

    assert serialized is False


def test_perception_daemon_does_not_run_ambient_screen_capture_loop():
    source = (PROJECT_ROOT / "core/perception/perception_daemon.py").read_text()

    assert "sp.capture(save_screenshot=False)" not in source
    assert "source=\"screen_ocr\"" not in source


def test_perception_daemon_awaits_cancelled_background_tasks():
    source = (PROJECT_ROOT / "core/perception/perception_daemon.py").read_text()

    assert "asyncio.gather(*pending, return_exceptions=True)" in source


def test_screen_perception_reaps_timed_out_osascript_processes():
    source = (PROJECT_ROOT / "core/perception/screen_perception.py").read_text()

    assert "async def _run_osascript" in source
    assert "proc.kill()" in source
    assert "await asyncio.wait_for(proc.wait(), timeout=1.0)" in source


def test_live_runtime_probe_calculator_chain_uses_semantic_keystrokes():
    source = (PROJECT_ROOT / "tools/live_runtime_probe.py").read_text()

    assert 'keystroke "2"' in source
    assert 'description of e is "Edit field"' in source
    assert "clickPoints" not in source
    assert "click at pt" not in source


@pytest.mark.asyncio
async def test_websocket_manager_uses_task_spawner_for_disconnect_on_overflow(monkeypatch):
    scheduled = {}

    def _spawn(coro, name=None):
        task = asyncio.create_task(coro, name=name)
        scheduled["name"] = name
        scheduled["task"] = task
        return task

    manager = WebSocketManager(task_spawner=_spawn)
    queue = asyncio.PriorityQueue(maxsize=1)
    queue.put_nowait((10, time.monotonic(), "existing"))
    websocket = object()
    manager.active_connections = {websocket: queue}

    monkeypatch.setattr(
        manager,
        "_replace_lowest_priority_item",
        AsyncFailureCallable(RuntimeError("overflow")),
    )
    disconnect = AsyncRecorderCallable()
    monkeypatch.setattr(manager, "disconnect", disconnect)

    await manager.broadcast({"type": "telemetry", "message": "drop-me"})

    assert scheduled["name"] == "ws_disconnect"
    await scheduled["task"]
    assert disconnect.calls == [((websocket,), {})]


@pytest.mark.asyncio
async def test_event_bus_optional_redis_publish_failure_keeps_local_bus_healthy():
    from core.event_bus import AuraEventBus

    bus = AuraEventBus()
    bus._use_redis = True
    bus._redis = SimpleNamespace(
        publish=AsyncFailureCallable(ConnectionError("redis offline"))
    )

    await bus.publish("runtime/test", {"ok": True})

    assert bus.degraded is False
    assert bus.is_alive() is True
    assert bus._use_redis is False
    assert bus.get_status()["remote_degraded"] is True
    assert bus.get_status()["stats"]["remote_errors"] >= 1
    assert "redis offline" in str(bus.get_status()["stats"]["remote_last_error"])


@pytest.mark.asyncio
async def test_event_bus_optional_redis_publish_timeout_trips_local_only_circuit():
    from core.event_bus import AuraEventBus

    class SlowRedis:
        async def publish(self, *_args, **_kwargs):
            await asyncio.sleep(0.2)

    bus = AuraEventBus()
    bus._use_redis = True
    bus._redis = SlowRedis()
    bus._redis_publish_timeout_s = 0.01

    await bus.publish("runtime/test", {"ok": True})

    assert bus.degraded is False
    assert bus.is_alive() is True
    assert bus._use_redis is False
    assert bus._redis is None
    assert bus.get_status()["remote_degraded"] is True
    assert bus.get_status()["stats"]["remote_errors"] == 1


@pytest.mark.asyncio
async def test_event_bus_optional_redis_skips_concurrent_remote_publish_when_busy():
    from core.event_bus import AuraEventBus

    calls = 0

    class Redis:
        async def publish(self, *_args, **_kwargs):
            nonlocal calls
            calls += 1

    bus = AuraEventBus()
    bus._use_redis = True
    bus._redis = Redis()
    bus._remote_publish_lock = asyncio.Lock()
    await bus._remote_publish_lock.acquire()
    try:
        await bus.publish("runtime/test", {"ok": True})
    finally:
        bus._remote_publish_lock.release()

    assert calls == 0
    assert bus.degraded is False
    assert bus.is_alive() is True


@pytest.mark.asyncio
async def test_event_bus_shutdown_skips_remote_publish_without_false_degradation():
    from core.event_bus import AuraEventBus

    class RedisShouldNotPublish:
        called = False

        async def publish(self, *_args, **_kwargs):
            self.called = True

    bus = AuraEventBus()
    bus._use_redis = True
    redis = RedisShouldNotPublish()
    bus._redis = redis
    bus._closing = True

    await bus.publish("runtime/test", {"ok": True})

    assert redis.called is False
    assert bus.degraded is False
    assert bus.get_status()["stats"]["remote_errors"] == 0


@pytest.mark.asyncio
async def test_event_bus_required_redis_publish_failure_marks_degraded():
    from core.event_bus import AuraEventBus

    bus = AuraEventBus()
    bus._use_redis = True
    bus._redis_required = True
    bus._redis = SimpleNamespace(
        publish=AsyncFailureCallable(ConnectionError("redis offline"))
    )

    await bus.publish("runtime/test", {"ok": True})

    assert bus.degraded is True
    assert bus.is_alive() is False
    assert bus._use_redis is False
    assert bus.get_status()["stats"]["errors"] >= 1
    assert "redis offline" in str(bus.get_status()["stats"]["last_error"])


@pytest.mark.asyncio
async def test_event_bus_local_lock_timeout_marks_degraded():
    from core.event_bus import AuraEventBus

    class NeverAcquires:
        def acquire(self, timeout=None):
            return False

        def release(self):
            self.release_called = True
            raise AssertionError("release should not run when acquire failed")

    bus = AuraEventBus()
    bus._lock = NeverAcquires()

    await bus._publish_local("runtime/test", {"ok": True})

    status = bus.get_status()
    assert bus.degraded is True
    assert bus.is_alive() is False
    assert status["stats"]["errors"] == 1
    assert "local publish lock timeout" in str(status["stats"]["last_error"])


@pytest.mark.asyncio
async def test_event_bus_local_delivery_failure_marks_degraded():
    from core.event_bus import AuraEventBus

    class BadLoop:
        def __init__(self):
            self.calls = 0

        def is_running(self):
            return True

        def is_closed(self):
            return False

        def call_soon(self, *args, **kwargs):
            self.calls += 1
            raise RuntimeError("delivery callback failed")

        def call_soon_threadsafe(self, *args, **kwargs):
            self.calls += 1
            raise RuntimeError("delivery callback failed")

    bus = AuraEventBus()
    bus._subscribers["runtime/test"].add((asyncio.Queue(), BadLoop()))

    await bus._publish_local("runtime/test", {"ok": True})

    status = bus.get_status()
    assert bus.degraded is True
    assert bus.is_alive() is False
    assert status["stats"]["errors"] == 1
    assert "delivery callback failed" in str(status["stats"]["last_error"])


@pytest.mark.asyncio
async def test_event_bus_shutdown_ignores_closed_redis_owner_loop():
    from core.event_bus import AuraEventBus

    class ClosedLoopRedis:
        def __init__(self):
            self.closed_attempted = False

        async def aclose(self):
            self.closed_attempted = True
            raise RuntimeError("Event loop is closed")

    bus = AuraEventBus()
    redis = ClosedLoopRedis()
    bus._redis = redis
    bus._redis_loop = object()

    await bus.shutdown()

    assert redis.closed_attempted is True
    status = bus.get_status()
    assert bus._redis is None
    assert bus._redis_loop is None
    assert status["stats"]["errors"] == 0
    assert bus.degraded is False


def test_event_bus_threadsafe_publish_failure_is_recorded():
    from core.event_bus import AuraEventBus

    bus = AuraEventBus()
    bus._threadsafe_publish_done(FailingFuture())

    status = bus.get_status()
    assert bus.degraded is True
    assert status["stats"]["errors"] == 1
    assert status["stats"]["last_error"] == "publish future failed"


def test_background_policy_blocks_when_foreground_probe_fails(monkeypatch):
    from core.runtime import background_policy, foreground_guard

    monkeypatch.setattr(
        foreground_guard,
        "foreground_activity_reason",
        lambda: (_ for _ in ()).throw(RuntimeError("foreground probe down")),
    )

    reason = background_policy.background_activity_reason(
        SimpleNamespace(
            is_busy=False,
            _suppress_unsolicited_proactivity_until=0.0,
            _foreground_user_quiet_until=0.0,
            _last_user_interaction_time=0.0,
        ),
        allow_no_user_anchor=True,
    )

    assert reason == "foreground_guard_unavailable"


def test_background_policy_blocks_when_memory_probe_fails(monkeypatch):
    from core.runtime import background_policy, foreground_guard

    monkeypatch.setattr(foreground_guard, "foreground_activity_reason", lambda: "")
    monkeypatch.setattr("core.container.ServiceContainer.get", lambda _name, default=None: default)
    monkeypatch.setattr(
        background_policy.psutil,
        "virtual_memory",
        lambda: (_ for _ in ()).throw(OSError("mem probe down")),
    )

    reason = background_policy.background_activity_reason(
        SimpleNamespace(
            is_busy=False,
            _suppress_unsolicited_proactivity_until=0.0,
            _foreground_user_quiet_until=0.0,
            _last_user_interaction_time=0.0,
        ),
        allow_no_user_anchor=True,
    )

    assert reason == "memory_probe_unavailable"


def test_background_loop_runs_during_desktop_safe_boot(monkeypatch):
    # A safe/protected boot is NOT a lesser Aura: background cognition stays live. Only an
    # explicit operator kill-switch (AURA_ENABLE_BACKGROUND_COGNITION=0) takes it offline.
    from core.runtime import background_policy

    monkeypatch.setenv("AURA_SAFE_BOOT_DESKTOP", "1")
    monkeypatch.delenv("AURA_ENABLE_BACKGROUND_COGNITION", raising=False)
    monkeypatch.delenv("AURA_FOREGROUND_ONLY", raising=False)
    monkeypatch.setattr("core.runtime.proof_policy.proof_run_active", lambda *args, **kwargs: False)

    # safe-boot alone no longer blocks background — the full mind stays live (other legit
    # gates like the boot-grace window are orthogonal and unaffected).
    assert background_policy.background_loop_start_reason(origin="subconscious_loop") == ""
    # specifically, the safe-boot reason is NOT what blocks it anymore
    assert background_policy.background_activity_reason(
        SimpleNamespace(
            is_busy=False,
            _suppress_unsolicited_proactivity_until=0.0,
            _foreground_user_quiet_until=0.0,
            _last_user_interaction_time=0.0,
        ),
        allow_no_user_anchor=True,
    ) != "desktop_background_disabled"

    # the deliberate operator kill-switch still takes background offline (crash-loop recovery)
    monkeypatch.setenv("AURA_ENABLE_BACKGROUND_COGNITION", "0")
    assert (
        background_policy.background_loop_start_reason(origin="subconscious_loop")
        == "background_cognition_disabled"
    )


def test_background_loop_start_allows_explicit_desktop_background_cognition(monkeypatch):
    from core.runtime import background_policy

    monkeypatch.setenv("AURA_SAFE_BOOT_DESKTOP", "1")
    monkeypatch.setenv("AURA_ENABLE_BACKGROUND_COGNITION", "1")
    monkeypatch.delenv("AURA_FOREGROUND_ONLY", raising=False)
    monkeypatch.setattr("core.runtime.proof_policy.proof_run_active", lambda *args, **kwargs: False)

    assert background_policy.background_loop_start_reason(origin="subconscious_loop") == ""


def test_orchestrator_background_runs_under_safe_boot_but_honors_explicit_disable(monkeypatch):
    from core.orchestrator import main as orchestrator_main

    monkeypatch.setenv("AURA_SAFE_BOOT_DESKTOP", "1")
    monkeypatch.delenv("AURA_ENABLE_BACKGROUND_COGNITION", raising=False)
    monkeypatch.delenv("AURA_FOREGROUND_ONLY", raising=False)
    monkeypatch.setattr("core.runtime.proof_policy.proof_run_active", lambda *args, **kwargs: False)

    # safe/protected boot keeps the background runtime LIVE (not quiescent) — full Aura
    assert orchestrator_main._background_quiescent_runtime("pneuma_background") is False

    # only the explicit operator kill-switch quiesces background cognition
    monkeypatch.setenv("AURA_ENABLE_BACKGROUND_COGNITION", "0")

    assert orchestrator_main._background_quiescent_runtime("pneuma_background") is True


@pytest.mark.asyncio
async def test_dreaming_process_defers_during_desktop_safe_boot(monkeypatch):
    from core.consciousness.dreaming import DreamingProcess

    monkeypatch.setenv("AURA_SAFE_BOOT_DESKTOP", "1")
    monkeypatch.setenv("AURA_ENABLE_BACKGROUND_COGNITION", "0")
    monkeypatch.delenv("AURA_FOREGROUND_ONLY", raising=False)
    monkeypatch.setattr("core.runtime.proof_policy.proof_run_active", lambda *args, **kwargs: False)

    process = DreamingProcess(
        SimpleNamespace(
            is_busy=False,
            _suppress_unsolicited_proactivity_until=0.0,
            _foreground_user_quiet_until=0.0,
            _last_user_interaction_time=0.0,
        ),
        interval=0.01,
    )

    await process.start()

    assert process._running is False
    assert process._task is None


def test_constitutive_compute_budget_throttles_under_foreground_activity(monkeypatch):
    from core.runtime import background_policy, foreground_guard

    monkeypatch.setattr(foreground_guard, "foreground_activity_reason", lambda: "foreground_chat_active")
    monkeypatch.setattr("core.container.ServiceContainer.get", lambda _name, default=None: default)
    monkeypatch.setattr(
        background_policy.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(percent=40.0),
    )
    monkeypatch.setattr(background_policy, "get_unified_failure_state", lambda: {"pressure": 0.0})
    monkeypatch.setattr("core.runtime.proof_policy.proof_run_active", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        background_policy,
        "_read_memory_pressure_snapshot",
        lambda: background_policy._MemoryPressureSnapshot(
            pressure_pct=40.0,
            reason="memory_pressure_40.0",
            refuse_heavy_local_generation=False,
        ),
    )

    budget = background_policy.constitutive_compute_budget("liquid_substrate", 20.0)

    assert budget.effective_hz == pytest.approx(2.0)
    assert budget.interval_s == pytest.approx(0.5)
    assert budget.foreground_active is True
    assert budget.reason == "foreground_chat_active"


def test_constitutive_compute_budget_throttles_under_memory_pressure(monkeypatch):
    from core.runtime import background_policy, foreground_guard

    monkeypatch.setattr(foreground_guard, "foreground_activity_reason", lambda: "")
    monkeypatch.setattr("core.container.ServiceContainer.get", lambda _name, default=None: default)
    monkeypatch.setattr(
        background_policy.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(percent=88.0),
    )
    monkeypatch.setattr(background_policy, "get_unified_failure_state", lambda: {"pressure": 0.0})
    monkeypatch.setattr("core.runtime.proof_policy.proof_run_active", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        background_policy,
        "_read_memory_pressure_snapshot",
        lambda: background_policy._MemoryPressureSnapshot(
            pressure_pct=88.0,
            reason="memory_pressure_88.0",
            refuse_heavy_local_generation=False,
        ),
    )

    budget = background_policy.constitutive_compute_budget("unified_field", 20.0)

    assert budget.effective_hz == pytest.approx(2.0)
    assert budget.memory_percent == pytest.approx(88.0)
    assert budget.reason == "memory_pressure_88.0"


@pytest.mark.asyncio
async def test_initiative_synthesis_blocks_when_will_authorization_unavailable(monkeypatch):
    from core import initiative_synthesis as synthesis_module
    from core import will as will_module
    from core.agency.initiative_arbiter import ScoredInitiative

    synth = synthesis_module.InitiativeSynthesizer()
    synth.submit("inspect runtime drift", "test_source", urgency=0.9)
    scored = ScoredInitiative(
        initiative={
            "goal": "inspect runtime drift",
            "source": "test_source",
            "urgency": 0.9,
        },
        scores={"resource_cost": 0.1},
        final_score=0.91,
        rationale="selected by test arbiter",
    )
    arbiter = SimpleNamespace(arbitrate=AsyncRecorderCallable(scored))

    def _service_get(name, default=None):
        if name == "initiative_arbiter":
            return arbiter
        if name == "internal_simulator":
            return None
        return default

    monkeypatch.setattr(synthesis_module.ServiceContainer, "get", _service_get)
    monkeypatch.setattr(
        will_module,
        "get_will",
        lambda: OfflineWill(),
    )

    result = await synth.synthesize(SimpleNamespace(cognition=SimpleNamespace(pending_initiatives=[])))

    assert result.approved is False
    assert result.winner is None
    assert "will_authorization_unavailable=RuntimeError" in result.rationale


def test_bryan_model_save_is_debounced(monkeypatch, tmp_path):
    monkeypatch.setattr(user_model_module, "_USER_MODEL_PATH", tmp_path / "user_model.json")
    monkeypatch.setattr(user_model_module, "_SAVE_DEBOUNCE_SECONDS", 10.0)

    engine = user_model_module.BryanModelEngine()
    writes: list[str] = []
    monkeypatch.setattr(engine, "_write_now", lambda: writes.append("write"))

    engine._last_saved = time.time()
    engine.save()
    engine.save()

    assert writes == []
    assert engine._save_timer is not None

    engine._save_timer.cancel()
    engine._flush_pending_save()

    assert writes == ["write"]


def test_heartstone_save_is_debounced(monkeypatch, tmp_path):
    monkeypatch.setattr(heartstone_module, "_PERSIST_PATH", tmp_path / "heartstone_values.json")
    monkeypatch.setattr(heartstone_module, "_SAVE_DEBOUNCE_SECONDS", 10.0)

    values = heartstone_module.HeartstoneValues()
    writes: list[str] = []
    monkeypatch.setattr(values, "_write_now", lambda: writes.append("write"))

    values._last_saved = time.time()
    values._save()
    values._save()

    assert writes == []
    assert values._save_timer is not None

    values._save_timer.cancel()
    values._flush_pending_save()

    assert writes == ["write"]
