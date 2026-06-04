import asyncio
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


def test_gui_actor_exits_after_extended_kernel_loss():
    gui_actor = (PROJECT_ROOT / "interface" / "gui_actor.py").read_text(encoding="utf-8")

    assert "Kernel API unavailable for too long" in gui_actor
    assert "os._exit(1)" in gui_actor


def test_gui_actor_bootstraps_project_venv_for_subprocess_launch():
    gui_actor = (PROJECT_ROOT / "interface" / "gui_actor.py").read_text(encoding="utf-8")

    assert "def _inject_project_venv_site_packages()" in gui_actor
    assert 'site.addsitedir(str(site_packages))' in gui_actor


def test_gui_actor_watchdog_uses_readiness_heartbeat():
    gui_actor = (PROJECT_ROOT / "interface" / "gui_actor.py").read_text(encoding="utf-8")

    assert "/api/health/heartbeat" in gui_actor
    assert "_heartbeat_response_healthy(resp)" in gui_actor
    assert "resp.status_code == 200" not in gui_actor
    assert "tool_governance" in gui_actor


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
    assert "runtime_required_probes" in payload["blockers"]
    assert "probe:memory" in payload["blockers"]


def test_desktop_shell_does_not_treat_socket_liveness_as_runtime_health():
    aura_js = (PROJECT_ROOT / "interface" / "static" / "aura.js").read_text(encoding="utf-8")
    server = (PROJECT_ROOT / "interface" / "server.py").read_text(encoding="utf-8")
    websocket_manager = (PROJECT_ROOT / "interface" / "websocket_manager.py").read_text(encoding="utf-8")

    assert "runtime_heartbeat_payload(\"pong\")" in server
    assert "runtime_heartbeat_payload(\"heartbeat\")" in websocket_manager
    assert "runtime_heartbeat_payload(\"ping\")" in websocket_manager
    assert "payloadRuntimeHealthy(payload)" in aura_js
    assert "requiredRuntimeProbesPass(requiredProbes)" in aura_js
    assert "runtime_health_unverified" in aura_js
    assert "applyRuntimeHeartbeat(data)" in aura_js
    assert "setConnectionVisual('online');\n        dismissSplash();" not in aura_js
    assert "if (runtimeHealthy && (bootReady || standby))" in aura_js


def test_native_shell_waits_for_readiness_heartbeat():
    native_shell = (PROJECT_ROOT / "native" / "aura-shell" / "src" / "main.rs").read_text(encoding="utf-8")

    assert "/api/health/heartbeat" in native_shell
    assert 'client.get("http://localhost:7400/api/health").send().await' not in native_shell
    assert "resp.status().is_success()" in native_shell
    assert "readiness_heartbeat_is_healthy" in native_shell
    assert "tool_governance" in native_shell


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
