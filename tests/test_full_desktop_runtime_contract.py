from types import SimpleNamespace

from core.container import ServiceContainer
from interface.routes.system import _collect_full_runtime_status


def _status_service(method="get_status", **status):
    return SimpleNamespace(**{method: lambda: dict(status)})


def _full_services():
    swarm = _status_service(
        available=True,
        active_shards=0,
        capacity=2,
        last_deliberation={"status": "never_run"},
    )
    return {
        "pneuma": object(),
        "mhaf": object(),
        "curiosity_engine": _status_service(running=True),
        "proactive_comm": _status_service(running=True),
        "autonomous_initiative_loop": _status_service(
            running=True,
            core_tasks={
                "world": True,
                "knowledge": True,
                "self_development": True,
                "social": True,
                "mission": True,
            },
        ),
        "research_cycle": _status_service(running=True),
        "self_healing": _status_service(running=True),
        "self_modification_engine": _status_service(
            method="runtime_status",
            running=True,
            mode="validation_quarantine",
        ),
        "consciousness": SimpleNamespace(_running=True),
        "autonomy_conductor": _status_service(
            method="status",
            active=True,
            jobs={
                "overt_action_cycle": {},
                "internal_deliberation_cycle": {},
            },
        ),
        "agency_core": SimpleNamespace(swarm=swarm),
        "overt_action_loop": _status_service(method="status", enabled=True),
        "wake_word": _status_service(running=True),
        "screen_perception": _status_service(running=True, captures=1, last_hash="screen"),
        "perceptual_pump": _status_service(
            running=True,
            frames_produced=3,
            substrate_injections=1,
            latest_frame={"active_app": "Aura Zenith"},
        ),
        "cognitive_situation": _status_service(
            running=True,
            frames_built=1,
            latest={"semantic_flexibility": 0.55, "sensorimotor_grounding": 0.62},
        ),
        "imagination_engine": _status_service(
            method="snapshot",
            running=True,
            status="active",
            frames=1,
        ),
        "timescale_bridge": _status_service(
            running=True,
            observations=2,
            last_reconciliation={"foreground_anchor_required": False},
        ),
        "ambient_developer_stream": _status_service(
            running=True,
            frames=1,
            latest_frame={"summary": "ambient developer stream observed no material changes"},
        ),
        "autonomic_reflection_loop": _status_service(
            running=True,
            reflections_written=1,
            latest_reflection={"ambient_summary": "ambient developer stream observed no material changes"},
        ),
    }


def _install_services(monkeypatch, services):
    monkeypatch.setattr(
        ServiceContainer,
        "get",
        classmethod(lambda _cls, name, default=None: services.get(name, default)),
    )


def test_full_desktop_runtime_reports_every_canonical_background_organ(monkeypatch):
    monkeypatch.setenv("AURA_LAUNCHED_FROM_APP", "1")
    monkeypatch.setenv("AURA_DESKTOP_RESOURCE_GUARD", "1")
    monkeypatch.delenv("AURA_SAFE_BOOT_DESKTOP", raising=False)
    monkeypatch.delenv("AURA_FOREGROUND_ONLY", raising=False)
    monkeypatch.delenv("AURA_ENABLE_BACKGROUND_COGNITION", raising=False)
    _install_services(monkeypatch, _full_services())

    status = _collect_full_runtime_status(
        {"online": True, "tick_count": 4},
        {"online": True, "tick_count": 2},
    )

    assert status["profile"] == "full_desktop"
    assert status["resource_guard_enabled"] is True
    assert status["full_runtime_expected"] is True
    assert status["ready"] is True
    assert status["blockers"] == []
    assert status["components"]["self_modification"]["mode"] == "validation_quarantine"
    assert status["components"]["autonomous_initiative"]["core_tasks"]["social"] is True
    assert status["components"]["overt_action"]["scheduled"] is True
    assert status["components"]["deliberation"]["scheduled"] is True
    assert status["components"]["screen_perception"]["running"] is True
    assert status["components"]["perceptual_pump"]["running"] is True
    assert status["components"]["cognitive_situation"]["running"] is True
    assert status["components"]["imagination_engine"]["running"] is True
    assert status["components"]["timescale_bridge"]["running"] is True
    assert status["components"]["ambient_developer_stream"]["running"] is True
    assert status["components"]["autonomic_reflection_loop"]["running"] is True


def test_protected_desktop_boot_still_expects_full_background_runtime(monkeypatch):
    monkeypatch.setenv("AURA_LAUNCHED_FROM_APP", "1")
    monkeypatch.setenv("AURA_DESKTOP_RESOURCE_GUARD", "1")
    monkeypatch.setenv("AURA_SAFE_BOOT_DESKTOP", "1")
    monkeypatch.delenv("AURA_FOREGROUND_ONLY", raising=False)
    monkeypatch.delenv("AURA_ENABLE_BACKGROUND_COGNITION", raising=False)
    _install_services(monkeypatch, _full_services())

    status = _collect_full_runtime_status(
        {"online": True, "tick_count": 4},
        {"online": True, "tick_count": 2},
    )

    assert status["profile"] == "protected_full_desktop"
    assert status["resource_guard_enabled"] is True
    assert status["full_runtime_expected"] is True
    assert status["ready"] is True
    assert status["blockers"] == []
    assert status["components"]["autonomic_reflection_loop"]["running"] is True


def test_full_desktop_runtime_fails_readiness_when_background_organ_is_missing(monkeypatch):
    monkeypatch.setenv("AURA_LAUNCHED_FROM_APP", "1")
    monkeypatch.setenv("AURA_DESKTOP_RESOURCE_GUARD", "1")
    monkeypatch.delenv("AURA_SAFE_BOOT_DESKTOP", raising=False)
    monkeypatch.delenv("AURA_FOREGROUND_ONLY", raising=False)
    monkeypatch.delenv("AURA_ENABLE_BACKGROUND_COGNITION", raising=False)
    services = _full_services()
    services.pop("research_cycle")
    _install_services(monkeypatch, services)

    status = _collect_full_runtime_status({"online": True}, {"online": True})

    assert status["ready"] is False
    assert "research" in status["blockers"]


def test_full_desktop_runtime_fails_readiness_when_screen_perception_is_missing(monkeypatch):
    monkeypatch.setenv("AURA_LAUNCHED_FROM_APP", "1")
    monkeypatch.setenv("AURA_DESKTOP_RESOURCE_GUARD", "1")
    monkeypatch.delenv("AURA_SAFE_BOOT_DESKTOP", raising=False)
    monkeypatch.delenv("AURA_FOREGROUND_ONLY", raising=False)
    monkeypatch.delenv("AURA_ENABLE_BACKGROUND_COGNITION", raising=False)
    services = _full_services()
    services.pop("screen_perception")
    _install_services(monkeypatch, services)

    status = _collect_full_runtime_status({"online": True}, {"online": True})

    assert status["ready"] is False
    assert "screen_perception" in status["blockers"]


def test_full_desktop_runtime_fails_readiness_when_perceptual_pump_is_stopped(monkeypatch):
    monkeypatch.setenv("AURA_LAUNCHED_FROM_APP", "1")
    monkeypatch.setenv("AURA_DESKTOP_RESOURCE_GUARD", "1")
    monkeypatch.delenv("AURA_SAFE_BOOT_DESKTOP", raising=False)
    monkeypatch.delenv("AURA_FOREGROUND_ONLY", raising=False)
    monkeypatch.delenv("AURA_ENABLE_BACKGROUND_COGNITION", raising=False)
    services = _full_services()
    services["perceptual_pump"] = _status_service(running=False, frames_produced=0)
    _install_services(monkeypatch, services)

    status = _collect_full_runtime_status({"online": True}, {"online": True})

    assert status["ready"] is False
    assert "perceptual_pump" in status["blockers"]


def test_full_desktop_runtime_fails_readiness_when_cognitive_situation_missing(monkeypatch):
    monkeypatch.setenv("AURA_LAUNCHED_FROM_APP", "1")
    monkeypatch.setenv("AURA_DESKTOP_RESOURCE_GUARD", "1")
    monkeypatch.delenv("AURA_SAFE_BOOT_DESKTOP", raising=False)
    monkeypatch.delenv("AURA_FOREGROUND_ONLY", raising=False)
    monkeypatch.delenv("AURA_ENABLE_BACKGROUND_COGNITION", raising=False)
    services = _full_services()
    services.pop("cognitive_situation")
    _install_services(monkeypatch, services)

    status = _collect_full_runtime_status({"online": True}, {"online": True})

    assert status["ready"] is False
    assert "cognitive_situation" in status["blockers"]


def test_full_desktop_runtime_fails_readiness_when_imagination_engine_missing(monkeypatch):
    monkeypatch.setenv("AURA_LAUNCHED_FROM_APP", "1")
    monkeypatch.setenv("AURA_DESKTOP_RESOURCE_GUARD", "1")
    monkeypatch.delenv("AURA_SAFE_BOOT_DESKTOP", raising=False)
    monkeypatch.delenv("AURA_FOREGROUND_ONLY", raising=False)
    monkeypatch.delenv("AURA_ENABLE_BACKGROUND_COGNITION", raising=False)
    services = _full_services()
    services.pop("imagination_engine")
    _install_services(monkeypatch, services)

    status = _collect_full_runtime_status({"online": True}, {"online": True})

    assert status["ready"] is False
    assert "imagination_engine" in status["blockers"]


def test_full_desktop_runtime_fails_readiness_when_timescale_bridge_missing(monkeypatch):
    monkeypatch.setenv("AURA_LAUNCHED_FROM_APP", "1")
    monkeypatch.setenv("AURA_DESKTOP_RESOURCE_GUARD", "1")
    monkeypatch.delenv("AURA_SAFE_BOOT_DESKTOP", raising=False)
    monkeypatch.delenv("AURA_FOREGROUND_ONLY", raising=False)
    monkeypatch.delenv("AURA_ENABLE_BACKGROUND_COGNITION", raising=False)
    services = _full_services()
    services.pop("timescale_bridge")
    _install_services(monkeypatch, services)

    status = _collect_full_runtime_status({"online": True}, {"online": True})

    assert status["ready"] is False
    assert "timescale_bridge" in status["blockers"]


def test_full_desktop_runtime_fails_readiness_when_ambient_stream_missing(monkeypatch):
    monkeypatch.setenv("AURA_LAUNCHED_FROM_APP", "1")
    monkeypatch.setenv("AURA_DESKTOP_RESOURCE_GUARD", "1")
    monkeypatch.delenv("AURA_SAFE_BOOT_DESKTOP", raising=False)
    monkeypatch.delenv("AURA_FOREGROUND_ONLY", raising=False)
    monkeypatch.delenv("AURA_ENABLE_BACKGROUND_COGNITION", raising=False)
    services = _full_services()
    services.pop("ambient_developer_stream")
    _install_services(monkeypatch, services)

    status = _collect_full_runtime_status({"online": True}, {"online": True})

    assert status["ready"] is False
    assert "ambient_developer_stream" in status["blockers"]


def test_full_desktop_runtime_fails_readiness_when_autonomic_reflection_loop_missing(monkeypatch):
    monkeypatch.setenv("AURA_LAUNCHED_FROM_APP", "1")
    monkeypatch.setenv("AURA_DESKTOP_RESOURCE_GUARD", "1")
    monkeypatch.delenv("AURA_SAFE_BOOT_DESKTOP", raising=False)
    monkeypatch.delenv("AURA_FOREGROUND_ONLY", raising=False)
    monkeypatch.delenv("AURA_ENABLE_BACKGROUND_COGNITION", raising=False)
    services = _full_services()
    services.pop("autonomic_reflection_loop")
    _install_services(monkeypatch, services)

    status = _collect_full_runtime_status({"online": True}, {"online": True})

    assert status["ready"] is False
    assert "autonomic_reflection_loop" in status["blockers"]


def test_full_desktop_runtime_fails_readiness_when_initiative_loop_is_missing(monkeypatch):
    monkeypatch.setenv("AURA_LAUNCHED_FROM_APP", "1")
    monkeypatch.setenv("AURA_DESKTOP_RESOURCE_GUARD", "1")
    monkeypatch.delenv("AURA_SAFE_BOOT_DESKTOP", raising=False)
    monkeypatch.delenv("AURA_FOREGROUND_ONLY", raising=False)
    monkeypatch.delenv("AURA_ENABLE_BACKGROUND_COGNITION", raising=False)
    services = _full_services()
    services.pop("autonomous_initiative_loop")
    _install_services(monkeypatch, services)

    status = _collect_full_runtime_status({"online": True}, {"online": True})

    assert status["ready"] is False
    assert "autonomous_initiative" in status["blockers"]


def test_full_desktop_runtime_fails_readiness_when_initiative_task_is_dead(monkeypatch):
    monkeypatch.setenv("AURA_LAUNCHED_FROM_APP", "1")
    monkeypatch.setenv("AURA_DESKTOP_RESOURCE_GUARD", "1")
    monkeypatch.delenv("AURA_SAFE_BOOT_DESKTOP", raising=False)
    monkeypatch.delenv("AURA_FOREGROUND_ONLY", raising=False)
    monkeypatch.delenv("AURA_ENABLE_BACKGROUND_COGNITION", raising=False)
    services = _full_services()
    services["autonomous_initiative_loop"] = _status_service(
        running=False,
        enabled=True,
        core_tasks={
            "world": True,
            "knowledge": True,
            "self_development": False,
            "social": True,
            "mission": True,
        },
    )
    _install_services(monkeypatch, services)

    status = _collect_full_runtime_status({"online": True}, {"online": True})

    assert status["ready"] is False
    assert status["components"]["autonomous_initiative"]["core_tasks"]["self_development"] is False
    assert "autonomous_initiative" in status["blockers"]
