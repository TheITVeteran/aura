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
    assert status["components"]["overt_action"]["scheduled"] is True
    assert status["components"]["deliberation"]["scheduled"] is True


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
