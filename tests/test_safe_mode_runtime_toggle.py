"""tests/test_safe_mode_runtime_toggle.py
=============================================
The user-facing "Safe mode" toggle must actually do something. It was a dead
setting: boot always applied full mode and nothing read safety.safe_mode. These
tests lock in the live runtime toggle and the settings → runtime bridge.
"""
from __future__ import annotations

from core.safe_mode import is_safe_mode, runtime_feature_enabled, set_safe_mode


class _Kernel:
    def __init__(self, volition: int):
        self.volition_level = volition


class _Orch:
    def __init__(self, volition: int = 3):
        self.kernel = _Kernel(volition)
        self.conversation_history: list = []


def test_set_safe_mode_disables_self_directed_behavior():
    orch = _Orch(volition=3)
    out = set_safe_mode(orch, True)
    assert out["safe_mode"] is True
    assert is_safe_mode(orch) is True
    # Consumers read these via runtime_feature_enabled on their next tick.
    assert runtime_feature_enabled(orch, "self_modification") is False
    assert runtime_feature_enabled(orch, "persona_evolution") is False
    assert runtime_feature_enabled(orch, "dream_cycle") is False
    assert runtime_feature_enabled(orch, "memory_consolidation") is False


def test_full_mode_restores_capabilities_at_high_volition():
    orch = _Orch(volition=3)
    set_safe_mode(orch, True)
    out = set_safe_mode(orch, False)
    assert out["safe_mode"] is False
    assert is_safe_mode(orch) is False
    # Volition 3 in full mode re-enables self-modification.
    assert runtime_feature_enabled(orch, "self_modification") is True
    assert runtime_feature_enabled(orch, "persona_evolution") is True


def test_safe_mode_holds_even_at_high_volition():
    # Safe mode must override volition — a "paused" kill switch.
    orch = _Orch(volition=3)
    set_safe_mode(orch, True)
    assert runtime_feature_enabled(orch, "self_modification") is False


def test_toggle_is_idempotent():
    orch = _Orch(volition=1)
    set_safe_mode(orch, True)
    set_safe_mode(orch, True)
    assert is_safe_mode(orch) is True
    assert runtime_feature_enabled(orch, "self_modification") is False


def test_is_safe_mode_defaults_false_on_unpatched_orch():
    assert is_safe_mode(_Orch()) is False


def test_settings_bridge_applies_to_live_orchestrator(monkeypatch):
    """Changing safety.safe_mode in the store must apply to the live runtime."""
    import core.container as container
    import interface.routes.settings as settings

    orch = _Orch(volition=3)
    monkeypatch.setattr(container.ServiceContainer, "get", staticmethod(
        lambda name, default=None: orch if name == "orchestrator" else default
    ))

    # Drive the bridge directly (it is what the store subscriber invokes).
    settings._apply_safe_mode_to_runtime("safety.safe_mode", False, True)
    assert is_safe_mode(orch) is True
    settings._apply_safe_mode_to_runtime("safety.safe_mode", True, False)
    assert is_safe_mode(orch) is False


def test_settings_bridge_ignores_other_keys(monkeypatch):
    import core.container as container
    import interface.routes.settings as settings

    orch = _Orch(volition=3)
    set_safe_mode(orch, True)
    monkeypatch.setattr(container.ServiceContainer, "get", staticmethod(
        lambda name, default=None: orch if name == "orchestrator" else default
    ))
    # An unrelated key must not touch safe mode.
    settings._apply_safe_mode_to_runtime("theme.mode", "auto", "dark")
    assert is_safe_mode(orch) is True


def test_settings_store_registers_safe_mode_subscriber(monkeypatch, tmp_path):
    """get_settings() must wire the safe-mode bridge as a subscriber."""
    import interface.routes.settings as settings

    monkeypatch.setattr(settings, "_STORE", None)
    store = settings.get_settings()
    assert settings._apply_safe_mode_to_runtime in store._subscribers
