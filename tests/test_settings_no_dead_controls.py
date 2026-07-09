"""tests/test_settings_no_dead_controls.py
==============================================
Backstop for the dead-control finding (see docs/SETTINGS_WIRING_AUDIT.md): the
settings panel persists every toggle but the runtime read almost none, so most
controls looked functional and did nothing. This test forces a conscious
classification of EVERY settings key — wired, frontend-only, or known-dead
(tracked debt) — so a NEW setting cannot be added as a silent no-op, and the
known-dead set cannot grow unnoticed.

When you wire a known-dead setting, move it from KNOWN_DEAD to WIRED. When you
add a new setting, classify it here (and wire it, or justify it as frontend-only)
or this test fails.
"""
from __future__ import annotations

from interface.routes.settings import _RUNTIME_MODE_KEYS, SCHEMA

# Settings the runtime actually enforces today (verified by other tests).
WIRED = {
    "safety.safe_mode",       # core.runtime.safe_mode.set_safe_mode via settings bridge
    "autonomy.level",         # "paused" -> restricted runtime via the same bridge
}

# Legitimately client-side only — the desktop/web shell renders these; there is
# nothing for the Python runtime to enforce.
FRONTEND_ONLY = {
    "theme.mode",
    "theme.reduced_motion",
}

# Tracked debt: persisted but NOT yet enforced by the runtime. Each needs a
# bridge+consumer+test (see docs/SETTINGS_WIRING_AUDIT.md). Shrink this set as
# settings get wired; do NOT grow it without a deliberate decision.
KNOWN_DEAD = {
    "model.local_path",
    "model.deep_path",
    "model.cloud_fallback_enabled",
    "voice.input_enabled",
    "voice.output_enabled",
    "voice.output_rate",
    "permissions.camera",
    "permissions.screen",
    "permissions.files_workspace",
    "autonomy.proactive_messaging",
    "autonomy.self_modification",
    "memory.retention_days",
    "memory.review_window",
    "privacy.mode",
    "dev.developer_mode",
    "dev.diagnostics_enabled",
    "notify.enabled",
    "notify.quiet_hours_start",
    "notify.quiet_hours_end",
}


def test_every_setting_is_classified():
    """No setting may be unclassified — a new one must be wired or justified."""
    classified = WIRED | FRONTEND_ONLY | KNOWN_DEAD
    schema_keys = {s.key for s in SCHEMA}
    unclassified = schema_keys - classified
    assert not unclassified, (
        f"New/unclassified settings: {sorted(unclassified)}. Wire them (preferred), "
        "mark frontend-only, or add to KNOWN_DEAD in docs/SETTINGS_WIRING_AUDIT.md."
    )
    # And no stale classifications pointing at removed keys.
    stale = classified - schema_keys
    assert not stale, f"Classifications for keys no longer in SCHEMA: {sorted(stale)}"


def test_classification_buckets_are_disjoint():
    assert not (WIRED & FRONTEND_ONLY)
    assert not (WIRED & KNOWN_DEAD)
    assert not (FRONTEND_ONLY & KNOWN_DEAD)


def test_wired_safety_controls_are_actually_bridged():
    # The two kill switches must be in the runtime-mode bridge key set.
    assert "safety.safe_mode" in _RUNTIME_MODE_KEYS
    assert "autonomy.level" in _RUNTIME_MODE_KEYS
    for key in _RUNTIME_MODE_KEYS:
        assert key in WIRED, f"{key} drives runtime but isn't marked WIRED"
