"""Injection-point safety contracts for the affective steering engine.

Root incident (2026-07-19): fluent spliced-dialog nonsense served live. The
steering stack had two structural holes: (1) hooks trusted whatever alpha the
20Hz substrate sync last wrote — a stalled/dead sync froze steering at its
last-hot value (or the install-time DEFAULT_ALPHA=5.0, which the governor
itself would never emit; its own clip is 3.0); (2) only explicitly-marked
jobs got the user-surface clamp, so a dropped flag decoded hot. These tests
pin the fixes: a hard ceiling and staleness derating AT THE INJECTION POINT,
and fail-safe clamping for unmarked worker jobs.
"""
from __future__ import annotations

import time

from core.brain.llm.mlx_worker import (
    _surface_control_alpha,
    _surface_generation_contract_enabled,
)
from core.consciousness.affective_steering import DEFAULT_ALPHA, AffectiveSteeringHook


def _bare_hook(alpha: float) -> AffectiveSteeringHook:
    hook = AffectiveSteeringHook.__new__(AffectiveSteeringHook)
    hook._alpha = alpha
    hook._last_substrate_sync_monotonic = 0.0
    return hook


class TestEffectiveAlphaGuards:
    def test_install_time_default_exceeds_governor_ceiling(self):
        # The premise of the ceiling: DEFAULT_ALPHA really is hotter than the
        # governor's own clip. If this stops being true, revisit the guard.
        assert DEFAULT_ALPHA > 3.0

    def test_never_synced_hook_derates_to_safe_alpha(self):
        hook = _bare_hook(DEFAULT_ALPHA)
        # No substrate sync has EVER run — injection must not run hot.
        assert hook._effective_alpha() <= hook._STALE_SAFE_ALPHA

    def test_fresh_sync_allows_governor_range_but_caps_at_ceiling(self):
        hook = _bare_hook(DEFAULT_ALPHA)
        hook._last_substrate_sync_monotonic = time.monotonic()
        effective = hook._effective_alpha()
        assert effective == hook._INJECTION_ALPHA_CEILING
        hook._alpha = 1.7
        assert hook._effective_alpha() == 1.7

    def test_stale_sync_derates(self):
        hook = _bare_hook(2.5)
        hook._last_substrate_sync_monotonic = (
            time.monotonic() - hook._SYNC_STALE_AFTER_S - 1.0
        )
        assert hook._effective_alpha() <= hook._STALE_SAFE_ALPHA

    def test_low_alpha_untouched_by_staleness(self):
        hook = _bare_hook(0.12)
        assert hook._effective_alpha() == 0.12

    def test_invalid_alpha_disables_injection(self):
        assert _bare_hook(float("nan"))._effective_alpha() == 0.0
        assert _bare_hook(-1.0)._effective_alpha() == 0.0
        assert _bare_hook(0.0)._effective_alpha() == 0.0


class TestFailSafeJobClamping:
    def test_unmarked_job_is_clamped(self):
        assert _surface_generation_contract_enabled({}) is True

    def test_explicit_opt_out_is_honored(self):
        assert (
            _surface_generation_contract_enabled(
                {"allow_full_affective_steering": True}
            )
            is False
        )

    def test_unmarked_job_gets_prose_tier_alpha(self):
        # Unmarked jobs clamp at the ordinary user-prose tier, not the
        # near-off strict tier — internal cognition keeps affective color.
        alpha = _surface_control_alpha({}, 5.0)
        assert 0.2 <= alpha <= 0.4

    def test_strict_tier_remains_near_off(self):
        assert _surface_control_alpha({"strict_answer_contract": True}, 5.0) <= 0.1
