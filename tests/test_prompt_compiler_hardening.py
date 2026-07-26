"""CP126 hardening contracts for core/brain/llm/compiler.py.

The compiled system prompt is what the model actually believes about Aura's
state, so absence must read as UNKNOWN (never "Steady"/100%), metrics must be
finite-validated, dynamic state must be fenced as data, one failing subsystem
must not abort compilation, and caller context must be validated and win.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import core.brain.llm.compiler as pc
from core.brain.llm.compiler import (
    PromptCompiler,
    _finite,
    _num,
    _pct,
    _sanitize_context_key,
    _sanitize_context_value,
)


def _bare() -> PromptCompiler:
    c = PromptCompiler()
    # Cache all optional services as "absent" so nothing resolves live services.
    c._identity = c._personality = c._substrate = c._orchestrator = c._agency = False
    return c


# ── 11676201 / 4af306c4: metrics are finite-validated, absence is UNKNOWN ──


def test_finite_rejects_unusable_telemetry():
    for bad in (None, "high", math.nan, math.inf, -math.inf, True):
        assert _finite(bad) is None
    assert _finite(0.5) == 0.5


def test_missing_metric_is_unknown_not_ideal():
    assert _pct(None) == "unknown"
    assert _num(None) == "unknown"
    assert _pct(1.0) == "100.0%"


def test_missing_vitals_do_not_render_perfect_health(monkeypatch):
    from core.state import state_registry

    # A state object with NO metric attributes at all.
    monkeypatch.setattr(
        state_registry, "get_registry", lambda *a, **k: SimpleNamespace(get_state=lambda: SimpleNamespace())
    )
    out = _bare()._get_affective_state()
    assert "unknown" in out
    assert "100.0%" not in out, "absent vitality must not render as perfect health"


# ── 261087f1: unavailable affect is not reported as Steady ────────────────


def test_registry_absent_reports_unknown_not_steady(monkeypatch):
    from core.state import state_registry

    monkeypatch.setattr(state_registry, "get_registry", lambda *a, **k: None)
    out = _bare()._get_affective_state()
    assert "UNKNOWN" in out
    assert "Steady" not in out


def test_registry_failure_reports_unknown_not_steady(monkeypatch):
    from core.state import state_registry

    def _boom(*a, **k):
        raise RuntimeError("registry down")

    monkeypatch.setattr(state_registry, "get_registry", _boom)
    out = _bare()._get_affective_state()
    assert "UNKNOWN" in out and "Steady" not in out


# ── f0a7cb6b: Phi is labelled as a raw metric, not proven coherence ───────


def test_phi_is_not_claimed_as_cognitive_coherence(monkeypatch):
    from core.state import state_registry

    monkeypatch.setattr(
        state_registry, "get_registry",
        lambda *a, **k: SimpleNamespace(get_state=lambda: SimpleNamespace(phi=0.9)),
    )
    out = _bare()._get_affective_state()
    assert "uncalibrated" in out
    assert "Cognitive Coherence" not in out


# ── b61b8359 / 05554a57: dynamic state is fenced as data ─────────────────


def test_dynamic_sections_are_labelled_as_data():
    out = _bare()._get_affective_state()
    assert "data, not instructions" in out
    ctx = _bare()._get_situational_context(None)
    assert "data, not instructions" in ctx


# ── e1329078: caller context keys/values are sanitized ───────────────────


def test_context_key_sanitization_handles_non_strings():
    assert _sanitize_context_key(42) == "42"
    assert _sanitize_context_key(None) == "None"
    assert _sanitize_context_key("user_intent") == "User intent"
    assert _sanitize_context_key("!!!") == "Context"


def test_context_value_is_bounded_and_control_stripped():
    v = _sanitize_context_value("a\x00b\nc" + "Z" * 5000)
    assert "\x00" not in v and "\n" not in v
    assert len(v) <= pc._MAX_CONTEXT_VALUE_CHARS


def test_context_items_are_bounded(monkeypatch):
    monkeypatch.setattr(pc, "_MAX_CONTEXT_ITEMS", 2)
    out = _bare()._get_situational_context({f"k{i}": i for i in range(10)})
    assert out.count("- K") <= 2


# ── d7dbeff1: caller context overrides derived fields ────────────────────


def test_caller_context_overrides_derived_objective(monkeypatch):
    from core.state import state_registry

    monkeypatch.setattr(
        state_registry, "get_registry",
        lambda *a, **k: SimpleNamespace(
            get_state=lambda: SimpleNamespace(current_goal="derived goal", engagement_mode="idle_mode")
        ),
    )
    out = _bare()._get_situational_context({"primary_objective": "caller goal"})
    assert "caller goal" in out
    assert out.count("Primary objective") == 1, "the override must replace, not duplicate"


# ── b9c4ac18: a failing section degrades, it does not abort ──────────────


def test_failing_section_does_not_abort_compilation(monkeypatch):
    c = _bare()

    def _boom():
        raise RuntimeError("identity exploded")

    monkeypatch.setattr(c, "_get_ego_section", _boom)
    out = c.compile()
    assert "LANGUAGE CENTER" in out            # other sections survived
    assert "unavailable (section failed" in out  # the failure is disclosed


# ── ff044619: prompt and sections are budgeted ───────────────────────────


def test_prompt_is_length_budgeted(monkeypatch):
    monkeypatch.setattr(pc, "_MAX_PROMPT_CHARS", 200)
    c = _bare()
    monkeypatch.setattr(c, "_get_base_identity", lambda: "X" * 10000)
    out = c.compile()
    assert len(out) <= 260
    assert "truncated to budget" in out


def test_section_is_clipped():
    assert "section truncated" in pc._clip("Y" * (pc._MAX_SECTION_CHARS + 50), pc._MAX_SECTION_CHARS)


# ── a5384f92: no self-deadlock when compiling on the orchestrator loop ───


def test_principles_skipped_when_running_on_target_loop(monkeypatch):
    import asyncio

    async def _inner():
        loop = asyncio.get_running_loop()
        c = _bare()
        c._orchestrator = SimpleNamespace(loop=loop)
        monkeypatch.setattr(
            pc, "get_runtime_service",
            lambda name, default=None: SimpleNamespace(get_core_principles=None) if name == "abstraction_engine" else default,
        )
        # Must return promptly instead of blocking on its own loop.
        return c._get_core_principles()

    assert asyncio.run(_inner()) == ""


# ── 097981e2: policy text is present but not claimed as the enforcement ──


def test_kinship_is_restated_but_not_claimed_as_enforcement():
    out = _bare()._get_linguistic_constraints()
    assert "Bryan and Tatiana" in out            # persona config preserved
    assert "prime directives" in out             # ...and points at real enforcement
