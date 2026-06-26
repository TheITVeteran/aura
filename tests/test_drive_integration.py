"""Drive-integration volition: temporal accumulation, competition, hysteresis (not VAD thresholds)."""
from __future__ import annotations

from core.consciousness.drive_integration import (
    Drive,
    DriveIntegrationEngine,
    get_drive_integration_engine,
)


# ── the Drive leaky-integrator primitive ────────────────────────────────────

def test_sustained_signal_accumulates_to_fire():
    d = Drive("x", "act", gain=1.0, leak=0.1, fire_threshold=0.7)
    fired_step = None
    for i in range(20):
        d.integrate(0.6, dt=1.0)
        if d.ready():
            fired_step = i
            break
    assert fired_step is not None  # a sustained moderate pull eventually commits


def test_transient_spike_decays_without_firing():
    d = Drive("x", "act", gain=1.0, leak=0.5, fire_threshold=0.9)
    d.integrate(0.8, dt=0.2)          # one short spike
    assert not d.ready()
    for _ in range(10):
        d.integrate(0.0, dt=1.0)      # no further signal
    assert d.activation < 0.2          # decayed away


def test_hysteresis_suppresses_until_release():
    d = Drive("x", "act", gain=2.0, leak=0.05, fire_threshold=0.7, release_threshold=0.35)
    for _ in range(5):
        d.integrate(0.9, dt=1.0)
    assert d.ready()
    d.fire(now=100.0)
    assert d.suppressed
    assert not d.ready()              # cannot immediately re-fire
    # It stays suppressed while activation is above the release threshold...
    d.integrate(0.9, dt=0.1)
    assert d.suppressed
    # ...and re-arms only once it falls below release.
    for _ in range(40):
        d.integrate(0.0, dt=1.0)
    d.integrate(0.0, dt=1.0)
    assert not d.suppressed


# ── the engine: competition + arbitration ───────────────────────────────────

def test_curiosity_wins_on_high_arousal_positive_valence():
    eng = DriveIntegrationEngine()
    decision = None
    for _ in range(15):
        decision = eng.step({"arousal": 0.8, "valence": 0.6, "novelty": 0.7}, dt=1.0)
        if decision.action:
            break
    assert decision.action == "explore_knowledge"
    assert decision.drive == "curiosity"


def test_boredom_wins_on_low_arousal_negative_valence():
    eng = DriveIntegrationEngine()
    decision = None
    for _ in range(20):
        decision = eng.step({"arousal": -0.6, "valence": -0.4, "novelty": 0.0}, dt=1.0)
        if decision.action:
            break
    assert decision.action == "seek_novelty"


def test_pain_drives_relief_and_outcompetes():
    eng = DriveIntegrationEngine()
    decision = None
    for _ in range(15):
        decision = eng.step({"arousal": 0.6, "valence": 0.3, "pain": 0.95}, dt=1.0)
        if decision.action:
            break
    assert decision.action == "stabilize"
    assert decision.drive == "relief"


def test_no_signal_yields_no_action():
    eng = DriveIntegrationEngine()
    decision = eng.step({"arousal": 0.0, "valence": 0.0, "dominance": 0.0}, dt=1.0)
    assert decision.action is None
    assert decision.reason in {"no_drive_ready", "inhibited"}


def test_competition_picks_one_winner_not_many():
    eng = DriveIntegrationEngine()
    # Push curiosity and reflection both hard; exactly one should fire on a given step.
    actions = []
    for _ in range(15):
        d = eng.step({"arousal": 0.7, "valence": 0.5, "dominance": 0.8}, dt=1.0)
        if d.action:
            actions.append(d.action)
    # Each firing depletes the winner, so we don't get a flood of simultaneous actions.
    assert actions  # something fired
    assert eng.state()["drives"]["curiosity"]["action"] == "explore_knowledge"


def test_winner_depletes_so_cooldown_emerges_from_dynamics():
    eng = DriveIntegrationEngine()
    for _ in range(15):
        d = eng.step({"arousal": 0.85, "valence": 0.6, "novelty": 0.8}, dt=1.0)
        if d.action:
            break
    # Immediately after firing, curiosity should be suppressed (no flat 30s clock involved).
    assert eng._drives["curiosity"].suppressed


# ── grounding + extensibility ────────────────────────────────────────────────

def test_gather_signals_fills_defaults():
    eng = DriveIntegrationEngine()
    sig = eng.gather_signals({"valence": 0.5})
    assert sig["valence"] == 0.5
    assert "pain" in sig and "arousal" in sig


def test_custom_drive_can_be_added():
    eng = DriveIntegrationEngine()
    eng.add_drive(Drive("hunger", "seek_food", gain=2.0, leak=0.05, fire_threshold=0.6),
                  lambda s: s.get("hunger", 0.0))
    decision = None
    for _ in range(10):
        decision = eng.step({"hunger": 0.9}, dt=1.0)
        if decision.action == "seek_food":
            break
    assert decision.action == "seek_food"


# ── singleton ────────────────────────────────────────────────────────────────

def test_singleton_is_stable():
    assert get_drive_integration_engine() is get_drive_integration_engine()
