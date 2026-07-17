"""Contract tests: typed cognitive ingress for latent allocation.

Allocation must come from the organs that know — memory, body, goals, Will,
affect, self-model, world model — with per-signal receipts, and degrade to
stable baselines (not crashes, not zeros) when organs are absent.
"""
from __future__ import annotations

import pytest

from core.brain.cognitive_ingress import (
    COGNITIVE_INGRESS_SCHEMA,
    assemble_cognitive_ingress,
)


@pytest.fixture()
def registry(monkeypatch):
    """A controllable runtime-service registry."""
    services: dict[str, object] = {}

    import core.brain.cognitive_ingress as ingress_mod

    monkeypatch.setattr(
        ingress_mod, "_get_service", lambda name: services.get(name)
    )
    return services


def test_cold_registry_yields_baselines_with_absent_receipts(registry):
    ingress = assemble_cognitive_ingress(None, "explain the scheduler")
    receipt = ingress.to_receipt()
    assert receipt["schema"] == COGNITIVE_INGRESS_SCHEMA
    assert 0.0 < ingress.stakes < 1.0 and 0.0 < ingress.uncertainty < 1.0
    assert ingress.stakes == pytest.approx(receipt["base_stakes"], abs=1e-9)
    # Every organ is reported absent, none silently skipped.
    assert set(receipt["absent_sources"]) >= {
        "memory", "goals", "will", "affect", "world_model"
    }


def test_memory_familiarity_lowers_uncertainty_and_blankness_raises_it(registry):
    class FamiliarMemory:
        def search(self, objective, limit=4):
            return ["hit"] * 4

    class BlankMemory:
        def search(self, objective, limit=4):
            return []

    registry["memory_facade"] = FamiliarMemory()
    familiar = assemble_cognitive_ingress(None, "the scheduler design")
    registry["memory_facade"] = BlankMemory()
    blank = assemble_cognitive_ingress(None, "the scheduler design")
    assert familiar.uncertainty < blank.uncertainty
    familiar_signal = next(s for s in familiar.signals if s.source == "memory")
    assert familiar_signal.present and familiar_signal.value == 1.0


def test_goal_overlap_raises_stakes(registry):
    class Goals:
        def active_goals(self):
            return ["Ship the scheduler arbitration redesign safely"]

    registry["goal_engine"] = Goals()
    on_goal = assemble_cognitive_ingress(
        None, "How should the scheduler arbitration redesign handle deadlines?"
    )
    off_goal = assemble_cognitive_ingress(None, "What rhymes with orange?")
    assert on_goal.stakes > off_goal.stakes
    signal = next(s for s in on_goal.signals if s.source == "goals")
    assert signal.present and signal.value > 0


def test_affect_uncertainty_and_will_preference_flow_through(registry):
    class Affect:
        felt_uncertainty = 0.8

    class Will:
        deliberation_preference = 0.9

    registry["affect_engine"] = Affect()
    registry["volition"] = Will()
    ingress = assemble_cognitive_ingress(None, "resolve the conflict")
    baseline = ingress.to_receipt()["base_uncertainty"]
    assert ingress.uncertainty > baseline
    will_signal = next(s for s in ingress.signals if s.source == "will")
    assert will_signal.present and will_signal.stakes_delta > 0


def test_self_model_terms_raise_stakes(registry):
    identity = assemble_cognitive_ingress(
        None, "Should you change your own governance weights, Aura?"
    )
    mundane = assemble_cognitive_ingress(None, "Summarize the quarterly report")
    assert identity.stakes > mundane.stakes
    signal = next(s for s in identity.signals if s.source == "self_model")
    assert signal.present


def test_broken_organ_is_absent_not_fatal(registry):
    class ExplodingMemory:
        def search(self, objective, limit=4):
            raise RuntimeError("index corrupted")

    registry["memory_facade"] = ExplodingMemory()
    ingress = assemble_cognitive_ingress(None, "anything")
    signal = next(s for s in ingress.signals if s.source == "memory")
    assert signal.present is False


def test_receipt_proves_which_sources_moved_the_allocation(registry):
    class Goals:
        def active_goals(self):
            return ["Verify the latent cortex consolidation pipeline"]

    registry["goal_engine"] = Goals()
    ingress = assemble_cognitive_ingress(
        None, "Verify the latent cortex consolidation pipeline end to end"
    )
    receipt = ingress.to_receipt()
    moved = [s for s in receipt["signals"] if s["present"] and s["stakes_delta"] > 0]
    assert any(s["source"] == "goals" for s in moved)
    assert all("detail" in s for s in receipt["signals"])
