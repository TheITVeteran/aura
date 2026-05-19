import sys
import types
from types import SimpleNamespace

import pytest

from core.orchestrator.mixins.boot.boot_autonomy import (
    BootAutonomyMixin,
    _safe_priority,
)


class Startable:
    def __init__(self, label: str, starts: list[str]):
        self.label = label
        self.starts = starts

    async def start(self):
        self.starts.append(self.label)


def _module(name: str, **attrs):
    module = types.ModuleType(name)
    for attr_name, value in attrs.items():
        setattr(module, attr_name, value)
    return module


@pytest.mark.asyncio
async def test_autonomous_evolution_continues_after_failed_boot_step():
    calls: list[str] = []

    class Harness(BootAutonomyMixin):
        async def _ok_first(self):
            calls.append("first")

        async def _fails(self):
            calls.append("fails")
            raise RuntimeError("transient boot failure")

        async def _ok_last(self):
            calls.append("last")

    harness = Harness()
    harness._init_self_modification_engine = harness._ok_first
    harness._init_transcendence_layer = harness._fails
    harness._init_cognitive_modulators = harness._ok_last
    for name in (
        "_init_meta_learning",
        "_init_meta_optimization",
        "_init_concept_bridge",
        "_init_advanced_ontology",
        "_init_motivation_engine",
        "_init_reflex_engine",
        "_init_identity_gate",
        "_init_lazarus_brainstem",
        "_init_persona_evolver",
        "_init_live_learner",
        "_init_autonomous_task_engine",
        "_init_continuous_learner",
        "_init_fictional_synthesis",
        "_init_final_foundations",
        "_init_evolution_orchestrator",
        "_init_singularity_loops",
    ):
        setattr(harness, name, harness._ok_last)

    await harness._init_autonomous_evolution()

    assert calls[:3] == ["first", "fails", "last"]
    assert calls.count("last") == 17


@pytest.mark.asyncio
async def test_singularity_loop_disable_still_wires_tier4_boot(monkeypatch):
    starts: list[str] = []
    registered: dict[str, object] = {}

    monkeypatch.setenv("AURA_ENABLE_SINGULARITY_LOOPS", "0")
    monkeypatch.delenv("AURA_FOREGROUND_ONLY", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "core.world_state",
        _module("core.world_state", get_world_state=lambda: Startable("world_state", starts)),
    )
    monkeypatch.setitem(
        sys.modules,
        "core.initiative_synthesis",
        _module(
            "core.initiative_synthesis",
            get_initiative_synthesizer=lambda: Startable("initiative", starts),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "core.simulation.internal_simulator",
        _module(
            "core.simulation.internal_simulator",
            InternalSimulator=lambda: SimpleNamespace(label="internal_simulator"),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "core.continuous_cognition",
        _module(
            "core.continuous_cognition",
            get_continuous_cognition=lambda: Startable("continuous_cognition", starts),
        ),
    )
    monkeypatch.setattr(
        "core.orchestrator.mixins.boot.boot_autonomy.ServiceContainer.get",
        staticmethod(lambda _name, default=None: default),
    )
    monkeypatch.setattr(
        "core.orchestrator.mixins.boot.boot_autonomy.ServiceContainer.register_instance",
        staticmethod(lambda name, instance, required=True: registered.setdefault(name, instance)),
    )

    await BootAutonomyMixin()._init_singularity_loops()

    assert starts == ["world_state", "initiative", "continuous_cognition"]
    assert "internal_simulator" in registered


def test_safe_priority_falls_back_for_bad_goal_priority():
    assert _safe_priority("not-a-number") == 0.6
    assert _safe_priority("0.2") == 0.6
    assert _safe_priority("0.9") == 0.9
