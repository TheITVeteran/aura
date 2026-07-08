import asyncio
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

        def __getattr__(self, name):
            # Every boot step the mixin enumerates that this test hasn't
            # pinned resolves to the benign recorder — the contract under
            # test is "a failed step never halts the sequence", and it must
            # not break every time a new autonomy step joins the boot list.
            if name.startswith("_init_"):
                return self._ok_last
            raise AttributeError(name)

    harness = Harness()
    harness._init_self_modification_engine = harness._ok_first
    harness._init_transcendence_layer = harness._fails
    harness._init_cognitive_modulators = harness._ok_last

    await harness._init_autonomous_evolution()

    assert calls[:3] == ["first", "fails", "last"]
    # every step after the failure still ran; a floor (not an exact count)
    # pins the contract without re-breaking on each new boot step. Steps the
    # mixin defines for real aren't stubbed by __getattr__ and don't append —
    # the floor covers the stubbed remainder of the sequence.
    assert calls.count("last") >= 10


@pytest.mark.asyncio
async def test_final_foundations_initialization_is_idempotent(monkeypatch):
    registrations: list[object] = []
    salvaged_calls = 0

    class Harness(BootAutonomyMixin):
        async def _init_salvaged_subsystems(self):
            nonlocal salvaged_calls
            salvaged_calls += 1

    sentinel = object()
    monkeypatch.setitem(
        sys.modules,
        "core.final_engines",
        _module(
            "core.final_engines",
            register_final_engines=lambda **_kwargs: registrations.append(sentinel) or sentinel,
        ),
    )
    harness = Harness()

    await harness._init_final_foundations()
    await harness._init_final_foundations()

    assert registrations == [sentinel]
    assert salvaged_calls == 1
    assert harness.final_engines is sentinel


def test_boot_has_one_owner_for_autonomy_substeps():
    from pathlib import Path

    boot_source = (
        Path(__file__).parents[1] / "core" / "orchestrator" / "boot.py"
    ).read_text(encoding="utf-8")

    assert "_spawn_boot_task(self._init_fictional_synthesis()" not in boot_source
    assert "_spawn_boot_task(self._init_final_foundations()" not in boot_source


@pytest.mark.asyncio
async def test_motivation_boot_rebinds_drive_aliases_to_started_engine(monkeypatch):
    from core.motivation import engine as motivation_module
    from core.orchestrator.mixins.boot import boot_autonomy

    registered: dict[str, object] = {}

    def _register(name, instance, *args, **kwargs):
        registered[name] = instance

    monkeypatch.setattr(
        boot_autonomy.ServiceContainer,
        "register_instance",
        staticmethod(_register),
    )
    monkeypatch.setattr(
        motivation_module,
        "_background_autonomy_block_reason",
        lambda _orchestrator: "unit_test_hold",
    )

    class Harness(BootAutonomyMixin):
        pass

    harness = Harness()
    await harness._init_motivation_engine()
    try:
        assert registered["motivation_engine"] is harness.motivation
        assert registered["drive_engine"] is harness.motivation
        assert registered["drives"] is harness.motivation
        assert harness.motivation.is_alive() is True
    finally:
        await harness.motivation.stop()
        await asyncio.sleep(0)


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
