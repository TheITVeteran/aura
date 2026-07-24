"""Contract tests for the Homeostate engine (Salt fusion).

Covers lowstate compilation (requisite DAG, cycles, determinism), the four
requisite semantics, idempotent apply with honest dry-run, the degradation
beacon, and the event-bus reactor roundtrip.
"""
from __future__ import annotations

import asyncio

import pytest

from core.runtime.homeostate import (
    CompileError,
    DegradationBeacon,
    HomeostateEngine,
    HomeostateReactor,
    StateSpec,
    compile_lowstate,
    install_default_catalog,
    reset_homeostate_for_test,
)


@pytest.fixture(autouse=True)
def _fresh_homeostate():
    reset_homeostate_for_test()
    yield
    reset_homeostate_for_test()


# ── compiler ──────────────────────────────────────────────────────────────

def test_compile_orders_by_requisites_then_declaration():
    specs = [
        StateSpec(id="c", fn="noop", require=("a", "b")),
        StateSpec(id="a", fn="noop"),
        StateSpec(id="b", fn="noop", require=("a",)),
    ]
    ordered = [s.id for s in compile_lowstate(specs)]
    assert ordered == ["a", "b", "c"]


def test_compile_rejects_duplicate_ids():
    with pytest.raises(CompileError, match="duplicate"):
        compile_lowstate([StateSpec(id="x", fn="noop"), StateSpec(id="x", fn="noop")])


def test_compile_rejects_unknown_requisite():
    with pytest.raises(CompileError, match="unknown state"):
        compile_lowstate([StateSpec(id="x", fn="noop", require=("ghost",))])


def test_compile_rejects_cycles():
    with pytest.raises(CompileError, match="cycle"):
        compile_lowstate([
            StateSpec(id="a", fn="noop", require=("b",)),
            StateSpec(id="b", fn="noop", require=("a",)),
        ])


# ── requisite semantics ───────────────────────────────────────────────────

def _engine_with(fns: dict[str, object]) -> HomeostateEngine:
    engine = HomeostateEngine()
    for name, fn in fns.items():
        engine.registry.register(name, fn)  # type: ignore[arg-type]
    return engine


def test_require_gates_downstream_on_failure():
    calls: list[str] = []

    def failing(test=False, watch_triggered=False, **_):
        calls.append("failing")
        return {"result": False, "changes": {}, "comment": "broken"}

    def downstream(test=False, watch_triggered=False, **_):
        calls.append("downstream")
        return {"result": True, "changes": {}, "comment": ""}

    engine = _engine_with({"t.fail": failing, "t.down": downstream})
    engine.define("hs", [
        StateSpec(id="one", fn="t.fail"),
        StateSpec(id="two", fn="t.down", require=("one",)),
    ])
    report = engine.apply("hs")
    assert not report.ok
    assert report.failed == ["one"]
    assert report.not_run == ["two"]
    assert calls == ["failing"]           # downstream never executed


def test_onchanges_runs_only_after_upstream_change():
    runs: list[str] = []

    def changer(test=False, watch_triggered=False, *, change: bool, **_):
        return {"result": True, "changes": {"did": True} if change else {}, "comment": ""}

    def reactor_state(test=False, watch_triggered=False, **_):
        runs.append("ran")
        return {"result": True, "changes": {}, "comment": ""}

    engine = _engine_with({"t.change": changer, "t.react": reactor_state})
    engine.define("quiet", [
        StateSpec(id="up", fn="t.change", args={"change": False}),
        StateSpec(id="down", fn="t.react", onchanges=("up",)),
    ])
    engine.define("noisy", [
        StateSpec(id="up", fn="t.change", args={"change": True}),
        StateSpec(id="down", fn="t.react", onchanges=("up",)),
    ])
    quiet = engine.apply("quiet")
    assert quiet.ok and runs == [] and "skipped" in quiet.results[1].comment
    noisy = engine.apply("noisy")
    assert noisy.ok and runs == ["ran"]


def test_onfail_runs_remediation_only_on_failure():
    remedies: list[str] = []

    def maybe(test=False, watch_triggered=False, *, ok: bool, **_):
        return {"result": ok, "changes": {}, "comment": ""}

    def remedy(test=False, watch_triggered=False, **_):
        remedies.append("remedied")
        return {"result": True, "changes": {"healed": True}, "comment": ""}

    engine = _engine_with({"t.maybe": maybe, "t.remedy": remedy})
    engine.define("healthy", [
        StateSpec(id="probe", fn="t.maybe", args={"ok": True}),
        StateSpec(id="heal", fn="t.remedy", onfail=("probe",)),
    ])
    engine.define("sick", [
        StateSpec(id="probe", fn="t.maybe", args={"ok": False}),
        StateSpec(id="heal", fn="t.remedy", onfail=("probe",)),
    ])
    assert engine.apply("healthy").ok and remedies == []
    sick = engine.apply("sick")
    assert remedies == ["remedied"]
    assert sick.results[1].changes == {"healed": True}


def test_watch_passes_trigger_flag():
    seen: list[bool] = []

    def changer(test=False, watch_triggered=False, **_):
        return {"result": True, "changes": {"c": 1}, "comment": ""}

    def watcher(test=False, watch_triggered=False, **_):
        seen.append(watch_triggered)
        return {"result": True, "changes": {}, "comment": ""}

    engine = _engine_with({"t.change": changer, "t.watch": watcher})
    engine.define("hs", [
        StateSpec(id="up", fn="t.change"),
        StateSpec(id="down", fn="t.watch", watch=("up",)),
    ])
    assert engine.apply("hs").ok
    assert seen == [True]


def test_state_exception_is_failure_not_crash():
    def boom(test=False, watch_triggered=False, **_):
        raise RuntimeError("kaput")

    engine = _engine_with({"t.boom": boom})
    engine.define("hs", [StateSpec(id="b", fn="t.boom")])
    report = engine.apply("hs")
    assert report.failed == ["b"]
    assert "kaput" in report.results[0].comment


def test_unknown_state_function_fails_cleanly():
    engine = HomeostateEngine()
    engine.define("hs", [StateSpec(id="x", fn="no.such")])
    report = engine.apply("hs")
    assert report.failed == ["x"]


# ── idempotence + dry-run ─────────────────────────────────────────────────

def test_file_directory_idempotent_and_dry_run(tmp_path):
    target = tmp_path / "organ" / "cavity"
    engine = HomeostateEngine()
    engine.define("dirs", [StateSpec(id="d", fn="file.directory", args={"path": str(target)})])

    dry = engine.apply("dirs", test=True)
    assert dry.ok and dry.changed == ["d"]
    assert not target.exists()                     # dry-run mutated nothing

    wet = engine.apply("dirs")
    assert wet.ok and wet.changed == ["d"]
    assert target.is_dir()

    again = engine.apply("dirs")
    assert again.ok and again.changed == []        # idempotent: no second change


def test_file_directory_reports_non_directory_conflict(tmp_path):
    target = tmp_path / "occupied"
    target.write_text("i am a file")
    engine = HomeostateEngine()
    engine.define("dirs", [StateSpec(id="d", fn="file.directory", args={"path": str(target)})])
    report = engine.apply("dirs")
    assert report.failed == ["d"]


def test_service_available_dry_run_does_not_instantiate():
    engine = HomeostateEngine()
    engine.define("svc", [
        StateSpec(id="missing", fn="service.available", args={"name": "no_such_service_xyz"}),
    ])
    report = engine.apply("svc", test=True)
    assert report.failed == ["missing"]
    assert "NOT registered" in report.results[0].comment


def test_default_catalog_compiles():
    engine = HomeostateEngine()
    install_default_catalog(engine)
    assert engine.catalog()["runtime_baseline"] >= 5
    # Dry-run of the real baseline never mutates and never raises.
    report = engine.apply("runtime_baseline", test=True)
    assert isinstance(report.ok, bool)


# ── beacon ────────────────────────────────────────────────────────────────

def test_degradation_beacon_fires_on_threshold_and_cools_down():
    from core.runtime.errors import record_degradation

    beacon = DegradationBeacon(window_s=300.0, threshold=3, cooldown_s=600.0)
    for _ in range(3):
        record_degradation(
            "homeostate_beacon_test_subsystem",
            RuntimeError("synthetic"),
            severity="warning",
            action="test fixture",
        )
    events = beacon.poll_once()
    mine = [e for e in events if e["subsystem"] == "homeostate_beacon_test_subsystem"]
    assert len(mine) == 1
    assert mine[0]["serious_count"] >= 3
    # Cooldown: an immediate second poll stays quiet for that subsystem.
    again = beacon.poll_once()
    assert not [e for e in again if e["subsystem"] == "homeostate_beacon_test_subsystem"]


# ── reactor ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_reactor_converges_on_bus_event(tmp_path):
    from core.event_bus import reset_event_bus

    bus = await reset_event_bus()
    target = tmp_path / "healed"
    engine = HomeostateEngine()
    engine.define("heal", [StateSpec(id="d", fn="file.directory", args={"path": str(target)})])
    reactor = HomeostateReactor(engine)
    reactor.bind("homeostate.beacon.degradation", "heal", cooldown_s=0.0)
    reactor.start()
    try:
        await asyncio.sleep(0.05)                 # listener subscribes
        await bus.publish("homeostate.beacon.degradation", {"subsystem": "x"})
        for _ in range(40):
            if target.is_dir():
                break
            await asyncio.sleep(0.05)
        assert target.is_dir(), "reactor did not converge the bound highstate"
        assert reactor.reactions_fired >= 1
    finally:
        await reactor.stop()


@pytest.mark.asyncio
async def test_reactor_cooldown_suppresses_storms(tmp_path):
    from core.event_bus import reset_event_bus

    bus = await reset_event_bus()
    engine = HomeostateEngine()
    runs: list[float] = []

    def counting(test=False, watch_triggered=False, **_):
        runs.append(1.0)
        return {"result": True, "changes": {}, "comment": ""}

    engine.registry.register("t.count", counting)
    engine.define("hs", [StateSpec(id="c", fn="t.count")])
    reactor = HomeostateReactor(engine)
    reactor.bind("homeostate.beacon.degradation", "hs", cooldown_s=600.0)
    reactor.start()
    try:
        await asyncio.sleep(0.05)
        for _ in range(5):
            await bus.publish("homeostate.beacon.degradation", {"subsystem": "x"})
        await asyncio.sleep(0.5)
        assert len(runs) == 1, "cooldown must collapse an event storm to one convergence"
    finally:
        await reactor.stop()


def test_homeostate_service_names():
    from core.service_names import ServiceNames

    assert ServiceNames.HOMEOSTATE == "homeostate"
    assert ServiceNames.HOMEOSTATE_REACTOR == "homeostate_reactor"
