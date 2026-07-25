"""Contract tests for the LLVM-derived disciplines.

core/verify/{invariants,runtime_invariants}.py,
core/pipeline/pass_manager.py, core/runtime/sanitizers.py.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from core.pipeline.pass_manager import (
    AnalysisManager,
    PassManager,
    PreservedAnalyses,
    bisect_pipeline,
    get_instrumentation,
    reset_pass_manager_for_test,
)
from core.runtime import sanitizers as san_mod
from core.runtime import taint as taint_mod
from core.runtime.sanitizers import (
    PoisonPool,
    SequenceChecker,
    UseAfterReleaseError,
    check_finite,
    sanitize_finite,
    sanitizer_report,
)
from core.verify.invariants import (
    InvariantRegistry,
    Severity,
    Violation,
    get_registry,
    invariant,
    verify,
)


@pytest.fixture(autouse=True)
def _clean():
    reset_pass_manager_for_test()
    san_mod.reset_sanitizers_for_test()
    taint_mod.reset_taint_for_test()
    yield
    reset_pass_manager_for_test()
    san_mod.reset_sanitizers_for_test()
    taint_mod.reset_taint_for_test()


# ── verifier ──────────────────────────────────────────────────────────

def _isolated_registry(monkeypatch) -> InvariantRegistry:
    registry = InvariantRegistry()
    monkeypatch.setattr("core.verify.invariants._REGISTRY", registry)
    return registry


def test_verifier_fills_invariant_name_and_severity_from_the_spec(monkeypatch):
    _isolated_registry(monkeypatch)

    @invariant("demo.always_broken", scope="demo", owner="test")
    def _broken():
        yield Violation(subject="thing", message="it is broken")

    report = verify("demo", record=False)
    assert report.ok is False
    violation = report.violations[0]
    assert violation.invariant == "demo.always_broken"
    assert violation.severity is Severity.ERROR


def test_a_check_that_raises_is_itself_a_violation(monkeypatch):
    _isolated_registry(monkeypatch)

    @invariant("demo.explodes", scope="demo", owner="test")
    def _explodes():
        raise ValueError("probe unavailable")
        yield  # pragma: no cover

    report = verify("demo", record=False)
    assert report.ok is False
    assert "the invariant check itself failed" in report.violations[0].message
    assert "demo.explodes" in report.skipped


def test_warnings_do_not_make_a_report_fail(monkeypatch):
    _isolated_registry(monkeypatch)

    @invariant("demo.nit", scope="demo", severity=Severity.WARNING, owner="test")
    def _nit():
        yield Violation(subject="x", message="minor")

    report = verify("demo", record=False)
    assert report.ok is True
    assert len(report.warnings) == 1


def test_scopes_isolate_verification_cost(monkeypatch):
    _isolated_registry(monkeypatch)
    ran: list[str] = []

    @invariant("a.one", scope="a", owner="test")
    def _a():
        ran.append("a")
        return ()

    @invariant("b.one", scope="b", owner="test")
    def _b():
        ran.append("b")
        return ()

    verify("a", record=False)
    assert ran == ["a"]


def test_duplicate_invariant_name_is_refused(monkeypatch):
    _isolated_registry(monkeypatch)

    @invariant("dup.name", scope="d", owner="test")
    def _one():
        return ()

    with pytest.raises(ValueError, match="already registered"):

        @invariant("dup.name", scope="d", owner="test")
        def _two():
            return ()


def test_runtime_invariants_are_registered_and_run_clean():
    from core.verify import runtime_invariants  # noqa: F401 — import registers

    specs = {s.name for s in get_registry().specs()}
    for expected in (
        "container.alias_resolves",
        "container.dependency_graph_acyclic",
        "locks.no_open_splats",
        "oom.spine_is_immune",
        "integrity.untainted",
    ):
        assert expected in specs

    # Over a clean process these must not report ERRORs; a verifier that
    # cries wolf at rest is one nobody reads.
    report = verify("container", "locks", "memory", record=False)
    assert report.ok, report.summary()


def test_container_alias_invariant_catches_a_dangling_alias(monkeypatch):
    from core.verify import runtime_invariants  # noqa: F401

    monkeypatch.setattr(
        "core.verify.runtime_invariants._container_state",
        lambda: ({"real": object()}, {"legacy": "gone"}),
    )
    report = verify("container", record=False)
    assert not report.ok
    assert any("legacy" in v.subject for v in report.errors)


def test_container_dependency_cycle_is_caught(monkeypatch):
    from core.verify import runtime_invariants  # noqa: F401

    class _D:
        def __init__(self, deps):
            self.dependencies = deps

    monkeypatch.setattr(
        "core.verify.runtime_invariants._container_state",
        lambda: ({"a": _D(["b"]), "b": _D(["a"])}, {}),
    )
    report = verify("container", record=False)
    cycles = [v for v in report.errors if v.invariant == "container.dependency_graph_acyclic"]
    assert cycles, report.summary()


# ── pass manager ──────────────────────────────────────────────────────

def test_preserved_analyses_semantics():
    assert PreservedAnalyses.all().preserved("anything") is True
    assert PreservedAnalyses.none().preserved("anything") is False
    assert PreservedAnalyses.none().preserve("affect").preserved("affect") is True
    # abandon beats all() — an explicit invalidation is never overridden.
    assert PreservedAnalyses.all().abandon("affect").preserved("affect") is False
    # Two passes preserve only what both preserved.
    combined = PreservedAnalyses.all().intersect(PreservedAnalyses.none().preserve("evidence"))
    assert combined.preserved("evidence") is True
    assert combined.preserved("affect") is False


def test_analysis_manager_caches_and_invalidates_precisely():
    am: AnalysisManager[dict] = AnalysisManager()
    computed = {"count": 0}

    def salience(unit):
        computed["count"] += 1
        return unit["value"] * 2

    am.register_fn("salience", salience)
    unit = {"value": 3}

    assert am.get("salience", unit) == 6
    assert am.get("salience", unit) == 6
    assert computed["count"] == 1, "second get must be a cache hit"

    am.invalidate(PreservedAnalyses.all())
    assert am.cached("salience") is True

    dropped = am.invalidate(PreservedAnalyses.none())
    assert dropped == ["salience"]
    am.get("salience", unit)
    assert computed["count"] == 2


def test_analysis_manager_refuses_unknown_analysis():
    am: AnalysisManager[dict] = AnalysisManager()
    with pytest.raises(KeyError, match="no analysis named"):
        am.get("nope", {})


def test_pass_manager_runs_in_order_and_invalidates():
    order: list[str] = []
    am: AnalysisManager[dict] = AnalysisManager()
    am.register_fn("evidence", lambda u: list(u.get("docs", ())))

    pm: PassManager[dict] = PassManager("test")
    pm.add_fn("retrieve", lambda u, a: (order.append("retrieve"), PreservedAnalyses.none())[1])
    pm.add_fn(
        "rank",
        lambda u, a: (order.append("rank"), a.get("evidence", u), PreservedAnalyses.all())[2],
    )
    pm.add_fn("respond", lambda u, a: (order.append("respond"), PreservedAnalyses.all())[1])

    unit = {"docs": ["a", "b"]}
    preserved = pm.run(unit, am)
    assert order == ["retrieve", "rank", "respond"]
    # The pipeline as a whole preserves nothing, because `retrieve` did not.
    assert preserved.preserved("evidence") is False


def test_opt_bisect_skips_passes_past_the_limit():
    ran: list[str] = []
    pm: PassManager[dict] = PassManager("bisect")
    for name in ("one", "two", "three", "four"):
        pm.add_fn(name, lambda u, a, n=name: (ran.append(n), PreservedAnalyses.all())[1])

    get_instrumentation().set_bisect_limit(2)
    pm.run({})
    assert ran == ["one", "two"]

    report = get_instrumentation().report()
    assert report["skips"] == 2
    skipped = [r for r in report["recent"] if r["skipped"]]
    assert skipped and "opt-bisect" in skipped[0]["reason"]


def test_bisect_pipeline_finds_the_offending_pass():
    state: dict[str, int] = {}

    def run():
        state.clear()
        pm: PassManager[dict] = PassManager("cog")
        pm.add_fn("good_one", lambda u, a: PreservedAnalyses.all())
        pm.add_fn("good_two", lambda u, a: PreservedAnalyses.all())
        pm.add_fn(
            "the_culprit",
            lambda u, a: (state.__setitem__("bad", 1), PreservedAnalyses.all())[1],
        )
        pm.add_fn("good_three", lambda u, a: PreservedAnalyses.all())
        pm.run({})
        return dict(state)

    result = bisect_pipeline(run, lambda out: "bad" not in out, max_ordinal=4)
    assert result["found"] is True
    assert result["pass"] == "the_culprit"


def test_bisect_restores_the_previous_limit():
    get_instrumentation().set_bisect_limit(7)
    bisect_pipeline(lambda: {}, lambda _: True, max_ordinal=3)
    assert get_instrumentation().bisect_limit() == 7


def test_before_hook_can_skip_a_pass():
    ran: list[str] = []
    get_instrumentation().add_before_hook(lambda name, ordinal: "skipme" not in name)
    pm: PassManager[dict] = PassManager("hooked")
    pm.add_fn("keepme", lambda u, a: (ran.append("keepme"), PreservedAnalyses.all())[1])
    pm.add_fn("skipme", lambda u, a: (ran.append("skipme"), PreservedAnalyses.all())[1])
    pm.run({})
    assert ran == ["keepme"]


def test_pass_failure_is_recorded_before_it_propagates():
    pm: PassManager[dict] = PassManager("boom")
    pm.add_fn("explodes", lambda u, a: (_ for _ in ()).throw(RuntimeError("nope")))
    with pytest.raises(RuntimeError):
        pm.run({})
    records = get_instrumentation().records()
    assert records[-1].error.startswith("RuntimeError")


def test_kernel_tick_loop_consults_the_instrumentation():
    """The live phase loop must actually go through the seam."""
    import inspect

    from core.kernel import aura_kernel

    source = inspect.getsource(aura_kernel.AuraKernel.tick)
    assert "_pass_instrumentation()" in source
    assert "_record_pass(" in source


# ── sanitizers ────────────────────────────────────────────────────────

def test_poison_pool_reports_use_after_release():
    pool: PoisonPool[dict] = PoisonPool("buffers", lambda: {"data": None})
    obj = pool.acquire("turn_buffer")
    obj["data"] = "live"
    stale = pool.release(obj)

    assert stale.anything is None  # the read is answered, and reported
    report = sanitizer_report()
    assert report["clean"] is False
    assert any("use-after-release" in f["message"] for f in report["findings"])
    assert taint_mod.is_tainted(taint_mod.TaintFlag.SANITIZER)


def test_poison_strict_mode_raises():
    pool: PoisonPool[dict] = PoisonPool("strict", lambda: {}, strict=True)
    stale = pool.release(pool.acquire())
    with pytest.raises(UseAfterReleaseError):
        _ = stale.field


def test_poison_pool_detects_double_release():
    pool: PoisonPool[dict] = PoisonPool("dbl", lambda: {})
    obj = pool.acquire()
    pool.release(obj)
    pool.release(obj)
    assert pool.double_releases == 1
    assert any("double release" in f["message"] for f in sanitizer_report()["findings"])


def test_poison_pool_borrow_releases_on_exception():
    pool: PoisonPool[dict] = PoisonPool("borrow", lambda: {})
    with pytest.raises(ValueError):
        with pool.borrow():
            raise ValueError("boom")
    assert pool.releases == 1


def test_check_finite_catches_nan_in_nested_vectors():
    assert check_finite("affect", [0.1, 0.2, 0.3]) is True
    assert check_finite("affect", [[0.1, float("nan")], [0.3]]) is False
    findings = sanitizer_report()["findings"]
    assert any("NaN" in f["message"] for f in findings)


def test_check_finite_catches_infinity_and_can_raise():
    assert check_finite("reward", float("inf")) is False
    with pytest.raises(ValueError, match="non-finite"):
        check_finite("reward", [float("-inf")], strict=True)


def test_check_finite_ignores_strings_and_bools():
    assert check_finite("label", "hello") is True
    assert check_finite("flag", [True, False]) is True


def test_sanitize_finite_replaces_poison():
    cleaned = sanitize_finite("steering", [1.0, float("nan"), 3.0])
    assert cleaned == [1.0, 0.0, 3.0]


def test_sequence_checker_reports_cross_thread_access():
    checker = SequenceChecker("turn_buffer")
    assert checker.check("first") is True

    result: list[bool] = []
    thread = threading.Thread(target=lambda: result.append(checker.check("from_worker")))
    thread.start()
    thread.join(timeout=5)

    assert result == [False]
    assert any("sequence violation" in f["message"] for f in sanitizer_report()["findings"])


def test_sequence_checker_distinguishes_tasks():
    async def scenario():
        checker = SequenceChecker("lane_state")

        async def touch(label):
            return checker.check(label)

        first = await asyncio.create_task(touch("task_a"))
        second = await asyncio.create_task(touch("task_b"))
        return first, second

    first, second = asyncio.run(scenario())
    assert first is True
    assert second is False, "two tasks are two sequences"


def test_sanitizer_findings_deduplicate_but_count():
    pool: PoisonPool[dict] = PoisonPool("dedupe", lambda: {})
    for _ in range(4):
        stale = pool.release(pool.acquire("same_label"))
        _ = stale.field
    report = sanitizer_report()
    assert report["distinct_findings"] == 1
    assert report["total_occurrences"] == 4


def test_health_integrity_block_carries_sanitizer_and_verifier_state():
    from core.runtime.health_contract import _runtime_integrity_block

    block = _runtime_integrity_block()
    assert "sanitizers" in block
    assert "passes" in block
    assert "verifier" in block
