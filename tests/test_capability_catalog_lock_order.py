"""The skill catalog's two locks must be taken in one order, always.

On 2026-08-03 the live desktop could not finish booting. Three stall dumps
sixty seconds apart showed the same pair of parked threads:

  * the boot loop, in ``_ensure_catalog_loaded`` holding ``catalog_load``,
    waiting for ``catalog`` inside ``_reload_skills_transaction``;
  * a health-probe worker, in ``is_ready`` holding ``catalog``, waiting for
    ``catalog_load`` behind the lazy ``skills`` property.

Neither could proceed. ``/health`` timed out, the desktop sat on "RESUMING
LIVE SURFACE", the reaper killed the kernel, the launcher restarted it, and
the race ran again — a boot loop, not a slow boot.

The invariant these tests pin: nothing reads the lazily-loading ``skills`` or
``instances`` properties while holding the catalog guard.
"""
import re
import threading
from pathlib import Path

import pytest

from core.capability_engine import CapabilityEngine

CATALOG_BUILD_BUDGET_S = 60.0


@pytest.fixture
def engine():
    return CapabilityEngine()


class TestConcurrentFirstLoadAndReadinessProbe:
    """The exact two-thread race from the boot dumps."""

    def test_readiness_probe_during_first_catalog_load_does_not_deadlock(self, engine):
        entered_load = threading.Event()
        release_load = threading.Event()
        finished: dict[str, object] = {}

        original_transaction = engine._reload_skills_transaction

        def paced_transaction() -> None:
            # We are inside _ensure_catalog_loaded, holding catalog_load.
            entered_load.set()
            # Hold it open until the probe thread is committed to its own path.
            release_load.wait(CATALOG_BUILD_BUDGET_S)
            original_transaction()

        engine._reload_skills_transaction = paced_transaction

        def loader() -> None:
            finished["skills"] = len(engine.skills)

        def probe() -> None:
            finished["ready"] = engine.is_ready()

        # Daemons on purpose: when this regresses the threads park forever, and
        # a non-daemon thread would hang interpreter shutdown instead of
        # failing the test. A deadlock must be reported, not re-enacted.
        loader_thread = threading.Thread(target=loader, name="catalog-loader", daemon=True)
        loader_thread.start()
        assert entered_load.wait(CATALOG_BUILD_BUDGET_S), "loader never reached the transaction"

        probe_thread = threading.Thread(target=probe, name="health-probe", daemon=True)
        probe_thread.start()
        # Give the probe long enough to reach whichever lock it takes first.
        probe_thread.join(1.0)

        release_load.set()

        loader_thread.join(CATALOG_BUILD_BUDGET_S)
        probe_thread.join(CATALOG_BUILD_BUDGET_S)

        assert not loader_thread.is_alive(), (
            "the catalog loader is still parked — the boot-side half of the deadlock"
        )
        assert not probe_thread.is_alive(), (
            "the readiness probe is still parked — the health-side half of the deadlock"
        )
        assert finished["skills"], "catalog loaded but registered no skills"
        assert finished["ready"] is True, "a fully loaded catalog must report ready"

    def test_probe_first_then_loader_does_not_deadlock(self, engine):
        """The same race with the threads started in the opposite order."""

        finished: dict[str, object] = {}
        barrier = threading.Barrier(2, timeout=CATALOG_BUILD_BUDGET_S)

        def probe() -> None:
            barrier.wait()
            finished["ready"] = engine.is_ready()

        def loader() -> None:
            barrier.wait()
            finished["skills"] = len(engine.skills)

        threads = [
            threading.Thread(target=probe, name="health-probe", daemon=True),
            threading.Thread(target=loader, name="catalog-loader", daemon=True),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(CATALOG_BUILD_BUDGET_S)

        alive = [thread.name for thread in threads if thread.is_alive()]
        assert not alive, f"threads still parked on the catalog locks: {alive}"
        assert finished["skills"], "catalog loaded but registered no skills"


class TestLoadUnderTheGuardIsRefusedNotHung:
    def test_ensure_catalog_loaded_refuses_to_invert_the_order(self, engine):
        """Holding the guard and asking for a load must return, not block."""

        with engine._catalog_guard():
            # Before the fix this either recursed or waited for a lock whose
            # owner was waiting for this thread. It must now decline and serve
            # whatever generation is published.
            engine._ensure_catalog_loaded()
            assert engine._skills == {}, (
                "the inverted load path must not populate the catalog behind the guard"
            )

        # The refusal must not have poisoned the engine: a clean load still works.
        assert len(engine.skills) > 0


class TestRegistrationSurvivesTheLazyLoad:
    """Removing the inversion made a latent defect reachable.

    register_skill wrote into a catalog that had not loaded yet, so the first
    later read rebuilt _skills from discovery and the registration vanished.
    It used to work by accident: _refresh_active_skills read the lazy `skills`
    property from inside the guard, which loaded the catalog as a side effect
    of the very lock inversion that deadlocked boot.
    """

    def test_a_registered_skill_is_still_there_after_the_catalog_loads(self, engine):
        from core.skills.base_skill import BaseSkill

        class RuntimeSkill(BaseSkill):
            name = "lock_order_probe_skill"
            description = "Registered at runtime, then read back after a lazy load."
            effect_scope = "pure_compute"

            async def execute(self, params, context=None):
                return {"ok": True}

        skill = RuntimeSkill()
        assert engine.register_skill(skill) is True
        # The property is the first thing that triggers the catalog load.
        assert "lock_order_probe_skill" in engine.skills
        assert engine.skills["lock_order_probe_skill"].instance is skill
        # And the discovered catalog is present too — registration adds, not replaces.
        assert len(engine.skills) > 1


class TestNoInvertedReadsInSource:
    """A source ratchet, so the next edit cannot reintroduce the class."""

    LAZY_PROPERTY = re.compile(r"self\.(skills|instances)\b")

    def test_no_guarded_block_reads_a_lazily_loading_property(self):
        source = Path("core/capability_engine.py").read_text(encoding="utf-8").split("\n")
        guard_starts = [
            index
            for index, line in enumerate(source)
            if "_catalog_guard()" in line and line.lstrip().startswith("with")
        ]
        assert guard_starts, "the catalog guard disappeared — update this ratchet"

        offenders: list[str] = []
        for start in guard_starts:
            indent = len(source[start]) - len(source[start].lstrip())
            for index in range(start + 1, len(source)):
                line = source[index]
                if line.strip() and (len(line) - len(line.lstrip())) <= indent:
                    break
                if self.LAZY_PROPERTY.search(line):
                    offenders.append(f"{index + 1}: {line.strip()}")

        assert not offenders, (
            "these reads take catalog_load while holding catalog — the boot deadlock:\n"
            + "\n".join(offenders)
        )
