"""The reason classification must not contradict itself.

Theseus's ``Objective`` refuses to assemble a problem whose declarations
conflict — a variable that is both an optimization variable and an auxiliary
one, a cost weight that depends on an optimization variable ("the jacobians
computed by our optimizers will be incorrect"). It rejects at assembly instead
of returning a subtly wrong answer at solve time.

``surface_disposition``'s reason sets are the same kind of declaration, read by
~115 call sites through three different doors: ``assessment.ok``,
``disposition_for``, and raw ``.reasons``. Twice those doors have disagreed and
a reply was served by one gate and destroyed by another. Nothing checked.

Now four invariants do, and these tests prove each one both HOLDS on the real
declarations and FIRES when the declaration is broken — a check that cannot
fail is not a check.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import core.conversation.disposition_invariants  # noqa: F401 — registers them
from core.verify.invariants import verify

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SCOPE = "conversation"
_EXPECTED = {
    "surface.advisory_never_destroys",
    "surface.advisory_keeps_the_exchange",
    "surface.reason_names_are_comparable",
    "surface.disposition_agrees_with_the_sets",
}


def _report():
    return verify(SCOPE, record=False)


def _fired(report) -> set[str]:
    return {v.invariant for v in report.violations}


class TestTheyAreDeclared:
    def test_all_four_are_registered(self):
        from core.verify.invariants import get_registry

        names = {spec.name for spec in get_registry().specs((SCOPE,))}
        assert _EXPECTED <= names

    def test_surface_disposition_registers_them_on_import(self):
        """A check nothing imports is a check nothing runs.

        In a fresh interpreter, so the answer is about the import graph rather
        than about what some earlier test already pulled in. Deliberately NOT
        an ``importlib.reload``: reloading this module mints a second
        ``SurfaceDisposition`` enum, and every later ``is`` comparison in the
        process then fails against an identical-looking member.
        """
        import subprocess
        import sys

        probe = (
            "import core.conversation.surface_disposition\n"
            "from core.verify.invariants import get_registry\n"
            "names = {s.name for s in get_registry().specs(('conversation',))}\n"
            "print(sorted(names))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0, result.stderr[-2000:]
        registered = set(eval(result.stdout.strip()))  # noqa: S307 — our own literal
        assert _EXPECTED <= registered, (
            f"importing surface_disposition registered {registered}"
        )


class TestTheyHoldToday:
    def test_the_live_declaration_is_consistent(self):
        report = _report()
        assert report.ok, [v.to_dict() for v in report.violations]

    def test_no_check_was_skipped(self):
        """A check that raised counts as a violation, not as a pass."""
        report = _report()
        assert not report.skipped, report.skipped


class TestEachOneCanFire:
    """Break the declaration; the matching invariant must object."""

    @pytest.fixture()
    def sets(self, monkeypatch):
        import core.conversation.surface_disposition as module

        return module, monkeypatch

    def test_an_advisory_that_also_destroys_is_caught(self, sets):
        module, monkeypatch = sets
        monkeypatch.setattr(
            module,
            "ADVISORY_ONLY_REASONS",
            frozenset({"reply_abandons_thread", "empty_reply"}),
        )
        assert "surface.advisory_never_destroys" in _fired(_report())

    def test_an_advisory_that_can_erase_the_turn_is_caught(self, sets):
        module, monkeypatch = sets
        monkeypatch.setattr(
            module,
            "ADVISORY_ONLY_REASONS",
            frozenset({"reply_abandons_thread", "some_new_soft_heuristic"}),
        )
        fired = _fired(_report())
        assert "surface.advisory_keeps_the_exchange" in fired

    def test_an_uncomparable_reason_name_is_caught(self, sets):
        module, monkeypatch = sets
        monkeypatch.setattr(
            module,
            "ADVISORY_ONLY_REASONS",
            frozenset({"reply_abandons_thread", "Reply Abandons Thread"}),
        )
        assert "surface.reason_names_are_comparable" in _fired(_report())

    def test_a_disposition_that_ignores_the_sets_is_caught(self, sets):
        """The drift that actually happened: disposition_for not knowing.

        Before the fix, `disposition_for` returned REPAIR for every advisory
        reason, and at the conversation-learning gate REPAIR meant "do not
        remember this exchange".
        """
        module, monkeypatch = sets

        def _blind(reasons):
            found = module._reason_set(reasons)
            if not found:
                return module.SurfaceDisposition.SERVE
            if found & module.UNSPEAKABLE_REASONS:
                return module.SurfaceDisposition.DISCARD
            return module.SurfaceDisposition.REPAIR  # the old, set-blind rule

        monkeypatch.setattr(module, "disposition_for", _blind)
        assert "surface.disposition_agrees_with_the_sets" in _fired(_report())
