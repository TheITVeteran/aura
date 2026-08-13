"""The assumption ledger must not be able to lie about itself.

The ledger's whole value is that a reader can trust the status column. Two ways
that fails: an assumption claiming a checker that does not exist, and an
assumption that is not discharged and does not say what is missing. Both are
the same underlying defect this repository keeps finding — the absence of a
check reported as a passed check — so both are refused rather than warned about.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.verify import system_assumptions  # noqa: F401  (import populates registry)
from core.verify.assumptions import (
    Assumption,
    AssumptionRegistry,
    AssumptionStatus,
    assumption_report,
    get_assumption_registry,
    verify_dischargers,
)

ROOT = Path(__file__).resolve().parent.parent


def test_every_discharged_assumption_names_a_checker_that_exists():
    """The gate. This caught a bad reference the hour the ledger was written."""
    failures = verify_dischargers(root=ROOT)
    assert not failures, "\n".join(str(f) for f in failures)


def test_the_ledger_is_not_empty():
    """An empty ledger would pass every check and mean nothing."""
    registry = get_assumption_registry()
    assert len(registry) >= 8
    assert registry.by_status(AssumptionStatus.OUTSIDE_THE_SYSTEM), (
        "a system with no assumptions outside itself has not looked underneath it"
    )


def test_a_discharged_assumption_must_name_its_checker():
    with pytest.raises(ValueError, match="names no checker"):
        Assumption(
            id="x",
            statement="s",
            breaks="b",
            status=AssumptionStatus.DISCHARGED,
            owner="o",
        )


@pytest.mark.parametrize(
    "status", [AssumptionStatus.UNDISCHARGED, AssumptionStatus.OUTSIDE_THE_SYSTEM]
)
def test_an_undischarged_assumption_must_say_what_is_missing(status):
    with pytest.raises(ValueError, match="says nothing about what is missing"):
        Assumption(id="x", statement="s", breaks="b", status=status, owner="o")


def test_a_missing_test_file_is_caught():
    registry = AssumptionRegistry()
    registry.register(
        Assumption(
            id="ghost",
            statement="s",
            breaks="b",
            status=AssumptionStatus.DISCHARGED,
            owner="o",
            discharged_by="tests/test_this_does_not_exist.py::test_nope",
        )
    )
    failures = verify_dischargers(registry.all(), root=ROOT)
    assert len(failures) == 1
    assert "no such file" in failures[0].reason


def test_a_renamed_test_function_is_caught():
    """The realistic rot: the file survives, the function is renamed away."""
    registry = AssumptionRegistry()
    registry.register(
        Assumption(
            id="renamed",
            statement="s",
            breaks="b",
            status=AssumptionStatus.DISCHARGED,
            owner="o",
            discharged_by="tests/test_assumption_ledger.py::test_that_was_renamed",
        )
    )
    failures = verify_dischargers(registry.all(), root=ROOT)
    assert len(failures) == 1
    assert "defines no" in failures[0].reason


def test_a_missing_make_target_is_caught():
    registry = AssumptionRegistry()
    registry.register(
        Assumption(
            id="nomake",
            statement="s",
            breaks="b",
            status=AssumptionStatus.DISCHARGED,
            owner="o",
            discharged_by="make target-that-does-not-exist",
        )
    )
    failures = verify_dischargers(registry.all(), root=ROOT)
    assert len(failures) == 1
    assert "no target" in failures[0].reason


def test_reusing_an_id_with_a_different_body_is_refused():
    """Ids are a contract; silently rebinding one rewrites history."""
    registry = AssumptionRegistry()
    first = Assumption(
        id="dup",
        statement="one",
        breaks="b",
        status=AssumptionStatus.UNDISCHARGED,
        owner="o",
        note="n",
    )
    registry.register(first)
    registry.register(first)  # idempotent re-registration is fine
    with pytest.raises(ValueError, match="already registered"):
        registry.register(
            Assumption(
                id="dup",
                statement="something else entirely",
                breaks="b",
                status=AssumptionStatus.UNDISCHARGED,
                owner="o",
                note="n",
            )
        )


def test_the_report_is_serialisable_and_counts_add_up():
    report = assumption_report()
    total = sum(report["counts"].values())
    assert total == report["total"]
    assert report["discharge_failures"] == []


def test_the_known_platform_exposure_is_recorded():
    """macOS fsync() does not force the drive cache, and the code calls it.

    Pinned because it is the clearest case of the ledger doing its job: a real
    durability gap that is stated rather than silently carried, with the reason
    the obvious repair is not automatically the right one.
    """
    registry = get_assumption_registry()
    ids = {a.id for a in registry.all()}
    assert "host.fsync_reaches_stable_storage" in ids
    entry = next(a for a in registry.all() if a.id == "host.fsync_reaches_stable_storage")
    assert entry.status is AssumptionStatus.OUTSIDE_THE_SYSTEM
    assert "F_FULLFSYNC" in entry.note
