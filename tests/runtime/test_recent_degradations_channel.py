"""The channel that explains a live failure must actually return records.

DEFECT, found 2026-08-10 while asking the live desktop "anything feel off?".

``DegradationTracker`` defined ``recent`` twice. Python keeps the last
definition, so the module-level ``recent_degradations`` — which passes
``subsystem_prefixes`` — raised ``TypeError`` on every call it had ever made.
Its callers wrap it in broad excepts, so a total failure of the explanation
channel presented to users as "nothing to report".

That is why Aura, one minute after boot and while carrying a MARGINAL fault,
an open incident and a repeating governance refusal, answered "My substrate is
stable. My drives are aligned."
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clean_tracker():
    from core.runtime.errors import get_degradation_tracker

    get_degradation_tracker().reset()
    yield
    get_degradation_tracker().reset()


def _record(subsystem: str, message: str, severity: str = "degraded") -> None:
    from core.runtime.errors import record_degradation

    try:
        raise RuntimeError(message)
    except RuntimeError as exc:
        record_degradation(
            subsystem,
            exc,
            severity=severity,
            action="test record",
            enforce_failure_policy=False,
        )


def test_tracker_defines_recent_exactly_once() -> None:
    """A second definition would silently shadow the first again."""
    import inspect

    from core.runtime import errors

    source = inspect.getsource(errors.DegradationTracker)
    definitions = [line for line in source.splitlines() if line.strip().startswith("def recent(")]

    assert len(definitions) == 1, definitions


def test_recent_degradations_returns_records() -> None:
    from core.runtime.errors import recent_degradations

    _record("latent_cortex", "receipt_contract_failed:decode_bridge_unapplied")

    records = recent_degradations(limit=6)

    assert len(records) == 1
    assert records[0]["subsystem"] == "latent_cortex"
    assert "decode_bridge_unapplied" in records[0]["error"]
    assert records[0]["action"] == "test record"


def test_recent_degradations_accepts_prefixes() -> None:
    """The argument whose absence made every call raise."""
    from core.runtime.errors import recent_degradations

    _record("latent_cortex", "one")
    _record("autonomous_task_engine", "two")

    filtered = recent_degradations(limit=6, subsystem_prefixes=("latent_",))

    assert [r["subsystem"] for r in filtered] == ["latent_cortex"]


def test_recent_degradations_is_newest_last() -> None:
    from core.runtime.errors import recent_degradations

    _record("first_subsystem", "older")
    _record("second_subsystem", "newer")

    records = recent_degradations(limit=6)

    assert [r["subsystem"] for r in records] == [
        "first_subsystem",
        "second_subsystem",
    ]


def test_limit_is_honoured_with_and_without_prefixes() -> None:
    from core.runtime.errors import recent_degradations

    for index in range(6):
        _record("latent_cortex", f"event-{index}")

    assert len(recent_degradations(limit=2)) == 2
    assert len(recent_degradations(limit=2, subsystem_prefixes=("latent_",))) == 2


def test_tracker_recent_still_returns_objects_for_its_callers() -> None:
    """health_contract, proof_kernel_bridge and certify_boot read attributes."""
    from core.runtime.errors import get_degradation_tracker

    _record("latent_cortex", "object-shape")

    records = get_degradation_tracker().recent(limit=5)

    assert records
    assert records[0].subsystem == "latent_cortex"
    assert records[0].severity


def test_voice_self_report_surfaces_a_live_degradation() -> None:
    """The end of the wire: what the identity contract now hands the voice."""
    from core.conversation.chat_preflight import _live_health_summary

    _record("latent_cortex", "receipt_contract_failed:decode_bridge_unapplied")

    lines = _live_health_summary()
    blob = " ".join(lines)

    assert "latent_cortex" in blob
    assert "decode_bridge_unapplied" in blob
    # And it must forbid the answer that started this investigation.
    assert "never describe" in blob


def test_clean_runtime_makes_no_degradation_claim() -> None:
    """No invented all-clear either: absent evidence yields no line."""
    from core.conversation.chat_preflight import _live_health_summary

    blob = " ".join(_live_health_summary())

    assert "Degradations recorded recently" not in blob


def test_recovered_degradation_is_not_asserted_to_be_current() -> None:
    from core.conversation.chat_preflight import _live_health_summary
    from core.resilience.incident_manager import get_incident_manager

    subsystem = "recovered_voice_probe"
    _record(subsystem, "temporary failure")
    assert get_incident_manager().resolve(f"degradation:{subsystem}") is not None

    blob = " ".join(_live_health_summary())

    assert subsystem not in blob
    assert "These active incidents are mine and current" not in blob


def test_stale_warning_is_not_presented_as_current(monkeypatch) -> None:
    from core.conversation import chat_preflight

    observed_at = 10_000.0
    monkeypatch.setattr(chat_preflight.time, "time", lambda: observed_at)
    active, unconfirmed = chat_preflight._current_degradation_records(
        [
            {
                "subsystem": "old_warning",
                "severity": "warning",
                "error": "old",
                "at": observed_at - chat_preflight._DEGRADATION_CURRENT_WINDOW_S - 1.0,
            }
        ],
        observed_at=observed_at,
    )

    assert active == []
    assert unconfirmed == []


def test_old_but_active_incident_remains_current(monkeypatch) -> None:
    from core.conversation import chat_preflight

    monkeypatch.setattr(
        chat_preflight,
        "_active_degradation_categories",
        lambda: {"degradation:still_broken"},
    )
    active, unconfirmed = chat_preflight._current_degradation_records(
        [
            {
                "subsystem": "still_broken",
                "severity": "degraded",
                "error": "persistent",
                "at": 1.0,
            }
        ],
        observed_at=10_000.0,
    )

    assert [record["subsystem"] for record in active] == ["still_broken"]
    assert unconfirmed == []
