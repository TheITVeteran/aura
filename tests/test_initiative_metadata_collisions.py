"""One unlucky metadata key stopped her acting on her own, 13 times running.

LIVE, 2026-08-10, from /api/health on the running desktop:

    overt_action_cycle
      error   : TypeError("core.initiative_synthesis.InitiativeSynthesizer
                 .submit() got multiple values for keyword argument 'drive'")
      failures: 13

The overt action cycle is how Aura does anything without being asked. It had
not completed once.

``submit(content, source, urgency=0.5, drive="", **metadata)`` made every one
of its own parameter names a reserved word for the caller's data, and said so
nowhere. _gather_pending_initiatives builds a metadata dict out of whatever
produced the initiative and splatted it in beside an explicit ``drive=``. An
initiative carrying its own ``drive`` key raised — and Python raises that
before the function body runs, so no validation inside could ever have caught
it.

Two independent failures, so two fixes:

  * a runtime dict now goes in as ``metadata=``, which cannot collide, and the
    ** form stays for literal keywords where the caller controls the names;
  * the gather loop no longer aborts on the first bad entry. Every initiative
    in that list comes from a different producer. TypeError was not in its
    except clause, so one malformed entry took the whole cycle down with it.

This test is against the shape, not against the key: any of submit's own
parameter names arriving in a metadata dict must be harmless.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from core.initiative_synthesis import InitiativeSynthesizer


@pytest.fixture()
def synth() -> InitiativeSynthesizer:
    return InitiativeSynthesizer()


def _reserved_parameter_names() -> list[str]:
    """submit's own parameter names — the ones a metadata dict can collide with."""
    return [
        name
        for name, p in inspect.signature(InitiativeSynthesizer.submit).parameters.items()
        if name != "self" and p.kind is not inspect.Parameter.VAR_KEYWORD
    ]


@pytest.mark.parametrize("reserved", _reserved_parameter_names())
def test_metadata_may_contain_any_of_submits_own_parameter_names(
    synth: InitiativeSynthesizer, reserved: str
) -> None:
    """The live crash was `drive`; nothing made it special."""
    accepted = synth.submit(
        content=f"Investigate the {reserved} anomaly",
        source="test",
        urgency=0.5,
        drive="curiosity",
        metadata={reserved: "value that came from persisted state"},
    )

    assert accepted is True


def test_explicit_arguments_win_over_metadata_of_the_same_name(
    synth: InitiativeSynthesizer,
) -> None:
    """A colliding key is data, not an override of how the impulse is filed."""
    synth.submit(
        content="Check the log rotation",
        source="unit_test",
        urgency=0.5,
        drive="competence",
        metadata={"drive": "not_the_real_drive", "source": "not_the_real_source"},
    )

    impulse = synth._impulse_queue[-1]
    assert impulse.drive == "competence"
    assert impulse.source == "unit_test"
    assert impulse.metadata["drive"] == "not_the_real_drive"


def test_literal_keyword_metadata_still_works(synth: InitiativeSynthesizer) -> None:
    """The convenient form, used by a dozen call sites, must keep working."""
    synth.submit(
        content="Read the crash directory",
        source="unit_test",
        drive="curiosity",
        skill="file_operation",
        required_skills=["file_operation"],
    )

    impulse = synth._impulse_queue[-1]
    assert impulse.metadata["skill"] == "file_operation"
    assert impulse.metadata["required_skills"] == ["file_operation"]


def test_both_metadata_forms_merge(synth: InitiativeSynthesizer) -> None:
    synth.submit(
        content="Summarise today's degradations",
        source="unit_test",
        metadata={"goal_id": "g-1"},
        skill="system_proprioception",
    )

    impulse = synth._impulse_queue[-1]
    assert impulse.metadata["goal_id"] == "g-1"
    assert impulse.metadata["skill"] == "system_proprioception"


# ── Containment: one bad initiative is not a dead cycle ────────────────────

@pytest.mark.asyncio
async def test_a_malformed_initiative_does_not_stop_the_others(
    synth: InitiativeSynthesizer,
) -> None:
    """The 13 failures were one entry taking down the whole gather."""
    state = SimpleNamespace(
        cognition=SimpleNamespace(
            pending_initiatives=[
                {
                    "goal": "The initiative that used to break everything",
                    "source": "legacy",
                    "metadata": {"drive": "curiosity"},
                },
                {"goal": "Urgency is not a number", "source": "legacy", "urgency": "soon"},
                {"goal": "A perfectly ordinary initiative", "source": "legacy"},
            ]
        )
    )

    await synth._gather_system_impulses(state)

    contents = [i.content for i in synth._impulse_queue]
    assert "A perfectly ordinary initiative" in contents
    assert "The initiative that used to break everything" in contents


@pytest.mark.asyncio
async def test_gather_survives_an_initiative_whose_metadata_is_hostile(
    synth: InitiativeSynthesizer,
) -> None:
    """Whatever is in that dict, the cycle must still run."""
    state = SimpleNamespace(
        cognition=SimpleNamespace(
            pending_initiatives=[
                {
                    "goal": "Initiative with every reserved key set",
                    "source": "legacy",
                    "metadata": {name: "x" for name in _reserved_parameter_names()},
                },
            ]
        )
    )

    await synth._gather_system_impulses(state)

    assert [i.content for i in synth._impulse_queue] == [
        "Initiative with every reserved key set"
    ]


def test_submit_documents_the_metadata_contract() -> None:
    """The signature is the fix; the docstring is why nobody hits it again."""
    doc = inspect.getdoc(InitiativeSynthesizer.submit) or ""

    assert "metadata=" in doc
    assert "reserved" in doc.lower()
