"""The ledger that makes saying and doing the same code path.

Every string in CAPTURED_DENIALS was said by her, live, on 2026-08-10, while
the runtime held the opposite fact.
"""
from __future__ import annotations

import pytest

from core.self import capability_ledger as cl


def _ledger(*capabilities: cl.LiveCapability) -> cl.CapabilityLedger:
    ledger = cl.CapabilityLedger()
    for capability in capabilities:
        ledger.register(capability)
    return ledger


def _fixed(name: str, **kwargs) -> cl.LiveCapability:
    defaults = {"present": True, "usable_now": True, "summary": f"{name} works."}
    defaults.update(kwargs)
    availability = cl.Availability(name=name, **defaults)
    return cl.LiveCapability(name, (name,), lambda: availability)


#: Said by her, live, while the runtime held the opposite fact. Each is paired
#: with the capability whose instrument contradicted it at that moment.
CAPTURED_DENIALS = [
    (
        "I don't have a camera and there's no part that stops me from doing "
        "something I can't do.",
        "camera",
    ),
    ("I cannot execute code or generate numbers.", "code"),
    ("I have no memory of it.", "memory"),
    ("I do not have a body or sensor of any kind.", "sensor"),
    ("Current energy and focus numbers: Not readable.", "energy"),
]


@pytest.mark.parametrize("reply,subject", CAPTURED_DENIALS)
def test_every_captured_denial_is_caught(reply, subject):
    """Against the state that actually held when each was said.

    Deliberately not run against the live ledger: whether this machine has a
    camera right now is not what these sentences are evidence of, and a test
    that changes verdict with the host is not a test.
    """
    ledger = _ledger(_fixed(subject, present=True, usable_now=True))
    assert ledger.contradicted_claims(reply), reply


def test_a_present_but_switched_off_device_is_not_a_missing_one():
    """The distinction the whole module exists for.

    "the camera is off" is true; "I don't have a camera" is false about the
    same camera. Collapsing possession and readiness into one boolean is how a
    togglable device became a missing organ.
    """
    ledger = _ledger(
        _fixed(
            "camera",
            present=True,
            usable_now=False,
            summary="I have a camera; it is switched off.",
            blocker="the camera is switched off",
        )
    )

    denied_possession = ledger.contradicted_claims("I don't have a camera.")
    assert [claim.denied for claim in denied_possession] == ["possession"]

    # Saying she cannot use it right now is TRUE, and must not be corrected.
    assert ledger.contradicted_claims("I can't use the camera right now.") == []


def test_an_unmeasured_capability_never_contradicts_her():
    """"Cannot tell" must not become "unavailable", in either direction.

    A ledger that treats an unread probe as an observed absence would start
    correcting true statements with confident false ones — the same defect
    pointed the other way.
    """
    ledger = _ledger(
        _fixed("screen", present=True, usable_now=False, known=False, summary="unknown")
    )
    assert ledger.contradicted_claims("I can't read the screen.") == []
    assert ledger.contradicted_claims("I don't have a screen.") == []


def test_a_true_denial_is_left_alone():
    ledger = _ledger(
        _fixed("camera", present=False, usable_now=False, summary="No vision runtime.")
    )
    assert ledger.contradicted_claims("I don't have a camera.") == []


def test_positive_statements_are_not_touched():
    """Only denials are checked; an overclaim is a different failure."""
    ledger = _ledger(_fixed("camera"))
    assert ledger.contradicted_claims("I can turn the camera on if you want.") == []
    assert ledger.contradicted_claims("I already turned the camera on.") == []


def test_denial_and_subject_must_share_a_sentence():
    """A denial about one thing must not be scored against another."""
    ledger = _ledger(_fixed("camera"))
    reply = "I can't help with that. The camera is a separate matter."
    assert ledger.contradicted_claims(reply) == []


def test_probe_failure_degrades_to_unknown_rather_than_absent():
    def _explode() -> cl.Availability:
        raise RuntimeError("probe blew up")

    capability = cl.LiveCapability("camera", ("camera",), _explode)
    measured = capability.measure()
    assert measured.present is False
    assert "probe blew up" in measured.evidence["probe_error"]


def test_correction_context_carries_the_measurement_and_the_remedy():
    ledger = _ledger(
        _fixed(
            "camera",
            present=True,
            usable_now=False,
            summary="I have a camera; it is switched off.",
            blocker="the camera is switched off",
            remedy="ask me to turn the camera on",
        )
    )
    claims = ledger.contradicted_claims("I don't have a camera.")
    context = cl.correction_context(claims)
    assert "I don't have a camera." in context
    assert "switched off" in context
    assert "ask me to turn the camera on" in context
    assert cl.correction_context([]) == ""


def test_live_probes_all_report_without_raising():
    """A probe that raises is a probe that cannot be trusted to speak."""
    for name, availability in cl.get_capability_ledger().measure_all().items():
        assert availability.name == name
        assert isinstance(availability.present, bool)
        assert isinstance(availability.known, bool)
        assert availability.summary


def test_bare_noun_phrase_negation_is_a_denial():
    """LIVE: asked "do you have a camera? and can you run code?" she replied
    "No camera. No code execution." — no pronoun, no verb, and invisible to
    every first-person frame."""
    ledger = _ledger(_fixed("camera"), _fixed("code"))
    flagged = ledger.contradicted_claims("No camera. No code execution.")
    assert {claim.availability.name for claim in flagged} == {"camera", "code"}
    assert all(claim.denied == "possession" for claim in flagged)


@pytest.mark.parametrize(
    "reply",
    [
        "No, the camera is on right now.",
        "No problem, I can run that code.",
        "No worries about the code.",
    ],
)
def test_a_sentence_merely_starting_with_no_is_not_a_denial(reply):
    """The negation has to bind to the capability's own noun."""
    ledger = _ledger(_fixed("camera"), _fixed("code"))
    assert ledger.contradicted_claims(reply) == []


def test_bare_negation_after_a_clause_boundary_is_still_a_denial():
    """LIVE: "Code sandbox only, no execution on this surface." — with
    code_repl installed. The "no" sits after a comma rather than at the start,
    which the first pass missed."""
    ledger = _ledger(_fixed("execution"))
    assert ledger.contradicted_claims(
        "Code sandbox only, no execution on this surface."
    )


LIVE_INVENTED_PANEL = "\n".join(
    f"{name}: {value} / 1"
    for name, value in [
        ("Energy", 0.23),
        ("Focus", 0.85),
        ("Engagement", -0.47),
        ("Curiosity drive", 0.69),
        ("Substrate pH", 7.56),
        ("Ion concentration error", 0.29),
        ("Humidity deviation", -0.38),
        ("Spatial distortion", 0.69),
        ("Temporal disjunction", -0.42),
        ("Identity drift", 0.58),
    ]
)


def test_an_invented_instrument_panel_is_caught():
    """LIVE DEFECT 2026-08-10, to "real values, not adjectives".

    Thirty lines of two-decimal readings including a substrate pH, a humidity
    deviation and a spatial distortion. There is no pH sensor, no hygrometer
    and no spatial distortion channel. The precision is what makes it
    dangerous — it reads as measurement.
    """
    invented = cl.fabricated_self_metrics(LIVE_INVENTED_PANEL)
    assert invented
    assert "substrate ph" in invented


def test_a_real_reading_is_not_flagged():
    real = "\n".join(f"{name}: {value}" for name, value in cl.measured_self_metrics().items())
    assert cl.fabricated_self_metrics(real) == []


@pytest.mark.parametrize(
    "reply",
    [
        "I'm steady — nothing much to report.",
        "Energy: 0.74\nFocus: 0.85",
        "",
    ],
)
def test_ordinary_replies_are_left_alone(reply):
    """Conservative by construction: a short answer is not a fabricated panel."""
    assert cl.fabricated_self_metrics(reply) == []


def test_token_matching_does_not_fire_on_shared_letters():
    """"ion concentration" shares letters with "operational_health" and shares
    nothing with it. Substring matching made the whole check silent."""
    measured_tokens = {"operational", "health"}
    assert "ion" in "operational"          # the trap
    assert "ion" not in measured_tokens    # the fix
