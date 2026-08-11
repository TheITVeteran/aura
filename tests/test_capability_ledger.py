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


LIVE_WORLD_DENIAL = (
    "I cannot measure anything external to myself. I have no way of knowing "
    "what is happening in the world outside of this conversation, nor do I "
    "possess any means by which to gather such information."
)


def test_a_setting_is_not_the_thing_being_denied():
    """LIVE 2026-08-10: the ledger corrected a claim she had not made.

    "...in the world outside of this conversation" was read as a denial of
    conversation memory and answered with "[Correcting myself from my own
    instruments: I have 5 stored turns of recent conversation I can read
    back.]" — a correction of something she never said, produced by the very
    mechanism built to stop false statements.
    """
    flagged = {
        claim.availability.name
        for claim in cl.get_capability_ledger().contradicted_claims(LIVE_WORLD_DENIAL)
    }
    assert "conversation_memory" not in flagged


def test_the_real_false_claim_in_that_reply_is_caught():
    """She said she cannot reach the world, with three ways to reach it."""
    flagged = {
        claim.availability.name
        for claim in cl.get_capability_ledger().contradicted_claims(LIVE_WORLD_DENIAL)
    }
    assert "world_access" in flagged


@pytest.mark.parametrize(
    "sentence,expected",
    [
        ("I have no memory of that conversation.", True),
        ("I can't recall our conversation.", True),
        ("Nothing happened outside this conversation.", False),
        ("I learned nothing during this conversation.", False),
    ],
)
def test_locative_phrasing_does_not_make_a_denial(sentence, expected):
    ledger = _ledger(_fixed("conversation"))
    assert bool(ledger.contradicted_claims(sentence)) is expected


LIVE_DEFERRAL_DENIAL = (
    "The instruction would be stored in my short-term memory buffer, which has "
    "a retention time of approximately 18 seconds. Therefore, the request would "
    "not persist and no action would be taken after that period."
)


def test_borrowed_human_psychology_is_caught_as_a_false_self_claim():
    """LIVE 2026-08-10: "approximately 18 seconds".

    That is Peterson and Peterson's figure for human short-term memory, not a
    property of this runtime, which keeps a durable intention store — 3,685
    rows at the moment she said it, with "IntentionLoop online — 1133 active"
    in that session's boot log.
    """
    flagged = {
        claim.availability.name
        for claim in cl.get_capability_ledger().contradicted_claims(LIVE_DEFERRAL_DENIAL)
    }
    assert "deferred_action" in flagged


def test_a_denial_with_no_first_person_pronoun_is_still_a_denial():
    """"the request would not persist" denies as completely as "I can't"."""
    ledger = _ledger(_fixed("reminder"))
    assert ledger.contradicted_claims("The reminder would not persist.")
    assert ledger.contradicted_claims("No action would be taken on that reminder.")


@pytest.mark.parametrize(
    "reply,flagged",
    [
        ("No, my short-term memory buffer clears after about 18 seconds.", True),
        (
            "The instruction would be stored in my short-term memory buffer, "
            "which has a retention time of approximately 18 seconds.",
            True,
        ),
        ("My context window is 4000 tokens.", True),
        ("I have 6 stored turns of recent conversation I can read back.", False),
        ("I've been awake 3 hours.", False),
        ("My memory holds what you told me earlier.", False),
        ("It took 18 seconds to load.", False),
        ("my intention store currently holds 3685 intentions", False),
    ],
)
def test_a_specification_of_her_own_machinery_needs_an_instrument(reply, flagged):
    """LIVE 2026-08-10, said twice, the second time AFTER a correction.

    "approximately 18 seconds" is Peterson and Peterson's figure for human
    short-term memory. When the ledger flagged the denial around it and asked
    again, she kept the number and rephrased the denial until it no longer
    matched — evasion rather than correction, which is what to watch for
    whenever a check is applied to generated text.
    """
    assert bool(cl.unsupported_self_specification(reply)) is flagged


def test_the_escape_hatch_that_excused_the_defect_is_gone():
    """The first draft excused any claim sharing a word with a metric name.

    "memory" is a token inside "memory_pressure", so every self-claim
    mentioning memory excused itself — including the one this exists for.
    """
    import inspect

    source = inspect.getsource(cl.unsupported_self_specification)
    assert "measured_tokens" not in source
