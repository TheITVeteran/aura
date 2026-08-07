"""Was that meant for her?

An open microphone is only usable if the answer is right almost always, and
the two errors are not symmetric. Missing a turn costs a repeat, usually with
her name attached, and the user learns nothing bad about the system. Answering
a phone call, a television, or a sentence about her interrupts something real
and teaches the user to switch the feature off.

So these tests are mostly about the second kind of error. The cold-open rungs
are deliberately strict, and every case below that should stay silent has a
plausible-sounding transcript — because the ones that read obviously wrong on
the page are not the ones that ship.
"""
from __future__ import annotations

import pytest

from core.voice.duplex.addressivity import (
    AddressContext,
    AddressivityGate,
)


@pytest.fixture()
def gate() -> AddressivityGate:
    return AddressivityGate(names=("aura",))


def cold() -> AddressContext:
    """Nothing said for a while; she has not spoken."""
    return AddressContext(since_last_reply_s=None, duration_s=2.0)


def just_spoke(seconds: float = 3.0) -> AddressContext:
    return AddressContext(since_last_reply_s=seconds, duration_s=2.0)


# ── her name ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "Aura, what's the weather",
        "hey Aura can you open my notes",
        "Aura",
        "okay Aura",
        "so what do you think, Aura?",
    ],
)
def test_called_by_name_is_answered(gate: AddressivityGate, text: str) -> None:
    verdict = gate.evaluate(text, cold())
    assert verdict.addressed, verdict.narrative()
    assert verdict.rung == "named"


@pytest.mark.parametrize(
    "text",
    [
        "Aura is being weird today",
        "Aura said something strange earlier",
        "Aura keeps interrupting me",
        "Aura was down all morning",
    ],
)
def test_talking_about_her_is_not_talking_to_her(gate: AddressivityGate, text: str) -> None:
    """The classic false accept, and the one users find most uncanny.

    Pattern-matching on the name alone answers all of these, which is exactly
    how an assistant ends up joining a conversation that was about it.
    """
    verdict = gate.evaluate(text, cold())
    assert not verdict.addressed, verdict.narrative()


def test_name_does_not_override_a_third_party_address(gate: AddressivityGate) -> None:
    verdict = gate.evaluate("tell him Aura can do it", cold())
    assert not verdict.addressed, verdict.narrative()


# ── the open floor ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    ["yeah", "no, the other one", "and what about Tuesday", "actually, never mind"],
)
def test_bare_replies_land_while_the_floor_is_open(gate: AddressivityGate, text: str) -> None:
    """The reason dropping the wake word is worth anything.

    None of these mean a thing out of context, and all of them are obviously
    hers seconds after she asked a question. Requiring her name here is what
    makes talking to an assistant feel like operating one.
    """
    verdict = gate.evaluate(text, just_spoke(2.0))
    assert verdict.addressed, verdict.narrative()
    assert verdict.rung == "open_floor"


def test_the_floor_closes(gate: AddressivityGate) -> None:
    """A stale open floor is how a room's background chatter gets answered."""
    verdict = gate.evaluate("yeah", just_spoke(seconds=600.0))
    assert not verdict.addressed, verdict.narrative()


def test_open_floor_still_refuses_a_third_party_aside(gate: AddressivityGate) -> None:
    verdict = gate.evaluate("hold on, I'm on the phone", just_spoke(2.0))
    assert not verdict.addressed, verdict.narrative()


# ── cold opens ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "can you pull up my calendar",
        "what's the fastest way to do this",
        "play something quiet",
        "remind me to call the dentist",
    ],
)
def test_a_real_request_is_answered_cold(gate: AddressivityGate, text: str) -> None:
    """Walk up and talk. This is the case the whole feature exists for."""
    verdict = gate.evaluate(text, cold())
    assert verdict.addressed, verdict.narrative()
    assert verdict.rung == "cold_open"


@pytest.mark.parametrize(
    "text",
    [
        "I was thinking about that thing from yesterday",
        "the traffic was unbelievable",
        "no way, seriously",
        "that's what I told them",
    ],
)
def test_overheard_speech_is_not_answered_cold(gate: AddressivityGate, text: str) -> None:
    """Ordinary talking, in a room with a microphone in it."""
    verdict = gate.evaluate(text, cold())
    assert not verdict.addressed, verdict.narrative()


def test_a_fragment_is_not_a_request(gate: AddressivityGate) -> None:
    verdict = gate.evaluate("what's", cold())
    assert not verdict.addressed, verdict.narrative()


def test_distant_speech_is_not_answered_cold(gate: AddressivityGate) -> None:
    """Compared against this speaker's own baseline, never an absolute level.

    An absolute threshold measures the microphone and the room; only the
    deviation says anything about whether it was aimed here.
    """
    context = cold()
    context.loudness_z = -2.4
    verdict = gate.evaluate("can you pull up my calendar", context)
    assert not verdict.addressed, verdict.narrative()
    assert "quieter" in " ".join(verdict.vetoes)


def test_a_crowded_room_is_not_answered_cold(gate: AddressivityGate) -> None:
    context = cold()
    context.competing_speech = True
    verdict = gate.evaluate("can you pull up my calendar", context)
    assert not verdict.addressed, verdict.narrative()


def test_unknown_confidence_does_not_invent_one(gate: AddressivityGate) -> None:
    """A rung with no reading skips rather than defaulting.

    The recogniser does not report a per-utterance confidence today. Filling
    the field with 1.0 would let a real check pass on a fabricated number,
    which is worse than not running it.
    """
    context = cold()
    assert context.asr_confidence is None
    verdict = gate.evaluate("can you pull up my calendar", context)
    assert verdict.addressed
    assert not any("confidence" in reason for reason in verdict.reasons)


def test_low_confidence_vetoes_when_it_is_known(gate: AddressivityGate) -> None:
    context = cold()
    context.asr_confidence = 0.3
    verdict = gate.evaluate("can you pull up my calendar", context)
    assert not verdict.addressed, verdict.narrative()


# ── the explicit floor ───────────────────────────────────────────────────


def test_opening_the_floor_deliberately_outranks_everything(gate: AddressivityGate) -> None:
    """Push-to-talk and focused voice mode are consent, not evidence.

    If a heuristic could veto an explicit act of address, the control would
    not be a control.
    """
    context = cold()
    context.floor_explicitly_open = True
    verdict = gate.evaluate("the traffic was unbelievable", context)
    assert verdict.addressed
    assert verdict.rung == "explicit"


# ── the verdict explains itself ──────────────────────────────────────────


def test_every_verdict_carries_its_reasons(gate: AddressivityGate) -> None:
    """A decision nobody can interrogate is a decision nobody can fix.

    The only report a user can give is "it answered when it shouldn't have";
    without the reasons attached, that is unactionable.
    """
    for text, context in (
        ("Aura, hello", cold()),
        ("the traffic was unbelievable", cold()),
        ("yeah", just_spoke(2.0)),
        ("hold on, I'm on the phone", just_spoke(2.0)),
    ):
        verdict = gate.evaluate(text, context)
        assert verdict.reasons or verdict.vetoes, f"silent verdict for {text!r}"
        assert verdict.narrative()
        assert verdict.rung


def test_empty_input_is_not_addressed(gate: AddressivityGate) -> None:
    assert not gate.evaluate("", cold()).addressed
    assert not gate.evaluate("   ", cold()).addressed


# ── the floor command ────────────────────────────────────────────────────


def test_the_session_stands_the_gate_down_when_the_floor_is_opened() -> None:
    """Focused voice mode is consent, and consent outranks the heuristic.

    Without this the full-screen surface would still be deciding whether it
    was being spoken to — the one place that judgement is not wanted, because
    the user opened it precisely to talk.
    """
    import asyncio

    from core.voice.duplex.session import DuplexVoiceSession

    async def exercise() -> None:
        async def send_json(_payload):
            return None

        async def send_binary(_payload):
            return None

        session = DuplexVoiceSession(
            session_id="floor", send_json=send_json, send_binary=send_binary
        )
        assert session._ambient_gate is not None
        assert session._floor_explicitly_open is False

        await session.handle_command({"command": "set_floor", "open": True})
        assert session._floor_explicitly_open is True

        # Ordinary overheard speech is answered while the floor is open.
        verdict = session._evaluate_addressivity("the traffic was unbelievable", None)
        assert verdict.addressed
        assert verdict.rung == "explicit"

        await session.handle_command({"command": "set_floor", "open": False})
        assert session._floor_explicitly_open is False
        assert not session._evaluate_addressivity("the traffic was unbelievable", None).addressed

        await session.close()

    asyncio.run(exercise())


def test_a_disabled_gate_cannot_be_switched_back_on_by_a_message() -> None:
    """With ambient gating off, the floor is permanently open.

    A stale or hostile message must not be able to introduce a gate the user
    never asked for and cannot see.
    """
    import asyncio

    from core.voice.duplex.config import DuplexConfig
    from core.voice.duplex.session import DuplexVoiceSession

    async def exercise() -> None:
        async def send_json(_payload):
            return None

        async def send_binary(_payload):
            return None

        config = DuplexConfig()
        config.ambient.enabled = False
        session = DuplexVoiceSession(
            session_id="no-gate",
            send_json=send_json,
            send_binary=send_binary,
            config=config,
        )
        assert session._ambient_gate is None
        assert session._floor_explicitly_open is True

        await session.handle_command({"command": "set_floor", "open": False})
        assert session._floor_explicitly_open is True
        assert session._evaluate_addressivity("anything at all", None).addressed

        await session.close()

    asyncio.run(exercise())
