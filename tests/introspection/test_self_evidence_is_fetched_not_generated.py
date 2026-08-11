"""Her own telemetry must reach the reply, and absence must be typed.

LIVE, 2026-08-10. Two failures on the same runtime, minutes apart, with one
cause between them.

Asked "which of your subsystems is degraded or failing right now? ... If a job
of yours has been failing repeatedly, I want the name and the count", she said:

    "I couldn't get to an answer I'd stand behind on that one, and I won't send
    you a thinner one and pass it off as the real thing."

/api/health at that moment: integrity=degraded, CRSM manifest stale, and
overt_action_cycle with failures=13 carrying its exact TypeError. The answer was
structured, live, and hers.

Asked what was on the screen — a sense health reports as granted, bridged and
directly probed — she said "a web browser interface with multiple tabs", then
"no applications running in the foreground", then "nothing displayed except
generic desktop wallpaper". Three claims that cannot all hold. Nothing had
handed her a reading, and nothing had told her that either.

So: evidence that exists does not reach the reply, and an absent reading is
indistinguishable from an unremarkable one. Generation fills the space, and
agrees with whatever the question implied — confident where there was nothing,
refusing where there was plenty.

The fix is a fetch, not a phrasing. resolve_self_health() calls the real
sources; every channel comes back as a value with provenance or as one of four
distinct absences; and the answer is BUILT from those values, so it cannot
describe a state she is not in.
"""

from __future__ import annotations

import pytest

from core.introspection.self_evidence import (
    EvidenceBundle,
    Reading,
    ReadingState,
    asks_about_own_operational_state,
    render_self_health_answer,
    resolve_self_health,
    self_health_answer,
)


# ── The demand predicate: narrow on purpose ────────────────────────────────

@pytest.mark.parametrize(
    "message",
    [
        "Which of your subsystems is degraded or failing right now?",
        "If a job of yours has been failing repeatedly, I want the name and the count.",
        "how are your internals holding up?",
        "is your substrate healthy",
        "what is the status of your runtime",
        "are any of your loops stuck?",
    ],
)
def test_questions_about_her_own_state_are_recognised(message: str) -> None:
    assert asks_about_own_operational_state(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "my deploy is failing again",
        "my server is degraded",
        "how are the kids doing",
        "what is the capital of Peru",
        "the build is broken",
        "",
    ],
)
def test_questions_about_other_things_are_not(message: str) -> None:
    """A false positive answers a question nobody asked with telemetry."""
    assert asks_about_own_operational_state(message) is False


# ── Readings are fetched, and absences are distinguishable ─────────────────

def test_resolution_reads_the_real_sources() -> None:
    bundle = resolve_self_health()

    assert bundle.demand == "self_health"
    channels = {r.channel for r in bundle.readings}
    assert {"runtime_health", "failing_jobs", "degradations"} <= channels
    for reading in bundle.readings:
        assert reading.provenance, f"{reading.channel} claims no source"


def test_the_four_absences_are_not_the_same_fact() -> None:
    """Collapsing these is what produced both live failures."""
    states = {
        ReadingState.READ,
        ReadingState.ABSENT_NEVER_SAMPLED,
        ReadingState.ABSENT_UNAVAILABLE,
        ReadingState.ABSENT_NOT_INSTRUMENTED,
    }
    assert len({str(s) for s in states}) == 4

    never = Reading(channel="camera", state=ReadingState.ABSENT_NEVER_SAMPLED)
    missing = Reading(channel="latency", state=ReadingState.ABSENT_NOT_INSTRUMENTED)
    assert never.present is False
    assert missing.present is False
    assert never.state is not missing.state


def test_an_ungrounded_bundle_cannot_produce_an_answer() -> None:
    """No reading, no text. This is what stops reassurance being manufactured."""
    bundle = EvidenceBundle(
        demand="self_health",
        readings=(
            Reading(
                channel="runtime_health",
                state=ReadingState.ABSENT_UNAVAILABLE,
                provenance="runtime_health_report()",
                detail="RuntimeError: no container",
            ),
        ),
    )

    assert bundle.grounded is False
    rendered = render_self_health_answer(bundle)
    assert "not readable" in rendered
    # And it names which channel failed rather than implying health.
    assert "runtime_health" in rendered
    for word in ("stable", "nominal", "healthy", "fine"):
        assert word not in rendered.lower()


# ── The answer is a function of the values ─────────────────────────────────

def test_a_repeatedly_failing_job_is_named_with_its_count_and_error() -> None:
    """The exact question she refused: the name and the count."""
    bundle = EvidenceBundle(
        demand="self_health",
        readings=(
            Reading(
                channel="runtime_health",
                state=ReadingState.READ,
                value="degraded",
                provenance="runtime_health_report().status",
            ),
            Reading(
                channel="failing_jobs",
                state=ReadingState.READ,
                unit="jobs",
                provenance="runtime_health_report()",
                value=[{
                    "job": "overt_action_cycle",
                    "failures": 13,
                    "error": "TypeError(\"submit() got multiple values for keyword argument 'drive'\")",
                }],
            ),
        ),
    )

    rendered = render_self_health_answer(bundle)

    assert "degraded" in rendered
    assert "overt_action_cycle" in rendered
    assert "13" in rendered
    assert "drive" in rendered


def test_failing_jobs_are_extracted_from_the_real_health_shape() -> None:
    """Pinned to the structure /api/health actually serves."""
    from core.introspection.self_evidence import _failing_jobs

    report = {
        "full_runtime": {"components": {"autonomy_conductor": {"jobs": {
            "overt_action_cycle": {
                "failures": 13,
                "last_result": {"error": "TypeError: drive"},
            },
            "reasoning_self_improve": {"failures": 0, "last_result": {}},
            "architecture_auto_cycle": {"failures": 2, "last_result": {"error": "x"}},
        }}}}
    }

    rows = _failing_jobs(report)

    assert [r["job"] for r in rows] == ["overt_action_cycle", "architecture_auto_cycle"]
    assert rows[0]["failures"] == 13
    assert "drive" in rows[0]["error"]


def test_a_malformed_health_report_yields_no_rows_rather_than_raising() -> None:
    from core.introspection.self_evidence import _failing_jobs

    assert _failing_jobs({}) == []
    assert _failing_jobs({"full_runtime": {"components": {"autonomy_conductor": {"jobs": []}}}}) == []


# ── The causal seam: the refusal path consults the readings ────────────────

def test_self_health_answer_is_empty_for_an_unrelated_turn() -> None:
    assert self_health_answer("what is the capital of Peru") == ""


def test_the_refusal_path_asks_whether_the_runtime_holds_the_answer() -> None:
    """Without this the module is a library nobody calls.

    The live refusal was emitted with the answer sitting in runtime_health_report().
    """
    import inspect

    from interface.routes import chat

    source = inspect.getsource(chat)
    assert source.count("_self_health_answer_or_empty(") >= 3  # definition + both sites

    # Anchor on the assignment that BUILDS the refusal, not on the sentence —
    # which also appears in comments describing past defects.
    marker = "failure_reply = (\n"
    sites = _positions(source, marker)
    assert sites, "the refusal is no longer built where this test expects"
    for index in sites:
        window = source[index : index + 1800]
        assert "_self_health_answer_or_empty" in window


def test_helper_returns_empty_rather_than_raising(monkeypatch) -> None:
    """It runs on the path that already failed; it may not make things worse."""
    from interface.routes.chat import _self_health_answer_or_empty

    import core.introspection.self_evidence as module

    def explode(_message):
        raise RuntimeError("resolver is broken")

    monkeypatch.setattr(module, "self_health_answer", explode)

    assert _self_health_answer_or_empty("is your runtime healthy") == ""


def _positions(haystack: str, needle: str) -> list[int]:
    found: list[int] = []
    start = haystack.find(needle)
    while start != -1:
        found.append(start)
        start = haystack.find(needle, start + 1)
    return found


# ── The shared present: a sense that never looked must say so ──────────────

@pytest.mark.parametrize(
    "message",
    [
        "Without me telling you anything: what am I doing right now, and am I alone?",
        "am I alone?",
        "whats playing on my screen right now",
    ],
)
def test_questions_about_the_shared_present_are_recognised(message: str) -> None:
    from core.introspection.self_evidence import asks_about_the_shared_present

    assert asks_about_the_shared_present(message) is True


@pytest.mark.parametrize("message", ["what is the capital of Peru", "explain recursion", ""])
def test_other_questions_do_not_wake_the_senses(message: str) -> None:
    from core.introspection.self_evidence import asks_about_the_shared_present

    assert asks_about_the_shared_present(message) is False


def test_a_never_sampled_sense_is_not_a_negative_reading() -> None:
    """The live answer was "you seem to be alone", then "I cannot determine
    if there are other people present" — one sentence apart."""
    from core.introspection.self_evidence import _signal_reading

    reading = _signal_reading("camera", {"vision": {"updated_at": 0.0, "face_count": 0}}, "vision")

    assert reading.state is ReadingState.ABSENT_NEVER_SAMPLED
    assert reading.present is False


def test_a_sampled_sense_reads_normally() -> None:
    from core.introspection.self_evidence import _signal_reading

    reading = _signal_reading(
        "camera", {"vision": {"updated_at": 1786440000.0, "face_count": 2}}, "vision"
    )

    assert reading.state is ReadingState.READ
    assert reading.value["face_count"] == 2


def test_missing_sense_service_still_names_every_channel() -> None:
    """Omitting them would rebuild the defect one level up."""
    from core.introspection.self_evidence import resolve_shared_present

    bundle = resolve_shared_present()
    channels = {r.channel for r in bundle.readings}

    assert {"camera", "microphone", "typing"} <= channels


def test_the_present_answer_never_asserts_solitude_without_a_camera_reading() -> None:
    from core.introspection.self_evidence import shared_present_answer

    answer = shared_present_answer("what am I doing right now, and am I alone?")

    assert "alone" not in answer.lower().replace("anyone else is here", "")
    assert "never produced a sample" in answer
