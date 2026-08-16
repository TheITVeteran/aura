"""CP126 ``core/brain/imagination.py`` — fifteen findings, four critical.

The frame emits novelty, salience, memory pressure, verification pressure,
routing directives, a selected attractor, a recurrent depth, ablation
predictions and a field named ``causal_effects``. Every one is produced by
regex matches, keyword counts and fixed coefficients. The four criticals
are one sentence: the names promise measurement and the code performs
lexical scoring.

Renaming is not available — ``causal_effects`` is read by
``cognitive_engine``, ``task_decomposer`` and ``cognitive_situation_frame``,
and breaking three live readers to fix a wording problem is not a fix. So
the basis travels beside the value, and the basis is load-bearing: nothing
below MEASURED may durably change what gets selected.
"""

from __future__ import annotations

import json

import pytest

from core.brain.imagination import (
    DEFAULT_BASES,
    LEARNING_EVIDENCE_FLOOR,
    ImaginationEngine,
)
from core.brain.imagination_basis import Basis, basis_of, meets


def _engine() -> ImaginationEngine:
    return ImaginationEngine()


# ── f44b0770 / ee92ae29 / 4945e371: the names do not outrun the code ────────


def test_every_emitted_quantity_declares_what_it_rests_on():
    frame = _engine().imagine("draw me a picture and tell me what happens next")
    for field in ("salience", "novelty_pressure", "causal_effects", "attractor_state"):
        assert field in frame.bases, f"{field} is emitted with no recorded basis"


def test_nothing_in_the_frame_claims_to_be_measured():
    """Not one number here comes from a reading, a corpus or a fit model."""
    for field, raw in DEFAULT_BASES.items():
        basis = Basis(raw)
        assert not meets(basis, Basis.MEASURED), (
            f"{field} is declared {basis.value}; nothing in this frame is "
            "measured, and a basis that overstates is worse than none"
        )


def test_the_templates_are_declared_templates():
    """Simulation, novelty and counterfactuals are fixed phrases."""
    for field in (
        "novel_thoughts",
        "simulation_steps",
        "counterfactuals",
        "experiments",
        "mental_canvas",
        "visual_model",
    ):
        assert basis_of(DEFAULT_BASES, field) is Basis.TEMPLATE, (
            f"{field} is assembled from the first few keywords and fixed "
            "phrases; declaring it anything else claims a world model"
        )


def test_the_attractor_says_no_loop_ran():
    """A softmax over authored scores is not a recurrent mechanism."""
    frame = _engine().imagine("what if the answer were different")
    state = frame.attractor_state
    assert state["mechanism"] == "softmax_over_authored_scores"
    assert state["recurrent_depth_executed"] is False, (
        "recurrent_depth is computed and returned; nothing shows it reached "
        "downstream cognition"
    )
    assert state["recurrent_depth"] >= 1  # still emitted, for the callers that use it


def test_an_unlabelled_field_reads_as_the_weakest_basis():
    """Defaulting DOWN is the point: unlabelled means unmeasured."""
    assert basis_of({}, "anything") is Basis.LEXICAL
    assert basis_of(None, "anything") is Basis.LEXICAL
    assert basis_of({"x": "not a basis"}, "x") is Basis.LEXICAL


# ── 04a745b8: a caller's reward does not reshape cognition ──────────────────


def test_an_unevidenced_reward_is_recorded_and_applied_to_nothing():
    engine = _engine()
    frame = engine.imagine("think about tool governance", context={"subject": "bryan"})
    record = engine.learn_from_feedback(
        frame, reward=1.0, outcome="great", subject="bryan"
    )

    assert record is not None
    assert record["applied"] is False
    assert record["updated_bias"] == pytest.approx(0.0)
    assert LEARNING_EVIDENCE_FLOOR.value in record["refusal"]


def test_an_observed_outcome_does_reshape_it():
    """The floor is a floor, not a wall."""
    engine = _engine()
    frame = engine.imagine("think about tool governance", context={"subject": "bryan"})
    record = engine.learn_from_feedback(
        frame,
        reward=1.0,
        outcome="the task completed",
        subject="bryan",
        evidence_basis=Basis.MEASURED.value,
        evidence_id="turn-1",
    )
    assert record["applied"] is True
    assert record["updated_bias"] > 0.0
    assert record["evidence_id"] == "turn-1"


def test_a_fabricated_frame_still_teaches_nothing():
    engine = _engine()
    assert engine.learn_from_feedback(
        {"frame_id": "never-issued", "mode": "creative_synthesis"},
        reward=1.0,
        outcome="great",
        evidence_basis=Basis.MEASURED.value,
    ) is None


# ── f1ef7cfb: one subject's rewards do not steer another's ──────────────────


def test_learning_is_partitioned_by_subject():
    engine = _engine()
    frame = engine.imagine("a question about foxes", context={"subject": "bryan"})
    engine.learn_from_feedback(
        frame,
        reward=1.0,
        outcome="good",
        subject="bryan",
        evidence_basis=Basis.MEASURED.value,
    )

    assert engine.snapshot(subject="bryan")["attractor_bias"], "the subject learned nothing"
    assert engine.snapshot(subject="someone_else")["attractor_bias"] == {}, (
        "one person's rewards changed another person's selection probabilities"
    )


def test_an_unattributed_reward_teaches_the_anonymous_subject_only():
    engine = _engine()
    frame = engine.imagine("a question")
    engine.learn_from_feedback(
        frame, reward=1.0, outcome="ok", evidence_basis=Basis.MEASURED.value
    )
    assert engine.snapshot(subject="anonymous")["attractor_bias"]
    assert engine.snapshot(subject="bryan")["attractor_bias"] == {}


# ── 91ea5bfa: imagining is not evidence about imagining ─────────────────────


def test_imagining_alone_reinforces_nothing():
    """Every frame used to strengthen the association that produced it."""
    engine = _engine()
    for _ in range(20):
        engine.imagine("a creative visual question", context={"subject": "bryan"})

    snapshot = engine.snapshot(subject="bryan")
    assert snapshot["eligibility_trace"] == {}, (
        "twenty invocations built a durable trace with no outcome anywhere — "
        "a loop that closes on itself"
    )
    assert snapshot["attractor_bias"] == {}


def test_the_frame_still_carries_its_candidate_trace():
    """The proposal survives; only its promotion moved."""
    frame = _engine().imagine("a creative visual question")
    assert frame.eligibility_trace, "the frame proposes nothing to reinforce"


# ── 566e64ff / dcc6fd02: a status route is not a place for a scratchpad ─────


def test_the_global_snapshot_does_not_publish_the_latest_private_frame():
    engine = _engine()
    engine.imagine(
        "my recovery phrase is correct horse battery staple, help me remember it",
        context={"subject": "bryan"},
    )
    published = json.dumps(engine.snapshot())
    assert "correct horse battery staple" not in published, (
        "the complete latest frame — objective, memories, canvas, novel "
        "thoughts — was returned to any caller with no authorization"
    )
    assert engine.snapshot()["latest"]["content_withheld"] is True


def test_a_named_subject_can_still_read_its_own_frame():
    engine = _engine()
    engine.imagine("something of mine", context={"subject": "bryan"})
    own = json.dumps(engine.snapshot(subject="bryan", include_content=True))
    assert "something of mine" in own


def test_one_subject_cannot_read_anothers_frame():
    engine = _engine()
    engine.imagine("bryan's private thought", context={"subject": "bryan"})
    other = json.dumps(engine.snapshot(subject="someone_else", include_content=True))
    assert "private thought" not in other


def test_outcomes_are_filtered_to_the_asking_subject():
    engine = _engine()
    frame = engine.imagine("q", context={"subject": "bryan"})
    engine.learn_from_feedback(
        frame,
        reward=1.0,
        outcome="bryan's outcome",
        subject="bryan",
        evidence_basis=Basis.MEASURED.value,
    )
    assert engine.snapshot(subject="bryan")["recent_outcomes"]
    assert engine.snapshot(subject="someone_else")["recent_outcomes"] == []


# ── f58115e3: the queue metrics say they are a model ────────────────────────


def test_the_queue_numbers_do_not_claim_to_measure_a_queue():
    frame = _engine().imagine("anything at all")
    working = frame.working_memory
    assert working["measures_a_real_queue"] is False
    assert working["model"] == "synthetic_load_model"
    # The numbers are still there — they damp admission usefully.
    assert "queue_load" in working and "expected_wait_s" in working


# ── 7975bf24: the monitor decides admission ─────────────────────────────────


def test_the_gate_records_whether_pressure_was_read_or_asserted():
    frame = _engine().imagine("anything")
    assert frame.working_memory["runtime_memory_basis"] in {
        Basis.MEASURED.value,
        Basis.CALLER_ASSERTED.value,
        Basis.LEXICAL.value,
    }


def test_a_caller_percentage_is_clamped():
    engine = _engine()
    reading = engine._runtime_memory_pressure(
        {"memory_pressure": {"level": "high", "pressure_pct": 10_000.0}}
    )
    # With a monitor present this is the monitor's reading; without one it is
    # the caller's, clamped. Either way the number cannot be out of range.
    assert 0.0 <= reading["pressure_pct"] <= 100.0


# ── 3369210a: predictions are labelled hypotheses ───────────────────────────


def test_ablation_predictions_say_they_are_untested():
    frame = _engine().imagine(
        "search the web for a file and imagine what could go wrong"
    )
    assert frame.ablation_predictions
    for name, text in frame.ablation_predictions.items():
        assert text.startswith("UNTESTED HYPOTHESIS:"), (
            f"{name} reads as a behavioural result; no paired run, baseline "
            "or metric tests it"
        )


# ── b95c4d62: a regex is not a governance decision ──────────────────────────


def test_tool_governance_is_labelled_as_an_inference():
    frame = _engine().imagine("search the web and run a shell command for me")
    effects = frame.causal_effects
    assert effects["tool_governance"] is True
    assert effects["tool_governance_is_a_decision"] is False, (
        "a keyword match was presented as a governance decision; Will, scoped "
        "authority, permissions and consent were none of them consulted"
    )
    assert effects["tool_governance_basis"] == Basis.LEXICAL.value


# ── 9d4f7016: a truncated objective says so ─────────────────────────────────


def test_a_cut_objective_is_flagged():
    engine = _engine()
    short = engine.imagine("a short objective")
    assert short.objective_truncated is False

    long = engine.imagine("x" * 900 + " and never use the network")
    assert long.objective_truncated is True, (
        "a constraint written past the 500-character cut vanished silently"
    )


# ── 4195ed46: the status reports something that can vary ────────────────────


def test_the_status_summarises_what_its_numbers_rest_on():
    summary = _engine().snapshot()["bases"]
    assert summary["fields"] > 0
    assert summary["measured_or_better"] == 0, (
        "the status claims a measured field where none exists"
    )
    assert summary["lexical_or_template"] == summary["fields"]
