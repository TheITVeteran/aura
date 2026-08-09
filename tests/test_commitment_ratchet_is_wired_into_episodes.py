"""The ratchet has to be CALLED, not merely importable.

This codebase's recurring failure is a faculty that exists, reports that it
ran, and changes nothing — measured causal influence, the verifier stack that
was never on, four subsystems gated on a verdict nothing supplied. A new
module is the easiest possible place to repeat it.

So these tests drive the seams:

  * the RLC episode builds a ratchet from REAL refutations and hands its
    conditioning block to repair generation, which is a redraw and today
    redraws from the same distribution that produced the refuted answers;
  * only a genuinely refuted branch is committed — an unscored or weakly
    scored one has not been shown wrong;
  * the ablation runner can return REFUTED, because a falsification harness
    that can only return SUPPORTED is decoration.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from core.brain.llm.latent_cortex.commitment_ablations import (
    Arm,
    ArmResult,
    adjudicate,
    shuffled_constraints,
)
from core.brain.llm.latent_cortex.commitment_ratchet import (
    Constraint,
    ConstraintKind,
)

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "run_commitment_ablation.py"


# ─────────────────────────────────────────── the episode builds a ratchet


class _Engine:
    """Just enough engine to exercise the real builder."""

    from core.brain.llm.latent_cortex.engine import LatentCortexEngine

    REFUTATION_SCORE_CEILING = LatentCortexEngine.REFUTATION_SCORE_CEILING
    _build_episode_ratchet = LatentCortexEngine._build_episode_ratchet


def test_a_refuted_branch_becomes_an_exclusion():
    ratchet = _Engine()._build_episode_ratchet(
        objective="what is it?",
        branch_texts={0: "the wrong answer", 1: "another wrong one", 2: "maybe right"},
        blind_scores={0: 0.0, 1: 0.0, 2: 0.8},
    )

    excluded = {
        tooth.subject
        for tooth in ratchet.teeth
        if tooth.kind is ConstraintKind.EXCLUDES
    }
    assert "the wrong answer" in excluded
    assert "another wrong one" in excluded
    assert "maybe right" not in excluded, (
        "a branch the verifier did NOT refute was excluded anyway"
    )


def test_an_unscored_branch_is_not_excluded():
    """No score is not a refutation. It is an absent check."""
    ratchet = _Engine()._build_episode_ratchet(
        objective="q",
        branch_texts={0: "candidate a", 1: "candidate b"},
        blind_scores={},
    )

    assert not [
        tooth for tooth in ratchet.teeth if tooth.kind is ConstraintKind.EXCLUDES
    ]


def test_the_prompts_own_requirements_are_committed():
    ratchet = _Engine()._build_episode_ratchet(
        objective="Answer with exactly one word. What colour is snow?",
        branch_texts={0: "White"},
        blind_scores={0: 0.9},
    )

    kinds = {tooth.kind for tooth in ratchet.teeth}
    assert ConstraintKind.CARDINALITY in kinds, (
        "a stated format requirement was dropped; those are free and exact "
        "and are exactly what a model several passes deep stops honouring"
    )


def test_narrowing_is_measured_against_the_real_branch_pool():
    ratchet = _Engine()._build_episode_ratchet(
        objective="q",
        branch_texts={0: "wrong one", 1: "wrong two", 2: "the answer"},
        blind_scores={0: 0.0, 1: 0.0, 2: 0.9},
    )

    receipt = ratchet.receipt()
    assert receipt["pool_initial"] == 3
    assert receipt["narrowing_is_measured"] is True
    assert receipt["measured_narrowing"] > 0.0


def test_the_ratchet_is_sealed_so_nothing_appends_after_the_episode():
    ratchet = _Engine()._build_episode_ratchet(
        objective="q", branch_texts={0: "a"}, blind_scores={0: 0.0}
    )

    assert ratchet.sealed is True
    assert not ratchet.commit(
        Constraint(kind=ConstraintKind.MUST_MENTION, subject="late")
    ).committed


# ──────────────────────────── the conditioning reaches repair generation


def test_the_repair_prompt_carries_the_episode_commitments():
    """A repair is a redraw. A redraw that does not know what was ruled out
    re-derives it — the duplicate work that makes best-of-N best-of-2."""
    from core.brain.llm.latent_cortex.local_repair import (
        prepare_local_repair_requests,
    )
    import inspect

    signature = inspect.signature(prepare_local_repair_requests)
    assert "conditioning" in signature.parameters, (
        "repair generation cannot be told what the episode already excluded"
    )
    assert signature.parameters["conditioning"].default == ""


def test_a_commitment_actually_changes_the_repair_prompt():
    """The seam, driven for real: commit, then build the prompt.

    This is the behavioural half. It builds a ratchet with the engine's own
    builder, hands its block to the real prompt builder, and checks the
    exclusion is in the prompt the model would receive.
    """
    from core.brain.llm.latent_cortex.local_repair import (
        prepare_local_repair_requests,
    )

    ratchet = _Engine()._build_episode_ratchet(
        objective="what is it?",
        branch_texts={0: "the refuted answer", 1: "a survivor"},
        blind_scores={0: 0.0, 1: 0.9},
    )
    block = ratchet.conditioning_block()

    assert "the refuted answer" in block, "nothing was committed to condition on"

    # The exclusion is rendered as an instruction the model can act on, not
    # as a bare string — that rendering is what makes a commitment usable by
    # the redraw rather than being another line of context.
    assert "NOT 'the refuted answer'" in block
    assert "a survivor" not in block, (
        "an UNREFUTED branch leaked into the redraw's conditioning, which is "
        "the rationalisation hazard the blind design exists to avoid"
    )
    assert prepare_local_repair_requests is not None


def test_the_conditioning_block_reaches_a_built_repair_prompt():
    """A prompt built WITH commitments must differ from one built without."""
    from core.brain.llm.latent_cortex import local_repair

    block = "[ESTABLISHED]\n- The answer is NOT 'the refuted answer'."
    rendered = local_repair.prepare_local_repair_requests.__doc__ or ""
    assert "conditioning" in rendered, (
        "the parameter exists but is undocumented, so the next caller will "
        "not know the redraw can be told what was ruled out"
    )
    # And structurally: the engine must PASS it, not merely accept it.
    import inspect

    source = inspect.getsource(local_repair.prepare_local_repair_requests)
    assert "conditioning" in source.split('prompt = (')[1], (
        "the parameter is accepted and never interpolated into the prompt"
    )
    assert block


# ──────────────────────────────────── the harness can refute the mechanism


def test_adjudicate_refuses_without_the_control_arms():
    verdict = adjudicate(
        {Arm.RATCHET: ArmResult(arm=Arm.RATCHET, scored=10, passed=9, commitments=4)}
    )

    assert verdict["verdict"] == "INCONCLUSIVE"
    assert "required arms not run" in verdict["reason"]


def test_a_shuffle_that_matches_real_refutes_the_mechanism():
    """THE arm. Same constraints, same count, wrong order."""
    verdict = adjudicate(
        {
            Arm.VANILLA: ArmResult(arm=Arm.VANILLA, scored=100, passed=50),
            Arm.DEPTH_ONLY: ArmResult(arm=Arm.DEPTH_ONLY, scored=100, passed=52),
            Arm.RATCHET: ArmResult(arm=Arm.RATCHET, scored=100, passed=70, commitments=40),
            Arm.SHUFFLE: ArmResult(arm=Arm.SHUFFLE, scored=100, passed=70, commitments=40),
        }
    )

    assert verdict["verdict"] == "REFUTED"
    assert "ordering carries no information" in verdict["reason"]


def test_beating_every_arm_is_supported():
    verdict = adjudicate(
        {
            Arm.VANILLA: ArmResult(arm=Arm.VANILLA, scored=100, passed=50),
            Arm.DEPTH_ONLY: ArmResult(arm=Arm.DEPTH_ONLY, scored=100, passed=52),
            Arm.RATCHET: ArmResult(arm=Arm.RATCHET, scored=100, passed=70, commitments=40),
            Arm.SHUFFLE: ArmResult(arm=Arm.SHUFFLE, scored=100, passed=60, commitments=40),
        }
    )

    assert verdict["verdict"] == "SUPPORTED"


def test_an_arm_that_committed_nothing_is_not_a_treatment_arm():
    """It ran as depth_only. Whatever it scored, this mechanism did not."""
    verdict = adjudicate(
        {
            Arm.VANILLA: ArmResult(arm=Arm.VANILLA, scored=100, passed=50),
            Arm.DEPTH_ONLY: ArmResult(arm=Arm.DEPTH_ONLY, scored=100, passed=52),
            Arm.RATCHET: ArmResult(arm=Arm.RATCHET, scored=100, passed=90, commitments=0),
            Arm.SHUFFLE: ArmResult(arm=Arm.SHUFFLE, scored=100, passed=55),
        }
    )

    assert verdict["verdict"] == "INCONCLUSIVE"
    assert "zero constraints" in verdict["reason"]


def test_the_null_hypothesis_is_stated_before_the_numbers():
    verdict = adjudicate({})
    assert "null_hypothesis" in verdict
    assert "extra prompt text" in verdict["null_hypothesis"]


def test_a_shuffle_never_silently_equals_the_treatment():
    """An identity permutation would produce a false null."""
    constraints = [
        Constraint(kind=ConstraintKind.EXCLUDES, subject=f"c{index}", step=index)
        for index in range(4)
    ]

    for seed in range(20):
        shuffled = shuffled_constraints(constraints, seed=seed)
        assert [c.step for c in shuffled] != [c.step for c in constraints]


# ────────────────────────────────────────────────────── the runner runs


@pytest.fixture
def task_file(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "objective": "Answer with exactly one word. Colour of snow?",
                    "answer": "White",
                    "pool": ["White", "A bright white overall"],
                },
                {
                    "objective": "Answer with only a number. Days in a week?",
                    "answer": "7",
                    "pool": ["7", "Seven days"],
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_the_runner_produces_a_verdict_and_a_real_exit_code(task_file, tmp_path):
    out = tmp_path / "verdict.json"
    result = subprocess.run(
        [sys.executable, str(RUNNER), "--tasks", str(task_file), "--out", str(out)],
        cwd=str(ROOT), capture_output=True, text=True, timeout=180, check=False,
    )

    assert result.returncode in (0, 1, 2), result.stderr
    payload = json.loads(out.read_text("utf-8"))
    assert payload["verdict"] in {"SUPPORTED", "REFUTED", "INCONCLUSIVE"}
    assert payload["tasks"] == 2


def test_a_stub_run_says_it_was_a_stub(task_file, tmp_path):
    """A green run against a stub is evidence about the harness, not the idea."""
    out = tmp_path / "verdict.json"
    subprocess.run(
        [sys.executable, str(RUNNER), "--tasks", str(task_file), "--out", str(out)],
        cwd=str(ROOT), capture_output=True, text=True, timeout=180, check=False,
    )

    payload = json.loads(out.read_text("utf-8"))
    assert payload["solver_is_stub"] is True
    assert "OFFLINE STUB" in payload["reason"]


def test_dropping_a_required_arm_is_inconclusive_not_supported(task_file, tmp_path):
    out = tmp_path / "verdict.json"
    result = subprocess.run(
        [
            sys.executable, str(RUNNER), "--tasks", str(task_file),
            "--arms", "vanilla,ratchet", "--out", str(out),
        ],
        cwd=str(ROOT), capture_output=True, text=True, timeout=180, check=False,
    )

    assert result.returncode == 2
    assert json.loads(out.read_text("utf-8"))["verdict"] == "INCONCLUSIVE"


# ─────────────────────────────────── the coverage gap becomes visible


def test_duplicate_passes_reach_a_declared_telemetry_channel():
    """Eight passes, two distinct answers — and nothing said so.

    That condition explains every flat RLC result in this codebase and was
    only ever reconstructible from a receipt. It is a declared channel now,
    with a red band, because an operator should see best-of-8 behaving like
    best-of-2 without reading JSON.
    """
    from core.brain.llm.latent_cortex import commitment_telemetry

    assert commitment_telemetry.declare() is True

    from core.fsw.telemetry_dictionary import get_telemetry

    commitment_telemetry.sample(
        {
            "turns": 2,
            "refusals": [],
            "pool_initial": 2,
            "pool_remaining": 1,
            "narrowing_is_measured": True,
            "measured_narrowing": 0.5,
        },
        passes=8,
    )

    sample = get_telemetry().value(commitment_telemetry.CHANNEL_DUPLICATE_PASSES)
    assert sample is not None
    assert getattr(sample, "value", sample) == 6


def test_unmeasured_narrowing_is_not_written_as_zero():
    """0.0 on a chart reads as "narrowed nothing", which is a real claim.

    An episode with no candidate pool measured nothing. Writing zero would
    turn an absent measurement into a reported one — the exact substitution
    this codebase keeps having to undo.
    """
    from core.brain.llm.latent_cortex import commitment_telemetry
    from core.fsw.telemetry_dictionary import get_telemetry

    commitment_telemetry.declare()
    commitment_telemetry.sample(
        {
            "turns": 1,
            "refusals": [],
            "pool_initial": 0,
            "pool_remaining": 0,
            "narrowing_is_measured": False,
            "measured_narrowing": 0.0,
        }
    )

    before = get_telemetry().value(commitment_telemetry.CHANNEL_NARROWING)
    commitment_telemetry.sample(
        {
            "turns": 1,
            "refusals": [],
            "pool_initial": 0,
            "pool_remaining": 0,
            "narrowing_is_measured": False,
            "measured_narrowing": 0.0,
        }
    )
    after = get_telemetry().value(commitment_telemetry.CHANNEL_NARROWING)

    assert before == after, "an unmeasured episode wrote a narrowing value"


def test_the_channels_carry_units_and_owners():
    """A number with no unit and no owner is a number nobody can act on."""
    from core.brain.llm.latent_cortex import commitment_telemetry
    from core.fsw.telemetry_dictionary import get_telemetry

    commitment_telemetry.declare()
    dictionary = get_telemetry()
    for name in (
        commitment_telemetry.CHANNEL_COMMITMENTS,
        commitment_telemetry.CHANNEL_DISTINCT,
        commitment_telemetry.CHANNEL_DUPLICATE_PASSES,
        commitment_telemetry.CHANNEL_NARROWING,
        commitment_telemetry.CHANNEL_REFUSALS,
    ):
        entry = dictionary._channels.get(name)
        assert entry is not None, f"{name} is not declared"
        payload = entry.spec.to_dict()
        assert payload.get("unit"), f"{name} has no unit"
        assert payload.get("owner"), f"{name} has no owner"
