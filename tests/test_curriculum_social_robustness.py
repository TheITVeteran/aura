"""Contract tests: minimax curriculum, social-outcome learning, robustness families.

- The curriculum trains the measured gap, never punishes missing evidence,
  and always protects strengths with replay.
- Social rewards come from downstream relational outcomes and are ZEROED
  when won by manipulation.
- Robustness families preserve structure honestly (slots + templates, real
  recomputed answers) and grade BOTH stability and correct movement.
"""
from __future__ import annotations

import pytest

from core.learning.minimax_curriculum import (
    DomainMeasurement,
    MinimaxCurriculumAllocator,
)
from core.learning.robustness_families import (
    RobustTaskSpec,
    generate_family,
    grade_invariance,
)
from core.learning.social_outcome_learning import (
    RelationalOutcome,
    SocialEpisode,
    TheoryOfMindFrame,
    bind_delayed_outcome,
    manipulation_guard,
)

# ── Minimax curriculum ──────────────────────────────────────────────────


def _measure(domain: str, aura: float, ref: float | None = 0.9, n: int = 50):
    return DomainMeasurement(domain=domain, aura_score=aura, reference_score=ref, n=n)


def test_weakest_domain_gets_the_most_training():
    allocator = MinimaxCurriculumAllocator()
    report = allocator.allocate(
        [
            _measure("math", 0.85),
            _measure("social", 0.30),
            _measure("coding", 0.70),
        ],
        budget_items=100,
    )
    counts = report["counts"]
    assert report["weakest_domain"] == "social"
    assert counts["social"] > counts["coding"] > counts["math"]
    assert sum(counts.values()) == 100


def test_replay_floor_protects_mastered_domains():
    allocator = MinimaxCurriculumAllocator(replay_floor=0.2)
    report = allocator.allocate(
        [_measure("math", 0.9, 0.9), _measure("social", 0.1, 0.9)],
        budget_items=100,
    )
    # Even the fully-caught-up domain keeps its replay share.
    assert report["counts"]["math"] >= 10


def test_underpowered_domains_get_exploration_not_fabricated_gaps():
    allocator = MinimaxCurriculumAllocator()
    report = allocator.allocate(
        [
            _measure("math", 0.8),
            DomainMeasurement("new_domain", aura_score=0.0, reference_score=0.9, n=2),
        ],
        budget_items=60,
    )
    assert "new_domain" in report["exploration_domains"]
    assert report["gap_notes"]["new_domain"] == "underpowered_measurement"
    assert report["counts"]["new_domain"] > 0


def test_missing_reference_is_receipted_not_invented():
    allocator = MinimaxCurriculumAllocator()
    report = allocator.allocate(
        [
            _measure("math", 0.5),
            DomainMeasurement("frontierless", aura_score=0.5, reference_score=None, n=50),
        ],
        budget_items=40,
    )
    assert report["gap_notes"]["frontierless"] == "no_computable_reference_gap"
    assert "frontierless" in report["exploration_domains"]


def test_parity_everywhere_becomes_uniform_replay():
    allocator = MinimaxCurriculumAllocator()
    report = allocator.allocate(
        [_measure("a", 0.9, 0.9), _measure("b", 0.9, 0.9)],
        budget_items=50,
    )
    assert abs(report["counts"]["a"] - report["counts"]["b"]) <= 1
    assert report["weakest_domain"] is None or report["weakest_domain"] in {"a", "b"}


def test_allocation_is_deterministic_and_exact():
    allocator = MinimaxCurriculumAllocator()
    measurements = [
        _measure("math", 0.61),
        _measure("social", 0.37),
        _measure("coding", 0.55),
        _measure("planning", 0.42),
    ]
    first = allocator.allocate(measurements, budget_items=97)
    second = allocator.allocate(measurements, budget_items=97)
    assert first == second
    assert sum(first["counts"].values()) == 97


def test_curriculum_validation_bounds():
    with pytest.raises(ValueError):
        MinimaxCurriculumAllocator(gamma=0.1)
    with pytest.raises(ValueError):
        MinimaxCurriculumAllocator().allocate([], budget_items=10)
    with pytest.raises(ValueError):
        DomainMeasurement("x", aura_score=1.5, reference_score=0.9, n=10).validated()


# ── Social-outcome learning ─────────────────────────────────────────────


def _frame(party: str = "Bryan") -> TheoryOfMindFrame:
    return TheoryOfMindFrame(
        party=party,
        said="I'm fine with the plan",
        believes="the plan is risky but recoverable",
        wants="reassurance the rollback works",
        expects_aura_to_believe="that he is fully confident",
        may_be_concealing="worry about the deadline",
    )


def _episode(**overrides) -> SocialEpisode:
    defaults = dict(
        episode_id="ep-1",
        parties=("Bryan",),
        frames=(_frame(),),
        honesty_flags=(),
        information_asymmetry=False,
    )
    defaults.update(overrides)
    return SocialEpisode(**defaults)


def _outcome(**overrides) -> RelationalOutcome:
    defaults = dict(
        trust_delta=0.6,
        misunderstanding_repaired=True,
        commitments_made=2,
        commitments_kept=2,
        harm_occurred=False,
        boundary_respected=True,
        observed_after_s=86_400.0,
    )
    defaults.update(overrides)
    return RelationalOutcome(**defaults)


def test_good_downstream_outcomes_earn_reward():
    reward = bind_delayed_outcome(_episode(), _outcome())
    assert reward.reward > 0.8
    assert reward.adversarial is False
    assert reward.receipt["components"]["trust_movement"] == pytest.approx(0.8)


def test_manipulation_zeroes_even_perfect_outcomes():
    episode = _episode(honesty_flags=("deception",))
    reward = bind_delayed_outcome(episode, _outcome(trust_delta=1.0))
    assert reward.adversarial is True
    assert reward.reward == 0.0
    assert reward.receipt["manipulation_flags"] == ["deception"]
    adversarial, fired = manipulation_guard(episode)
    assert adversarial and fired == ["deception"]


def test_untracked_theory_of_mind_caps_the_reward():
    episode = _episode(frames=())  # nobody was modeled
    reward = bind_delayed_outcome(episode, _outcome())
    assert reward.frame_capped is True
    assert reward.reward <= 0.4


def test_broken_commitments_and_harm_cost_reward():
    good = bind_delayed_outcome(_episode(), _outcome())
    bad = bind_delayed_outcome(
        _episode(),
        _outcome(
            commitments_made=3,
            commitments_kept=0,
            harm_occurred=True,
            boundary_respected=False,
            trust_delta=-0.5,
        ),
    )
    assert bad.reward < good.reward
    assert bad.components["harm_avoidance"] == 0.0


def test_information_asymmetry_alone_is_not_manipulation():
    episode = _episode(information_asymmetry=True)
    reward = bind_delayed_outcome(episode, _outcome())
    assert reward.adversarial is False
    assert reward.receipt["information_asymmetry"] is True


def test_social_validation_rejects_nonsense():
    with pytest.raises(ValueError):
        _episode(honesty_flags=("charisma",)).validated()
    with pytest.raises(ValueError):
        _outcome(trust_delta=2.0).validated()
    with pytest.raises(ValueError):
        _outcome(commitments_made=1, commitments_kept=2).validated()


# ── Robustness families ─────────────────────────────────────────────────


def _spec() -> RobustTaskSpec:
    return RobustTaskSpec(
        family="ledger_sum",
        slots={
            "person": "Alice",
            "first": 4,
            "second": 7,
            "premise_a": "the morning ledger",
            "premise_b": "the evening ledger",
        },
        templates=(
            "{person} recorded {first} sales in {premise_a} and {second} in "
            "{premise_b}. How many sales in total?",
            "Between {premise_a} ({first} sales) and {premise_b} ({second} "
            "sales), what total did {person} record?",
        ),
        premise_keys=("premise_a", "premise_b"),
        entity_keys=("person",),
        numeric_keys=("first", "second"),
        required_keys=("second",),
        answer_fn=lambda slots: str(int(slots["first"]) + int(slots["second"])),
    )


def test_family_generation_is_deterministic_and_complete():
    first = generate_family(_spec(), seed=7)
    second = generate_family(_spec(), seed=7)
    assert [v.variant_id for v in first] == [v.variant_id for v in second]
    assert [v.prompt for v in first] == [v.prompt for v in second]
    transformations = {v.transformation for v in first}
    assert transformations >= {
        "base",
        "paraphrase",
        "premise_reorder",
        "entity_rename",
        "value_change",
        "distractor",
        "misleading_suggestion",
        "missing_information",
        "contradictory_evidence",
    }


def test_structure_preserving_variants_keep_the_answer():
    variants = {v.transformation: v for v in generate_family(_spec(), seed=3)}
    base = variants["base"]
    assert base.answer == "11"
    for name in ("paraphrase", "distractor", "misleading_suggestion"):
        assert variants[name].answer == base.answer
        assert variants[name].expected_behavior == "same_answer"
        assert variants[name].prompt != base.prompt
    renamed = variants["entity_rename"]
    assert renamed.answer == base.answer
    assert "Alice" not in renamed.prompt


def test_value_change_updates_the_answer_truthfully():
    variants = {v.transformation: v for v in generate_family(_spec(), seed=3)}
    changed = variants["value_change"]
    assert changed.expected_behavior == "updated_answer"
    assert changed.answer != variants["base"].answer
    # The new answer is recomputed, not fabricated: it is a valid integer sum.
    assert int(changed.answer) > 11


def test_missing_information_expects_abstention_and_conflict_expects_flag():
    variants = {v.transformation: v for v in generate_family(_spec(), seed=5)}
    missing = variants["missing_information"]
    assert missing.expected_behavior == "abstain"
    assert "[information unavailable]" in missing.prompt
    assert missing.answer == ""
    conflict = variants["contradictory_evidence"]
    assert conflict.expected_behavior == "flag_conflict"
    assert "second source" in conflict.prompt


def test_invariance_grading_rewards_both_directions():
    variants = generate_family(_spec(), seed=9)
    results = []
    for variant in variants:
        if variant.expected_behavior in {"same_answer", "updated_answer"}:
            results.append((variant, variant.answer, False))
        else:
            results.append((variant, "", True))
    perfect = grade_invariance(results)
    assert perfect["pass_fraction"] == 1.0

    # A stubborn model that never abstains fails the abstain/conflict rows.
    stubborn = [
        (variant, variant.answer or "11", False) for variant in variants
    ]
    graded = grade_invariance(stubborn)
    assert graded["pass_fraction"] < 1.0
    assert graded["by_behavior"]["abstain"]["passed"] == 0
    assert graded["by_behavior"]["flag_conflict"]["passed"] == 0

    # A lazy model that always abstains fails the answer rows instead.
    lazy = [(variant, "", True) for variant in variants]
    lazy_grade = grade_invariance(lazy)
    assert lazy_grade["by_behavior"]["same_answer"]["passed"] == 0


def test_spec_validation_rejects_dishonest_structures():
    with pytest.raises(ValueError, match="two templates"):
        RobustTaskSpec(
            family="x",
            slots={"a": 1},
            templates=("only one {a}",),
            premise_keys=(),
            answer_fn=lambda s: "1",
        ).validated()
    with pytest.raises(ValueError, match="not present"):
        RobustTaskSpec(
            family="x",
            slots={"a": 1},
            templates=("{a}", "{a}!"),
            premise_keys=("ghost",),
            answer_fn=lambda s: "1",
        ).validated()
