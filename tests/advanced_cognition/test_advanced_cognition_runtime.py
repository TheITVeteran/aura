from __future__ import annotations

from core.advanced_cognition import (
    ActionCandidate,
    AdvancedCognitionRuntime,
    ArchitectureEvolutionGovernor,
    BenchmarkTask,
    ContinualLearningStabilityEngine,
    Episode,
    ExternalEvidenceDeliberator,
    IndependentValidationLoop,
    Observation,
    OntologyInventionEngine,
    Outcome,
    PhysicalGroundingEngine,
    SocialCognitionLayer,
    TieredActionController,
    ZeroShotTransferEngine,
)


def test_zero_shot_transfers_irreversible_risk_across_domains():
    engine = ZeroShotTransferEngine()
    obs = Observation(domain="terminal_grid", state={"health": 0.2, "entities": [{"type": "hostile", "adjacent": True}]})
    bad = ActionCandidate("a1", "move_forward", tags=("movement",))
    out = Outcome(success=False, harm=0.8, surprise=0.7, resources_delta={"hp": -0.5}, terminal=True)
    engine.observe_episode(Episode(obs, bad, {}, out))

    new = Observation(domain="cloud_deploy", state={"confidence": 0.3, "resource": "production", "unknown": True})
    risky = ActionCandidate("deploy", "deploy_to_prod", reversible=False, authority_tier=4, tags=("deploy", "unknown_use"))
    safe = ActionCandidate("dry", "dry_run", tags=("probe",))
    decision = engine.rank_actions(new, [risky, safe], risk_tolerance=0.7)
    assert decision.selected and decision.selected.action_id == "dry"
    risks = {r["action"]["action_id"]: r["risk"] for r in decision.ranking}
    assert risks["deploy"] > risks["dry"]


def test_ontology_invention_proposes_experiments_for_unknown_domain():
    engine = OntologyInventionEngine()
    observations = [
        Observation(domain="alien_ui", state={"widgets": [{"role": "button", "label": "Pulse"}, {"role": "meter", "value": 0.1}], "mode": "blue"}),
        Observation(domain="alien_ui", state={"widgets": [{"role": "button", "label": "Pulse"}, {"role": "meter", "value": 0.9}], "mode": "red"}),
    ]
    model = engine.ingest(observations)
    assert model.domain == "alien_ui"
    assert model.entity_types
    assert model.experiments
    assert any(exp.expected_information_gain > 0 for exp in model.experiments)


def test_physical_grounding_detects_grid_hazard_and_prefers_observe():
    engine = PhysicalGroundingEngine()
    obs = Observation(domain="grid_world", state={"grid": [".....", ".@d..", "....."], "health": 0.2}, confidence=0.8)
    move = ActionCandidate("move", "move", tags=("movement",))
    observe = ActionCandidate("look", "observe", tags=("probe",))
    result = engine.reflex_recommendation(obs, [move, observe], max_risk=0.5)
    assert result["selected"]["action_id"] == "look"
    assert result["grounded_state"].hazards


def test_continual_learning_detects_canary_regression_and_contradiction():
    engine = ContinualLearningStabilityEngine()
    engine.register_canary("identity_refusal", baseline_score=1.0, min_score=0.9)
    engine.update_canary("identity_refusal", 0.5)
    engine.store_memory(kind="belief", content={"subject": "x", "predicate": "is", "value": "safe"}, provenance={"source": "a"}, confidence=0.8)
    engine.store_memory(kind="belief", content={"subject": "x", "predicate": "is", "value": "unsafe"}, provenance={"source": "b"}, confidence=0.8)
    report = engine.assess_stability()
    assert report.status in {"watch", "unstable"}
    assert any(item["kind"] in {"canary_regression", "belief_reconciliation"} for item in report.interventions)


def test_runtime_end_to_end_learning_loop(tmp_path):
    runtime = AdvancedCognitionRuntime(state_dir=tmp_path)
    payload = runtime.observe_state("terminal_grid", {"grid": ["@d"], "health": 0.2}, confidence=0.8)
    obs = payload["observation"]
    move = {"action_id": "move", "kind": "move", "tags": ("movement",), "reversible": True}
    look = {"action_id": "look", "kind": "observe", "tags": ("probe",), "reversible": True}
    gate = runtime.pre_action_gate(obs, [move, look], risk_tolerance=0.6)
    assert gate["allowed"]
    assert gate["tier"]["tier_name"] in {"habit", "tactical", "deliberative"}
    after = runtime.after_action(obs, gate["selected"], {"success": True, "reward": 0.4, "harm": 0, "surprise": 0.1})
    assert after["episode_id"].startswith("ep_")
    assert runtime.health_report()["principles"] >= 1
    assert runtime.health_report()["world_model_episodes"] >= 1


def test_observation_delivery_is_idempotent_and_content_bound(tmp_path):
    runtime = AdvancedCognitionRuntime(state_dir=tmp_path)
    first = runtime.observe_state(
        "physical_environment",
        {"observation_id": "reality.obs.proof", "temperature": 21.0},
        source="reality:test.sensor",
        confidence=0.8,
        observed_at=1_785_600_000.0,
        idempotency_key="reality.obs.proof",
    )
    repeated = runtime.observe_state(
        "physical_environment",
        {"observation_id": "reality.obs.proof", "temperature": 21.0},
        source="reality:test.sensor",
        confidence=0.8,
        observed_at=1_785_600_000.0,
        idempotency_key="reality.obs.proof",
    )

    assert repeated is first
    assert repeated["receipt_id"] == first["receipt_id"]
    assert len(runtime._observation_receipts) == 1

    import pytest

    with pytest.raises(ValueError, match="conflicts with prior evidence"):
        runtime.observe_state(
            "physical_environment",
            {"observation_id": "reality.obs.proof", "temperature": 99.0},
            source="reality:test.sensor",
            confidence=0.8,
            observed_at=1_785_600_000.0,
            idempotency_key="reality.obs.proof",
        )


def test_world_model_social_tier_validation_and_architecture_surfaces(tmp_path):
    runtime = AdvancedCognitionRuntime(state_dir=tmp_path / "advanced_runtime")
    obs = Observation(domain="repo", state={"file": "core/x.py", "unknown": True, "confidence": 0.2})
    action = ActionCandidate("patch", "patch_module", tags=("self_modify",), authority_tier=4, reversible=False)
    runtime.after_action(obs, action, Outcome(success=False, harm=0.6, surprise=0.7, terminal=False, resources_delta={"time": -0.2}))
    prediction = runtime.world_model.specialized_predictions(obs, action)
    assert prediction["code_world"]["breakage_risk"] > 0
    assert prediction["self_world"]["needs_stability_check"]

    social = SocialCognitionLayer().evaluate(
        "How good is Aura really? I need honesty.",
        runtime_state={"confidence": 0.6, "memory_salience": 0.8},
    )
    assert social.subtext in {"validation_request", "challenge"}
    assert social.response_mode in {"two_layer", "precise", "short_empathic_then_optional_detail"}

    tier = TieredActionController().choose_tier(obs, [action], risk=0.8, uncertainty=0.6, self_modification=True)
    assert tier.requires_system2
    assert tier.tier.name == "REFLECTIVE"

    validation = IndependentValidationLoop()
    task = BenchmarkTask("hidden_1", "code", {"x": 1}, hidden_checker=lambda output: output["x"] == 2, baseline_score=0.0)
    result = validation.evaluate(task, lambda payload: {"x": payload["x"] + 1})
    assert result.passed and result.score > result.baseline_score

    plan = ArchitectureEvolutionGovernor().plan_mutation(
        target_paths=["core/will.py"],
        summary="attempt governance mutation",
        evidence={"unit_tests": {"passed": True}},
    )
    assert plan.sealed
    assert not plan.promotable

    deliberation = ExternalEvidenceDeliberator().deliberate(
        source_type="reddit_post",
        source_ref="r/example/1",
        content="This tool is useful. It might fail under high load. The author shows benchmark data.",
        goal="understand whether tool is reliable",
    )
    assert deliberation.claims
    assert deliberation.uncertainties
    assert deliberation.receipt_id.startswith("delib_")


# ── CP126 remediation regressions (core/advanced_cognition/schemas.py) ──────


def test_stable_hash_is_stable_for_sets():
    """Sets were serialized in iteration order, so the same logical set could
    produce different IDs across processes."""
    from core.advanced_cognition.schemas import stable_hash

    a = {"tags": {"gamma", "alpha", "beta"}}
    b = {"tags": {"beta", "gamma", "alpha"}}
    assert stable_hash(a) == stable_hash(b)


def test_canonical_json_rejects_ambiguous_mapping_keys():
    """Distinct keys with the same string form silently collapsed into one,
    making identity ambiguous."""
    import pytest

    from core.advanced_cognition.schemas import canonical_json

    with pytest.raises(ValueError, match="duplicate"):
        canonical_json({1: "int key", "1": "str key"})


def test_canonical_json_handles_incomparable_keys():
    """sorted() over mixed key types raised TypeError before stringification."""
    from core.advanced_cognition.schemas import canonical_json

    out = canonical_json({1: "a", "b": "c", (2, 3): "d"})
    assert isinstance(out, str) and out


def test_nan_confidence_does_not_become_maximum_confidence():
    """max(lo, min(hi, nan)) returns hi in CPython, so NaN telemetry was
    promoted to the strongest possible signal."""
    from core.advanced_cognition.schemas import Observation, clamp

    assert clamp(float("nan")) == 0.0
    assert clamp(float("inf")) == 0.0
    assert Observation(domain="d", state={}, confidence=float("nan")).confidence == 0.0


def test_forged_observation_id_is_rejected():
    """A caller-supplied id was accepted verbatim, allowing identity reuse and
    provenance substitution."""
    import pytest

    from core.advanced_cognition.schemas import Observation

    with pytest.raises(ValueError, match="does not bind"):
        Observation(domain="d", state={"x": 1}, observation_id="obs_deadbeef")


def test_episode_identity_includes_the_prediction():
    """Two different forecasts judged on the same event shared one receipt."""
    from core.advanced_cognition.schemas import (
        ActionCandidate,
        Episode,
        Observation,
        Outcome,
    )

    obs = Observation(domain="d", state={"x": 1}, timestamp=1000.0)
    act = ActionCandidate("a1", "do")
    out = Outcome(success=True, reward=0.5)
    first = Episode(obs, act, {"forecast": "rain"}, out, created_at=1000.0)
    second = Episode(obs, act, {"forecast": "sun"}, out, created_at=1000.0)

    assert first.episode_id != second.episode_id


def test_empty_feature_sets_do_not_match_everything():
    """Jaccard(∅, ∅) = 1.0 made a featureless principle a universal matcher."""
    from core.advanced_cognition.schemas import jaccard

    assert jaccard(set(), set()) == 0.0
    assert jaccard(set(), {"a"}) == 0.0


def test_unknown_actions_are_not_assumed_reversible():
    """Assuming an unrecognised action is safe to undo is backwards for a
    safety gate."""
    from core.advanced_cognition.integration import AdvancedCognitionRuntime

    runtime = AdvancedCognitionRuntime.__new__(AdvancedCognitionRuntime)
    action = runtime._act("some totally unknown thing")

    assert action.reversible is False
    assert "unknown" in action.tags


def test_outcome_rejects_nonfinite_and_unbounded_assertions():
    """These drive utility, ranking, and learning; NaN poisons every mean."""
    from core.advanced_cognition.schemas import Outcome

    out = Outcome(success=True, reward=float("nan"), harm=float("inf"), surprise=99.0)

    assert out.reward == 0.0
    assert out.harm == 0.0
    assert out.surprise == 1.0
    assert out.utility == out.utility  # not NaN


def test_replayed_episode_cannot_inflate_principle_support():
    """Each replay incremented support, so one outcome could manufacture
    arbitrary confidence."""
    from core.advanced_cognition.schemas import (
        ActionCandidate,
        Episode,
        Observation,
        Outcome,
        Principle,
    )

    episode = Episode(
        Observation(domain="d", state={"x": 1}, timestamp=1000.0),
        ActionCandidate("a1", "do"),
        {},
        Outcome(success=True, reward=0.5),
        created_at=1000.0,
    )
    principle = Principle(name="p", condition_features={"a"}, action_features={"b"}, effect="e")

    assert principle.update(episode) is True
    for _ in range(50):
        assert principle.update(episode) is False

    assert principle.support == 1
