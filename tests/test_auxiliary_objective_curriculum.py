"""SPARK-062 contracts: seven declared terms, and proof each one is live.

The tests that carry this file are the ones about a term that is declared,
weighted, logged, and doing nothing. That failure is invisible in a loss curve
by construction — the composite descends either way — so it has to be caught
structurally or not at all.

The second group is the depth curriculum. Its contracts are that advancement is
bound to measured competence rather than a step counter, that it can move back,
that a receipt replays exactly, and that a stage the inference configuration
cannot execute is refused before a campaign rather than discovered after one.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from core.learning.auxiliary_objective_curriculum import (  # noqa: E402
    AUXILIARY_COMPOSITE_SCHEMA,
    DEPTH_CURRICULUM_SCHEMA,
    SHARES_CALLER_SUPPLIED,
    SHARES_DERIVED_FROM_COMPOSITE,
    SPARK062_TERMS,
    AuxiliaryObjectiveError,
    AuxiliaryTerm,
    DepthCurriculum,
    DepthStage,
    TermTarget,
    base_weight_loss,
    build_liveness_report,
    canonical_sha256,
    liveness_from_composite,
    parity_binding,
    require_parity,
    validate_curriculum_receipt,
    validate_liveness_report,
    validate_term_set,
)


class _Spec:
    """Stand-in for RLCExecutionSpec's two parity-relevant fields."""

    def __init__(self, recurrent_steps: int, alpha_schedule: str = "constant"):
        self.recurrent_steps = recurrent_steps
        self.alpha_schedule = alpha_schedule


def _base(name: str, weight: float = 1.0) -> AuxiliaryTerm:
    return AuxiliaryTerm(
        name=name,
        target=TermTarget.BASE_WEIGHTS,
        weight=weight,
        source_module="core.learning.example",
    )


def _head(name: str, weight: float = 1.0) -> AuxiliaryTerm:
    return AuxiliaryTerm(
        name=name,
        target=TermTarget.AUXILIARY_HEAD,
        weight=weight,
        source_module="core.learning.example",
    )


def _head_evidence(name: str) -> dict[str, dict[str, object]]:
    return {
        "head_gradient_norms": {name: 0.4},
        "head_before_sha256s": {name: "a" * 64},
        "head_after_sha256s": {name: "b" * 64},
        "head_optimizer_update_counts": {name: 1},
    }


# ── The seven declarations ──────────────────────────────────────────────


def test_the_seven_spark062_objectives_are_declared_and_typed():
    names = [term.name for term in SPARK062_TERMS]
    assert names == [
        "process",
        "improvement",
        "diversity",
        "stopping",
        "causality",
        "mistake_location",
        "accept_discard",
    ]
    validate_term_set(SPARK062_TERMS)
    # Three of the seven train separate heads. Getting this wrong is the
    # category error the registry exists to prevent, so it is pinned.
    head_terms = {term.name for term in SPARK062_TERMS if term.target is TermTarget.AUXILIARY_HEAD}
    assert head_terms == {"process", "mistake_location", "accept_discard"}
    # Every term names a real, importable module rather than a description.
    for term in SPARK062_TERMS:
        assert term.source_module.startswith("core.")
        __import__(term.source_module)


def test_term_declarations_reject_incoherent_configurations():
    with pytest.raises(AuxiliaryObjectiveError, match="diagnostic"):
        AuxiliaryTerm(
            name="probe",
            target=TermTarget.DIAGNOSTIC,
            weight=1.0,
            source_module="core.learning.example",
        )
    with pytest.raises(AuxiliaryObjectiveError, match="inert by construction"):
        AuxiliaryTerm(
            name="probe",
            target=TermTarget.BASE_WEIGHTS,
            weight=0.0,
            source_module="core.learning.example",
        )
    with pytest.raises(AuxiliaryObjectiveError, match="dotted module"):
        AuxiliaryTerm(
            name="probe",
            target=TermTarget.BASE_WEIGHTS,
            weight=1.0,
            source_module="somewhere",
        )
    with pytest.raises(AuxiliaryObjectiveError, match="duplicate"):
        validate_term_set([_base("a"), _base("a")])


# ── The central contract: an inert term must be caught ──────────────────


def test_a_declared_term_with_no_gradient_path_is_inert_and_refuses():
    """The failure the module exists for.

    Every term is declared, weighted and reported. One of them has no path to
    the parameters at all. The composite's loss is unaffected by that fact, so
    only a structural check can see it.
    """
    terms = [_base("improvement"), _base("diversity")]
    report = build_liveness_report(
        terms,
        shares={"improvement": 0.4, "diversity": 0.4},
        gradient_norms={"improvement": 0.7, "diversity": 0.0},
    )

    rows = {row["name"]: row["liveness"] for row in report["terms"]}
    assert rows["improvement"] == "live"
    assert rows["diversity"] == "inert_zero_gradient"
    assert report["inert_required_terms"] == ["diversity"]
    assert report["supports_training"] is False
    validate_liveness_report(report)


def test_a_term_too_small_to_move_the_optimizer_is_inert():
    """v3's diversity term sat at 0.037% of the loss and was reported active."""
    terms = [_base("improvement"), _base("diversity")]
    report = build_liveness_report(
        terms,
        shares={"improvement": 0.6, "diversity": 0.00037},
        gradient_norms={"improvement": 0.7, "diversity": 1e-4},
    )
    rows = {row["name"]: row["liveness"] for row in report["terms"]}
    assert rows["diversity"] == "inert_negligible_share"
    assert report["supports_training"] is False


def test_a_head_term_that_reached_the_base_weights_is_misdeclared():
    """A critic objective summed into the base loss steers the wrong model."""
    terms = [_base("improvement"), _head("process")]
    report = build_liveness_report(
        terms,
        shares={"improvement": 0.5, "process": 0.5},
        gradient_norms={"improvement": 0.7, "process": 0.3},
    )
    rows = {row["name"]: row["liveness"] for row in report["terms"]}
    assert rows["process"] == "misdeclared_target"
    assert report["supports_training"] is False


def test_a_head_without_own_gradient_and_mutation_evidence_is_unmeasured():
    report = build_liveness_report(
        [_head("process")],
        shares={"process": 0.5},
        gradient_norms={"process": 0.0},
    )

    assert report["terms"][0]["liveness"] == "unmeasured"
    assert report["supports_training"] is False


def test_a_head_without_measured_base_isolation_is_unmeasured():
    report = build_liveness_report(
        [_head("process")],
        shares={"process": 0.5},
        **_head_evidence("process"),
    )

    assert report["terms"][0]["liveness"] == "unmeasured"
    assert report["supports_training"] is False


def test_a_head_with_gradient_but_no_parameter_change_is_inert():
    evidence = _head_evidence("process")
    evidence["head_after_sha256s"] = {"process": "a" * 64}
    report = build_liveness_report(
        [_head("process")],
        shares={"process": 0.5},
        gradient_norms={"process": 0.0},
        **evidence,
    )

    assert report["terms"][0]["liveness"] == "inert_head_not_updated"
    assert report["supports_training"] is False


@pytest.mark.parametrize(
    ("shares", "gradients"),
    [
        ({"improvement": float("nan")}, {"improvement": 0.5}),
        ({"improvement": 0.5}, {"improvement": float("inf")}),
        ({"improvement": 1.1}, {"improvement": 0.5}),
        ({"improvement": 0.5}, {"improvement": -0.1}),
    ],
)
def test_liveness_refuses_nonfinite_or_out_of_range_measurements(shares, gradients):
    with pytest.raises(AuxiliaryObjectiveError):
        build_liveness_report(
            [_base("improvement")],
            shares=shares,
            gradient_norms=gradients,
        )


def test_a_base_term_without_a_gradient_measurement_is_unmeasured_not_live():
    """Absence of a check must never read as a passed check."""
    report = build_liveness_report([_base("improvement")], shares={"improvement": 0.5})
    assert report["terms"][0]["liveness"] == "unmeasured"
    assert report["supports_training"] is False


def test_a_fully_live_composite_supports_training():
    terms = [_base("improvement"), _base("diversity"), _head("process")]
    report = build_liveness_report(
        terms,
        shares={"improvement": 0.4, "diversity": 0.3, "process": 0.3},
        gradient_norms={"improvement": 0.7, "diversity": 0.5, "process": 0.0},
        **_head_evidence("process"),
    )
    assert report["live_terms"] == ["diversity", "improvement", "process"]
    assert report["inert_required_terms"] == []
    assert report["supports_training"] is True
    validate_liveness_report(report)


def test_liveness_report_rejects_forged_verdicts():
    terms = [_base("improvement"), _base("diversity")]
    report = build_liveness_report(
        terms,
        shares={"improvement": 0.4, "diversity": 0.4},
        gradient_norms={"improvement": 0.7, "diversity": 0.0},
    )
    forged = {key: value for key, value in report.items() if key != "receipt_sha256"}
    forged["inert_required_terms"] = []
    forged["supports_training"] = True
    forged["receipt_sha256"] = canonical_sha256(forged)
    with pytest.raises(AuxiliaryObjectiveError, match="does not replay"):
        validate_liveness_report(forged)

    tampered = dict(report)
    tampered["live_terms"] = ["improvement", "diversity"]
    with pytest.raises(AuxiliaryObjectiveError, match="commitment"):
        validate_liveness_report(tampered)


# ── The composite excludes head terms from the base loss ────────────────


def test_base_loss_excludes_head_terms_and_reports_shares():
    terms = [_base("improvement", 2.0), _head("process", 1.0)]
    primary = mx.array(2.0)
    total, telemetry = base_weight_loss(
        terms,
        {"improvement": mx.array(0.5), "process": mx.array(10.0)},
        primary=primary,
    )
    mx.eval(total)
    # 2.0 + 2.0*0.5 == 3.0; the head term's 10.0 must NOT appear.
    assert float(total) == pytest.approx(3.0, abs=1e-6)
    assert telemetry["base_weight_terms"] == ["improvement"]
    assert telemetry["excluded_from_base_loss"] == ["process"]
    assert telemetry["weighted_contributions"]["process"] == pytest.approx(10.0)
    assert telemetry["shares"]["improvement"] > 0.0


def test_base_loss_refuses_missing_and_undeclared_terms():
    terms = [_base("improvement")]
    with pytest.raises(AuxiliaryObjectiveError, match="no computed value"):
        base_weight_loss(terms, {}, primary=mx.array(1.0))
    with pytest.raises(AuxiliaryObjectiveError, match="undeclared"):
        base_weight_loss(
            terms,
            {"improvement": mx.array(0.1), "mystery": mx.array(0.1)},
            primary=mx.array(1.0),
        )


def test_base_loss_gradient_reaches_only_the_base_terms():
    """The exclusion is real, not cosmetic: the head term has no gradient path."""
    terms = [_base("improvement"), _head("process")]

    def loss(values):
        total, _ = base_weight_loss(
            terms,
            {"improvement": values["improvement"], "process": values["process"]},
            primary=mx.array(1.0),
        )
        return total

    gradients = mx.grad(loss)({"improvement": mx.array(0.5), "process": mx.array(0.5)})
    mx.eval(gradients)
    assert float(gradients["improvement"]) != 0.0
    assert float(gradients["process"]) == 0.0


# ── Depth curriculum ────────────────────────────────────────────────────


def _stages() -> list[DepthStage]:
    return [
        DepthStage(depth=2, min_samples=4, competence_threshold=0.6),
        DepthStage(depth=4, min_samples=4, competence_threshold=0.6),
        DepthStage(depth=8, min_samples=4, competence_threshold=0.6),
    ]


def test_curriculum_advances_only_on_measured_competence():
    curriculum = DepthCurriculum(_stages())
    assert curriculum.stage.depth == 2

    # Enough competence but not enough samples: a step counter would advance.
    assert curriculum.observe(competence=0.9, samples=1) == ("held_insufficient_samples")
    assert curriculum.stage.depth == 2

    # Enough samples, competence below threshold, already at the first stage.
    assert curriculum.observe(competence=0.2, samples=8) == "held_below_threshold"
    assert curriculum.stage.depth == 2

    assert curriculum.observe(competence=0.8, samples=8) == "advanced"
    assert curriculum.stage.depth == 4


def test_curriculum_regresses_when_competence_collapses():
    """A curriculum that can only advance turns a dip into a permanent one."""
    curriculum = DepthCurriculum(_stages())
    curriculum.observe(competence=0.9, samples=8)
    assert curriculum.stage.depth == 4

    assert curriculum.observe(competence=0.1, samples=8) == "regressed"
    assert curriculum.stage.depth == 2


def test_curriculum_holds_at_the_final_stage():
    curriculum = DepthCurriculum(_stages())
    curriculum.observe(competence=0.9, samples=8)
    curriculum.observe(competence=0.9, samples=8)
    assert curriculum.stage.depth == 8
    assert curriculum.observe(competence=0.9, samples=8) == "held_at_final_stage"
    assert curriculum.stage.depth == 8


def test_curriculum_rejects_incoherent_stage_sets():
    with pytest.raises(AuxiliaryObjectiveError, match="at least one stage"):
        DepthCurriculum([])
    with pytest.raises(AuxiliaryObjectiveError, match="increasing"):
        DepthCurriculum(
            [
                DepthStage(depth=4, min_samples=2, competence_threshold=0.5),
                DepthStage(depth=2, min_samples=2, competence_threshold=0.5),
            ]
        )
    with pytest.raises(AuxiliaryObjectiveError, match="competence"):
        DepthCurriculum(_stages()).observe(competence=1.4, samples=4)


def test_curriculum_receipt_replays_every_transition():
    curriculum = DepthCurriculum(_stages())
    curriculum.observe(competence=0.9, samples=8)
    curriculum.observe(competence=0.2, samples=8)
    curriculum.observe(competence=0.95, samples=8)
    receipt = curriculum.to_receipt()

    assert receipt["schema"] == DEPTH_CURRICULUM_SCHEMA
    assert [row["transition"] for row in receipt["history"]] == [
        "advanced",
        "regressed",
        "advanced",
    ]
    validate_curriculum_receipt(receipt)

    # A forged final position must not survive the replay.
    forged = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    forged["current_depth"] = 8
    forged["current_index"] = 2
    forged["receipt_sha256"] = canonical_sha256(forged)
    with pytest.raises(AuxiliaryObjectiveError, match="final position"):
        validate_curriculum_receipt(forged)


def test_curriculum_receipt_rejects_a_forged_advancement_policy():
    """A curriculum advanced by a step counter is not this contract."""
    receipt = DepthCurriculum(_stages()).to_receipt()
    forged = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    forged["advancement_policy"] = "fixed_step_schedule_v1"
    forged["receipt_sha256"] = canonical_sha256(forged)
    with pytest.raises(AuxiliaryObjectiveError, match="any other policy"):
        validate_curriculum_receipt(forged)


# ── Train/inference depth parity ────────────────────────────────────────


def test_parity_accepts_an_executable_stage():
    stage = DepthStage(depth=4, min_samples=4, competence_threshold=0.6)
    binding = parity_binding(
        stage,
        spec=_Spec(recurrent_steps=4),
        inference_max_steps=8,
        inference_min_steps=2,
        inference_fixed_depth=True,
    )
    assert binding["parity"] is True
    assert binding["problems"] == []
    require_parity(binding)


def test_parity_refuses_a_stage_the_inference_config_cannot_run():
    """Training a depth production never reaches tunes an unreachable regime."""
    stage = DepthStage(depth=16, min_samples=4, competence_threshold=0.6)
    binding = parity_binding(
        stage,
        spec=_Spec(recurrent_steps=16),
        inference_max_steps=8,
        inference_fixed_depth=True,
    )
    assert binding["parity"] is False
    assert "stage_depth_exceeds_inference_max_steps" in binding["problems"]
    with pytest.raises(AuxiliaryObjectiveError, match="parity refused"):
        require_parity(binding)


def test_parity_refuses_a_spec_that_disagrees_with_its_stage():
    stage = DepthStage(depth=4, min_samples=4, competence_threshold=0.6)
    binding = parity_binding(
        stage,
        spec=_Spec(recurrent_steps=2),
        inference_max_steps=8,
        inference_fixed_depth=True,
    )
    assert "execution_spec_depth_differs_from_stage" in binding["problems"]
    assert binding["parity"] is False


def test_parity_refuses_adaptive_halting_against_a_trained_fixed_depth():
    """Early halting runs a different computation than the one trained."""
    stage = DepthStage(depth=4, min_samples=4, competence_threshold=0.6)
    binding = parity_binding(
        stage,
        spec=_Spec(recurrent_steps=4),
        inference_max_steps=8,
        inference_fixed_depth=False,
    )
    assert "adaptive_halting_breaks_trained_depth_parity" in binding["problems"]
    assert binding["parity"] is False


def test_require_parity_fails_closed_on_garbage():
    with pytest.raises(AuxiliaryObjectiveError):
        require_parity({"parity": "yes"})
    with pytest.raises(AuxiliaryObjectiveError):
        require_parity(None)


def test_liveness_and_composite_schemas_are_pinned():
    report = build_liveness_report(
        [_base("improvement")],
        shares={"improvement": 0.5},
        gradient_norms={"improvement": 0.5},
    )
    assert report["schema"] == AUXILIARY_COMPOSITE_SCHEMA
    validate_liveness_report(report)


def test_a_relabelled_inert_row_cannot_pass_as_live():
    """Adversarial self-review finding, the SPARK-062 half.

    Relabelling a zero-gradient row `live` and updating `live_terms` to match
    passed every aggregate check in the first version of this validator,
    because those aggregates were derived from the labels they were supposed
    to police. Verdicts are now recomputed from each row's own share and
    gradient.
    """
    terms = [_base("improvement"), _base("diversity")]
    report = build_liveness_report(
        terms,
        shares={"improvement": 0.4, "diversity": 0.4},
        gradient_norms={"improvement": 0.7, "diversity": 0.0},
    )
    forged = {key: value for key, value in report.items() if key != "receipt_sha256"}
    rows = [dict(row) for row in forged["terms"]]
    for row in rows:
        if row["name"] == "diversity":
            row["liveness"] = "live"
    forged["terms"] = rows
    forged["live_terms"] = ["diversity", "improvement"]
    forged["inert_required_terms"] = []
    forged["supports_training"] = True
    forged["receipt_sha256"] = canonical_sha256(forged)

    with pytest.raises(AuxiliaryObjectiveError, match="does not replay"):
        validate_liveness_report(forged)


def test_a_forged_share_is_the_boundary_the_receipt_names_honestly():
    """The limit of self-validation, stated rather than papered over.

    Recomputing verdicts from the rows closes label forgery. It cannot close
    INPUT forgery: the rebuild consumes the same share it is checking, so a
    forged share is reproduced faithfully and the report validates. Pretending
    otherwise would be the exact "absence of a check reported as a passed
    check" pattern this codebase keeps finding. The receipt therefore names
    where its shares came from, and `liveness_from_composite` removes the
    input from the caller's hands entirely.
    """
    terms = [_base("improvement"), _base("diversity")]
    report = build_liveness_report(
        terms,
        shares={"improvement": 0.6, "diversity": 0.00037},
        gradient_norms={"improvement": 0.7, "diversity": 1e-4},
    )
    forged = {key: value for key, value in report.items() if key != "receipt_sha256"}
    rows = [dict(row) for row in forged["terms"]]
    for row in rows:
        if row["name"] == "diversity":
            row["liveness"] = "live"
            row["share"] = 0.4
    forged["terms"] = rows
    forged["live_terms"] = ["diversity", "improvement"]
    forged["inert_required_terms"] = []
    forged["supports_training"] = True
    forged["receipt_sha256"] = canonical_sha256(forged)

    # It validates — and says plainly that a human supplied the shares.
    validated = validate_liveness_report(forged)
    assert validated["shares_source"] == SHARES_CALLER_SUPPLIED

    # Derived shares cannot be forged this way: the composite computes them
    # from measured contributions, so the input leaves the caller's hands.
    terms_live = [_base("improvement"), _head("process")]
    total, telemetry = base_weight_loss(
        terms_live,
        {"improvement": mx.array(0.5), "process": mx.array(0.5)},
        primary=mx.array(2.0),
    )
    mx.eval(total)
    derived = liveness_from_composite(
        terms_live,
        telemetry,
        gradient_norms={"improvement": 0.7, "process": 0.0},
        **_head_evidence("process"),
    )
    assert derived["shares_source"] == SHARES_DERIVED_FROM_COMPOSITE
    validate_liveness_report(derived)
