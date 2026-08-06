from __future__ import annotations

import pytest

from core.brain.llm.latent_cortex.exact_paired_grade import (
    EXACT_INTERACTION_SCHEMA,
    EXACT_PAIRED_COMPARISON_SCHEMA,
    ExactPairedGradeError,
    ExactPairedObservation,
    exact_campaign_power_plan,
    exact_group_sequential_power_plan,
    exact_interaction_proven,
    exact_interaction_refuted,
    grade_exact_interaction,
    grade_exact_paired_comparison,
    minimum_zero_loss_noninferiority_observations,
)
from core.brain.llm.latent_cortex.exact_paired_statistics import Rational
from core.brain.llm.latent_cortex.experiments import CONJECTURE, PROVEN, REFUTED

DEFAULT_TOLERANCE = Rational(1, 1)


def _assert_no_floats(value) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _assert_no_floats(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_floats(child)
    else:
        assert not isinstance(value, float)


def _family(
    name: str,
    *,
    count: int = 20,
    treatment: bool = True,
    control: bool = False,
    treatment_compute: int = 100,
    control_compute: int = 100,
) -> list[ExactPairedObservation]:
    return [
        ExactPairedObservation(
            task_id=f"{name}-{index:03d}",
            family=name,
            treatment_success=treatment,
            control_success=control,
            treatment_compute=treatment_compute,
            control_compute=control_compute,
        )
        for index in range(count)
    ]


def _grade(
    families,
    *,
    require_compute: bool = False,
    tolerance: Rational = DEFAULT_TOLERANCE,
):
    return grade_exact_paired_comparison(
        experiment="adapter_rlc_gain",
        statement="adapter RLC improves over adapter vanilla",
        treatment="adapter_rlc",
        control="adapter_vanilla",
        observations_by_family=families,
        require_compute=require_compute,
        compute_tolerance=tolerance,
        global_bound_family_count=len(families) + 1,
    )


def test_exact_comparison_proves_broad_perfect_gain_with_full_certificates():
    grade = _grade(
        {
            "zeta": _family("zeta"),
            "alpha": _family("alpha"),
            "beta": _family("beta"),
        }
    )
    assert grade["tier"] == PROVEN
    _assert_no_floats(grade)
    evidence = grade["evidence"]
    assert evidence["schema"] == EXACT_PAIRED_COMPARISON_SCHEMA
    assert list(evidence["families"]) == ["alpha", "beta", "zeta"]
    assert evidence["positive_families"] == ["alpha", "beta", "zeta"]
    assert evidence["regressed_families"] == []
    assert evidence["global_bound_family_count"] == 4
    assert evidence["holm"]["family_size"] == 3
    assert [
        entry["hypothesis"] for entry in evidence["holm"]["ordered"]
    ] == ["alpha", "beta", "zeta"]
    for family in evidence["families"].values():
        assert family["effect_bounds"]["certified"] is True
        assert family["effect_bounds"]["component_alpha"] == {
            "numerator": 1,
            "denominator": 320,
        }
        assert len(family["effect_bounds"]["components"]) == 4
        assert family["holm_adjusted_p"] is not None
    assert evidence["pooled"]["effect_bounds"]["certified"] is True


def test_exact_comparison_underpowered_family_has_empty_holm_not_exception():
    grade = _grade({"small": _family("small", count=10)})
    assert grade["tier"] == CONJECTURE
    assert grade["evidence"]["underpowered_families"] == ["small"]
    assert grade["evidence"]["holm"] == {
        "method": "Holm step-down, exact rational",
        "family_size": 0,
        "ordered": [],
    }
    assert (
        grade["evidence"]["families"]["small"]["holm_adjusted_p"] is None
    )


def test_zero_observed_regressions_do_not_fake_powered_noninferiority():
    grade = _grade(
        {
            "alpha": _family(
                "alpha",
                treatment=False,
                control=False,
            ),
            "beta": _family(
                "beta",
                treatment=False,
                control=False,
            ),
        }
    )
    assert grade["tier"] == CONJECTURE
    assert grade["evidence"]["regressed_families"] == []
    assert grade["evidence"]["noninferior_families"] == []
    assert grade["evidence"]["all_families_noninferior"] is False


def test_exact_compute_gate_accepts_boundary_and_rejects_one_unit_over():
    within = _grade(
        {
            "alpha": _family(
                "alpha",
                treatment_compute=120,
                control_compute=100,
            ),
            "beta": _family(
                "beta",
                treatment_compute=120,
                control_compute=100,
            ),
        },
        require_compute=True,
        tolerance=Rational(1, 5),
    )
    assert within["evidence"]["invalid_compute_families"] == []

    outside = _grade(
        {
            "alpha": _family(
                "alpha",
                treatment_compute=121,
                control_compute=100,
            ),
            "beta": _family(
                "beta",
                treatment_compute=121,
                control_compute=100,
            ),
        },
        require_compute=True,
        tolerance=Rational(1, 5),
    )
    assert outside["tier"] == CONJECTURE
    assert outside["evidence"]["invalid_compute_families"] == [
        "alpha",
        "beta",
    ]
    assert len(
        outside["evidence"]["families"]["alpha"][
            "compute_mismatch_task_ids"
        ]
    ) == 20


def test_exact_comparison_refutes_uniform_regression():
    grade = _grade(
        {
            "alpha": _family(
                "alpha",
                treatment=False,
                control=True,
            ),
            "beta": _family(
                "beta",
                treatment=False,
                control=True,
            ),
        }
    )
    assert grade["tier"] == REFUTED
    assert grade["evidence"]["regressed_families"] == ["alpha", "beta"]


def test_exact_interaction_certifies_positive_and_negative_extremes():
    positive = grade_exact_interaction(
        adapter_differences=[1] * 20,
        base_differences=[0] * 20,
        global_bound_family_count=2,
    )
    assert positive["schema"] == EXACT_INTERACTION_SCHEMA
    assert positive["mean"] == {"numerator": 1, "denominator": 1}
    assert exact_interaction_proven(positive) is True
    assert exact_interaction_refuted(positive) is False
    assert positive["one_sided_exact_sign_flip_p"] == {
        "numerator": 1,
        "denominator": 1 << 20,
    }

    negative = grade_exact_interaction(
        adapter_differences=[-1] * 20,
        base_differences=[0] * 20,
        global_bound_family_count=2,
    )
    assert exact_interaction_proven(negative) is False
    assert exact_interaction_refuted(negative) is True


def test_exact_interaction_retains_zero_observations_without_changing_tail():
    interaction = grade_exact_interaction(
        adapter_differences=[1, 0, 1, 0],
        base_differences=[0, 0, 0, 0],
        global_bound_family_count=2,
    )
    assert interaction["interaction_values"] == [1, 0, 1, 0]
    assert interaction["sign_flip"]["observations"] == 4
    assert interaction["one_sided_exact_sign_flip_p"] == {
        "numerator": 1,
        "denominator": 4,
    }


def test_exact_interaction_supports_powered_campaigns_above_old_64_task_cap():
    interaction = grade_exact_interaction(
        adapter_differences=[1] * 80,
        base_differences=[0] * 80,
        global_bound_family_count=2,
    )
    assert interaction["n"] == 80
    assert interaction["sign_flip"]["observations"] == 80
    assert exact_interaction_proven(interaction) is True


def test_noninferiority_power_certificate_proves_exact_minimum():
    power = minimum_zero_loss_noninferiority_observations(
        global_bound_family_count=50,
    )
    assert power["certified"] is True
    assert 300 < power["minimum_observations"] < 600
    assert power["selected_lower"]["numerator"] * 50 > (
        -power["selected_lower"]["denominator"]
    )
    assert power["prior_lower"]["numerator"] * 50 <= (
        -power["prior_lower"]["denominator"]
    )


def test_full_campaign_exact_power_boundary_is_410_fail_411_pass():
    underpowered = exact_campaign_power_plan(
        domain_count=7,
        comparison_count=6,
        arm_count=6,
        planned_observations_per_domain=410,
    )
    powered = exact_campaign_power_plan(
        domain_count=7,
        comparison_count=6,
        arm_count=6,
        planned_observations_per_domain=411,
    )

    assert underpowered["minimum_observations"] == 411
    assert underpowered["powered_for_zero_loss_noninferiority"] is False
    assert powered["powered_for_zero_loss_noninferiority"] is True
    assert powered["planned_total_tasks"] == 2_877
    assert powered["planned_total_cells"] == 17_262


def test_group_sequential_power_plan_conserves_alpha_and_freezes_each_look():
    plan = exact_group_sequential_power_plan(
        domain_count=7,
        comparison_count=6,
        arm_count=6,
        look_observations_per_domain=(160, 320, 480, 640),
        alpha_weights=(
            Rational(1, 100),
            Rational(4, 100),
            Rational(15, 100),
            Rational(80, 100),
        ),
    )

    assert plan["alpha_weight_sum"] == {"numerator": 1, "denominator": 1}
    assert plan["familywise_alpha"] == {"numerator": 1, "denominator": 20}
    assert [look["look"] for look in plan["looks"]] == [1, 2, 3, 4]
    assert [look["observations_per_domain"] for look in plan["looks"]] == [
        160,
        320,
        480,
        640,
    ]
    assert plan["looks"][0]["family_alpha"] == {
        "numerator": 1,
        "denominator": 2_000,
    }
    assert plan["looks"][-1]["family_alpha"] == {
        "numerator": 1,
        "denominator": 25,
    }
    assert plan["terminal_fixed_design"]["minimum_observations"] == 411


@pytest.mark.parametrize(
    ("looks", "weights", "error"),
    [
        ((100, 100), (Rational(1, 2), Rational(1, 2)), "contract_invalid"),
        ((100, 200), (Rational(1, 2), Rational(1, 3)), "alpha_not_conserved"),
        ((100,), (Rational(0, 1),), "contract_invalid"),
    ],
)
def test_group_sequential_power_plan_rejects_ambiguous_or_unfunded_looks(
    looks,
    weights,
    error,
):
    with pytest.raises(ExactPairedGradeError, match=error):
        exact_group_sequential_power_plan(
            domain_count=7,
            comparison_count=6,
            arm_count=6,
            look_observations_per_domain=looks,
            alpha_weights=weights,
        )


@pytest.mark.parametrize(
    "call",
    [
        lambda: _grade({}),
        lambda: _grade({"x": []}),
        lambda: grade_exact_interaction(
            adapter_differences=[],
            base_differences=[],
            global_bound_family_count=2,
        ),
        lambda: grade_exact_interaction(
            adapter_differences=[2],
            base_differences=[0],
            global_bound_family_count=2,
        ),
        lambda: grade_exact_interaction(
            adapter_differences=[1],
            base_differences=[0, 1],
            global_bound_family_count=2,
        ),
    ],
)
def test_exact_grade_malformed_inputs_fail_closed(call):
    with pytest.raises(ExactPairedGradeError):
        call()
