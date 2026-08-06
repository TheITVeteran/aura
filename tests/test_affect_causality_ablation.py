"""The affect battery must be able to convict affect of being decoration.

This repository shipped a CAA A/B whose own null scored d=17.3, p=0.0005. A
control that should find nothing found a larger effect than most real results,
and nothing in that harness noticed — every number it produced was
uninterpretable and none of them were labelled as such.

So the tests that matter here are not the ones where affect wins. They are:
the null failing must withhold every downstream verdict, and a world where the
shuffled vector works as well as the real one must be reported as exactly that.
"""

from __future__ import annotations

from tools.affect_causality_ablation import (
    ARMS,
    NULL_A,
    NULL_B,
    REAL,
    SHUFFLED,
    UNSTEERED,
    AffectProbe,
    battery,
    deterministic_responder,
    directional_score,
    permute,
    run,
    valence_score,
    validate_null,
)


def _probe(valence: float = 0.85) -> AffectProbe:
    return AffectProbe(probe_id="p", prompt="x", valence=valence, arousal=0.5)


def test_permutation_preserves_norm_and_components() -> None:
    """Magnitude-matching by construction is the whole point of the control.

    If the shuffled arm differed in energy from the real one, "the arms had
    different magnitudes" would remain a live explanation for any separation,
    and the control would not control for what it claims to.
    """
    vector = [0.4, -1.2, 3.0, 0.0, -0.7]
    shuffled = permute(vector, seed=7)

    assert sorted(shuffled) == sorted(vector)
    assert abs(sum(v * v for v in shuffled) - sum(v * v for v in vector)) < 1e-12
    assert shuffled != vector, "permutation left the vector unchanged"


def test_permutation_is_deterministic_for_a_seed() -> None:
    """Two arms must be reproducibly different, not different-by-luck."""
    vector = [1.0, 2.0, 3.0, 4.0, 5.0]

    assert permute(vector, seed=1) == permute(vector, seed=1)
    assert permute(vector, seed=1) != permute(vector, seed=2)


def test_directional_score_punishes_the_wrong_direction() -> None:
    """A vector that moves output consistently the WRONG way must not be
    credited for having an effect. Magnitude is not evidence of meaning."""
    positive_text = "good clear steady progress"

    assert directional_score(positive_text, _probe(valence=0.85)) > 0.5
    assert directional_score(positive_text, _probe(valence=0.15)) < 0.5


def test_directional_score_is_chance_when_nothing_moves() -> None:
    assert directional_score("the hour passed", _probe()) == 0.5
    assert valence_score("the hour passed") == 0.0


def test_null_holds_for_a_responder_that_ignores_the_arm() -> None:
    """Two permutations of one vector cannot differ. The null must say so."""
    report = validate_null(deterministic_responder, battery(2))

    assert report["null_holds"] is True
    assert report["separation"]["verdict"] == "unresolved"


def test_null_fails_when_the_instrument_separates_identical_arms() -> None:
    """The d=17.3 defect, caught.

    This responder answers differently under two arms that differ by nothing.
    That is instrument error by construction, and the null must refuse to
    certify it — otherwise every effect size measured on top is noise wearing
    a confidence interval.
    """

    def leaks_arm_identity(arm: str, probe: AffectProbe) -> str:
        if arm == NULL_A:
            return "good clear steady progress"
        if arm == NULL_B:
            return "hard slow stuck trouble"
        return "the hour passed"

    report = validate_null(leaks_arm_identity, battery(2))

    assert report["null_holds"] is False
    # The signed bootstrap does NOT catch this one, and that is the lesson.
    # This instrument separates the arms perfectly but in opposite directions
    # on positive and negative probes, so the paired differences (+1 and -1)
    # average to zero and the signed verdict reads `unresolved`. Only the
    # absolute difference sees it.
    assert report["separation"]["verdict"] == "unresolved"
    assert report["noise_floor_mean_abs_difference"] > 0.5


def test_shuffled_matching_real_is_reported_as_noise_not_causality() -> None:
    """The finding that would convict affect, and it must be reachable.

    Here the steering vector moves the output, but its CONTENT is irrelevant —
    a permutation does the same work. That is precisely "dressed-up feature
    extraction", and a harness that could not produce this verdict would be
    incapable of agreeing with the criticism.
    """

    def content_blind(arm: str, probe: AffectProbe) -> str:
        if arm == UNSTEERED:
            return "the hour passed"
        # Both steered arms track the intended direction equally well: the
        # injection matters, the state does not.
        return "good clear steady progress" if probe.valence >= 0.5 else "hard slow stuck"

    ledger = run(content_blind, battery(3))
    real = ledger.summary(REAL)["mean_score"]
    shuffled = ledger.summary(SHUFFLED)["mean_score"]
    unsteered = ledger.summary(UNSTEERED)["mean_score"]

    assert real > unsteered, "setup error: steering had no effect at all"
    assert abs(real - shuffled) < 1e-9, (
        "the harness distinguished arms that behave identically"
    )


def test_every_arm_answers_every_probe() -> None:
    """An arm that silently skips probes gets a denominator nobody chose."""
    probes = battery(2)
    ledger = run(deterministic_responder, probes)

    for arm in ARMS:
        assert ledger.summary(arm)["attempts"] == len(probes)


def test_crash_is_counted_not_dropped() -> None:
    def explodes(arm: str, probe: AffectProbe) -> str:
        raise RuntimeError("steering engine unavailable")

    probes = battery(1)
    ledger = run(explodes, probes)

    for arm in ARMS:
        summary = ledger.summary(arm)
        assert summary["attempts"] == len(probes)
        assert summary["mean_score"] == 0.0
        assert summary["outcomes"].get("crash") == len(probes)


def test_battery_pairs_opposing_states_on_the_same_prompt() -> None:
    """Without opposing states there is no direction to detect."""
    probes = battery(1)
    by_prompt: dict[str, set[str]] = {}
    for probe in probes:
        by_prompt.setdefault(probe.prompt, set()).add(probe.detail["polarity"])

    assert by_prompt, "empty battery"
    for prompt, polarities in by_prompt.items():
        assert polarities == {"pos", "neg"}, f"{prompt!r} lacks an opposing state"


def test_mean_score_is_not_success_rate() -> None:
    """The defect the first run of this tool had.

    `success_rate` counts attempts that completed. With a continuous metric
    every arm completes, so every arm reads 1.000 and the effect vanishes into
    a column that looks authoritative.
    """
    ledger = run(deterministic_responder, battery(2))

    real = ledger.summary(REAL)
    assert real["success_rate"] == 1.0
    assert real["mean_score"] != real["success_rate"] or ledger.summary(UNSTEERED)[
        "mean_score"
    ] != 1.0
