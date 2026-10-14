"""What it takes to let her substrate steer a person's turn.

The channel these tests guard is the one place her state reaches the model's
computation rather than its prompt. It is fail-closed by construction, and these
say so in each of the four ways it can be refused.
"""

from __future__ import annotations

import json

import pytest

from core.consciousness.fusion_certificate import (
    FUSION_MEASUREMENT_PROTOCOL,
    MIN_DISTRIBUTION_SHIFT,
    MIN_STATE_SEPARATION,
    FusionCertificate,
    certificate_for,
    certified_alpha,
    load_certificates,
    symmetric_kl,
    write_certificate,
)


def _certificate(**overrides) -> FusionCertificate:
    """A certificate that holds, so each test can break exactly one thing."""
    fields = {
        "measurement_protocol": FUSION_MEASUREMENT_PROTOCOL,
        "model_identity": "a" * 64,
        "model_name": "test-model",
        "alpha": 0.2,
        "distribution_shift": 0.29,
        "control_shift": 0.26,
        "state_separation": 1.98,
        "prompts_that_change": 8,
        "median_step_of_change": 5,
        "quality_delta": 0.0,
        "control_quality_delta": 0.0,
        "margin_delta": 0.009,
        "control_margin_delta": -0.006,
        "quality_scale": "forced-choice accuracy over 10 items",
        "prompts": 8,
        "steps": 24,
        "voice_delta": 0.004,
        "voice_replies": 12,
    }
    fields.update(overrides)
    return FusionCertificate(**fields)


def test_a_full_certificate_holds():
    certificate = _certificate()
    assert certificate.holds
    assert certificate.why_not() == ""


def test_old_measurement_is_readable_but_cannot_authorize_current_serving(tmp_path):
    old = _certificate(measurement_protocol="", basis_sha256="b" * 64)
    path = write_certificate(old, root=tmp_path)
    retained = path.read_bytes()
    assert certificate_for(old.model_identity, basis_sha256=old.basis_sha256, root=tmp_path) == old
    assert certified_alpha(old.model_identity, basis_sha256=old.basis_sha256, root=tmp_path) == 0.0
    current = _certificate(basis_sha256=old.basis_sha256)
    current_path = write_certificate(current, root=tmp_path)
    assert current_path != path and path.read_bytes() == retained
    assert certified_alpha(old.model_identity, basis_sha256=old.basis_sha256, root=tmp_path) == 0.2


def test_unchanged_answers_refuse_it():
    """The void A/B's own observation, kept as a refusal rather than a verdict."""
    certificate = _certificate(prompts_that_change=2)
    assert not certificate.arrives
    assert not certificate.holds
    assert "2 of 8 probes" in certificate.why_not()


def test_a_shift_too_small_to_matter_refuses_it():
    certificate = _certificate(distribution_shift=MIN_DISTRIBUTION_SHIFT / 2)
    assert not certificate.arrives
    assert not certificate.holds


def test_a_direction_worse_than_an_arbitrary_one_refuses_it():
    certificate = _certificate(margin_delta=-0.05, control_margin_delta=-0.01)
    assert certificate.arrives
    assert not certificate.beats_noise
    assert not certificate.holds
    assert "arbitrary displacement" in certificate.why_not()


@pytest.mark.parametrize("treatment,control", [(-0.05, -0.01), (0.0406, 0.0531), (-0.01, 0.02)])
def test_margin_diagnostic_preserves_improvement_and_regression_signs(treatment, control):
    certificate = _certificate(margin_delta=treatment, control_margin_delta=control)
    assert not certificate.beats_noise and not certificate.holds
    assert f"{treatment:+.4f}" in certificate.why_not()
    assert f"{control:+.4f}" in certificate.why_not()
    assert "costs" not in certificate.why_not()


def test_a_channel_that_carries_no_state_refuses_it():
    """A constant offset moves the model. It does not carry what she feels."""
    certificate = _certificate(state_separation=MIN_STATE_SEPARATION / 2)
    assert certificate.arrives
    assert certificate.beats_noise
    assert not certificate.carries_content
    assert not certificate.holds
    assert "constant offset" in certificate.why_not()


def test_a_channel_that_costs_answers_refuses_it():
    certificate = _certificate(quality_delta=-0.1)
    assert not certificate.costs_nothing
    assert not certificate.holds
    assert "cannot cost answers" in certificate.why_not()


def test_the_first_missing_thing_is_the_one_reported():
    certificate = _certificate(prompts_that_change=0, quality_delta=-0.5)
    assert "probes" in certificate.why_not()


def test_no_certificate_means_no_steering(tmp_path):
    assert certified_alpha("nothing-was-ever-measured-here", root=tmp_path) == 0.0
    assert certificate_for("nothing-was-ever-measured-here", root=tmp_path) is None


def test_a_failing_certificate_means_no_steering(tmp_path):
    certificate = _certificate(prompts_that_change=0)
    write_certificate(certificate, root=tmp_path)
    assert certified_alpha(certificate.model_identity, root=tmp_path) == 0.0


def test_a_holding_certificate_opens_the_channel(tmp_path):
    certificate = _certificate(alpha=0.15)
    write_certificate(certificate, root=tmp_path)
    assert certified_alpha(certificate.model_identity, root=tmp_path) == pytest.approx(0.15)


def test_a_certificate_is_bound_to_one_checkpoint(tmp_path):
    """Evidence earned by one set of weights says nothing about another."""
    certificate = _certificate(model_identity="a" * 64)
    write_certificate(certificate, root=tmp_path)
    assert certified_alpha("b" * 64, root=tmp_path) == 0.0


def test_an_unreadable_certificate_means_no_steering(tmp_path):
    directory = tmp_path / "artifacts" / "fusion"
    directory.mkdir(parents=True)
    (directory / "broken.json").write_text("{not json", encoding="utf-8")
    assert load_certificates(root=tmp_path) == {}


def test_a_round_trip_keeps_the_verdict(tmp_path):
    certificate = _certificate()
    path = write_certificate(certificate, root=tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["holds"] is True
    assert payload["beats_noise"] is True
    reloaded = load_certificates(root=tmp_path)[certificate.model_identity]
    assert reloaded.holds
    assert reloaded.alpha == pytest.approx(certificate.alpha)


def test_alpha_is_clamped_to_the_unit_interval(tmp_path):
    write_certificate(_certificate(alpha=7.0), root=tmp_path)
    assert certified_alpha("a" * 64, root=tmp_path) == 1.0


def test_symmetric_kl_is_zero_for_identical_distributions():
    assert symmetric_kl([0.25, 0.25, 0.5], [0.25, 0.25, 0.5]) == pytest.approx(0.0)


def test_symmetric_kl_is_symmetric():
    left = [0.7, 0.2, 0.1]
    right = [0.1, 0.3, 0.6]
    assert symmetric_kl(left, right) == pytest.approx(symmetric_kl(right, left))


def test_the_certificates_on_disk_were_measured_and_none_holds_without_her_voice():
    """What was measured stays a fact in the repo, and it is not yet enough.

    Each record on disk met the first four conditions. None carries a reading
    of her own replies under the steering, so none holds: on 13 October 2026
    the channel was open on those four and a reply came out in a voice that
    was not hers.
    """
    certificates = list(load_certificates().values())
    assert certificates, "no fusion certificate is on disk"
    measured = [c for c in certificates if c.arrives and c.carries_content and c.costs_nothing]
    assert measured, "no record on disk met the first four conditions"
    for certificate in certificates:
        if certificate.voice_delta is None:
            assert not certificate.holds
            assert "have not been scored" in certificate.why_not() or not (
                certificate.arrives and certificate.beats_noise
                and certificate.carries_content and certificate.costs_nothing
            )


def test_a_certificate_with_no_reading_of_her_voice_does_not_hold():
    certificate = _certificate(voice_delta=None, voice_replies=0)
    assert not certificate.holds and not certificate.keeps_her_voice
    assert "have not been scored" in certificate.why_not()


def test_a_direction_that_makes_her_own_words_less_likely_does_not_hold():
    certificate = _certificate(voice_delta=-0.31)
    assert not certificate.holds
    assert "less likely by 0.3100 nats a token" in certificate.why_not()


def test_certificates_are_found_from_the_repository_not_the_working_directory(
    tmp_path, monkeypatch
):
    """The worker is a forked child of a desktop app and its cwd is not this repo.

    A relative path would write the certificate somewhere and look for it
    somewhere else, and the channel would stay shut with no error anywhere.
    """
    from core.consciousness import fusion_certificate

    home = tmp_path / "repo"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setattr(fusion_certificate, "_REPO_ROOT", home)
    monkeypatch.chdir(elsewhere)

    write_certificate(_certificate(model_identity="9" * 64))
    assert certified_alpha("9" * 64) > 0.0
    assert (home / "artifacts" / "fusion").exists()
    assert not (elsewhere / "artifacts").exists()


def test_the_repository_root_holds_the_package():
    """A wrong number of parents would point at a directory that always reads empty."""
    from core.consciousness import fusion_certificate

    assert (fusion_certificate._REPO_ROOT / "core" / "consciousness").is_dir()


def test_a_new_basis_keeps_historical_evidence_but_cannot_inherit_it(tmp_path):
    old = _certificate()
    historical = write_certificate(old, root=tmp_path)
    original = historical.read_bytes()
    assert certificate_for(old.model_identity, basis_sha256="b" * 64, root=tmp_path) is None
    first = _certificate(basis_sha256="b" * 64)
    second = _certificate(basis_sha256="c" * 64, quality_delta=-0.2)
    first_path = write_certificate(first, root=tmp_path)
    second_path = write_certificate(second, root=tmp_path)
    assert first_path != second_path != historical
    assert historical.read_bytes() == original
    assert certified_alpha(old.model_identity, basis_sha256="b" * 64, root=tmp_path) == 0.2
    assert certified_alpha(old.model_identity, basis_sha256="c" * 64, root=tmp_path) == 0.0


def test_real_hook_basis_covers_layer_bytes_and_substrate_mapping():
    import numpy as np

    from core.consciousness.affective_steering import AffectiveSteeringHook, SteeringVector
    from core.consciousness.fusion_certificate import steering_basis_sha256

    vector = SteeringVector(key="energy", layer_idx=1, d_model=2,
                            v=np.array([1.0, 0.0]), substrate_idx=2, substrate_fn="linear")
    hook = AffectiveSteeringHook(object(), 1, {"energy": vector})
    original = steering_basis_sha256([hook])
    hook._alpha = 0.8
    assert steering_basis_sha256([hook]) == original
    hook._layer_idx = 2
    assert steering_basis_sha256([hook]) != original
    hook._layer_idx = 1
    vector.v[1] = 1.0
    assert steering_basis_sha256([hook]) != original
    vector.v[1] = 0.0
    vector.substrate_idx = 3
    assert steering_basis_sha256([hook]) != original
    with pytest.raises(ValueError, match="duplicate_layer"):
        steering_basis_sha256([hook, hook])
