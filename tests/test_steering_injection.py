"""Steering injection must actually modify hidden states when active.

The original live A/B runner assigned ``layer.__call__`` on the instance,
which Python's special-method lookup bypasses entirely — its "steered"
condition never injected anything. These tests pin the working mechanism.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from core.evaluation.steering_injection import (  # noqa: E402 — after importorskip
    ResidualSteeringInjector,
    load_production_vectors,
)


class _Layer:
    def __call__(self, h, *args, **kwargs):
        return h * 1.0  # identity-ish transform


class _Inner:
    def __init__(self, n_layers: int):
        self.layers = [_Layer() for _ in range(n_layers)]


class _Model:
    def __init__(self, n_layers: int = 4):
        self.model = _Inner(n_layers)

    def forward_through(self, h):
        for layer in self.model.layers:
            h = layer(h)
        return h


def test_injection_changes_hidden_state_only_when_active():
    model = _Model(n_layers=4)
    vec = np.zeros(8, dtype=np.float32)
    vec[0] = 1.0
    injector = ResidualSteeringInjector(model, {2: vec}, alpha=5.0)
    h = mx.ones((1, 3, 8))

    baseline = model.forward_through(h)
    with injector:
        injector.active = False
        unsteered = model.forward_through(h)
        injector.active = True
        steered = model.forward_through(h)
    restored = model.forward_through(h)

    assert bool(mx.allclose(baseline, unsteered).item())
    assert not bool(mx.allclose(baseline, steered).item()), (
        "active injection must change the hidden state — the instance "
        "__call__ assignment bug produced exactly this failure"
    )
    # Injection adds alpha on the steered axis at the hooked layer.
    delta = np.array(steered - baseline)
    assert delta[..., 0].max() == pytest.approx(5.0, rel=1e-3)
    assert np.abs(delta[..., 1:]).max() == pytest.approx(0.0, abs=1e-6)
    assert injector.injection_count > 0
    # Hooks removed: model behaves like baseline again.
    assert bool(mx.allclose(baseline, restored).item())


def test_calling_convention_actually_intercepts():
    """Guard against regressions to instance-attribute patching."""
    model = _Model(n_layers=2)
    vec = np.ones(4, dtype=np.float32)
    injector = ResidualSteeringInjector(model, {0: vec}, alpha=1.0)
    installed = injector.install()
    try:
        injector.active = True
        layer = model.model.layers[0]
        out = layer(mx.zeros((1, 1, 4)))
        assert float(np.array(out).sum()) != 0.0, (
            "layer(...) did not route through the injection subclass"
        )
    finally:
        injector.remove()
    assert installed == 1


def test_load_production_vectors_filters_and_normalizes(tmp_path):
    def _write(name, dimension, layer, vec, extracted=True):
        np.savez(
            tmp_path / name,
            v=vec.astype(np.float32),
            dimension=np.array(dimension),
            layer=np.array(layer),
            extracted=np.array(extracted),
        )

    _write("a.npz", "valence_positive", 5, np.array([3.0, 0.0, 0.0, 0.0]))
    _write("b.npz", "curiosity", 5, np.array([0.0, 4.0, 0.0, 0.0]))
    _write("c.npz", "valence_positive", 7, np.array([0.0, 0.0, 9.0, 0.0]))
    _write("d.npz", "frustration", 5, np.array([1.0, 1.0, 1.0, 1.0]))  # not requested
    _write("e.npz", "valence_positive", 9, np.array([2.0, 0.0, 0.0, 0.0]), extracted=False)

    vectors = load_production_vectors(tmp_path)

    assert set(vectors) == {5, 7}, "bootstrap and unrequested dimensions must be excluded"
    for vec in vectors.values():
        assert np.linalg.norm(vec) == pytest.approx(1.0, rel=1e-5)
    # Layer 5 averages the two unit axes then renormalizes.
    assert vectors[5][0] == pytest.approx(vectors[5][1], rel=1e-5)


def _validator_credits(artifact: dict) -> bool:
    """Would the CAA readiness chain accept this artifact as behavioral evidence?"""
    from training.caa_32b_validation import CAA32BValidator

    normalized = CAA32BValidator._normalize_behavioral_results(artifact)
    return bool(normalized.get("black_box_prompt_hygiene_passed", False))


def _live_ab_artifact(**overrides) -> dict:
    base = {
        "model": "models/Qwen2.5-32B-Instruct-4bit",
        "n_trials": 30,
        "held_out_tasks": ["a", "b", "c", "d", "e", "f"],
        "passes_adversarial_control": True,
        "sampling": {"temperature": 0.7, "top_p": 0.95, "paired_seeds": True},
        "injection_count": 480,
        "analysis": {
            "n_trials": 30,
            "steered_vs_rich": {"effect_size_d": 0.4, "observed_delta": 0.2},
            "steered_vs_terse": {"effect_size_d": 0.5, "observed_delta": 0.25},
            "steered_vs_baseline_mean_distance": 0.3,
            "passes_adversarial_control": True,
        },
    }
    base.update(overrides)
    return base


def test_validator_refuses_legacy_uninjected_artifact():
    """The pre-rebuild artifact (no sampling metadata, no injection count)
    came from a runner whose 'steered' condition never injected. The
    readiness chain must never credit it as behavioral evidence."""
    legacy = _live_ab_artifact()
    del legacy["sampling"]
    del legacy["injection_count"]
    assert not _validator_credits(legacy), (
        "theater artifact credited as behavioral evidence"
    )


def test_validator_refuses_artifact_whose_injection_never_fired():
    assert not _validator_credits(_live_ab_artifact(injection_count=0))


def test_validator_refuses_greedy_artifact():
    greedy = _live_ab_artifact()
    greedy["sampling"] = {"temperature": 0.0, "top_p": 1.0}
    assert not _validator_credits(greedy)


def test_validator_credits_injected_sampled_artifact():
    assert _validator_credits(_live_ab_artifact())


def test_committed_artifact_never_credited_without_injection_provenance():
    """Whatever artifact is committed right now: if it lacks injection
    provenance, the validator must classify it as non-evidence."""
    from pathlib import Path

    artifact_path = Path(__file__).resolve().parent / "CAA_32B_AB_LIVE_RESULTS.json"
    if not artifact_path.exists():
        return  # nothing committed; the synthetic cases above pin the logic
    data = json.loads(artifact_path.read_text(encoding="utf-8"))
    sampling = data.get("sampling") or {}
    injected = int(data.get("injection_count", 0) or 0) > 0
    sampled = float(sampling.get("temperature", 0.0) or 0.0) > 0.0
    if injected and sampled:
        steered = ((data.get("analysis") or {}).get("samples") or {}).get(
            "steered_black_box", []
        )
        assert len(set(steered)) > 1, "steered samples identical — greedy collapse"
    else:
        assert not _validator_credits(data), (
            "committed artifact lacks injection provenance but is still "
            "credited — regenerate via tests/run_32b_steering_ab_live.py"
        )
