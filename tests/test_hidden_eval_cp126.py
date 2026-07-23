"""CP126 contract tests for the sealed hidden evaluation suite."""
from __future__ import annotations

import json

import pytest

from core.architect import hidden_eval as module
from core.architect.hidden_eval import (
    PHI_POSITIVE_FLOOR,
    EvalScenario,
    HiddenEvalRunner,
    ProbeUnavailable,
    callable_fingerprint,
)


def _scenario(evaluate, *, scenario_id="s1", low=0.0, high=1.0, **kwargs) -> EvalScenario:
    return EvalScenario(
        scenario_id=scenario_id,
        name="Scenario",
        description="A sealed scenario",
        scenario_type="Test",
        expected_range=(low, high),
        evaluate=evaluate,
        **kwargs,
    )


def _runner(tmp_path, **kwargs) -> HiddenEvalRunner:
    return HiddenEvalRunner(data_dir=tmp_path, **kwargs)


# --- 878fed6e: the seal covers the evaluator ------------------------------


def test_replacing_the_evaluator_breaks_the_seal():
    scenario = _scenario(lambda: 0.5)
    assert scenario.verify_integrity() is True

    scenario.evaluate = lambda: 0.9

    assert scenario.verify_integrity() is False


def test_the_seal_covers_the_declared_range():
    scenario = _scenario(lambda: 0.5)
    scenario.expected_range = (0.0, 99.0)

    assert scenario.verify_integrity() is False


def test_the_seal_covers_fixtures():
    scenario = _scenario(lambda: 0.5, fixtures=("dataset-v1",))
    assert scenario.verify_integrity() is True

    scenario.fixtures = ("dataset-v2",)

    assert scenario.verify_integrity() is False


def test_the_seal_covers_the_code_version():
    scenario = _scenario(lambda: 0.5)
    scenario.code_version = "99"

    assert scenario.verify_integrity() is False


def test_an_external_key_makes_the_seal_unforgeable(monkeypatch):
    monkeypatch.setenv(module._SEAL_KEY_ENV, "operator-key")
    sealed = _scenario(lambda: 0.5)

    # An attacker who rewrites the metadata cannot recompute the seal without
    # the key, so verification still fails once the key is restored.
    sealed.description = "tampered"
    monkeypatch.delenv(module._SEAL_KEY_ENV)
    forged = module._sha256(sealed.seal_material())
    sealed.content_hash = forged
    monkeypatch.setenv(module._SEAL_KEY_ENV, "operator-key")

    assert sealed.verify_integrity() is False


def test_the_fingerprint_distinguishes_different_callables():
    def one():
        return 1.0

    def two():
        return 2.0

    assert callable_fingerprint(one) != callable_fingerprint(two)
    assert callable_fingerprint(one) == callable_fingerprint(one)


def test_a_tampered_scenario_is_counted_not_passed(tmp_path):
    runner = _runner(tmp_path)
    scenario = _scenario(lambda: 0.5)
    runner.register_scenario(scenario)
    scenario.evaluate = lambda: 0.5  # a different object with the same body

    result = runner.run_suite()

    assert result.tampered == 1
    assert result.passed == 0
    assert result.results[0].integrity_verified is False


# --- 9ed8db9b: unavailable probes are not passing --------------------------


def test_an_unavailable_probe_does_not_pass(tmp_path):
    runner = _runner(tmp_path)

    def unavailable():
        raise ProbeUnavailable("subsystem is not booted")

    runner.register_scenario(_scenario(unavailable))

    result = runner.run_suite()

    assert result.unavailable == 1
    assert result.passed == 0
    assert result.overall_health == 0.0
    assert result.results[0].available is False
    assert "not booted" in result.results[0].unavailable_reason


def test_a_dependency_failure_does_not_read_as_zero(tmp_path):
    runner = _runner(tmp_path)
    runner.register_scenario(
        _scenario(lambda: (_ for _ in ()).throw(ImportError("no module")), low=0.0, high=10.0)
    )

    result = runner.run_suite()

    # Under the old behaviour this returned 0.0, which is inside (0, 10).
    assert result.passed == 0
    assert result.results[0].available is False


def test_a_wholly_unavailable_suite_is_not_healthy(tmp_path):
    runner = _runner(tmp_path)
    for index in range(3):
        runner.register_scenario(
            _scenario(
                lambda: (_ for _ in ()).throw(ProbeUnavailable("down")),
                scenario_id=f"s{index}",
            )
        )

    result = runner.run_suite()

    assert result.overall_health == 0.0
    assert result.unavailable == 3


def test_a_working_probe_still_passes(tmp_path):
    runner = _runner(tmp_path)
    runner.register_scenario(_scenario(lambda: 0.5))

    result = runner.run_suite()

    assert result.passed == 1
    assert result.overall_health == 1.0
    assert result.results[0].available is True


def test_a_non_finite_measurement_fails(tmp_path):
    runner = _runner(tmp_path)
    runner.register_scenario(_scenario(lambda: float("inf")))

    assert runner.run_suite().passed == 0


# --- 0b4faa5b: the phi discriminator is true at its boundary --------------


def test_the_phi_scenario_rejects_zero(tmp_path):
    runner = HiddenEvalRunner.create_default_suite(data_dir=tmp_path)
    phi = next(s for s in runner._scenarios.values() if s.scenario_id == "phi_positive")

    assert phi.expected_range[0] == PHI_POSITIVE_FLOOR
    assert phi.expected_range[0] > 0.0


def test_the_default_suite_seals_cleanly(tmp_path):
    runner = HiddenEvalRunner.create_default_suite(data_dir=tmp_path)

    assert all(scenario.verify_integrity() for scenario in runner._scenarios.values())


# --- 131ab382: probes read the live runtime -------------------------------


def test_substrate_probe_refuses_without_a_live_substrate(monkeypatch):
    monkeypatch.setattr(module, "_live_service", lambda *names: None)

    with pytest.raises(ProbeUnavailable, match="continuous_substrate"):
        module._probe_substrate_energy()


def test_substrate_probe_reads_the_live_service(monkeypatch):
    import numpy as np

    class Live:
        def get_state_vector(self):
            return np.array([3.0, 4.0])

    monkeypatch.setattr(module, "_live_service", lambda *names: Live())

    assert module._probe_substrate_energy() == pytest.approx(5.0 / np.sqrt(2))


def test_world_model_probe_refuses_without_a_live_model(monkeypatch):
    monkeypatch.setattr(module, "_live_service", lambda *names: None)

    with pytest.raises(ProbeUnavailable, match="world model"):
        module._probe_world_model_surprise()


def test_world_model_probe_reads_the_live_model(monkeypatch):
    class Live:
        def get_mean_surprise(self):
            return 1.25

    monkeypatch.setattr(module, "_live_service", lambda *names: Live())

    assert module._probe_world_model_surprise() == pytest.approx(1.25)


def test_phi_probe_refuses_without_a_live_service(monkeypatch):
    monkeypatch.setattr(module, "_live_service", lambda *names: None)

    with pytest.raises(ProbeUnavailable, match="phi"):
        module._probe_phi_value()


def test_phi_probe_reads_the_live_service(monkeypatch):
    class Live:
        def latest_phi(self):
            return {"phi": 0.42}

    monkeypatch.setattr(module, "_live_service", lambda *names: Live())

    assert module._probe_phi_value() == pytest.approx(0.42)


def test_value_drift_probe_refuses_an_empty_graph(monkeypatch):
    monkeypatch.setattr(
        "core.adaptation.dynamic_value_graph.get_dynamic_value_graph",
        lambda: type("G", (), {"_nodes": {}})(),
    )

    with pytest.raises(ProbeUnavailable, match="no nodes|holds no nodes"):
        module._probe_value_drift()


# --- a400fb60: history survives a restart ---------------------------------


def test_history_is_restored_from_the_audit_chain(tmp_path):
    first = _runner(tmp_path)
    first.register_scenario(_scenario(lambda: 0.5))
    first.run_suite()
    first.run_suite()

    second = _runner(tmp_path)

    assert second.get_status()["history_length"] == 2
    assert second.get_status()["run_count"] == 2
    assert second.get_status()["latest_health"] == 1.0


def test_restored_history_enables_drift_detection(tmp_path):
    seeded = _runner(tmp_path, drift_window=2, drift_threshold=0.2)
    seeded.register_scenario(_scenario(lambda: 0.5))
    seeded.run_suite()
    seeded.run_suite()

    fresh = _runner(tmp_path, drift_window=2, drift_threshold=0.2)
    fresh.register_scenario(_scenario(lambda: 5.0))  # now outside the range

    result = fresh.run_suite()

    assert result.passed == 0
    assert result.drift_detected is True


def test_corrupt_history_lines_are_skipped(tmp_path):
    (tmp_path / "eval_history.jsonl").write_text(
        '{"total": 1, "results": []}\nnot json\n{"no_results_key": true}\n'
    )

    runner = _runner(tmp_path)

    assert runner.get_status()["history_length"] == 1


# --- ca19059e: a write failure is visible ---------------------------------


def test_a_persistence_failure_is_reported(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    runner.register_scenario(_scenario(lambda: 0.5))
    monkeypatch.setattr(
        runner, "_log_result", lambda result: (False, "OSError: disk full")
    )

    result = runner.run_suite()

    assert result.durable is False
    assert "disk full" in result.persistence_error
    assert result.to_dict()["durable"] is False


def test_a_successful_run_is_marked_durable(tmp_path):
    runner = _runner(tmp_path)
    runner.register_scenario(_scenario(lambda: 0.5))

    result = runner.run_suite()

    assert result.durable is True
    assert result.persistence_error == ""
    lines = (tmp_path / "eval_history.jsonl").read_text().strip().splitlines()
    assert json.loads(lines[0])["durable"] is True


def test_a_persistence_failure_raises_a_degradation(tmp_path, monkeypatch):
    recorded = []
    monkeypatch.setattr(
        "core.runtime.errors.record_degradation",
        lambda *args, **kwargs: recorded.append(args),
    )
    runner = _runner(tmp_path)
    runner.register_scenario(_scenario(lambda: 0.5))
    monkeypatch.setattr(runner, "_log_result", lambda result: (False, "boom"))

    runner.run_suite()

    assert recorded and recorded[0][0] == "hidden_eval"
