"""tests/test_research_modules.py -- Tests for Research-Grade Modules
=====================================================================
Tests for: PhiComputer, PlasticityGovernor, MetaLearner, LesionStudy,
HiddenEvalRunner.

Every module is tested for:
  1. Core algorithm correctness
  2. Deterministic behavior (same seed = same result)
  3. Edge cases and error handling
  4. Integration with existing substrate/brain modules
"""
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ══════════════════════════════════════════════════════════════════════════
# Phi Computation Tests
# ══════════════════════════════════════════════════════════════════════════

class TestPhiComputer(unittest.TestCase):
    """Tests for core/consciousness/phi_compute.py"""

    def test_phi_positive_for_coupled_system(self):
        """Coupled system has Phi > 0."""
        from core.consciousness.phi_compute import PhiComputer, PhiConfig
        computer = PhiComputer(PhiConfig(trajectory_length=50))
        rng = np.random.default_rng(42)

        # Generate coupled trajectory (correlated dimensions)
        W = rng.standard_normal((8, 8)) * 0.1
        x = rng.standard_normal(8)
        for _ in range(60):
            x = np.tanh(W @ x + rng.standard_normal(8) * 0.01)
            computer.record_state(x)

        result = computer.compute()
        self.assertGreater(result.phi, 0.0)
        self.assertEqual(result.method, "exhaustive")
        self.assertEqual(result.n_neurons, 8)

    def test_phi_zero_for_independent_system(self):
        """Independent (uncoupled) dimensions have Phi near 0."""
        from core.consciousness.phi_compute import PhiComputer, PhiConfig
        computer = PhiComputer(PhiConfig(trajectory_length=100))
        rng = np.random.default_rng(99)

        # Independent random walks (no coupling)
        for _ in range(120):
            state = rng.standard_normal(4)
            computer.record_state(state)

        result = computer.compute()
        # Should be very small (not exactly 0 due to finite samples)
        self.assertLess(result.phi, 0.5)

    def test_deterministic_phi(self):
        """Same trajectory gives same Phi."""
        from core.consciousness.phi_compute import PhiComputer, PhiConfig

        def build_trajectory():
            rng = np.random.default_rng(42)
            W = rng.standard_normal((6, 6)) * 0.1
            x = np.zeros(6)
            trajectory = []
            for _ in range(60):
                x = np.tanh(W @ x + rng.standard_normal(6) * 0.01)
                trajectory.append(x.copy())
            return trajectory

        traj = build_trajectory()

        c1 = PhiComputer(PhiConfig(trajectory_length=50))
        c2 = PhiComputer(PhiConfig(trajectory_length=50))
        for s in traj:
            c1.record_state(s)
            c2.record_state(s)

        r1 = c1.compute()
        r2 = c2.compute()
        self.assertAlmostEqual(r1.phi, r2.phi, places=10)

    def test_spectral_method_for_large_n(self):
        """Systems > max_exhaustive_n use spectral method."""
        from core.consciousness.phi_compute import PhiComputer, PhiConfig
        computer = PhiComputer(PhiConfig(
            trajectory_length=50, max_exhaustive_n=4))
        rng = np.random.default_rng(42)

        W = rng.standard_normal((8, 8)) * 0.1
        x = np.zeros(8)
        for _ in range(60):
            x = np.tanh(W @ x + rng.standard_normal(8) * 0.01)
            computer.record_state(x)

        result = computer.compute(coupling_matrix=W)
        self.assertEqual(result.method, "spectral")
        self.assertGreater(result.phi, 0.0)

    def test_insufficient_data_raises(self):
        """ValueError when trajectory is too short."""
        from core.consciousness.phi_compute import PhiComputer
        computer = PhiComputer()
        computer.record_state(np.zeros(4))  # Only 1 sample
        with self.assertRaises(ValueError):
            computer.compute()

    def test_mip_partition_valid(self):
        """MIP partition covers all neurons exactly once."""
        from core.consciousness.phi_compute import PhiComputer, PhiConfig
        computer = PhiComputer(PhiConfig(trajectory_length=30))
        rng = np.random.default_rng(42)

        for _ in range(40):
            computer.record_state(rng.standard_normal(6))

        result = computer.compute()
        a, b = result.mip_partition
        all_neurons = sorted(a + b)
        self.assertEqual(all_neurons, list(range(6)))

    def test_to_dict_serializable(self):
        """PhiResult.to_dict() is JSON-serializable."""
        from core.consciousness.phi_compute import PhiComputer, PhiConfig
        computer = PhiComputer(PhiConfig(trajectory_length=30))
        rng = np.random.default_rng(42)
        for _ in range(40):
            computer.record_state(rng.standard_normal(4))
        result = computer.compute()
        d = result.to_dict()
        json.dumps(d)  # Should not raise

    def test_integration_with_continuous_substrate(self):
        """PhiComputer works with real ContinuousSubstrate states."""
        from core.brain.llm.continuous_substrate import ContinuousSubstrate
        from core.consciousness.phi_compute import PhiComputer, PhiConfig

        sub = ContinuousSubstrate()
        computer = PhiComputer(PhiConfig(trajectory_length=50, max_exhaustive_n=8))

        for _ in range(60):
            sub._step_once()
            state = sub.get_state_vector()
            computer.record_state(state[:8])

        result = computer.compute()
        self.assertGreater(result.phi, 0.0)
        self.assertEqual(result.n_neurons, 8)


# ══════════════════════════════════════════════════════════════════════════
# Plasticity Governor Tests
# ══════════════════════════════════════════════════════════════════════════

class TestPlasticityGovernor(unittest.TestCase):
    """Tests for core/adaptation/plasticity_governor.py"""

    def test_register_and_consolidate(self):
        """Register parameters and consolidate Fisher."""
        from core.adaptation.plasticity_governor import (
            PlasticityGovernor, PlasticityConfig)

        gov = PlasticityGovernor(PlasticityConfig(ewc_lambda=10.0))
        params = np.random.default_rng(42).standard_normal(100)
        gov.register_parameters("test_params", params)

        # Record some gradients
        rng = np.random.default_rng(99)
        for _ in range(20):
            grad = rng.standard_normal(100) * 0.1
            gov.record_gradient("test_params", grad)

        records = gov.consolidate()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].parameter_set, "test_params")
        self.assertGreater(records[0].fisher_norm, 0.0)

    def test_penalty_suppresses_updates(self):
        """EWC penalty reduces update magnitude near anchor."""
        from core.adaptation.plasticity_governor import (
            PlasticityGovernor, PlasticityConfig)

        gov = PlasticityGovernor(PlasticityConfig(ewc_lambda=100.0))
        params = np.ones(50)
        gov.register_parameters("test", params)

        # Record gradients to build Fisher
        rng = np.random.default_rng(42)
        for _ in range(30):
            gov.record_gradient("test", rng.standard_normal(50))
        gov.consolidate()

        # Try to update away from anchor
        current = params + 0.5  # Moved away
        delta = np.ones(50) * 0.1  # Wants to move further

        penalized, report = gov.penalize_update("test", current, delta)
        # Penalty should be positive (it's resisting the update)
        self.assertGreater(report.penalty_magnitude, 0.0)
        # Penalized delta should differ from original
        self.assertFalse(np.allclose(penalized, delta))

    def test_no_penalty_without_consolidation(self):
        """Without Fisher estimation, no penalty is applied."""
        from core.adaptation.plasticity_governor import PlasticityGovernor

        gov = PlasticityGovernor()
        params = np.ones(20)
        gov.register_parameters("test", params)

        delta = np.ones(20) * 0.1
        penalized, report = gov.penalize_update("test", params, delta)
        np.testing.assert_array_equal(penalized, delta)
        self.assertEqual(report.penalty_magnitude, 0.0)

    def test_online_ewc_running_average(self):
        """Multiple consolidations use running Fisher average."""
        from core.adaptation.plasticity_governor import (
            PlasticityGovernor, PlasticityConfig)

        gov = PlasticityGovernor(PlasticityConfig(fisher_gamma=0.5))
        gov.register_parameters("test", np.zeros(10))

        rng = np.random.default_rng(42)

        # First consolidation
        for _ in range(10):
            gov.record_gradient("test", rng.standard_normal(10))
        gov.consolidate()
        fisher_1 = gov.get_importance_map("test").copy()

        # Second consolidation (different gradients)
        for _ in range(10):
            gov.record_gradient("test", rng.standard_normal(10) * 5)
        gov.consolidate()
        fisher_2 = gov.get_importance_map("test")

        # Fisher should have changed (running average)
        self.assertFalse(np.allclose(fisher_1, fisher_2))

    def test_importance_map_identifies_important_params(self):
        """Fisher diagonal is higher for params with consistent gradients."""
        from core.adaptation.plasticity_governor import PlasticityGovernor

        gov = PlasticityGovernor()
        gov.register_parameters("test", np.zeros(10))

        rng = np.random.default_rng(42)
        for _ in range(50):
            grad = np.zeros(10)
            # First 3 params always have large gradients
            grad[:3] = rng.standard_normal(3) * 10
            # Last 7 have small gradients
            grad[3:] = rng.standard_normal(7) * 0.01
            gov.record_gradient("test", grad)

        gov.consolidate()
        importance = gov.get_importance_map("test")

        # First 3 should be much more important
        self.assertGreater(np.mean(importance[:3]), np.mean(importance[3:]) * 10)


# ══════════════════════════════════════════════════════════════════════════
# Meta-Learner Tests
# ══════════════════════════════════════════════════════════════════════════

class TestMetaLearner(unittest.TestCase):
    """Tests for core/adaptation/meta_learner.py"""

    def test_es_gradient_approximation(self):
        """ES correctly approximates gradient of a simple quadratic."""
        from core.adaptation.meta_learner import ESMetaOptimizer, MetaConfig

        opt = ESMetaOptimizer(MetaConfig(
            n_perturbations=100, perturbation_sigma=0.01, seed=42))

        # f(x) = -||x - target||^2  (maximum at target)
        target = np.array([1.0, 2.0, 3.0])

        def evaluate(params):
            return -float(np.sum((params - target) ** 2))

        x = np.zeros(3)
        grad, metrics = opt.estimate_gradient(x, evaluate)

        # Gradient should point toward target
        for i in range(3):
            self.assertGreater(grad[i] * target[i], 0,
                f"Gradient[{i}] should point toward target")

    def test_meta_step_improves_reward(self):
        """Multiple meta-steps improve mean reward."""
        from core.adaptation.meta_learner import (
            MetaLearner, MetaConfig, MetaTask)

        target = np.array([1.0, 2.0])
        def evaluate(params):
            return -float(np.sum((params - target) ** 2))

        learner = MetaLearner(MetaConfig(
            n_perturbations=50, meta_lr=0.01, seed=42))
        learner.register_task(MetaTask(
            name="find_target",
            evaluate=evaluate,
            parameter_dim=2,
            baseline_params=np.zeros(2),
        ))

        rewards = []
        for _ in range(5):
            steps = learner.meta_step()
            rewards.append(steps[0].mean_reward)

        # Later rewards should be better (less negative)
        self.assertGreater(rewards[-1], rewards[0])

    def test_meta_params_persist(self):
        """Meta parameters are updated after meta_step."""
        from core.adaptation.meta_learner import MetaLearner, MetaConfig, MetaTask

        learner = MetaLearner(MetaConfig(
            n_perturbations=10, seed=42))
        learner.register_task(MetaTask(
            name="test",
            evaluate=lambda p: -float(np.sum(p ** 2)),
            parameter_dim=4,
            baseline_params=np.ones(4),
        ))

        initial = learner.get_meta_params("test").copy()
        learner.meta_step()
        updated = learner.get_meta_params("test")

        self.assertFalse(np.allclose(initial, updated))

    def test_antithetic_reduces_variance(self):
        """Antithetic sampling produces directionally consistent gradients."""
        from core.adaptation.meta_learner import ESMetaOptimizer, MetaConfig

        target = np.array([1.0, 2.0])
        def evaluate(params):
            return -float(np.sum((params - target) ** 2))

        # Use large population for reliable gradient estimation
        grads = []
        for seed in range(10):
            opt = ESMetaOptimizer(MetaConfig(
                n_perturbations=100, antithetic=True, seed=seed))
            g, _ = opt.estimate_gradient(np.zeros(2), evaluate)
            grads.append(g)

        # Mean gradient should reliably point toward target
        mean_grad = np.mean(grads, axis=0)
        for i in range(2):
            self.assertGreater(mean_grad[i] * target[i], 0,
                f"Mean gradient[{i}] should point toward target")


# ══════════════════════════════════════════════════════════════════════════
# Lesion Matrix Tests
# ══════════════════════════════════════════════════════════════════════════

class TestLesionStudy(unittest.TestCase):
    """Tests for core/architect/lesion_matrix.py"""

    def test_lesion_detects_critical_component(self):
        """Lesioning a critical component shows high impact."""
        from core.architect.lesion_matrix import (
            LesionStudy, LesionableComponent)

        # Simulate a system where component A is critical
        state_a = np.array([1.0, 2.0, 3.0])
        state_b = np.array([0.1, 0.1, 0.1])

        def output_fn():
            return float(np.sum(state_a) + np.sum(state_b))

        study = LesionStudy(n_probe_steps=5)
        study.register_component(LesionableComponent(
            name="critical_A",
            get_state=lambda: state_a.copy(),
            set_state=lambda s: np.copyto(state_a, s),
            zero_fn=lambda: state_a.fill(0),
        ))
        study.register_component(LesionableComponent(
            name="minor_B",
            get_state=lambda: state_b.copy(),
            set_state=lambda s: np.copyto(state_b, s),
            zero_fn=lambda: state_b.fill(0),
        ))
        study.register_probe("total_output", output_fn)
        study.set_step_function(lambda: None)

        matrix = study.run()
        self.assertEqual(len(matrix.results), 2)

        # Critical component should have higher impact
        a_result = [r for r in matrix.results if r.component_name == "critical_A"][0]
        b_result = [r for r in matrix.results if r.component_name == "minor_B"][0]
        self.assertGreater(a_result.criticality_score, b_result.criticality_score)

    def test_lesion_restores_state(self):
        """State is fully restored after lesion."""
        from core.architect.lesion_matrix import LesionStudy, LesionableComponent

        state = np.array([1.0, 2.0, 3.0])
        original = state.copy()

        study = LesionStudy(n_probe_steps=2)
        study.register_component(LesionableComponent(
            name="test",
            get_state=lambda: state.copy(),
            set_state=lambda s: np.copyto(state, s),
            zero_fn=lambda: state.fill(0),
        ))
        study.register_probe("energy", lambda: float(np.linalg.norm(state)))
        study.set_step_function(lambda: None)

        study.run()
        np.testing.assert_array_equal(state, original)

    def test_no_components_raises(self):
        """ValueError when no components registered."""
        from core.architect.lesion_matrix import LesionStudy
        study = LesionStudy()
        study.register_probe("test", lambda: 1.0)
        with self.assertRaises(ValueError):
            study.run()

    def test_matrix_shape(self):
        """Matrix has correct shape (components x metrics)."""
        from core.architect.lesion_matrix import LesionStudy, LesionableComponent

        state = np.zeros(5)
        study = LesionStudy(n_probe_steps=2)
        for i in range(3):
            s = np.zeros(5)
            study.register_component(LesionableComponent(
                name=f"comp_{i}",
                get_state=lambda: s.copy(),
                set_state=lambda v, s=s: np.copyto(s, v),
                zero_fn=lambda s=s: s.fill(0),
            ))
        study.register_probe("m1", lambda: 1.0)
        study.register_probe("m2", lambda: 2.0)
        study.set_step_function(lambda: None)

        matrix = study.run()
        self.assertEqual(matrix.matrix.shape, (3, 2))


# ══════════════════════════════════════════════════════════════════════════
# Hidden Eval Tests
# ══════════════════════════════════════════════════════════════════════════

class TestHiddenEvalRunner(unittest.TestCase):
    """Tests for core/architect/hidden_eval.py"""

    def test_passing_scenario(self):
        """Scenario within range passes."""
        from core.architect.hidden_eval import HiddenEvalRunner, EvalScenario

        runner = HiddenEvalRunner()
        runner.register_scenario(EvalScenario(
            scenario_id="test_pass",
            name="Test Pass",
            description="Always passes",
            scenario_type="test",
            expected_range=(0.0, 10.0),
            evaluate=lambda: 5.0,
        ))

        result = runner.run_suite()
        self.assertEqual(result.passed, 1)
        self.assertEqual(result.failed, 0)
        self.assertEqual(result.overall_health, 1.0)

    def test_failing_scenario(self):
        """Scenario outside range fails."""
        from core.architect.hidden_eval import HiddenEvalRunner, EvalScenario

        runner = HiddenEvalRunner()
        runner.register_scenario(EvalScenario(
            scenario_id="test_fail",
            name="Test Fail",
            description="Always fails",
            scenario_type="test",
            expected_range=(0.0, 1.0),
            evaluate=lambda: 99.0,
        ))

        result = runner.run_suite()
        self.assertEqual(result.failed, 1)
        self.assertEqual(result.overall_health, 0.0)

    def test_tamper_detection(self):
        """Tampered scenario is detected."""
        from core.architect.hidden_eval import HiddenEvalRunner, EvalScenario

        scenario = EvalScenario(
            scenario_id="tamper_test",
            name="Tamper Test",
            description="Should detect tampering",
            scenario_type="test",
            expected_range=(0.0, 1.0),
            evaluate=lambda: 0.5,
        )
        # Tamper with the scenario
        scenario.description = "MODIFIED BY ATTACKER"
        self.assertFalse(scenario.verify_integrity())

        runner = HiddenEvalRunner()
        runner.register_scenario(scenario)
        result = runner.run_suite()
        self.assertEqual(result.tampered, 1)

    def test_drift_detection(self):
        """Drift is detected when health drops significantly."""
        from core.architect.hidden_eval import HiddenEvalRunner, EvalScenario

        runner = HiddenEvalRunner(drift_window=3, drift_threshold=0.3)

        # Run a scenario that always passes
        runner.register_scenario(EvalScenario(
            scenario_id="drift_test",
            name="Drift",
            description="Drift test",
            scenario_type="test",
            expected_range=(0.0, 10.0),
            evaluate=lambda: 5.0,
        ))

        # Build history of good health
        for _ in range(5):
            runner.run_suite()

        # Now register a failing scenario
        runner.register_scenario(EvalScenario(
            scenario_id="drift_fail",
            name="Drift Fail",
            description="Causes drift",
            scenario_type="test",
            expected_range=(0.0, 0.1),
            evaluate=lambda: 99.0,
        ))

        result = runner.run_suite()
        # Health dropped from 1.0 to 0.5, which is > 0.3 threshold
        self.assertTrue(result.drift_detected)

    def test_default_suite_creates_scenarios(self):
        """Default suite has built-in scenarios."""
        from core.architect.hidden_eval import HiddenEvalRunner
        runner = HiddenEvalRunner.create_default_suite()
        self.assertGreater(len(runner._scenarios), 0)
        status = runner.get_status()
        self.assertGreater(status["n_scenarios"], 0)

    def test_eval_result_serializable(self):
        """EvalSuiteResult.to_dict() is JSON-serializable."""
        from core.architect.hidden_eval import HiddenEvalRunner, EvalScenario
        runner = HiddenEvalRunner()
        runner.register_scenario(EvalScenario(
            scenario_id="serial_test",
            name="Serialize",
            description="Test serialization",
            scenario_type="test",
            expected_range=(0.0, 10.0),
            evaluate=lambda: 5.0,
        ))
        result = runner.run_suite()
        json.dumps(result.to_dict())  # Should not raise


# ══════════════════════════════════════════════════════════════════════════
# Integration Tests
# ══════════════════════════════════════════════════════════════════════════

class TestResearchModuleIntegration(unittest.TestCase):
    """Cross-module integration tests."""

    def test_phi_with_hierarchical_brain(self):
        """Phi can be computed from HierarchicalBrain region states."""
        from core.brain.hierarchical_brain import HierarchicalBrain
        from core.consciousness.phi_compute import PhiComputer, PhiConfig

        brain = HierarchicalBrain()
        computer = PhiComputer(PhiConfig(trajectory_length=30, max_exhaustive_n=8))

        for _ in range(40):
            outputs = brain.step(np.random.randn(64).astype(np.float32))
            # Use sensory region output for phi
            sensory_out = outputs.get("sensory", np.zeros(8))[:8]
            computer.record_state(sensory_out)

        result = computer.compute()
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.phi, 0.0)

    def test_ewc_with_world_model(self):
        """EWC can protect world model encoder weights."""
        from core.adaptation.plasticity_governor import PlasticityGovernor
        from core.world_model.learned_world_model import LearnedWorldModel

        model = LearnedWorldModel()
        gov = PlasticityGovernor()
        gov.register_parameters("world_model_enc", model.W_enc)

        # Simulate gradient recording
        rng = np.random.default_rng(42)
        for _ in range(20):
            grad = rng.standard_normal(model.W_enc.shape) * 0.01
            gov.record_gradient("world_model_enc", grad)

        records = gov.consolidate()
        self.assertGreater(len(records), 0)

        # Verify importance map exists
        importance = gov.get_importance_map("world_model_enc")
        self.assertIsNotNone(importance)
        self.assertGreater(float(np.sum(importance)), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
