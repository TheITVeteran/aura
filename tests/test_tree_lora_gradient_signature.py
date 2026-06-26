from __future__ import annotations

import numpy as np

from core.learning.tree_lora_manager import TreeLoRAManager


def test_gradient_signature_depends_on_gradient_values_not_task_id() -> None:
    manager = TreeLoRAManager(signature_dim=8, layer_count=1)
    gradients = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float64)

    sig_a = manager.compute_gradient_signature("task_a", gradients)
    sig_b = manager.compute_gradient_signature("task_b", gradients)
    sig_c = manager.compute_gradient_signature("task_c", -gradients)

    np.testing.assert_allclose(sig_a.gradient_vector, sig_b.gradient_vector)
    assert float(np.dot(sig_a.gradient_vector, sig_c.gradient_vector)) < -0.99
    assert sig_a.loss_magnitude > 0.0


def test_large_gradient_signature_uses_all_values_deterministically() -> None:
    manager = TreeLoRAManager(signature_dim=4, layer_count=1)

    sig_1 = manager.compute_gradient_signature("same", np.arange(1, 17, dtype=np.float64))
    sig_2 = manager.compute_gradient_signature("same", np.arange(1, 17, dtype=np.float64))
    sig_3 = manager.compute_gradient_signature("same", np.arange(16, 0, -1, dtype=np.float64))

    np.testing.assert_allclose(sig_1.gradient_vector, sig_2.gradient_vector)
    assert not np.allclose(sig_1.gradient_vector, sig_3.gradient_vector)
    assert np.isclose(np.linalg.norm(sig_1.gradient_vector), 1.0)


def test_branch_adapter_delta_is_deterministic_from_signature() -> None:
    manager = TreeLoRAManager(signature_dim=8, layer_count=1)
    signature = manager.compute_gradient_signature("task", np.array([0.5, 1.5, -2.0, 3.0]))

    delta_a = manager._adapter_delta_from_signature((4, 2), signature.gradient_vector)
    delta_b = manager._adapter_delta_from_signature((4, 2), signature.gradient_vector)

    np.testing.assert_allclose(delta_a, delta_b)
    assert np.isclose(np.linalg.norm(delta_a), 0.01)
