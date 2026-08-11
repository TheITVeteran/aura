from __future__ import annotations

from tools.evaluate_unified_intrinsic_checkpoint import _sign_test_p_value


def test_sign_test_is_exact_and_refuses_ties() -> None:
    assert _sign_test_p_value([0.0, 0.0]) is None
    assert _sign_test_p_value([1.0] * 8) == 0.0078125
    assert _sign_test_p_value([1.0] * 4 + [-1.0] * 4) == 1.0
