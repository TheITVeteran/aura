from __future__ import annotations

from core.consciousness.caa.alpha_controller import AlphaController


def test_alpha_controller_dampens_for_bad_generation_health() -> None:
    controller = AlphaController(base_alpha=5.0)
    warm = controller.update(readiness_level="production", exact_match_ratio=1.0, extracted_ratio=1.0)

    unhealthy = controller.update(
        readiness_level="production",
        exact_match_ratio=1.0,
        extracted_ratio=1.0,
        generation_health=0.1,
        cross_entropy=9.0,
    )

    assert warm.current_alpha > 5.0
    assert unhealthy.current_alpha < warm.current_alpha
    assert unhealthy.dampening < 1.0
    assert "health dampening" in unhealthy.reason
