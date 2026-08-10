from __future__ import annotations

import pytest

from core.brain.llm.latent_cortex.plasticity_sites import (
    PLASTICITY_SITE_REGISTRY,
    PlasticitySiteRegistry,
    select_plasticity_layers,
)


def test_registry_covers_projection_and_layer_placement_product() -> None:
    receipt = PlasticitySiteRegistry().receipt()

    assert receipt["schema"] == "aura.rlc.plasticity_site_registry.v1"
    assert receipt["sites"] == [
        "o_proj:early",
        "o_proj:distributed",
        "o_proj:late",
        "down_proj:early",
        "down_proj:distributed",
        "down_proj:late",
    ]
    assert len(receipt["registry_sha256"]) == 64


def test_layer_placements_are_deterministic_and_span_the_window() -> None:
    assert select_plasticity_layers(7, 21, 4, placement="early") == (7, 8, 9, 10)
    assert select_plasticity_layers(7, 21, 4, placement="late") == (17, 18, 19, 20)
    assert select_plasticity_layers(7, 21, 4, placement="distributed") == (
        7,
        11,
        16,
        20,
    )


def test_registry_resolves_exact_site_and_rejects_unknowns() -> None:
    site = PLASTICITY_SITE_REGISTRY.resolve("down_proj", "late")
    assert site.site_id == "down_proj:late"
    assert site.layer_indices(2, 8, 2) == (6, 7)
    with pytest.raises(ValueError, match="unregistered plasticity site"):
        PLASTICITY_SITE_REGISTRY.resolve("q_proj", "late")


@pytest.mark.parametrize(
    ("start", "end", "maximum", "placement"),
    [(-1, 2, 1, "early"), (2, 2, 1, "early"), (1, 2, 0, "early"), (1, 2, 1, "middle")],
)
def test_layer_selection_rejects_invalid_contracts(
    start: int,
    end: int,
    maximum: int,
    placement: str,
) -> None:
    with pytest.raises(ValueError):
        select_plasticity_layers(start, end, maximum, placement=placement)
