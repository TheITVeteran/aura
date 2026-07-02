from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.adaptation.adaptive_immunity import AdaptiveImmuneSystem, CellKind, ImmuneCell
from core.adaptation.spatial_receptor_code import (
    SpatialReceptorMap,
    annotate_antigen_like,
    signal_from_antigen_like,
)


@dataclass
class FakeAntigen:
    danger: float
    resource_pressure: float
    error_load: float
    health_pressure: float
    temporal_pressure: float
    subsystem_need: float
    recurrence_pressure: float
    protected: bool = False
    source_domain: str = "substrate"
    source: str = "test"
    context: dict | None = None


def test_spatial_receptor_code_maps_continuous_pressure_to_stable_receptor():
    antigen = FakeAntigen(
        danger=0.66,
        resource_pressure=0.95,
        error_load=0.30,
        health_pressure=0.36,
        temporal_pressure=0.20,
        subsystem_need=0.72,
        recurrence_pressure=0.15,
    )

    signal = signal_from_antigen_like(antigen)
    distribution = SpatialReceptorMap().distribution(signal)

    assert distribution[0].receptor_id == "resource_pressure_responder"
    assert distribution[0].probability > distribution[1].probability
    assert "reduce_load" in distribution[0].downstream_targets


def test_spatial_receptor_code_marks_protected_identity_as_regulatory():
    antigen = FakeAntigen(
        danger=0.75,
        resource_pressure=0.18,
        error_load=0.80,
        health_pressure=0.30,
        temporal_pressure=0.20,
        subsystem_need=0.62,
        recurrence_pressure=0.25,
        protected=True,
    )

    annotation = annotate_antigen_like(antigen)

    assert annotation["top_receptor"]["receptor_id"] == "protected_identity_regulator"
    assert "regulatory_t" in annotation["top_receptor"]["preferred_cell_kinds"]


def test_adaptive_immunity_embeds_spatial_receptor_code_and_activation_prior(tmp_path):
    immune = AdaptiveImmuneSystem(state_dir=tmp_path, rng_seed=12)
    antigen = immune.present_antigen(
        {
            "type": "exception",
            "source": "test",
            "subsystem": "runtime",
            "text": "resource exhaustion memory pressure",
            "resource_pressure": 1.0,
            "error_count": 2,
        },
        state_snapshot={"resource_pressure": 1.0},
    )
    code = antigen.context.get("spatial_receptor_code")

    assert code
    assert code["top_receptor"]["receptor_id"] == "resource_pressure_responder"

    cell = ImmuneCell(
        cell_id="cell-test",
        lineage_id="lineage-test",
        kind=CellKind.CYTOTOXIC,
        receptor=np.ones_like(antigen.vector, dtype=np.float32),
    )
    prior = immune._spatial_receptor_activation_prior(cell, antigen)
    assert prior > 1.0
