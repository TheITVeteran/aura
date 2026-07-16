"""Integration contract for the allostasis organ: the causal seams.

The engine's math is covered in test_allostasis_engine.py. This file pins the
wiring that makes prediction *causal*:

  * BodyState carries anticipatory_pressure into pressure_vector() and
    total_pressure (→ affect, welfare, workspace, Will);
  * the metabolic coordinator consults the defer gate and treats an
    anticipatory signal as a resource constraint;
  * the health contract knows the organ (OPTIONAL tier, is_ready liveness);
  * the canonical service name matches the engine's SERVICE_NAME;
  * the HTTP receipt surface serves status and the falsifiable ledger;
  * the singleton registers itself in the ServiceContainer.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient


class _DeterministicAllostasis:
    def __init__(self, anticipatory=0.5, defer=False, reason="fixture"):
        self._anticipatory = anticipatory
        self._defer = defer
        self._reason = reason

    def felt_contribution(self):
        return {
            "anticipatory_pressure": self._anticipatory,
            "allostatic_load": 0.2,
            "nearest_crisis_eta_s": 900.0,
            "tier": "conserving",
        }

    def should_defer_heavy_work(self):
        return self._defer, self._reason

    def is_ready(self):
        return True

    def status(self):
        return {
            "service": "allostasis_engine",
            "tier": "conserving",
            "narrative": "deterministic fixture narrative",
            "open_forecasts": [],
            "recently_resolved": [],
            "calibration": {},
        }


# ─────────────────────────────────────────────────────────────────────────────
# BodyState: the felt seam
# ─────────────────────────────────────────────────────────────────────────────

class TestBodyStateSeam:
    def test_body_state_reads_engine_from_container(self, service_container):
        from core.being.aura_now import BodyState

        service_container.register_instance(
            "allostasis_engine",
            _DeterministicAllostasis(anticipatory=0.5),
            required=False,
        )
        body = BodyState.from_aura_state(None)
        assert body.anticipatory_pressure == pytest.approx(0.5)
        assert "allostasis_forecast" in body.telemetry_sources

    def test_no_engine_means_zero_and_no_source_tag(self, service_container):
        from core.being.aura_now import BodyState

        body = BodyState.from_aura_state(None)
        assert body.anticipatory_pressure == 0.0
        assert "allostasis_forecast" not in body.telemetry_sources

    def test_anticipation_flows_into_pressure_vector(self):
        from core.being.aura_now import BodyState

        body = BodyState(anticipatory_pressure=0.7)
        vector = body.pressure_vector()
        assert vector["anticipatory_pressure"] == pytest.approx(0.7)

    def test_anticipation_raises_total_pressure_while_present_is_green(self):
        from core.being.aura_now import BodyState

        calm = BodyState()
        anticipating = BodyState(anticipatory_pressure=0.8)
        assert calm.total_pressure == pytest.approx(0.0)
        # Peak-weighted blend: a lone anticipatory signal is a real pressure.
        assert anticipating.total_pressure > 0.4

    def test_broken_engine_degrades_to_zero(self, service_container):
        from core.being.aura_now import BodyState

        class Broken:
            def felt_contribution(self):
                raise RuntimeError("boom")

        service_container.register_instance("allostasis_engine", Broken(), required=False)
        body = BodyState.from_aura_state(None)
        assert body.anticipatory_pressure == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Metabolic coordinator: the regulation consumers
# ─────────────────────────────────────────────────────────────────────────────

class TestMetabolicConsumers:
    def _bare_coordinator(self):
        from core.coordinators.metabolic_coordinator import MetabolicCoordinator

        return object.__new__(MetabolicCoordinator)

    def test_defer_gate_consulted(self, service_container):
        coordinator = self._bare_coordinator()
        service_container.register_instance(
            "allostasis_engine",
            _DeterministicAllostasis(defer=True, reason="crisis in 9min"),
            required=False,
        )
        assert coordinator._allostasis_defers() is True

    def test_no_defer_when_engine_calm(self, service_container):
        coordinator = self._bare_coordinator()
        service_container.register_instance(
            "allostasis_engine", _DeterministicAllostasis(defer=False), required=False,
        )
        assert coordinator._allostasis_defers() is False

    def test_no_engine_never_defers(self, service_container):
        coordinator = self._bare_coordinator()
        assert coordinator._allostasis_defers() is False

    def test_anticipatory_signal_counts_as_resource_constraint(self, service_container):
        coordinator = self._bare_coordinator()
        service_container.register_instance(
            "allostasis_engine", _DeterministicAllostasis(defer=True), required=False,
        )
        assert coordinator._is_resource_constrained() is True


# ─────────────────────────────────────────────────────────────────────────────
# Health contract / service names / container
# ─────────────────────────────────────────────────────────────────────────────

class TestContracts:
    def test_health_contract_entry(self):
        from core.runtime.health_contract import RUNTIME_CONTRACT, ServiceTier

        entry = next(
            (r for r in RUNTIME_CONTRACT if r.container_key == "allostasis_engine"),
            None,
        )
        assert entry is not None, "allostasis_engine missing from RUNTIME_CONTRACT"
        assert entry.tier == ServiceTier.OPTIONAL
        assert entry.liveness_check == "is_ready"

    def test_canonical_service_name_matches_engine(self):
        from core.autonomic.allostasis import AllostasisEngine
        from core.service_names import ServiceNames

        assert ServiceNames.ALLOSTASIS == AllostasisEngine.SERVICE_NAME

    def test_singleton_registers_in_container(self, service_container, tmp_path, monkeypatch):
        monkeypatch.setenv("AURA_ALLOSTASIS_DIR", str(tmp_path / "allostasis"))
        from core.autonomic.allostasis import (
            get_allostasis_engine,
            reset_allostasis_engine_for_test,
        )

        reset_allostasis_engine_for_test()
        try:
            engine = get_allostasis_engine()
            assert get_allostasis_engine() is engine
            assert service_container.get("allostasis_engine", default=None) is engine
        finally:
            reset_allostasis_engine_for_test()

    def test_health_liveness_passes_with_real_engine(self, service_container, tmp_path):
        from core.autonomic.allostasis import AllostasisEngine
        from core.runtime.health_contract import RUNTIME_CONTRACT

        engine = AllostasisEngine(data_dir=tmp_path / "allostasis")
        entry = next(r for r in RUNTIME_CONTRACT if r.container_key == "allostasis_engine")
        check = getattr(engine, entry.liveness_check)
        assert check() is True


# ─────────────────────────────────────────────────────────────────────────────
# HTTP receipt surface
# ─────────────────────────────────────────────────────────────────────────────

class TestRoutes:
    def _client(self):
        from fastapi import FastAPI

        from interface.routes import allostasis as allostasis_routes

        app = FastAPI()
        app.include_router(allostasis_routes.router, prefix="/api")
        return TestClient(app)

    def test_status_503_when_unregistered(self, service_container):
        response = self._client().get("/api/allostasis")
        assert response.status_code == 503
        assert response.json()["available"] is False

    def test_status_serves_real_engine(self, service_container, tmp_path):
        from core.autonomic.allostasis import AllostasisEngine

        engine = AllostasisEngine(data_dir=tmp_path / "allostasis")
        service_container.register_instance("allostasis_engine", engine, required=False)
        response = self._client().get("/api/allostasis")
        assert response.status_code == 200
        payload = response.json()
        assert payload["available"] is True
        assert payload["tier"] == "settled"
        assert payload["narrative"]
        assert "vitals" in payload and "calibration" in payload

    def test_forecast_ledger_surface(self, service_container, tmp_path):
        from core.autonomic.allostasis import AllostasisEngine

        engine = AllostasisEngine(data_dir=tmp_path / "allostasis")
        service_container.register_instance("allostasis_engine", engine, required=False)
        response = self._client().get("/api/allostasis/forecasts")
        assert response.status_code == 200
        payload = response.json()
        assert payload["available"] is True
        assert payload["open"] == []
        assert "recently_resolved" in payload

    def test_route_shields_engine_errors(self, service_container):
        broken = SimpleNamespace(status=lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        service_container.register_instance("allostasis_engine", broken, required=False)
        response = self._client().get("/api/allostasis")
        assert response.status_code == 500
        assert response.json()["available"] is False
