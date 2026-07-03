"""Analogical leap engine — the horizontal-leap feeder for the FDE.

Pins: the OOD detector knows when the map runs out; structure mapping
selects schemas by dynamics, not surface words; transferred invariants
arrive as honest CONJECTUREs with real falsification plans where a
verifier exists and explicit unverifiability labels where none does.
"""
from __future__ import annotations

import json

from core.discovery.analogical_leap import (
    AnalogicalLeapEngine,
    ConjectureRecombinator,
    OutOfDistributionDetector,
    StructureMapper,
)
from core.discovery.frontier_discovery_engine import EpistemicStatus


class _NoSupportRetriever:
    def retrieve(self, intent):
        from types import SimpleNamespace
        return SimpleNamespace(hits=[])


class _StrongSupportRetriever:
    def retrieve(self, intent):
        from types import SimpleNamespace
        return SimpleNamespace(hits=[SimpleNamespace(score=0.9)])


class TestOODDetection:
    def test_unknown_problem_with_no_support_is_off_map(self):
        detector = OutOfDistributionDetector(retriever=_NoSupportRetriever())
        verdict = detector.assess(
            "qualia inversion metric for紫 unheard-of frobnication manifolds"
        )
        assert verdict.off_map is True
        assert verdict.retrieval_support == 0.0

    def test_strong_memory_support_keeps_problem_on_map(self):
        detector = OutOfDistributionDetector(retriever=_StrongSupportRetriever())
        verdict = detector.assess("routine question about known subsystem behavior")
        assert verdict.off_map is False
        assert verdict.retrieval_support >= 0.35

    def test_strong_schema_cues_keep_problem_on_map_even_without_memory(self):
        detector = OutOfDistributionDetector(retriever=_NoSupportRetriever())
        verdict = detector.assess(
            "the rumor spreads and propagates like a contagion, adoption "
            "saturates as the epidemic percolates through the network"
        )
        assert verdict.best_domain_match >= 0.30
        assert verdict.off_map is False
        assert "diffusion_spread" in verdict.nearest_schemas


class TestStructureMapping:
    def test_spread_dynamics_map_to_diffusion_schema(self):
        mapper = StructureMapper()
        mappings = mapper.map(
            "a glitch meme spreads between agents and propagates through "
            "the fleet, infecting schedulers until adoption saturates"
        )
        assert mappings
        assert mappings[0].schema.name == "diffusion_spread"
        assert mappings[0].matched_cues  # evidence, not vibes

    def test_congestion_dynamics_map_to_queueing_schema(self):
        mapper = StructureMapper()
        mappings = mapper.map(
            "the backlog grows, latency explodes under load, the bottleneck "
            "server saturates and throughput collapses as bursts arrive"
        )
        names = [m.schema.name for m in mappings]
        assert names[0] == "queueing_congestion"

    def test_role_bindings_use_problem_vocabulary(self):
        mapper = StructureMapper()
        mappings = mapper.map(
            "replication errors spread between databases and propagate "
            "through replicas, the contagion saturates the cluster"
        )
        bindings = mappings[0].role_bindings
        assert set(bindings) == set(mappings[0].schema.roles)
        # bound to salient problem nouns, not left as raw role names
        assert any(v not in mappings[0].schema.roles for v in bindings.values())


class TestConjectureDiscipline:
    def _mappings(self):
        return StructureMapper().map(
            "failures cascade and spread through services like a contagion, "
            "saturating the susceptible pool of healthy nodes"
        )

    def test_transferred_invariants_are_conjectures_never_asserted(self):
        conjectures = ConjectureRecombinator().recombine("p", self._mappings())
        assert conjectures
        for c in conjectures:
            assert c.status is EpistemicStatus.CONJECTURE
            assert c.status.is_assertable is False
            assert c.provenance == "analogical_leap"
            assert c.confidence <= 0.6  # analogies never arrive confident

    def test_verifiable_and_unverifiable_routes_are_labeled(self):
        conjectures = ConjectureRecombinator().recombine("p", self._mappings())
        plans = [c.falsification_plan for c in conjectures]
        assert any("test against observed data" in p for p in plans)
        # schemas without a numeric form must say so, not pretend
        unverifiable = [
            c for c in ConjectureRecombinator().recombine(
                "p",
                StructureMapper().map(
                    "agents exploit the shared commons, free-rider incentive "
                    "rewards overuse of the pool"
                ),
            )
        ]
        assert any(
            "no local verifier applies" in c.falsification_plan for c in unverifiable
        )


class TestLeapEngine:
    def test_leap_fires_only_off_map_unless_forced(self, tmp_path):
        engine = AnalogicalLeapEngine(
            detector=OutOfDistributionDetector(retriever=_StrongSupportRetriever()),
            artifact_path=tmp_path / "leaps.jsonl",
        )
        held = engine.leap("well-supported routine problem about spreads")
        assert held.verdict.off_map is False
        assert held.conjectures == []

        forced = engine.leap(
            "well-supported routine problem about spreads", force=True
        )
        assert forced.conjectures

    def test_off_map_leap_generates_and_logs_evidence(self, tmp_path):
        artifact = tmp_path / "leaps.jsonl"
        engine = AnalogicalLeapEngine(
            detector=OutOfDistributionDetector(retriever=_NoSupportRetriever()),
            artifact_path=artifact,
        )
        report = engine.leap(
            "an unprecedented anomaly where alerts cascade and spread through "
            "monitoring services, saturating every healthy channel"
        )
        assert report.verdict.off_map is True
        assert report.conjectures
        lines = artifact.read_text().strip().splitlines()
        assert len(lines) == 1
        payload = json.loads(lines[0])
        assert payload["schema"] == "aura.analogical_leap.v1"
        assert payload["off_map"] is True
        assert payload["conjectures"]
