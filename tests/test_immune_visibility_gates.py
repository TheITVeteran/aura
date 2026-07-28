"""Danger must not authorize itself when nobody can see the subsystem.

CP126 21af3c48: restart/restore/revoke/migration suppression was bypassed at
danger >= 0.88 with only a coverage RATIO floor — which any two channels can
satisfy. The two that matter for an irreversible action are a direct health
probe and a causal trace.

CP126 37f929c1: all_clear keyed on a channel-presence score, so it could be
issued without a direct component probe.
"""
from __future__ import annotations

import pytest

from core.adaptation.adaptive_immunity import (
    Antigen,
    EffectorArtifact,
    EffectorKind,
    ImmuneResponse,
    get_adaptive_immune_system,
)


@pytest.fixture()
def immune():
    return get_adaptive_immune_system()


def _antigen(danger=0.95):
    return Antigen.from_dict(
        {
            "antigen_id": "ag",
            "subsystem": "memory",
            "vector": [0.5] * 16,
            "danger": danger,
        }
    )


def _artifact(kind=EffectorKind.RESTART_COMPONENT):
    return EffectorArtifact(
        artifact_id="a1", kind=kind, component="memory", confidence=0.9,
        source_cell_id="c1", lineage_id="l1", bounded_payload={},
    )


def _response(artifact):
    response = ImmuneResponse.__new__(ImmuneResponse)
    response.artifacts = [artifact]
    return response


def _coverage(ratio, channels):
    return {"coverage_ratio": ratio, "observed_channels": list(channels)}


def test_extreme_danger_without_grounding_is_still_suppressed(immune):
    """danger 0.95 built from telemetry nobody corroborated."""
    artifact = _artifact()

    immune._apply_coverage_constraints(
        _response(artifact),
        _antigen(0.95),
        _coverage(0.3, ["subsystem_identity", "error_telemetry"]),
    )

    assert artifact.suppressed is True


@pytest.mark.parametrize("channel", ["health_probe", "causal_trace"])
def test_extreme_danger_with_grounding_may_proceed(immune, channel):
    artifact = _artifact()

    immune._apply_coverage_constraints(
        _response(artifact),
        _antigen(0.95),
        _coverage(0.3, ["subsystem_identity", channel]),
    )

    assert artifact.suppressed is False


def test_grounding_alone_does_not_bypass_the_ratio_floor(immune):
    artifact = _artifact()

    immune._apply_coverage_constraints(
        _response(artifact), _antigen(0.95), _coverage(0.1, ["health_probe"])
    )

    assert artifact.suppressed is True


def test_moderate_danger_is_suppressed_regardless(immune):
    artifact = _artifact()

    immune._apply_coverage_constraints(
        _response(artifact), _antigen(0.5), _coverage(0.3, ["health_probe"])
    )

    assert artifact.suppressed is True


@pytest.mark.parametrize(
    "kind",
    [
        EffectorKind.RESTART_COMPONENT,
        EffectorKind.RESTORE_CHECKPOINT,
        EffectorKind.REVOKE_TOOL,
        EffectorKind.SCHEMA_MIGRATION,
    ],
)
def test_every_irreversible_kind_is_gated(immune, kind):
    artifact = _artifact(kind)

    immune._apply_coverage_constraints(
        _response(artifact), _antigen(0.95), _coverage(0.3, ["error_telemetry"])
    )

    assert artifact.suppressed is True


def test_a_cheap_action_is_not_gated(immune):
    """Clearing a cache is reversible; it does not need the same evidence."""
    artifact = _artifact(EffectorKind.CLEAR_CACHE)

    immune._apply_coverage_constraints(
        _response(artifact), _antigen(0.95), _coverage(0.1, [])
    )

    assert artifact.suppressed is False


# --- 37f929c1: all_clear needs a direct probe ---------------------------


def test_all_clear_requires_a_direct_probe(immune):
    antigen = _antigen(0.05)
    verdict = immune._build_diagnostic_verdict(
        antigen,
        _response(_artifact()),
        coverage_report=_coverage(0.9, ["subsystem_identity", "error_telemetry"]),
        verification_report=immune._default_verification_report(
            status="not_executed", coverage_ratio=0.9
        ),
    )

    assert verdict["all_clear"] is False


def test_all_clear_is_reachable_with_a_probe(immune):
    antigen = _antigen(0.05)
    verdict = immune._build_diagnostic_verdict(
        antigen,
        _response(_artifact()),
        coverage_report=_coverage(
            0.9, ["subsystem_identity", "error_telemetry", "health_probe"]
        ),
        verification_report=immune._default_verification_report(
            status="not_executed", coverage_ratio=0.9
        ),
    )

    assert verdict["all_clear"] is True
