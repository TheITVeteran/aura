from __future__ import annotations

from core.ontogeny import telemetry


def test_authority_rank_defaults_to_observe_without_control_points():
    assert telemetry._authority_rank({}) == 0


def test_authority_rank_tracks_the_highest_learned_control_stage():
    report = {
        "control_points": {
            "cognition.effort": {"stage": "shadow"},
            "executive.admission": {"stage": "authority"},
        }
    }

    assert telemetry._authority_rank(report) == 3


def test_observation_alarm_boundary_requires_actual_authority(monkeypatch):
    writes: list[tuple[str, float | int]] = []
    monkeypatch.setattr(telemetry, "_declared", True)
    monkeypatch.setattr(telemetry, "_DEFERRED_SPECS", {})

    import core.fsw.telemetry_dictionary as dictionary

    monkeypatch.setattr(dictionary, "write", lambda name, value: writes.append((name, value)))

    base = {
        "episodes_seen": 50_000,
        "novelty": 0.2,
        "resolution": {"swept": 500, "observed": 0, "observation_rate": 0.0},
        "calibration": {},
        "world_model": {},
    }
    telemetry.sample(
        base
        | {
            "control_points": {
                "executive.admission": {
                    "stage": "shadow",
                    "corpus": {"evidence_rows": 0},
                }
            }
        }
    )
    assert not any(name == telemetry.CHANNEL_OBSERVATION_RATE for name, _ in writes)

    telemetry.sample(
        base
        | {
            "control_points": {
                "executive.admission": {
                    "stage": "authority",
                    "corpus": {"evidence_rows": 0},
                }
            }
        }
    )
    assert (telemetry.CHANNEL_OBSERVATION_RATE, 0.0) in writes
