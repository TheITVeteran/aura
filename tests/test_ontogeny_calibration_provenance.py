from __future__ import annotations

from pathlib import Path

from core.ontogeny import telemetry
from core.ontogeny.authority import AuthorityLedger
from core.ontogeny.calibration import (
    CANDIDATE_VALIDATION,
    OPERATIONAL_SHADOW,
    CalibrationMonitor,
    CalibrationObservation,
)
from core.ontogeny.experience import Episode, ExperienceSpine, Outcome, OutcomeKind, Provenance
from core.ontogeny.features import FeatureSchema
from core.ontogeny.heads import PredictionHead
from core.ontogeny.service import OntogenyCore
from core.ontogeny.trainer import Trainer, design_width


def _observation(
    episode_id: str,
    decided_at: float,
    *,
    revision: str = "rev-a",
    head_version: int = 3,
    correct: bool = True,
) -> CalibrationObservation:
    return CalibrationObservation(
        episode_id=episode_id,
        control_point="cp",
        confidence=0.95,
        correct=correct,
        decided_at=decided_at,
        observed_at=decided_at + 1.0,
        runtime_revision=revision,
        head_version=head_version,
        action="approved",
        provenance=OPERATIONAL_SHADOW,
    )


def test_operational_window_is_chronological_and_idempotent_by_episode_id():
    monitor = CalibrationMonitor(window=3, provenance=OPERATIONAL_SHADOW)
    monitor.activate("cp", runtime_revision="rev-a", head_version=3)

    for observation in reversed([_observation(f"e{i}", float(i)) for i in range(1, 5)]):
        monitor.observe(
            "cp",
            episode_id=observation.episode_id,
            confidence=observation.confidence,
            correct=observation.correct,
            decided_at=observation.decided_at,
            observed_at=observation.observed_at,
            runtime_revision=observation.runtime_revision,
            head_version=observation.head_version,
            action=observation.action,
            provenance=observation.provenance,
        )

    assert not monitor.observe(
        "cp", episode_id="e4", confidence=0.01, correct=False, decided_at=99.0,
        runtime_revision="rev-a", head_version=3, provenance=OPERATIONAL_SHADOW,
    )
    report = monitor.report("cp")
    assert report is not None
    assert report.samples == 3
    assert report.oldest_decision_at == 2.0
    assert report.newest_decision_at == 4.0
    assert report.accuracy == 1.0


def test_repaired_runtime_is_recovery_pending_while_red_history_is_preserved():
    monitor = CalibrationMonitor(provenance=OPERATIONAL_SHADOW)
    monitor.replace_observations(
        "cp",
        [_observation(f"old-{i}", float(i), revision="broken", correct=False) for i in range(80)],
        provenance=OPERATIONAL_SHADOW,
    )
    monitor.activate("cp", runtime_revision="repaired", head_version=3)

    current = monitor.report("cp")
    assert current is not None
    assert current.status == "recovery_pending"
    assert current.samples == 0
    assert current.runtime_revision == "repaired"

    history = monitor.cohort_reports("cp")["cp"]
    broken = next(row for row in history if row["runtime_revision"] == "broken")
    assert broken["status"] == "red"
    assert broken["active"] is False
    assert broken["samples"] == 80


def test_restart_rehydrates_source_and_head_cohort_without_duplication(
    tmp_path: Path, monkeypatch,
):
    monkeypatch.setenv("AURA_LAUNCH_EXPECTED_COMMIT", "a" * 40)
    db_path = tmp_path / "experience.db"
    spine = ExperienceSpine(db_path, autoflush=False)
    for index in range(3):
        episode = Episode(
            episode_id=f"persisted-{index}",
            control_point="executive.admission",
            features={"priority": 0.5},
            decision="approved",
            options=("approved", "deferred"),
            shadow={"approved": 0.9},
            shadow_version=7,
            provenance=Provenance.TEST,
            context={"runtime_revision": "a" * 40},
            decided_at=100.0 + index,
        )
        spine.record(episode)
        spine.resolve(
            episode.episode_id,
            Outcome(kind=OutcomeKind.SUCCESS, utility=1.0, resolver="test", resolved_at=200.0 + index),
        )
    spine.flush()
    spine.close()

    restored_spine = ExperienceSpine(db_path, autoflush=False)
    core = OntogenyCore(
        spine=restored_spine,
        authority=AuthorityLedger(tmp_path / "authority.json"),
        autostart=False,
    )
    cp = core._control_points["executive.admission"]
    for head in cp.heads.values():
        head.version = 7
    core._activate_operational_cohorts()

    assert core.rehydrate_operational_calibration() == {"executive.admission": 3}
    assert core.rehydrate_operational_calibration() == {"executive.admission": 3}
    report = core.report()["operational_calibration"]["executive.admission"]
    assert report["samples"] == 3
    assert report["runtime_revision"] == "a" * 40
    assert report["head_version"] == 7
    assert report["provenance"] == OPERATIONAL_SHADOW
    core.stop()


class _EpisodeSource:
    def __init__(self, episodes: list[Episode]):
        self._episodes = episodes

    def episodes(self, *_args, **_kwargs):
        return list(reversed(self._episodes))


def test_trainer_uses_disjoint_temporal_cohorts_and_replaces_candidate_evidence(tmp_path: Path):
    schema = FeatureSchema("cp", ("signal",))
    episodes: list[Episode] = []
    for index in range(450):
        success = index % 4 != 0
        episode = Episode(
            episode_id=f"train-{index}",
            control_point="cp",
            features={"signal": float(index % 7) / 7.0},
            decision="approved",
            options=("approved",),
            provenance=Provenance.TEST,
            feature_schema=schema.schema_id,
            decided_at=float(index + 1),
            context={"runtime_revision": "trainer-rev"},
        )
        episode.outcome = Outcome(
            kind=OutcomeKind.SUCCESS if success else OutcomeKind.FAILURE,
            utility=1.0 if success else 0.0,
            resolver="test",
            resolved_at=1000.0 + index,
        )
        episodes.append(episode)

    units = 4
    monitor = CalibrationMonitor(provenance=CANDIDATE_VALIDATION)
    authority = AuthorityLedger(tmp_path / "authority.json", calibration=monitor)
    trainer = Trainer(_EpisodeSource(episodes), authority, monitor, units=units, seed=7)
    head = PredictionHead("cp.approved", ("failure", "success"), design_width(schema, units))

    first = trainer.train("cp", schema, {"approved": head}, ("approved",))
    assert first.fitted
    cohorts = first.fit_evidence["temporal_cohorts"]
    assert cohorts["training"]["last_decided_at"] < cohorts["temperature"]["first_decided_at"]
    assert cohorts["temperature"]["last_decided_at"] < cohorts["evaluation"]["first_decided_at"]
    assert first.samples + first.temperature_samples + first.holdout_samples == len(episodes) - 50
    assert monitor.report("cp").samples == first.holdout_samples
    assert head.version == 1

    second = trainer.train("cp", schema, {"approved": head}, ("approved",))
    assert second.fitted
    assert head.version == 2
    assert monitor.report("cp").head_version == 2
    assert monitor.report("cp").samples == second.holdout_samples


def test_telemetry_reports_recovery_provenance_without_replaying_red_alarm(monkeypatch):
    writes: list[tuple[str, int | float]] = []
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(telemetry, "_declared", True)
    monkeypatch.setattr(telemetry, "_DEFERRED_SPECS", {})
    monkeypatch.setattr(telemetry, "_last_calibration_status", {})

    import core.fsw.telemetry_dictionary as dictionary

    monkeypatch.setattr(dictionary, "write", lambda name, value: writes.append((name, value)))
    monkeypatch.setattr(
        dictionary,
        "emit_event",
        lambda name, **kwargs: events.append((name, kwargs)),
    )
    telemetry.sample({
        "episodes_seen": 10,
        "novelty": 0.2,
        "resolution": {},
        "control_points": {},
        "world_model": {},
        "operational_calibration": {
            "cp": {
                "status": "recovery_pending",
                "samples": 0,
                "statistically_supported": False,
                "cohort_id": "operational_shadow:runtime=repaired:head=3",
                "provenance": OPERATIONAL_SHADOW,
                "overconfidence": 0.0,
            }
        },
    })

    assert not any(name == telemetry.CHANNEL_OVERCONFIDENCE for name, _ in writes)
    assert (telemetry.CHANNEL_CALIBRATION_SAMPLES, 0) in writes
    assert (telemetry.CHANNEL_CALIBRATION_SUPPORT, 0) in writes
    assert events[0][0] == telemetry.EVENT_CALIBRATION_STATUS
    assert events[0][1]["status"] == "recovery_pending"
    assert events[0][1]["provenance"] == OPERATIONAL_SHADOW
    assert events[0][1]["statistically_supported"] is False
