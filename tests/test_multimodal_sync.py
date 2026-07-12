from __future__ import annotations

import pytest

from core.perception.multimodal_sync import (
    Calibration,
    FusionPolicy,
    MissingReason,
    Modality,
    MultimodalSynchronizer,
    PerceptualClaim,
    PerceptualEvent,
    PrivacyClass,
    PrivacyPolicy,
)

SECOND_NS = 1_000_000_000


def _event(
    event_id: str,
    modality: Modality,
    *,
    now_ns: int,
    sequence: int = 1,
    source: str | None = None,
    confidence: float = 0.9,
    claims: tuple[PerceptualClaim, ...] = (),
    missing_reason: MissingReason | None = None,
    privacy: PrivacyPolicy | None = None,
) -> PerceptualEvent:
    return PerceptualEvent(
        event_id=event_id,
        modality=modality,
        source=source or f"sensor.{modality.value}",
        sequence=sequence,
        observed_at=1_700_000_000.0,
        observed_monotonic_ns=now_ns,
        summary="bounded summary",
        confidence=0.0 if missing_reason is not None else confidence,
        claims=claims,
        calibration=Calibration("test-calibration", status="valid", reliability=1.0),
        provenance=("unit-test",),
        privacy=privacy or PrivacyPolicy(),
        missing_reason=missing_reason,
    )


def test_fusion_exposes_every_missing_modality_and_causal_repair() -> None:
    now_ns = 10 * SECOND_NS
    sync = MultimodalSynchronizer(monotonic_clock=lambda: now_ns)
    sync.ingest(
        _event(
            "device-1",
            Modality.DEVICE,
            now_ns=now_ns,
            claims=(PerceptualClaim("device.cpu_pressure", 0.2, 0.95),),
        )
    )
    sync.ingest(
        _event(
            "body-denied",
            Modality.BODY,
            now_ns=now_ns,
            missing_reason=MissingReason.PERMISSION_DENIED,
        )
    )

    frame = sync.fuse("frame-1", anchor_monotonic_ns=now_ns, emitted_at=1_700_000_000.1)

    assert frame.has_usable(Modality.DEVICE) is True
    assert frame.missing[Modality.BODY] is MissingReason.PERMISSION_DENIED
    assert frame.missing[Modality.VISION] is MissingReason.NOT_OBSERVED
    assert set(frame.missing) == set(Modality) - {Modality.DEVICE}
    assert "request-consent:body" in frame.directives.repair_requirements
    assert "acquire-evidence:body" in frame.directives.planning_constraints
    assert frame.confidence < 0.65
    assert frame.uncertainty > 0.35


def test_stale_event_is_not_reused_as_current_truth() -> None:
    now_ns = 20 * SECOND_NS
    sync = MultimodalSynchronizer(monotonic_clock=lambda: now_ns)
    sync.ingest(
        _event(
            "audio-old",
            Modality.AUDIO,
            now_ns=now_ns - SECOND_NS,
            claims=(PerceptualClaim("audio.voice_activity", True, 0.9),),
        )
    )

    frame = sync.fuse(
        "frame-stale",
        anchor_monotonic_ns=now_ns + 3 * SECOND_NS,
        emitted_at=1_700_000_003.0,
    )

    assert frame.has_usable(Modality.AUDIO) is False
    assert frame.missing[Modality.AUDIO] is MissingReason.STALE
    assert "refresh-sensor:audio" in frame.directives.repair_requirements


def test_equal_conflicting_claims_abstain_instead_of_last_write_wins() -> None:
    now_ns = 30 * SECOND_NS
    sync = MultimodalSynchronizer(monotonic_clock=lambda: now_ns)
    sync.ingest(
        _event(
            "vision-app",
            Modality.VISION,
            now_ns=now_ns,
            source="camera-a",
            claims=(PerceptualClaim("scene.person_present", True, 0.9),),
        )
    )
    sync.ingest(
        _event(
            "vision-app-b",
            Modality.VISION,
            now_ns=now_ns,
            source="camera-b",
            claims=(PerceptualClaim("scene.person_present", False, 0.9),),
        )
    )

    frame = sync.fuse("frame-conflict", anchor_monotonic_ns=now_ns)

    belief = frame.belief("scene.person_present")
    assert belief is not None
    assert belief.status == "contested"
    assert belief.value is None
    assert len(frame.unresolved_contradictions) == 1
    assert "sensor-conflict:scene.person_present" in frame.directives.attention_targets
    assert "verify-before-action:scene.person_present" in frame.directives.planning_constraints
    assert "scene.person_present" not in frame.directives.memory_candidates


def test_failed_sensor_does_not_mask_an_independent_healthy_source() -> None:
    now_ns = 35 * SECOND_NS
    sync = MultimodalSynchronizer(monotonic_clock=lambda: now_ns)
    sync.ingest(
        _event(
            "vision-healthy",
            Modality.VISION,
            now_ns=now_ns - 10,
            source="camera-a",
            confidence=0.85,
            claims=(PerceptualClaim("scene.person_present", True, 0.9),),
        )
    )
    sync.ingest(
        _event(
            "vision-failed",
            Modality.VISION,
            now_ns=now_ns,
            source="camera-b",
            missing_reason=MissingReason.SENSOR_ERROR,
        )
    )

    frame = sync.fuse("frame-source-failover", anchor_monotonic_ns=now_ns)

    assert frame.has_usable(Modality.VISION) is True
    assert frame.observations[Modality.VISION].event_id == "vision-healthy"
    assert Modality.VISION not in frame.missing


def test_stronger_independent_evidence_can_resolve_but_retains_contradiction() -> None:
    now_ns = 40 * SECOND_NS
    sync = MultimodalSynchronizer(monotonic_clock=lambda: now_ns)
    sync.ingest(
        _event(
            "vision-strong",
            Modality.VISION,
            now_ns=now_ns,
            confidence=1.0,
            claims=(PerceptualClaim("scene.person_present", True, 1.0),),
        )
    )
    sync.ingest(
        _event(
            "spatial-weak",
            Modality.SPATIAL,
            now_ns=now_ns,
            confidence=0.3,
            claims=(PerceptualClaim("scene.person_present", False, 0.3),),
        )
    )

    frame = sync.fuse("frame-resolved", anchor_monotonic_ns=now_ns)

    belief = frame.belief("scene.person_present")
    assert belief is not None
    assert belief.status == "resolved"
    assert belief.value is True
    assert frame.contradictions[0].resolved is True
    assert frame.contradictions[0].selected_value is True
    assert "scene.person_present" in frame.directives.memory_candidates


def test_queue_is_bounded_and_duplicate_future_and_expired_events_are_rejected() -> None:
    now_ns = 100 * SECOND_NS
    policy = FusionPolicy(queue_limit=4, retained_event_age_s=10.0)
    sync = MultimodalSynchronizer(policy, monotonic_clock=lambda: now_ns)

    for sequence in range(1, 6):
        receipt = sync.ingest(
            _event(
                f"device-{sequence}",
                Modality.DEVICE,
                now_ns=now_ns,
                sequence=sequence,
            )
        )
    assert receipt.overflow_drop is True
    assert receipt.queue_depth == 4
    assert sync.get_status()["queue_overflow_drops"] == 1

    duplicate = sync.ingest(
        _event("device-5", Modality.DEVICE, now_ns=now_ns, sequence=5)
    )
    future = sync.ingest(
        _event("future", Modality.VISION, now_ns=now_ns + SECOND_NS)
    )
    expired = sync.ingest(
        _event("expired", Modality.AUDIO, now_ns=now_ns - 11 * SECOND_NS)
    )

    assert duplicate.accepted is False and duplicate.reason == "duplicate_event"
    assert future.accepted is False and future.reason == "future_event_time"
    assert expired.accepted is False and expired.reason == "expired_before_ingest"


def test_small_out_of_order_event_is_accepted_but_marked_late() -> None:
    now_ns = 200 * SECOND_NS
    sync = MultimodalSynchronizer(monotonic_clock=lambda: now_ns)
    sync.ingest(
        _event(
            "audio-2",
            Modality.AUDIO,
            now_ns=now_ns,
            source="microphone",
            sequence=2,
        )
    )
    receipt = sync.ingest(
        _event(
            "audio-1",
            Modality.AUDIO,
            now_ns=now_ns - SECOND_NS // 2,
            source="microphone",
            sequence=1,
        )
    )

    assert receipt.accepted is True
    assert receipt.out_of_order is True
    assert sync.get_status()["late_events"] == 1


def test_sensitive_unredacted_evidence_requires_explicit_consent() -> None:
    with pytest.raises(ValueError, match="requires explicit consent"):
        PrivacyPolicy(
            classification=PrivacyClass.SENSITIVE,
            retention="session",
            redacted=False,
            raw_retained=True,
        )

    policy = PrivacyPolicy(
        classification=PrivacyClass.SENSITIVE,
        retention="session",
        consent_scope="operator:microphone:session-7",
        redacted=False,
        raw_retained=True,
    )
    assert policy.consent_scope == "operator:microphone:session-7"


def test_status_never_exposes_event_summary_or_provenance_content() -> None:
    now_ns = 300 * SECOND_NS
    sync = MultimodalSynchronizer(monotonic_clock=lambda: now_ns)
    event = PerceptualEvent(
        event_id="speech-private",
        modality=Modality.SPEECH,
        source="microphone",
        sequence=1,
        observed_at=1_700_000_000.0,
        observed_monotonic_ns=now_ns,
        summary="the private transcript must not appear in diagnostics",
        confidence=0.9,
        claims=(PerceptualClaim("speech.private_value", "diagnostic secret", 0.9),),
        calibration=Calibration("mic", status="valid", reliability=1.0),
        provenance=("private-media-reference",),
        privacy=PrivacyPolicy(
            classification=PrivacyClass.SENSITIVE,
            retention="none",
            redacted=True,
        ),
    )
    sync.ingest(event)
    sync.fuse("frame-private", anchor_monotonic_ns=now_ns)

    rendered = repr(sync.get_status())
    assert "the private transcript" not in rendered
    assert "private-media-reference" not in rendered
    assert "diagnostic secret" not in rendered
    assert "speech-private" in rendered
