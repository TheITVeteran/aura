from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from core.runtime import settings_control_plane as control_plane
from core.runtime.settings_control_plane import (
    RuntimeSettingsStore,
    SettingsConflictError,
    SettingsControlPlaneError,
    SettingsIdempotencyError,
    SettingsIntegrityError,
)


def _store(tmp_path):
    return RuntimeSettingsStore(tmp_path / "runtime.json")


def test_first_commit_is_versioned_audited_and_acknowledged(tmp_path):
    store = _store(tmp_path)

    result = store.patch(
        {"theme.reduced_motion": True},
        expected_revision=0,
        actor="test",
        request_id="first-commit",
    )

    state = json.loads((tmp_path / "runtime.json").read_text(encoding="utf-8"))
    assert state["schema"] == "aura.runtime_settings"
    assert state["schema_version"] == 2
    assert state["revision"] == 1
    assert state["values"]["theme.reduced_motion"] is True
    assert result.receipt is not None
    assert result.application_receipt is not None
    assert result.application["theme.reduced_motion"]["status"] == "awaiting_frontend"
    assert store.verify_integrity() == {
        "ok": True,
        "error": None,
        "audit_entries": 1,
        "audit_head": result.receipt["receipt_hash"],
        "state_receipt_hash": result.receipt["receipt_hash"],
        "unapplied_audit_tail": 0,
        "application_entries": 1,
        "application_head": result.application_receipt["application_hash"],
        "unacknowledged_application_receipts": 0,
        "unacknowledged_revisions": [],
    }


@pytest.mark.parametrize(
    ("changes", "error_type"),
    [
        ({"theme.reduced_motion": "false"}, TypeError),
        ({"voice.output_rate": 99.0}, ValueError),
        ({"memory.retention_days": True}, TypeError),
        ({"unknown.setting": True}, KeyError),
    ],
)
def test_invalid_patch_is_rejected_without_partial_write(
    tmp_path,
    changes,
    error_type,
):
    store = _store(tmp_path)

    with pytest.raises(error_type):
        store.patch(
            {"notify.enabled": False, **changes},
            expected_revision=0,
            request_id="invalid-patch",
        )

    assert store.snapshot().revision == 0
    assert store.get("notify.enabled") is True
    assert not (tmp_path / "runtime.json").exists()
    assert not (tmp_path / "runtime.audit.jsonl").exists()


def test_empty_patch_is_rejected(tmp_path):
    store = _store(tmp_path)

    with pytest.raises(ValueError):
        store.patch({}, expected_revision=0, request_id="empty-patch")

    assert store.snapshot().revision == 0


def test_duplicate_json_keys_fail_integrity_instead_of_last_value_wins(tmp_path):
    path = tmp_path / "runtime.json"
    path.write_text(
        '{"notify.enabled":true,"notify.enabled":false}',
        encoding="utf-8",
    )

    store = RuntimeSettingsStore(path)

    assert store.describe()["integrity"]["ok"] is False
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        store.patch(
            {"theme.reduced_motion": True},
            expected_revision=0,
            request_id="duplicate-key-state",
        )


def test_retained_history_limit_is_enforced():
    entries = [
        {
            "revision": revision,
            "updated_at": float(revision),
            "last_receipt_hash": "",
            "values": {},
        }
        for revision in range(17)
    ]

    with pytest.raises(SettingsIntegrityError, match="retained limit"):
        RuntimeSettingsStore._validated_history(entries)


def test_compare_and_swap_rejects_stale_writers(tmp_path):
    first = _store(tmp_path)
    second = _store(tmp_path)
    first.patch(
        {"notify.enabled": False},
        expected_revision=0,
        request_id="winner",
    )

    with pytest.raises(SettingsConflictError) as error:
        second.patch(
            {"theme.reduced_motion": True},
            expected_revision=0,
            request_id="stale",
        )

    assert error.value.current_revision == 1
    assert second.snapshot().values["theme.reduced_motion"] is False


def test_concurrent_stores_commit_only_one_revision_zero_writer(tmp_path):
    stores = (_store(tmp_path), _store(tmp_path))
    changes = (
        {"notify.enabled": False},
        {"theme.reduced_motion": True},
    )

    def _write(index):
        try:
            return stores[index].patch(
                changes[index],
                expected_revision=0,
                request_id=f"writer-{index}",
            ).snapshot.revision
        except SettingsConflictError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(_write, (0, 1)))

    assert sorted(outcomes, key=str) == [1, "conflict"]
    assert _store(tmp_path).snapshot().revision == 1


def test_compatibility_set_has_bounded_conflict_retries(tmp_path, monkeypatch):
    store = _store(tmp_path)
    attempts = 0

    def _always_conflict(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        raise SettingsConflictError(0, 1)

    monkeypatch.setattr(store, "patch", _always_conflict)

    with pytest.raises(SettingsControlPlaneError, match="bounded conflict retry"):
        store.set("notify.enabled", False)

    assert attempts == 8


def test_request_id_is_idempotent_but_not_reusable(tmp_path):
    store = _store(tmp_path)
    first = store.patch(
        {"notify.enabled": False},
        expected_revision=0,
        request_id="stable-request",
    )
    replay = store.patch(
        {"notify.enabled": False},
        expected_revision=0,
        request_id="stable-request",
    )

    assert replay.replayed is True
    assert replay.snapshot.revision == 1
    assert replay.receipt == first.receipt
    with pytest.raises(SettingsIdempotencyError):
        store.patch(
            {"notify.enabled": True},
            expected_revision=1,
            request_id="stable-request",
        )


def test_idempotent_replay_reports_when_a_newer_revision_superseded_it(tmp_path):
    store = _store(tmp_path)
    first = store.patch(
        {"notify.enabled": False},
        expected_revision=0,
        request_id="first-request",
    )
    store.patch(
        {"notify.enabled": True},
        expected_revision=1,
        request_id="newer-request",
    )

    replay = store.patch(
        {"notify.enabled": False},
        expected_revision=0,
        request_id="first-request",
    )
    payload = replay.public()

    assert replay.receipt == first.receipt
    assert replay.replayed is True
    assert replay.superseded is True
    assert replay.superseded_by_revision == 2
    assert replay.snapshot.revision == 2
    assert replay.snapshot.values["notify.enabled"] is True
    assert payload["superseded"] is True
    assert payload["superseded_by_revision"] == 2


def test_commit_response_cannot_regress_after_concurrent_owner_dispatch(tmp_path):
    first = _store(tmp_path)
    second = _store(tmp_path)

    def _concurrent_owner(_key, _previous, _value):
        second.patch(
            {"notify.enabled": True},
            expected_revision=1,
            request_id="dispatch-race-winner",
        )
        return {"status": "applied", "detail": "older value was briefly applied"}

    first.subscribe(
        _concurrent_owner,
        owner="concurrent_owner",
        keys={"notify.enabled"},
    )

    result = first.patch(
        {"notify.enabled": False},
        expected_revision=0,
        request_id="dispatch-race-first",
    )

    assert result.snapshot.revision == 2
    assert result.snapshot.values["notify.enabled"] is True
    assert result.superseded is True
    assert result.superseded_by_revision == 2
    assert result.application["notify.enabled"] == {
        "owner": "desktop_notifications",
        "status": "superseded",
        "detail": "a newer durable revision replaced this value during live-owner dispatch",
    }


def test_legacy_flat_state_migrates_on_first_mutation(tmp_path):
    path = tmp_path / "runtime.json"
    path.write_text(
        json.dumps(
            {
                "autonomy.proactive_messaging": False,
                "voice.output_enabled": False,
                "retired.key": "ignored",
            }
        ),
        encoding="utf-8",
    )
    store = RuntimeSettingsStore(path)

    before = store.snapshot()
    assert before.migrated_from == "legacy_flat_v1"
    assert before.unknown_keys == ("retired.key",)
    assert before.values["autonomy.proactive_messaging"] == "never"
    result = store.patch(
        {"voice.output_enabled": True},
        expected_revision=0,
        request_id="legacy-migration",
    )

    assert result.snapshot.revision == 1
    assert result.receipt["metadata"] == {
        "dropped_unknown_keys": ["retired.key"],
        "migrated_from": "legacy_flat_v1",
    }
    assert store.verify_integrity()["ok"] is True


def test_rollback_creates_a_new_audited_revision(tmp_path):
    store = _store(tmp_path)
    store.patch(
        {"theme.reduced_motion": True},
        expected_revision=0,
        request_id="revision-1",
    )
    store.patch(
        {"notify.enabled": False},
        expected_revision=1,
        request_id="revision-2",
    )

    rolled_back = store.rollback(
        1,
        expected_revision=2,
        request_id="rollback-to-1",
    )

    assert rolled_back.snapshot.revision == 3
    assert rolled_back.snapshot.values["theme.reduced_motion"] is True
    assert rolled_back.snapshot.values["notify.enabled"] is True
    assert rolled_back.receipt["operation"] == "rollback"
    assert rolled_back.receipt["metadata"] == {"target_revision": 1}
    assert store.verify_integrity()["audit_entries"] == 3


def test_subscriber_failure_is_visible_and_does_not_undo_durable_state(tmp_path):
    store = _store(tmp_path)

    def _broken_owner(_key, _previous, _value):
        raise RuntimeError("owner unavailable")

    store.subscribe(_broken_owner, owner="broken_owner")
    result = store.patch(
        {"notify.enabled": False},
        expected_revision=0,
        request_id="subscriber-failure",
    )

    assert result.snapshot.values["notify.enabled"] is False
    assert result.application["notify.enabled"]["status"] == "failed"
    assert "owner unavailable" in result.application["notify.enabled"]["detail"]
    assert result.application_receipt is not None
    assert store.verify_integrity()["unacknowledged_application_receipts"] == 0


def test_invalid_subscriber_status_is_recorded_as_owner_failure(tmp_path):
    store = _store(tmp_path)
    store.subscribe(
        lambda *_args: {"status": "probably"},
        owner="invalid_status_owner",
    )

    result = store.patch(
        {"notify.enabled": False},
        expected_revision=0,
        request_id="invalid-subscriber-status",
    )

    assert result.snapshot.values["notify.enabled"] is False
    assert result.application["notify.enabled"]["status"] == "failed"
    assert "invalid settings application status" in result.application[
        "notify.enabled"
    ]["detail"]


def test_frontend_owner_can_append_application_acknowledgement(tmp_path):
    store = _store(tmp_path)
    result = store.patch(
        {"voice.auto_listen": True},
        expected_revision=0,
        request_id="frontend-setting",
    )

    acknowledged = store.acknowledge_application(
        result.receipt["receipt_hash"],
        {
            "voice.auto_listen": {
                "status": "applied",
                "detail": "desktop microphone lane started",
            }
        },
        actor="desktop_test",
    )

    assert acknowledged["application"]["voice.auto_listen"] == {
        "owner": "desktop_voice_shell",
        "status": "applied",
        "detail": "desktop microphone lane started",
    }
    report = store.verify_integrity()
    assert report["application_entries"] == 2
    assert report["unacknowledged_application_receipts"] == 0
    reloaded = _store(tmp_path)
    assert reloaded.describe()["application"]["voice.auto_listen"]["status"] == "applied"


def test_application_acknowledgement_cannot_cover_unrelated_setting(tmp_path):
    store = _store(tmp_path)
    result = store.patch(
        {"voice.auto_listen": True},
        expected_revision=0,
        request_id="frontend-setting-scope",
    )

    with pytest.raises(KeyError):
        store.acknowledge_application(
            result.receipt["receipt_hash"],
            {"notify.enabled": {"status": "applied"}},
        )

    assert store.verify_integrity()["application_entries"] == 1


def test_interrupted_commit_replays_prepared_receipt_without_revision_reuse(
    tmp_path,
    monkeypatch,
):
    store = _store(tmp_path)
    real_write = control_plane.atomic_write_text
    failed = False

    def _fail_state_once(path, text, **kwargs):
        nonlocal failed
        if not failed and str(path).endswith("runtime.json"):
            failed = True
            raise OSError("simulated power loss before state replace")
        return real_write(path, text, **kwargs)

    monkeypatch.setattr(control_plane, "atomic_write_text", _fail_state_once)
    with pytest.raises(OSError):
        store.patch(
            {"notify.enabled": False},
            expected_revision=0,
            request_id="prepared-transaction",
        )
    monkeypatch.setattr(control_plane, "atomic_write_text", real_write)

    recovered = _store(tmp_path)
    snapshot = recovered.snapshot()
    assert snapshot.revision == 1
    assert snapshot.values["notify.enabled"] is False
    report = recovered.verify_integrity()
    assert report["audit_entries"] == 1
    assert report["unacknowledged_application_receipts"] == 1

    replay = recovered.patch(
        {"notify.enabled": False},
        expected_revision=0,
        request_id="prepared-transaction",
    )
    assert replay.replayed is True
    assert replay.application["notify.enabled"]["status"] == "unconfirmed"
    assert recovered.snapshot().revision == 1


@pytest.mark.parametrize("target", ["mutation", "state", "application"])
def test_tampering_is_detected_and_blocks_future_mutation(tmp_path, target):
    store = _store(tmp_path)
    store.patch(
        {"notify.enabled": False},
        expected_revision=0,
        request_id="tamper-source",
    )
    paths = {
        "mutation": tmp_path / "runtime.audit.jsonl",
        "state": tmp_path / "runtime.json",
        "application": tmp_path / "runtime.application.jsonl",
    }
    path = paths[target]
    path.write_text(
        path.read_text(encoding="utf-8").replace("notify.enabled", "notify.enabld"),
        encoding="utf-8",
    )

    damaged = _store(tmp_path)
    assert damaged.describe()["integrity"]["ok"] is False
    with pytest.raises(SettingsIntegrityError):
        damaged.patch(
            {"theme.reduced_motion": True},
            expected_revision=0,
            request_id="must-not-write",
        )
