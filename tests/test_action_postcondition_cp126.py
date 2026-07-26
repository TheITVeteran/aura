"""CP126 contract tests for core/body/action_postcondition.py.

The module's job is to answer "did the action actually do what it claimed?"
CP126 found it asking the ACTUATOR that question. These pin the replacement:
independent evidence or an honest admission that there is none.

08943467 / 92b64654 / 2468cc73 / 36ae11a1 / ebc0d1eb.
"""
from __future__ import annotations

import asyncio
import inspect

import pytest

from core.body import action_postcondition as module
from core.body.action_postcondition import (
    MAX_TELEMETRY_KEYS,
    MAX_TELEMETRY_VALUE_CHARS,
    ActionPostconditionVerifier,
    redact_telemetry,
    snapshot_path,
)


class _State:
    def __init__(self, world_model=None):
        self.world_model = {} if world_model is None else world_model


def _verify(receipt, state=None, **kwargs):
    return asyncio.run(
        ActionPostconditionVerifier().verify(receipt, state or _State(), **kwargs)
    )


# --- 08943467: the actuator does not grade its own homework --------------


def test_the_receipts_claim_is_recorded_as_a_claim(tmp_path):
    target = tmp_path / "made.txt"
    target.write_text("after")

    result = _verify({"channel": "file", "status": "success", "path": str(target)})

    assert result["claimed_success"] is True
    assert "claimed_success" in result


def test_an_unbacked_claim_is_marked_unverified():
    """No snapshot, no filesystem target, nothing to corroborate: the claim
    stands but the result must not present it as verification."""
    result = _verify({"channel": "speech", "status": "success"})

    assert result["success"] is True
    assert result["verified"] is False
    assert result["verification_source"] == "actuator_claim"


def test_a_corroborated_claim_says_so(tmp_path):
    target = tmp_path / "f.txt"
    target.write_text("v1")
    before = snapshot_path(target)
    target.write_text("v2-different")

    result = _verify(
        {"channel": "file", "status": "success", "path": str(target)}, before=before
    )

    assert result["verified"] is True
    assert result["verification_source"] == "independent_evidence"


def test_a_refuted_claim_is_overturned(tmp_path):
    """The pre-existing contract: a missing output file beats status=success."""
    result = _verify(
        {"channel": "file", "status": "success", "path": str(tmp_path / "never.txt")}
    )

    assert result["claimed_success"] is True
    assert result["success"] is False


def test_the_verdict_is_not_read_off_the_status_field_alone():
    source = inspect.getsource(module.ActionPostconditionVerifier.verify)
    # The claim exists, but `success` is only ever assigned from evidence.
    assert "claimed_success = status ==" in source
    assignments = [
        line.strip() for line in source.splitlines()
        if line.strip().startswith("success = ")
    ]
    assert assignments == ["success = False if refuted else claimed_success"]


# --- 92b64654: preconditions, snapshots and expected effects -------------


def test_presence_without_a_snapshot_is_not_proof_of_a_change(tmp_path):
    target = tmp_path / "pre_existing.txt"
    target.write_text("untouched by this action")

    result = _verify({"channel": "file", "status": "success", "path": str(target)})

    check = result["checks"][0]
    assert check["verified"] is False
    assert "not proof of a change" in check["reason"]
    assert result["evidence"]["had_pre_action_snapshot"] is False


def test_an_unchanged_file_is_reported_as_unchanged(tmp_path):
    target = tmp_path / "same.txt"
    target.write_text("identical")
    before = snapshot_path(target)

    result = _verify(
        {"channel": "file", "status": "success", "path": str(target)}, before=before
    )

    assert result["checks"][0]["changed"] is False
    assert f"file_unchanged:{target}" in result["side_effects"]


def test_a_declared_expectation_can_refute_a_success(tmp_path):
    target = tmp_path / "written.txt"
    target.write_text("wrong content")

    result = _verify(
        {"channel": "file", "status": "success", "path": str(target)},
        expected_effect={"sha256": "0" * 64},
    )

    assert result["success"] is False
    assert result["evidence"]["had_expected_effect"] is True


def test_a_met_expectation_passes(tmp_path):
    target = tmp_path / "written.txt"
    target.write_text("right content")
    digest = snapshot_path(target)["sha256"]

    result = _verify(
        {"channel": "file", "status": "success", "path": str(target)},
        expected_effect={"exists": True, "sha256": digest},
    )

    assert result["success"] is True
    assert any(c["check"] == "expected_effect" and c["verified"] for c in result["checks"])


def test_an_expected_change_that_did_not_happen_is_caught(tmp_path):
    target = tmp_path / "stale.txt"
    target.write_text("v1")
    before = snapshot_path(target)

    result = _verify(
        {"channel": "file", "status": "success", "path": str(target)},
        before=before,
        expected_effect={"changed": True},
    )

    assert result["success"] is False


def test_a_snapshot_of_a_missing_path_is_honest(tmp_path):
    snap = snapshot_path(tmp_path / "nope")

    assert snap["exists"] is False
    assert "sha256" not in snap


# --- 2468cc73: side effects are observed, not inferred -------------------


def test_a_write_that_changed_nothing_is_not_called_modified(tmp_path):
    target = tmp_path / "noop.txt"
    target.write_text("same bytes")
    before = snapshot_path(target)

    result = _verify(
        {"channel": "file", "status": "success", "action": "write", "path": str(target)},
        before=before,
    )

    assert f"modified_file:{target}" not in result["side_effects"]


def test_a_write_that_did_change_is_called_modified(tmp_path):
    target = tmp_path / "real.txt"
    target.write_text("v1")
    before = snapshot_path(target)
    target.write_text("v2 is longer and different")

    result = _verify(
        {"channel": "file", "status": "success", "action": "write", "path": str(target)},
        before=before,
    )

    assert f"modified_file:{target}" in result["side_effects"]


def test_the_side_effect_is_no_longer_read_off_the_action_field():
    source = inspect.getsource(module.ActionPostconditionVerifier)
    assert 'receipt.get("action") == "write"' not in source


def test_a_missing_expected_file_is_a_side_effect(tmp_path):
    result = _verify(
        {"channel": "file", "status": "success", "path": str(tmp_path / "gone.txt")}
    )

    assert f"expected_file_missing:{tmp_path / 'gone.txt'}" in result["side_effects"]


# --- terminal channel (4bf25067 regression guard) ------------------------


def test_absent_exit_code_is_unknown_not_zero():
    result = _verify({"channel": "terminal", "status": "success"})

    assert "process_exit_code_unreported" in result["side_effects"]
    assert result["verified"] is False


def test_a_clean_exit_corroborates():
    result = _verify({"channel": "terminal", "status": "success", "exit_code": 0})

    assert result["side_effects"] == []
    assert result["verified"] is True
    assert result["success"] is True


def test_a_nonzero_exit_overturns_a_claimed_success():
    result = _verify({"channel": "terminal", "status": "success", "exit_code": 3})

    assert "process_failed_with_code:3" in result["side_effects"]
    assert result["success"] is False
    assert result["claimed_success"] is True


def test_an_unreadable_exit_code_is_not_silently_coerced():
    result = _verify({"channel": "terminal", "status": "success", "exit_code": "fine"})

    assert "process_exit_code_unreadable" in result["side_effects"]


def test_a_terminal_receipt_can_also_carry_a_file_postcondition(tmp_path):
    result = _verify(
        {
            "channel": "terminal",
            "status": "success",
            "exit_code": 0,
            "path": str(tmp_path / "artifact.txt"),
        }
    )

    assert result["success"] is False  # the artifact was never produced


# --- ebc0d1eb: the stored telemetry is redacted and bounded --------------


def test_command_output_is_not_persisted_verbatim():
    result = _verify(
        {
            "channel": "terminal",
            "status": "success",
            "exit_code": 0,
            "stdout": "ssh-rsa AAAAB3Nz... and the contents of ~/.aws/credentials",
        }
    )

    telemetry = result["evidence"]["telemetry"]
    assert telemetry["stdout"]["_redacted"] is True
    assert "credentials" not in str(telemetry)


@pytest.mark.parametrize(
    "field", ["output", "stdout", "stderr", "clipboard", "text", "spoken", "url", "prompt"],
)
def test_every_sensitive_channel_is_redacted(field):
    safe = redact_telemetry({field: "sensitive payload"})

    assert safe[field]["_redacted"] is True
    assert safe[field]["chars"] == len("sensitive payload")


def test_a_long_benign_string_is_truncated():
    safe = redact_telemetry({"note": "x" * 5000})

    assert len(safe["note"]) == MAX_TELEMETRY_VALUE_CHARS


def test_the_key_count_is_bounded():
    safe = redact_telemetry({f"k{i}": i for i in range(MAX_TELEMETRY_KEYS + 10)})

    assert safe["_truncated_keys"] == 10
    assert len(safe) <= MAX_TELEMETRY_KEYS + 1


def test_an_inline_secret_in_a_benign_field_is_scrubbed():
    safe = redact_telemetry({"note": "deploy with sk-abcdefghijklmnopqrstuvwx now"})

    assert "sk-abcdefghijklmnopqrstuvwx" not in safe["note"]
    assert "[REDACTED]" in safe["note"]


def test_nested_structures_are_not_walked_into_world_state():
    safe = redact_telemetry({"payload": {"deep": "secret"}})

    assert safe["payload"] == "<dict>"


def test_a_non_dict_receipt_does_not_crash():
    assert redact_telemetry("not a receipt")["_shape"] == "str"
    assert _verify("not a receipt")["channel"] == "unknown"


def test_the_full_receipt_is_no_longer_embedded():
    source = inspect.getsource(module.ActionPostconditionVerifier.verify)
    assert '"telemetry": receipt' not in source


# --- 36ae11a1: the world-model write is guarded -------------------------


def test_a_state_without_a_world_model_does_not_raise():
    class _Bare:
        pass

    result = _verify({"channel": "terminal", "status": "success", "exit_code": 0}, _Bare())

    assert result["success"] is True  # the verdict is still returned


def test_a_non_mapping_world_model_does_not_raise():
    class _Odd:
        world_model = "not a mapping"

    assert _verify({"channel": "speech", "status": "success"}, _Odd())["channel"] == "speech"


def test_a_write_failure_is_reported_not_swallowed_silently(caplog):
    class _Hostile(dict):
        def __setitem__(self, key, value):
            raise TypeError("read-only world model")

    state = _State(_Hostile())
    with caplog.at_level("WARNING"):
        _verify({"channel": "speech", "status": "success"}, state)

    assert any("could not be recorded" in record.message for record in caplog.records)


def test_the_verification_still_reaches_the_world_model():
    """Downstream readers (autobiography, executive kernel, preference
    provenance) all read state.world_model['last_verification']."""
    state = _State()

    _verify({"channel": "terminal", "status": "success", "exit_code": 0}, state)

    recorded = state.world_model["last_verification"]
    assert recorded["channel"] == "terminal"
    assert "side_effects" in recorded
    assert "success" in recorded


def test_the_record_helper_reports_whether_it_wrote():
    verifier = ActionPostconditionVerifier()

    class _Bare:
        pass

    assert verifier._record(_State(), {"x": 1}) is True
    assert verifier._record(_Bare(), {"x": 1}) is False


# --- the blocking-IO contract -------------------------------------------


def test_the_filesystem_read_is_offloaded_from_the_event_loop():
    """This verifier runs inside the life tick; stat + digest must not
    execute on the loop."""
    source = inspect.getsource(module.ActionPostconditionVerifier.verify)
    assert "await snapshot_path_async" in source
    assert "asyncio.to_thread" in inspect.getsource(module.snapshot_path_async)
