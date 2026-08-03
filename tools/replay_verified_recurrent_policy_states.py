#!/usr/bin/env python3
"""Execute exact recurrent policy-state replay in one isolated model process."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Never

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.execution_spec import (  # noqa: E402
    RLCExecutionSpec,
)
from core.learning.recurrent_grpo import (  # noqa: E402
    attach_recurrent_policy_adapters,
)
from core.learning.verified_recurrent_transition_repository import (  # noqa: E402
    EXTERNAL_POLICY_STATE_REPLAY_BATCH_SCHEMA,
    EXTERNAL_POLICY_STATE_REPLAY_REQUEST_PURPOSE,
    EXTERNAL_POLICY_STATE_REPLAY_REQUEST_SCHEMA,
    campaign_trust_policy_from_verifier_material,
    replay_recurrent_evidence_manifest_policy_states,
)
from core.learning.verified_transition_episode import (  # noqa: E402
    canonical_json_bytes,
)
from core.learning.verified_transition_policy_state_replay import (  # noqa: E402
    validate_policy_state_replay_contract,
)
from core.runtime.atomic_writer import (  # noqa: E402
    atomic_write_bytes_if_absent,
)
from core.runtime.file_read_gateway import read_stable_bytes  # noqa: E402

REQUEST_SCHEMA = EXTERNAL_POLICY_STATE_REPLAY_REQUEST_SCHEMA
REQUEST_PURPOSE = EXTERNAL_POLICY_STATE_REPLAY_REQUEST_PURPOSE
RESULT_SCHEMA = EXTERNAL_POLICY_STATE_REPLAY_BATCH_SCHEMA
_MAX_DOCUMENT_BYTES = 512 * 1024 * 1024
_REQUEST_KEYS = frozenset(
    {
        "schema",
        "purpose",
        "evidence_manifest",
        "policy_state_replay_contract",
        "campaign_trust_policy",
        "verifier_identity",
        "verified_at_unix",
        "request_sha256",
    }
)


class ExternalPolicyStateReplayError(RuntimeError):
    """The detached replay request or result is unsafe or inconsistent."""


def _fail(code: str) -> Never:
    raise ExternalPolicyStateReplayError(code)


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _private_regular(path: Path, *, must_exist: bool) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        _fail("policy_state_replay_path_not_absolute")
    if candidate.is_symlink():
        _fail("policy_state_replay_path_symlink_rejected")
    resolved = candidate.resolve(strict=must_exist)
    if must_exist:
        metadata = resolved.stat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
            or metadata.st_nlink != 1
        ):
            _fail("policy_state_replay_file_not_private")
    else:
        parent = resolved.parent.resolve(strict=True)
        metadata = parent.stat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            _fail("policy_state_replay_parent_not_private")
    return resolved


def _strict_document(path: Path) -> dict[str, Any]:
    resolved = _private_regular(path, must_exist=True)
    payload = read_stable_bytes(resolved, max_bytes=_MAX_DOCUMENT_BYTES)

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                _fail("policy_state_replay_document_duplicate_key")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload,
            object_pairs_hook=pairs,
            parse_constant=lambda _value: _fail("policy_state_replay_document_nonfinite"),
        )
    except ExternalPolicyStateReplayError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ExternalPolicyStateReplayError("policy_state_replay_document_invalid") from exc
    if not isinstance(value, dict) or payload != canonical_json_bytes(value):
        _fail("policy_state_replay_document_noncanonical")
    return value


def validate_request(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _REQUEST_KEYS:
        _fail("policy_state_replay_request_schema_invalid")
    document = json.loads(canonical_json_bytes(value))
    unsigned = dict(document)
    observed = unsigned.pop("request_sha256")
    if (
        document.get("schema") != REQUEST_SCHEMA
        or document.get("purpose") != REQUEST_PURPOSE
        or not isinstance(observed, str)
        or observed != _digest(unsigned)
        or not isinstance(document.get("evidence_manifest"), Mapping)
        or not isinstance(
            document.get("policy_state_replay_contract"),
            Mapping,
        )
        or not isinstance(document.get("campaign_trust_policy"), Mapping)
        or not isinstance(document.get("verifier_identity"), str)
        or not document["verifier_identity"]
        or document["verifier_identity"] != document["verifier_identity"].strip()
        or type(document.get("verified_at_unix")) is not int
        or document["verified_at_unix"] <= 0
    ):
        _fail("policy_state_replay_request_invalid")
    return document


def _validate_result(value: Any, *, request_sha256: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail("policy_state_replay_result_invalid")
    document = json.loads(canonical_json_bytes(value))
    transitions = document.get("transition_results")
    unsigned = dict(document)
    observed = unsigned.pop("result_sha256", None)
    if (
        set(document)
        != {
            "schema",
            "request_sha256",
            "policy_state_replay_contract_sha256",
            "evidence_manifest_sha256",
            "verifier_identity",
            "verified_at_unix",
            "transition_results",
            "transition_result_root_sha256",
            "completed_at_unix",
            "result_sha256",
        }
        or document.get("schema") != RESULT_SCHEMA
        or document.get("request_sha256") != request_sha256
        or not isinstance(transitions, list)
        or [row.get("sequence") for row in transitions]
        != sorted({row.get("sequence") for row in transitions})
        or document.get("transition_result_root_sha256")
        != _digest(
            [
                {
                    "sequence": row.get("sequence"),
                    "receipt_sha256": row.get("receipt_sha256"),
                }
                for row in transitions
            ]
        )
        or type(document.get("completed_at_unix")) is not int
        or document["completed_at_unix"] <= 0
        or observed != _digest(unsigned)
    ):
        _fail("policy_state_replay_result_invalid")
    return document


def _publish(path: Path, document: Mapping[str, Any]) -> None:
    resolved = _private_regular(path, must_exist=False)
    payload = canonical_json_bytes(dict(document))
    if atomic_write_bytes_if_absent(resolved, payload, mode=0o600):
        return
    if _strict_document(resolved) != dict(document):
        _fail("policy_state_replay_result_conflict")


def _execute(request_path: Path, result_path: Path) -> dict[str, Any]:
    request = validate_request(_strict_document(request_path))
    if result_path.exists():
        return _validate_result(
            _strict_document(result_path),
            request_sha256=request["request_sha256"],
        )
    contract = validate_policy_state_replay_contract(
        request["policy_state_replay_contract"],
        verify_files=True,
        verify_model=True,
    )
    policy = campaign_trust_policy_from_verifier_material(request["campaign_trust_policy"])
    try:
        from mlx_lm import load

        from core.runtime.model_lane_control import (
            estimate_model_job_footprint_gb,
            standalone_model_lane,
        )

        # Replay loads the same resident-class weights the live runtime uses.
        # Without a lane lease it can land beside a live model and double the
        # wired footprint.
        _model_path = contract["model"]["path"]
        with standalone_model_lane(
            owner_id="verified-recurrent-policy-replay",
            model_path=_model_path,
            purpose="replay",
            request_gb=estimate_model_job_footprint_gb(_model_path, purpose="replay"),
            metadata={"tool": "replay_verified_recurrent_policy_states"},
        ):
            loaded = load(_model_path)
        model = loaded[0] if isinstance(loaded, tuple) else loaded
        spec = RLCExecutionSpec.from_dict(json.loads(contract["execution_spec"]["document_json"]))
        adapter = contract["initial_policy_state_custody"]["adapter_initialization"]
        attach_recurrent_policy_adapters(
            model,
            spec,
            lora_rank=adapter["rank"],
            lora_layers=adapter["layers"],
            lora_targets=adapter["targets"],
            initialization_seed=adapter["seed"],
        )
        transition_results = list(
            replay_recurrent_evidence_manifest_policy_states(
                request["evidence_manifest"],
                policy_state_replay_contract=contract,
                campaign_trust_policy=policy,
                verifier_identity=request["verifier_identity"],
                verified_at_unix=request["verified_at_unix"],
                model=model,
            )
        )
    except ExternalPolicyStateReplayError:
        raise
    except Exception as exc:
        raise ExternalPolicyStateReplayError("policy_state_replay_execution_failed") from exc
    evidence = request["evidence_manifest"]
    body = {
        "schema": RESULT_SCHEMA,
        "request_sha256": request["request_sha256"],
        "policy_state_replay_contract_sha256": contract["contract_sha256"],
        "evidence_manifest_sha256": evidence["manifest_sha256"],
        "verifier_identity": request["verifier_identity"],
        "verified_at_unix": request["verified_at_unix"],
        "transition_results": transition_results,
        "transition_result_root_sha256": _digest(
            [
                {
                    "sequence": row["sequence"],
                    "receipt_sha256": row["receipt_sha256"],
                }
                for row in transition_results
            ]
        ),
        "completed_at_unix": max(
            int(time.time()),
            request["verified_at_unix"],
        ),
    }
    result = _validate_result(
        {**body, "result_sha256": _digest(body)},
        request_sha256=request["request_sha256"],
    )
    _publish(result_path, result)
    return result


def _resume_verdict(request_path: Path, result_path: Path) -> dict[str, Any]:
    request = validate_request(_strict_document(request_path))
    result_state = "absent"
    if result_path.exists():
        try:
            _validate_result(
                _strict_document(result_path),
                request_sha256=request["request_sha256"],
            )
            result_state = "complete"
        except ExternalPolicyStateReplayError:
            result_state = "invalid"
    verdict = "safe_to_resume" if result_state != "invalid" else "indeterminate"
    plan = os.environ["AURA_DETACHED_PLAN_SHA256"]
    command = os.environ["AURA_DETACHED_COMMAND_SHA256"]
    attempt = int(os.environ["AURA_DETACHED_PRIOR_ATTEMPT"])
    head = os.environ["AURA_DETACHED_PRIOR_JOURNAL_HEAD_SHA256"]
    if os.environ["AURA_DETACHED_RESUME_EVIDENCE_TRANSPORT"] != "stdout-v3":
        _fail("policy_state_replay_resume_evidence_transport_invalid")
    evidence = {
        "schema": "aura.detached_step.resume_evidence.v2",
        "plan_sha256": plan,
        "command_sha256": command,
        "prior_attempt": attempt,
        "prior_journal_head_sha256": head,
        "checkpoint_sequence": (
            len(request["evidence_manifest"]["updated_replay_sequences"])
            if result_state == "complete"
            else 0
        ),
        "request_sha256": request["request_sha256"],
        "result_state": result_state,
    }
    # The runner hashes the evidence object it receives over stdout-v3, so this
    # digest must not include the trailing newline the old file transport wrote.
    evidence_sha = hashlib.sha256(canonical_json_bytes(evidence)).hexdigest()
    checkpoint_sequence = evidence["checkpoint_sequence"]
    checkpoint_identity = _digest(
        {
            "prior_attempt": attempt,
            "prior_journal_head_sha256": head,
            "checkpoint_sequence": checkpoint_sequence,
            "evidence_sha256": evidence_sha,
        }
    )
    return {
        "schema": "aura.detached_step.resume_verdict.v3",
        "plan_sha256": plan,
        "command_sha256": command,
        "prior_attempt": attempt,
        "prior_journal_head_sha256": head,
        "checkpoint_sequence": checkpoint_sequence,
        "checkpoint_identity": checkpoint_identity,
        "verdict": verdict,
        "evidence_sha256": evidence_sha,
        "evidence": evidence,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for name in ("run", "resume-verdict"):
        command = subparsers.add_parser(name)
        command.add_argument("--request", required=True, type=Path)
        command.add_argument("--result", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.action == "run":
            result = _execute(args.request, args.result)
            summary = {
                "request_sha256": result["request_sha256"],
                "result_sha256": result["result_sha256"],
                "transition_count": len(result["transition_results"]),
            }
            print(canonical_json_bytes(summary).decode("ascii"))
        else:
            print(canonical_json_bytes(_resume_verdict(args.request, args.result)).decode("ascii"))
    except ExternalPolicyStateReplayError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
