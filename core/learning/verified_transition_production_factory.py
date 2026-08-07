"""Production construction and pre-sampling custody for verified training.

This module owns the boundary that cannot be represented by a preconstructed
test provider: the recurrent policy exists only after the resident model is
loaded and its scoped adapters are attached.  The factory validates that live
policy, binds the frozen task schedule, and wraps the provider with a durable
just-in-time plan issuer.  Every plan is externally signed and persisted
before the first model token for that group is sampled.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Never, Protocol, cast

from core.brain.llm.latent_cortex.campaign_trust import (
    EVIDENCE_VERIFIER,
    TASK_ISSUER,
    VerifiedCampaignTrustPolicy,
    assemble_role_attestation,
    operationally_isolated_roles,
    prepare_role_signature_request,
)
from core.learning.durable_external_verifier_job import (
    DurableExternalVerifierJob,
)
from core.learning.recurrent_grpo import (
    RecurrentSamplingConfig,
    recurrent_policy_sha256,
    recurrent_sampling_rng_root_sha256,
)
from core.learning.verified_token_trace import validate_tokenizer_bundle_identity
from core.learning.verified_training_task import (
    VerifiedTrainingTaskError,
    build_verified_training_task,
    validate_public_training_task,
)
from core.learning.verified_transition_causal_campaign import (
    validate_causal_campaign_evidence_manifest,
    validate_external_evidence_verification_receipt,
)
from core.learning.verified_transition_episode import canonical_json_bytes
from core.learning.verified_transition_group_admission import (
    TransitionGroupPlanEntry,
    build_transition_group_manifest,
    sampling_config_document_sha256,
    validate_transition_group_manifest,
)
from core.learning.verified_transition_policy_probe import (
    inspect_initial_adapter_snapshot,
    inspect_initial_optimizer_snapshot,
    validate_initial_policy_state_custody,
)
from core.learning.verified_transition_provider import (
    ProductionVerifiedTransitionGroupProvider,
    VerifiedTransitionProviderError,
    callable_source_sha256,
    validate_verified_transition_provider_contract,
)
from core.learning.verified_transition_reward import TransitionRewardConfig
from core.learning.verified_transition_trainer import (
    VerifiedTransitionGroupProvider,
    VerifiedTransitionProviderRuntime,
    VerifiedTransitionSamplingPlan,
    VerifiedTransitionTrainingScheduleEntry,
)
from core.runtime.atomic_writer import (
    atomic_write_bytes,
    atomic_write_bytes_if_absent,
    ensure_private_directory,
    interprocess_file_lock,
)
from core.runtime.detached_subprocess_broker import (
    DetachedBrokerError,
    broker_available,
    run_brokered_process,
)
from core.runtime.file_read_gateway import read_stable_bytes
from core.runtime.subprocess_gateway import get_subprocess_gateway

JIT_PROVIDER_CONFIG_SCHEMA = "aura.verified_transition.jit_provider_config.v1"
JIT_PLAN_PACKAGE_SCHEMA = "aura.verified_transition.jit_plan_package.v1"
JIT_PLAN_INTENT_SCHEMA = "aura.verified_transition.jit_plan_intent.v1"
SAMPLING_CONFIG_CONTRACT_SCHEMA = "aura.recurrent_sampling_config.fixed_point.v1"
COMMAND_SIGNER_REQUEST_SCHEMA = "aura.external_role_signer.request.v1"
COMMAND_SIGNER_RESPONSE_SCHEMA = "aura.external_role_signer.response.v1"
COMMAND_EVIDENCE_VERIFIER_REQUEST_SCHEMA = "aura.external_evidence_verifier.request.v2"
COMMAND_EVIDENCE_VERIFIER_RESPONSE_SCHEMA = "aura.external_evidence_verifier.response.v1"

_JIT_CONFIG_KEYS = frozenset(
    {
        "schema",
        "reward_config_sha256",
        "sampling_config",
        "branch_count",
        "signer_broker_identity",
        "signer_broker_source_sha256",
        "plan_store_root",
        "trainer_output_root",
        "transaction_root",
    }
)
_PLAN_PACKAGE_KEYS = frozenset(
    {
        "schema",
        "contract_sha256",
        "campaign_schedule_root_sha256",
        "sequence",
        "policy_before_sha256",
        "group_manifest",
        "group_manifest_attestation",
        "lineage_plan",
        "lineage_attestation",
        "admitted_at_unix_ns",
        "receipt_sha256",
    }
)
_SIGNER_RESPONSE_KEYS = frozenset({"schema", "request_sha256", "signature_b64"})
_VERIFIER_RESPONSE_KEYS = frozenset({"schema", "request_sha256", "verification_receipt"})
_SAMPLING_FIXED_FIELDS = (
    "temperature",
    "top_p",
    "max_abs_logprob_drift",
    "max_mean_abs_logprob_drift",
    "clip_epsilon",
    "max_clipped_token_fraction",
    "max_old_policy_approx_kl",
)
_SAMPLING_CONTRACT_KEYS = frozenset(
    {"schema", "max_tokens"} | {f"{field}_micros" for field in _SAMPLING_FIXED_FIELDS}
)
_TRAINING_ARGV_SHA256_KEY = "training_argv_sha256"


class VerifiedTransitionProductionFactoryError(RuntimeError):
    """Stable fail-closed production construction error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise VerifiedTransitionProductionFactoryError(code)


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256(value: Any, *, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{role}_invalid")
    return value


def _identifier(value: Any, *, role: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 256
        or not value[0].isalnum()
        or any(not (character.isalnum() or character in "._:/;=+-") for character in value)
    ):
        _fail(f"{role}_invalid")
    return value


def _integer(value: Any, *, role: str, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= (1 << 63) - 1:
        _fail(f"{role}_invalid")
    return value


def _clone(value: Any, *, role: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, UnicodeError, RecursionError, OverflowError):
        _fail(f"{role}_not_canonical_json")


def _private_directory(path: str | Path, *, role: str) -> Path:
    lexical = Path(path).expanduser().absolute()
    if lexical.is_symlink():
        _fail(f"{role}_symlink_rejected")
    directory = ensure_private_directory(lexical).resolve(strict=True)
    metadata = directory.stat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        _fail(f"{role}_not_private_owned_directory")
    return directory


def sampling_config_contract_document(
    config: RecurrentSamplingConfig,
) -> dict[str, Any]:
    """Encode sampling thresholds as exact fixed-point contract integers."""

    values = config.to_dict()
    document: dict[str, Any] = {
        "schema": SAMPLING_CONFIG_CONTRACT_SCHEMA,
        "max_tokens": config.max_tokens,
    }
    for field in _SAMPLING_FIXED_FIELDS:
        value = float(values[field])
        micros = round(value * 1_000_000)
        if abs(value - micros / 1_000_000) > 1e-12:
            _fail(f"sampling_config_{field}_not_fixed_point")
        document[f"{field}_micros"] = micros
    return document


def _sampling_config_from_contract(value: Any) -> RecurrentSamplingConfig:
    if not isinstance(value, Mapping) or set(value) != _SAMPLING_CONTRACT_KEYS:
        _fail("production_factory_sampling_config_schema_invalid")
    if value.get("schema") != SAMPLING_CONFIG_CONTRACT_SCHEMA:
        _fail("production_factory_sampling_config_schema_invalid")
    max_tokens = _integer(
        value.get("max_tokens"),
        role="production_factory_sampling_max_tokens",
        minimum=1,
    )
    fields: dict[str, float] = {}
    for field in _SAMPLING_FIXED_FIELDS:
        micros = _integer(
            value.get(f"{field}_micros"),
            role=f"production_factory_sampling_{field}",
        )
        fields[field] = micros / 1_000_000
    try:
        config = RecurrentSamplingConfig(max_tokens=max_tokens, **fields)
    except (TypeError, ValueError) as exc:
        raise VerifiedTransitionProductionFactoryError(
            "production_factory_sampling_config_invalid"
        ) from exc
    if sampling_config_contract_document(config) != dict(value):
        _fail("production_factory_sampling_config_reconstruction_mismatch")
    return config


class ExternalRoleSignerBroker(Protocol):
    """Externally custodied role signer; Aura never receives its private key."""

    @property
    def identity(self) -> str: ...

    @property
    def source_sha256(self) -> str: ...

    @property
    def implementation_sha256(self) -> str: ...

    @property
    def release_sha256(self) -> str: ...

    @property
    def custody_evidence_sha256(self) -> str: ...

    def attest(
        self,
        policy: VerifiedCampaignTrustPolicy,
        *,
        role: str,
        payload: Mapping[str, Any],
        signed_at_unix: int,
        purpose: str,
    ) -> Mapping[str, Any]: ...

    def verify_evidence_manifest(
        self,
        policy: VerifiedCampaignTrustPolicy,
        *,
        evidence_manifest: Mapping[str, Any],
        verified_at_unix: int,
        purpose: str,
    ) -> Mapping[str, Any]: ...


def detached_signer_broker_paths(
    identity: str,
    release_manifest: str | Path,
) -> tuple[Path, Path]:
    token = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    root = Path(release_manifest).expanduser().resolve(strict=True).parent
    return (
        root / f".signer-request-{token}.json",
        root / f".signer-response-{token}.json",
    )


class CommandRoleSignerBroker:
    """Call an absolute, pinned external signer command without a shell.

    The command receives one canonical JSON request on stdin and must return
    one canonical JSON response plus a trailing newline.  The private key is
    never accepted through argv, environment values, or this process.
    """

    def __init__(
        self,
        *,
        identity: str,
        executable: str | Path,
        executable_sha256: str,
        release_manifest: str | Path,
        custody_evidence: str | Path,
        arguments: Sequence[str] = (),
        timeout_seconds: float = 30.0,
        inherited_environment_names: Sequence[str] = ("HOME", "TMPDIR"),
        durable_policy_state_replay_job: DurableExternalVerifierJob | None = None,
    ) -> None:
        self._identity = _identifier(identity, role="signer_broker_identity")
        candidate = Path(executable).expanduser()
        if not candidate.is_absolute() or candidate.is_symlink():
            _fail("signer_broker_executable_invalid")
        try:
            resolved = candidate.resolve(strict=True)
            metadata = resolved.stat()
        except OSError as exc:
            raise VerifiedTransitionProductionFactoryError(
                "signer_broker_executable_unavailable"
            ) from exc
        if not stat.S_ISREG(metadata.st_mode) or not os.access(resolved, os.X_OK):
            _fail("signer_broker_executable_invalid")
        if (
            metadata.st_uid != os.getuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            _fail("signer_broker_executable_not_owned_immutable")
        self._executable = resolved
        self._executable_sha256 = _sha256(executable_sha256, role="signer_broker_executable_sha256")
        self._release_manifest = self._regular_artifact(
            release_manifest, role="signer_broker_release_manifest"
        )
        self._custody_evidence = self._regular_artifact(
            custody_evidence, role="signer_broker_custody_evidence"
        )
        self._arguments = tuple(arguments)
        if any(not isinstance(argument, str) or "\x00" in argument for argument in self._arguments):
            _fail("signer_broker_arguments_invalid")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not 0.1 <= float(timeout_seconds) <= 300.0
        ):
            _fail("signer_broker_timeout_invalid")
        self._timeout_seconds = float(timeout_seconds)
        names = tuple(inherited_environment_names)
        if any(
            not isinstance(name, str)
            or not name
            or not name.replace("_", "").isalnum()
            or name.startswith("AURA_")
            for name in names
        ):
            _fail("signer_broker_environment_names_invalid")
        self._environment_names = names
        self._durable_policy_state_replay_job = durable_policy_state_replay_job
        self._assert_executable_identity()

    @property
    def identity(self) -> str:
        return self._identity

    @property
    def source_sha256(self) -> str:
        return callable_source_sha256(type(self).attest)

    @property
    def implementation_sha256(self) -> str:
        return self._executable_sha256

    @property
    def release_sha256(self) -> str:
        return hashlib.sha256(
            read_stable_bytes(self._release_manifest, max_bytes=64 * 1024 * 1024)
        ).hexdigest()

    @property
    def custody_evidence_sha256(self) -> str:
        return hashlib.sha256(
            read_stable_bytes(self._custody_evidence, max_bytes=64 * 1024 * 1024)
        ).hexdigest()

    @staticmethod
    def _regular_artifact(value: str | Path, *, role: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute() or path.is_symlink():
            _fail(f"{role}_invalid")
        try:
            resolved = path.resolve(strict=True)
            metadata = resolved.stat()
        except OSError as exc:
            raise VerifiedTransitionProductionFactoryError(f"{role}_unavailable") from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            _fail(f"{role}_invalid")
        return resolved

    def _assert_executable_identity(self) -> None:
        observed = hashlib.sha256(
            read_stable_bytes(self._executable, max_bytes=512 * 1024 * 1024)
        ).hexdigest()
        if observed != self._executable_sha256:
            _fail("signer_broker_executable_identity_mismatch")

    def _execute(self, envelope: Mapping[str, Any]) -> dict[str, Any]:
        request_bytes = canonical_json_bytes(envelope) + b"\n"
        environment = {
            "LANG": "C",
            "LC_ALL": "C",
            **{name: os.environ[name] for name in self._environment_names if name in os.environ},
        }
        self._assert_executable_identity()
        if broker_available():
            request_path, response_path = detached_signer_broker_paths(
                self._identity,
                self._release_manifest,
            )
            atomic_write_bytes(request_path, request_bytes, mode=0o600)
            response_path.unlink(missing_ok=True)
            try:
                result = run_brokered_process(
                    [
                        str(self._executable),
                        *self._arguments,
                        "--request-file",
                        str(request_path),
                    ],
                    cwd=Path.cwd().resolve(strict=True),
                    stdout_path=response_path,
                    timeout_s=self._timeout_seconds,
                )
                stdout_bytes = read_stable_bytes(
                    response_path,
                    max_bytes=64 * 1024 + 1,
                )
            except (DetachedBrokerError, OSError) as exc:
                raise VerifiedTransitionProductionFactoryError(
                    "signer_broker_execution_failed"
                ) from exc
            finally:
                request_path.unlink(missing_ok=True)
            self._assert_executable_identity()
            if result.returncode != 0:
                _fail("signer_broker_rejected_request")
            stderr_bytes = b""
        else:
            try:
                with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
                    # Process ownership and signer independence are separate
                    # properties. The gateway owns admission and shutdown;
                    # pinned executable identity and the signed byte protocol
                    # preserve the broker's external trust boundary.
                    completed = get_subprocess_gateway().run(
                        [str(self._executable), *self._arguments],
                        input=request_bytes,
                        capture_output=False,
                        stdout=stdout,
                        stderr=stderr,
                        check=False,
                        timeout=self._timeout_seconds,
                        env=environment,
                        text=False,
                        offline_tooling=True,
                        source="training_tooling:verified_transition_external_signer",
                        accelerator_capability="none",
                    )
                    stdout.seek(0)
                    stdout_bytes = stdout.read(64 * 1024 + 1)
                    stderr.seek(0)
                    stderr_bytes = stderr.read(64 * 1024 + 1)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise VerifiedTransitionProductionFactoryError(
                    "signer_broker_execution_failed"
                ) from exc
            self._assert_executable_identity()
            if completed.returncode != 0:
                _fail("signer_broker_rejected_request")
        if len(stdout_bytes) > 64 * 1024 or len(stderr_bytes) > 64 * 1024:
            _fail("signer_broker_output_too_large")
        try:
            response = json.loads(stdout_bytes)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise VerifiedTransitionProductionFactoryError(
                "signer_broker_response_invalid"
            ) from exc
        if not isinstance(response, dict) or stdout_bytes != canonical_json_bytes(response) + b"\n":
            _fail("signer_broker_response_invalid")
        return response

    def verify_evidence_manifest(
        self,
        policy: VerifiedCampaignTrustPolicy,
        *,
        evidence_manifest: Mapping[str, Any],
        verified_at_unix: int,
        purpose: str,
    ) -> Mapping[str, Any]:
        """Require the pinned verifier process to replay exact package bytes."""

        purpose = _identifier(purpose, role="evidence_verifier_purpose")
        evidence = validate_causal_campaign_evidence_manifest(evidence_manifest)
        verifier_pin = policy.role_pin(EVIDENCE_VERIFIER)
        if (
            self.implementation_sha256 != verifier_pin["implementation_sha256"]
            or self.release_sha256 != verifier_pin["release_sha256"]
            or self.custody_evidence_sha256 != verifier_pin["custody_evidence_sha256"]
        ):
            _fail("signer_broker_artifact_identity_mismatch")
        body = {
            "schema": COMMAND_EVIDENCE_VERIFIER_REQUEST_SCHEMA,
            "purpose": purpose,
            "verifier_identity": self.identity,
            "verified_at_unix": _integer(
                verified_at_unix,
                role="evidence_verifier_verified_at",
                minimum=1,
            ),
            "evidence_manifest": evidence,
            "campaign_trust_policy": {
                "document": policy.document,
                "policy_sha256": policy.policy_sha256,
                "root_key_id": policy.root_key_id,
            },
        }
        request = {**body, "request_sha256": _digest(body)}
        response = self._execute(request)
        if (
            self.release_sha256 != verifier_pin["release_sha256"]
            or self.custody_evidence_sha256 != verifier_pin["custody_evidence_sha256"]
        ):
            _fail("signer_broker_artifact_identity_mismatch")
        if (
            set(response) != _VERIFIER_RESPONSE_KEYS
            or response.get("schema") != COMMAND_EVIDENCE_VERIFIER_RESPONSE_SCHEMA
            or response.get("request_sha256") != request["request_sha256"]
        ):
            _fail("evidence_verifier_response_invalid")
        receipt = validate_external_evidence_verification_receipt(
            response.get("verification_receipt"),
            evidence_manifest=evidence,
        )
        if (
            receipt["verifier_identity"] != self.identity
            or receipt["verified_at_unix"] != verified_at_unix
        ):
            _fail("evidence_verifier_receipt_identity_mismatch")
        return receipt

    def replay_policy_states(
        self,
        *,
        request: Mapping[str, Any],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        """Run exact policy-state replay under detached durable custody."""

        purpose = _identifier(
            request.get("purpose"),
            role="durable_evidence_verifier_purpose",
        )
        job = self._durable_policy_state_replay_job
        if job is None:
            _fail("durable_policy_state_replay_job_unavailable")
        return job.run_file_protocol(
            request,
            job.target_command,
            timeout_seconds,
            purpose,
        )

    def attest(
        self,
        policy: VerifiedCampaignTrustPolicy,
        *,
        role: str,
        payload: Mapping[str, Any],
        signed_at_unix: int,
        purpose: str,
    ) -> Mapping[str, Any]:
        purpose = _identifier(purpose, role="signer_broker_purpose")
        request = prepare_role_signature_request(
            policy,
            role=role,
            payload=payload,
            signed_at_unix=signed_at_unix,
            operation=(
                "campaign_close"
                if role == EVIDENCE_VERIFIER
                else "campaign_manifest"
                if purpose.endswith(":campaign-manifest")
                else "group_lineage"
                if purpose.endswith(":lineage")
                else "group_manifest"
            ),
            purpose=purpose,
        )
        return self.attest_prepared_request(
            policy,
            role=role,
            request=request,
        )

    def attest_prepared_request(
        self,
        policy: VerifiedCampaignTrustPolicy,
        *,
        role: str,
        request: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Sign an already-persisted request without reconstructing its bytes."""

        signed_payload = request.get("signed_payload")
        if not isinstance(signed_payload, Mapping):
            _fail("signer_broker_prepared_request_invalid")
        purpose = _identifier(
            signed_payload.get("purpose"),
            role="signer_broker_purpose",
        )
        envelope = {
            "schema": COMMAND_SIGNER_REQUEST_SCHEMA,
            "purpose": purpose,
            "signature_request": request,
        }
        self._assert_executable_identity()
        issuer_pin = policy.role_pin(role)
        expected_release = _sha256(
            issuer_pin.get("release_sha256"), role="signer_broker_release_pin"
        )
        expected_custody = _sha256(
            issuer_pin.get("custody_evidence_sha256"),
            role="signer_broker_custody_pin",
        )
        if (
            self.release_sha256 != expected_release
            or self.custody_evidence_sha256 != expected_custody
        ):
            _fail("signer_broker_artifact_identity_mismatch")
        response = self._execute(envelope)
        if (
            self.release_sha256 != expected_release
            or self.custody_evidence_sha256 != expected_custody
        ):
            _fail("signer_broker_artifact_identity_mismatch")
        if (
            set(response) != _SIGNER_RESPONSE_KEYS
            or response.get("schema") != COMMAND_SIGNER_RESPONSE_SCHEMA
            or response.get("request_sha256") != request["request_sha256"]
            or not isinstance(response.get("signature_b64"), str)
        ):
            _fail("signer_broker_response_invalid")
        try:
            signature = base64.b64decode(response["signature_b64"], validate=True)
        except (TypeError, ValueError) as exc:
            raise VerifiedTransitionProductionFactoryError(
                "signer_broker_signature_invalid"
            ) from exc
        if len(signature) != 64:
            _fail("signer_broker_signature_invalid")
        return assemble_role_attestation(
            policy,
            request,
            signature_b64=response["signature_b64"],
            role=role,
        )


class JITVerifiedTransitionPlanStore:
    """Create-once plan packages surviving signer or trainer process death."""

    def __init__(self, root: str | Path, *, contract_sha256: str) -> None:
        self.root = _private_directory(root, role="jit_plan_store")
        self.contract_sha256 = _sha256(contract_sha256, role="jit_plan_store_contract")

    def _path(self, sequence: int) -> Path:
        value = _integer(sequence, role="jit_plan_sequence")
        return self.root / f"plan-{value:08d}.json"

    def _intent_path(self, sequence: int) -> Path:
        value = _integer(sequence, role="jit_plan_sequence")
        return self.root / f"plan-{value:08d}.intent.json"

    def reserve_intent(
        self,
        *,
        sequence: int,
        campaign_schedule_root_sha256: str,
        policy_before_sha256: str,
        task_id: str,
        prompt_tokens_sha256: str,
        observed_at_unix_ns: int,
    ) -> dict[str, Any]:
        planned_at = (
            _integer(
                observed_at_unix_ns,
                role="jit_plan_intent_observed_at",
                minimum=1,
            )
            // 1_000_000_000
        ) * 1_000_000_000
        body = {
            "schema": JIT_PLAN_INTENT_SCHEMA,
            "contract_sha256": self.contract_sha256,
            "campaign_schedule_root_sha256": _sha256(
                campaign_schedule_root_sha256,
                role="jit_plan_intent_schedule_root",
            ),
            "sequence": _integer(sequence, role="jit_plan_sequence"),
            "policy_before_sha256": _sha256(
                policy_before_sha256,
                role="jit_plan_intent_policy",
            ),
            "task_id": _identifier(task_id, role="jit_plan_intent_task"),
            "prompt_tokens_sha256": _sha256(
                prompt_tokens_sha256,
                role="jit_plan_intent_prompt",
            ),
            "planned_at_unix_ns": planned_at,
            "admitted_at_unix_ns": planned_at + 1,
        }
        intent = {**body, "intent_sha256": _digest(body)}
        path = self._intent_path(sequence)
        with interprocess_file_lock(self.root / ".publish.lock"):
            if not atomic_write_bytes_if_absent(
                path,
                canonical_json_bytes(intent),
                mode=0o600,
            ):
                raw = read_stable_bytes(path, max_bytes=1 << 20)
                try:
                    existing = json.loads(raw)
                except (UnicodeError, json.JSONDecodeError) as exc:
                    raise VerifiedTransitionProductionFactoryError(
                        "jit_plan_intent_invalid"
                    ) from exc
                immutable = {
                    key: value
                    for key, value in intent.items()
                    if key not in {"planned_at_unix_ns", "admitted_at_unix_ns", "intent_sha256"}
                }
                existing_immutable = {
                    key: value
                    for key, value in existing.items()
                    if key not in {"planned_at_unix_ns", "admitted_at_unix_ns", "intent_sha256"}
                }
                existing_body = dict(existing)
                claimed = existing_body.pop("intent_sha256", None)
                if (
                    raw != canonical_json_bytes(existing)
                    or immutable != existing_immutable
                    or claimed != _digest(existing_body)
                ):
                    _fail("jit_plan_intent_conflict")
                intent = existing
        return intent

    def load(self, *, sequence: int) -> dict[str, Any] | None:
        path = self._path(sequence)
        if path.is_symlink():
            _fail("jit_plan_symlink_rejected")
        if not path.exists():
            return None
        metadata = path.stat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
            or metadata.st_nlink != 1
        ):
            _fail("jit_plan_not_private_owned_file")
        raw = read_stable_bytes(path, max_bytes=16 * 1024 * 1024)
        try:
            value = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise VerifiedTransitionProductionFactoryError("jit_plan_json_invalid") from exc
        if not isinstance(value, dict) or raw != canonical_json_bytes(value):
            _fail("jit_plan_json_noncanonical")
        return self.validate(value, sequence=sequence)

    def validate(self, value: Any, *, sequence: int) -> dict[str, Any]:
        if not isinstance(value, Mapping) or set(value) != _PLAN_PACKAGE_KEYS:
            _fail("jit_plan_package_schema_invalid")
        package = cast(dict[str, Any], _clone(value, role="jit_plan_package"))
        observed = _sha256(package.get("receipt_sha256"), role="jit_plan_receipt")
        unsigned = dict(package)
        unsigned.pop("receipt_sha256")
        if observed != _digest(unsigned):
            _fail("jit_plan_package_digest_mismatch")
        manifest = validate_transition_group_manifest(package.get("group_manifest"))
        if (
            package.get("schema") != JIT_PLAN_PACKAGE_SCHEMA
            or package.get("contract_sha256") != self.contract_sha256
            or package.get("sequence") != sequence
            or not isinstance(package.get("group_manifest_attestation"), Mapping)
            or not isinstance(package.get("lineage_plan"), Mapping)
            or not isinstance(package.get("lineage_attestation"), Mapping)
            or package.get("policy_before_sha256") != manifest["entries"][0]["policy_sha256"]
            or any(
                entry["policy_sha256"] != package["policy_before_sha256"]
                for entry in manifest["entries"]
            )
            or manifest["planned_at_unix_ns"]
            >= _integer(
                package.get("admitted_at_unix_ns"),
                role="jit_plan_admitted_at",
                minimum=1,
            )
        ):
            _fail("jit_plan_package_invalid")
        _sha256(
            package.get("campaign_schedule_root_sha256"),
            role="jit_plan_schedule_root",
        )
        package["group_manifest"] = manifest
        return package

    def publish(self, value: Mapping[str, Any], *, sequence: int) -> dict[str, Any]:
        package = self.validate(value, sequence=sequence)
        payload = canonical_json_bytes(package)
        path = self._path(sequence)
        with interprocess_file_lock(self.root / ".publish.lock"):
            if not atomic_write_bytes_if_absent(path, payload, mode=0o600):
                existing = self.load(sequence=sequence)
                if existing != package:
                    _fail("jit_plan_publication_conflict")
        loaded = self.load(sequence=sequence)
        if loaded != package:
            _fail("jit_plan_publication_mismatch")
        return package


@dataclass(frozen=True, slots=True)
class ProviderBoundTrainingTask:
    """Delegate task behavior while exposing its frozen public commitment."""

    source_task: Any
    task_commitment: Mapping[str, Any]

    def verified_transition_task_commitment(self) -> Mapping[str, Any]:
        return cast(dict[str, Any], _clone(self.task_commitment, role="bound_task"))

    def __getattr__(self, name: str) -> Any:
        return getattr(self.source_task, name)


class JITAdmittingVerifiedTransitionGroupProvider:
    """Admit one exact external plan before delegating model sampling."""

    def __init__(
        self,
        *,
        provider: ProductionVerifiedTransitionGroupProvider,
        policy: VerifiedCampaignTrustPolicy,
        signer_broker: ExternalRoleSignerBroker,
        plan_store: JITVerifiedTransitionPlanStore,
        sampling_config: RecurrentSamplingConfig,
        branch_count: int,
        reward_config_sha256: str,
        now_unix_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        if not operationally_isolated_roles(policy):
            _fail("jit_provider_operational_role_custody_required")
        if type(branch_count) is not int or not 1 <= branch_count <= 256:
            _fail("jit_provider_branch_count_invalid")
        if provider.contract_sha256 != plan_store.contract_sha256:
            _fail("jit_provider_plan_store_contract_mismatch")
        self._provider = provider
        self._policy = policy
        self._broker = signer_broker
        self._store = plan_store
        self._sampling = sampling_config
        self._branch_count = branch_count
        self._reward_config_sha256 = _sha256(
            reward_config_sha256, role="jit_provider_reward_config"
        )
        self._now_unix_ns = now_unix_ns
        self._lock = threading.RLock()

    @property
    def contract_sha256(self) -> str:
        return self._provider.contract_sha256

    def training_schedule_entry(self, *, sequence: int) -> VerifiedTransitionTrainingScheduleEntry:
        return self._provider.training_schedule_entry(sequence=sequence)

    def _bind_sampling_config(
        self, plan: VerifiedTransitionSamplingPlan
    ) -> VerifiedTransitionSamplingPlan:
        expected = self._sampling.to_dict()
        if plan.sampling_config and dict(plan.sampling_config) != expected:
            _fail("jit_provider_sampling_config_substitution")
        return VerifiedTransitionSamplingPlan(
            campaign_sequence=plan.campaign_sequence,
            group_manifest_sha256=plan.group_manifest_sha256,
            task_id=plan.task_id,
            policy_sha256=plan.policy_sha256,
            prompt_tokens_sha256=plan.prompt_tokens_sha256,
            execution_spec_sha256=plan.execution_spec_sha256,
            entries=plan.entries,
            sampling_config=expected,
        )

    def _build_package(
        self,
        *,
        sequence: int,
        task: Any,
        prompt_tokens: Sequence[int],
        policy_sha256: str,
        intent: Mapping[str, Any],
    ) -> dict[str, Any]:
        commitment = self._provider.task_commitment(sequence=sequence)
        prompt_sha256 = _digest(list(prompt_tokens))
        if (
            getattr(task, "task_id", None) != commitment["task_id"]
            or prompt_sha256 != commitment["prompt_tokens_sha256"]
            or policy_sha256 != self._provider.expected_policy_sha256
        ):
            _fail("jit_provider_runtime_schedule_mismatch")
        config_sha256 = sampling_config_document_sha256(self._sampling)
        entries = []
        for index, seed in enumerate(commitment["sample_seeds"]):
            episode_id = f"{self._provider.campaign_id}:s{sequence}:e{index}"
            branch_index = index % self._branch_count
            entries.append(
                TransitionGroupPlanEntry(
                    episode_id=episode_id,
                    task_id=cast(str, commitment["task_id"]),
                    rng_root_sha256=recurrent_sampling_rng_root_sha256(
                        episode_id=episode_id,
                        prompt_tokens_sha256=prompt_sha256,
                        policy_sha256=policy_sha256,
                        execution_spec_sha256=cast(
                            str, commitment["recurrent_execution_spec_sha256"]
                        ),
                        branch_index=branch_index,
                        seed=cast(int, seed),
                        sampling_config=self._sampling,
                    ),
                    policy_sha256=policy_sha256,
                    recurrent_execution_spec_sha256=cast(
                        str, commitment["recurrent_execution_spec_sha256"]
                    ),
                    producing_branch_index=branch_index,
                    sample_seed=cast(int, seed),
                    sampling_config_sha256=config_sha256,
                )
            )
        planned_at = _integer(
            intent.get("planned_at_unix_ns"),
            role="jit_provider_planned_at",
            minimum=1,
        )
        manifest = build_transition_group_manifest(
            group_id=f"{self._provider.campaign_id}:group:{sequence}",
            task_id=cast(str, commitment["task_id"]),
            entries=entries,
            reward_config_sha256=self._reward_config_sha256,
            planned_at_unix_ns=planned_at,
        )
        signed_at = planned_at // 1_000_000_000
        manifest_attestation = self._broker.attest(
            self._policy,
            role=TASK_ISSUER,
            payload=manifest,
            signed_at_unix=signed_at,
            purpose=f"{self._provider.campaign_id}:group:{sequence}:manifest",
        )
        lineage_plan = self._provider.lineage_plan_for_manifest(
            sequence=sequence,
            policy_before_sha256=policy_sha256,
            group_manifest=manifest,
        )
        lineage_attestation = self._broker.attest(
            self._policy,
            role=TASK_ISSUER,
            payload=lineage_plan,
            signed_at_unix=signed_at,
            purpose=f"{self._provider.campaign_id}:group:{sequence}:lineage",
        )
        admitted_at = _integer(
            intent.get("admitted_at_unix_ns"),
            role="jit_provider_admitted_at",
            minimum=planned_at + 1,
        )
        body = {
            "schema": JIT_PLAN_PACKAGE_SCHEMA,
            "contract_sha256": self.contract_sha256,
            "campaign_schedule_root_sha256": (self._provider.campaign_schedule_root_sha256),
            "sequence": sequence,
            "policy_before_sha256": policy_sha256,
            "group_manifest": manifest,
            "group_manifest_attestation": dict(manifest_attestation),
            "lineage_plan": dict(lineage_plan),
            "lineage_attestation": dict(lineage_attestation),
            "admitted_at_unix_ns": admitted_at,
        }
        return {**body, "receipt_sha256": _digest(body)}

    def _validate_runtime_package(
        self,
        package: Mapping[str, Any],
        *,
        sequence: int,
        task: Any,
        prompt_tokens: Sequence[int],
        policy_sha256: str,
    ) -> None:
        manifest = validate_transition_group_manifest(package["group_manifest"])
        commitment = self._provider.task_commitment(sequence=sequence)
        prompt_sha256 = _digest(list(prompt_tokens))
        config_sha256 = sampling_config_document_sha256(self._sampling)
        expected_lineage = self._provider.lineage_plan_for_manifest(
            sequence=sequence,
            policy_before_sha256=policy_sha256,
            group_manifest=manifest,
        )
        if (
            package.get("campaign_schedule_root_sha256")
            != self._provider.campaign_schedule_root_sha256
            or package.get("policy_before_sha256") != policy_sha256
            or package.get("lineage_plan") != expected_lineage
            or manifest["task_id"] != getattr(task, "task_id", None)
            or manifest["task_id"] != commitment["task_id"]
            or [entry["sample_seed"] for entry in manifest["entries"]] != commitment["sample_seeds"]
            or any(
                entry["sampling_config_sha256"] != config_sha256
                or entry["policy_sha256"] != policy_sha256
                or entry["recurrent_execution_spec_sha256"]
                != commitment["recurrent_execution_spec_sha256"]
                or entry["rng_root_sha256"]
                != recurrent_sampling_rng_root_sha256(
                    episode_id=entry["episode_id"],
                    prompt_tokens_sha256=prompt_sha256,
                    policy_sha256=policy_sha256,
                    execution_spec_sha256=entry["recurrent_execution_spec_sha256"],
                    branch_index=entry["producing_branch_index"],
                    seed=entry["sample_seed"],
                    sampling_config=self._sampling,
                )
                for entry in manifest["entries"]
            )
        ):
            _fail("jit_provider_persisted_plan_runtime_mismatch")

    def sampling_plan(
        self,
        *,
        sequence: int,
        task: Any,
        prompt_tokens: Sequence[int],
        policy_sha256: str,
    ) -> VerifiedTransitionSamplingPlan:
        with self._lock:
            try:
                return self._bind_sampling_config(
                    self._provider.sampling_plan(
                        sequence=sequence,
                        task=task,
                        prompt_tokens=prompt_tokens,
                        policy_sha256=policy_sha256,
                    )
                )
            except VerifiedTransitionProviderError as exc:
                if exc.code != "provider_sampling_plan_missing":
                    raise
            package = self._store.load(sequence=sequence)
            if package is None:
                prompt_sha256 = _digest(list(prompt_tokens))
                commitment = self._provider.task_commitment(sequence=sequence)
                intent = self._store.reserve_intent(
                    sequence=sequence,
                    campaign_schedule_root_sha256=(self._provider.campaign_schedule_root_sha256),
                    policy_before_sha256=policy_sha256,
                    task_id=cast(str, commitment["task_id"]),
                    prompt_tokens_sha256=prompt_sha256,
                    observed_at_unix_ns=self._now_unix_ns(),
                )
                package = self._store.publish(
                    self._build_package(
                        sequence=sequence,
                        task=task,
                        prompt_tokens=prompt_tokens,
                        policy_sha256=policy_sha256,
                        intent=intent,
                    ),
                    sequence=sequence,
                )
            self._validate_runtime_package(
                package,
                sequence=sequence,
                task=task,
                prompt_tokens=prompt_tokens,
                policy_sha256=policy_sha256,
            )
            self._provider.admit_group_plan(
                sequence=sequence,
                policy_before_sha256=policy_sha256,
                group_manifest=package["group_manifest"],
                group_manifest_attestation=package["group_manifest_attestation"],
                lineage_attestation=package["lineage_attestation"],
                admitted_at_unix_ns=package["admitted_at_unix_ns"],
            )
            return self._bind_sampling_config(
                self._provider.sampling_plan(
                    sequence=sequence,
                    task=task,
                    prompt_tokens=prompt_tokens,
                    policy_sha256=policy_sha256,
                )
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._provider, name)


class ProductionVerifiedTransitionProviderFactory:
    """Construct one externally rooted provider against the loaded policy."""

    def __init__(
        self,
        *,
        contract: Mapping[str, Any],
        provider_config: Mapping[str, Any],
        campaign_ledger: Any,
        campaign_trust_policy: VerifiedCampaignTrustPolicy,
        evidence_producer: Callable[..., Any],
        evidence_producer_identity: str,
        durable_artifact_loader: Callable[..., Any],
        durable_artifact_loader_identity: str,
        campaign_finalizer: Callable[..., Any],
        campaign_finalizer_identity: str,
        independent_scorer: Callable[..., Any],
        scorer_identity: str,
        token_encoder: Callable[..., Any],
        token_decoder: Callable[..., Any],
        token_codec_identity: str,
        task_issuer_signer_broker: ExternalRoleSignerBroker,
        evidence_verifier_signer_broker: ExternalRoleSignerBroker,
        task_commitments: Mapping[str, Mapping[str, Any]],
        task_answer_nonces: Mapping[str, bytes],
    ) -> None:
        frozen = validate_verified_transition_provider_contract(contract)
        config = cast(dict[str, Any], _clone(provider_config, role="provider_config"))
        if frozen["provider"]["config"] != config:
            _fail("production_factory_provider_config_mismatch")
        custody_value = config.get("initial_policy_state_custody")
        initial_policy_state_custody: dict[str, Any] | None = None
        if custody_value is not None:
            try:
                initial_policy_state_custody = validate_initial_policy_state_custody(custody_value)
                adapter_artifact = inspect_initial_adapter_snapshot(
                    initial_policy_state_custody["initial_adapter_path"],
                    execution_spec_sha256=(initial_policy_state_custody["execution_spec_sha256"]),
                )
                optimizer_artifact = inspect_initial_optimizer_snapshot(
                    initial_policy_state_custody["initial_optimizer_path"]
                )
            except Exception as exc:
                raise VerifiedTransitionProductionFactoryError(
                    "production_factory_initial_state_custody_unavailable"
                ) from exc
            if (
                initial_policy_state_custody["initial_policy_sha256"]
                != frozen["initial_policy_sha256"]
                or initial_policy_state_custody["execution_spec_sha256"]
                != frozen["task_schedule"][0]["recurrent_execution_spec_sha256"]
                or adapter_artifact != initial_policy_state_custody["initial_adapter_artifact"]
                or optimizer_artifact != initial_policy_state_custody["initial_optimizer_artifact"]
            ):
                _fail("production_factory_initial_state_custody_mismatch")
        training_argv = config.get("training_argv")
        if (
            not isinstance(training_argv, list)
            or len(training_argv) < 2
            or training_argv[0] != "tools/train_grpo.py"
            or any(
                not isinstance(argument, str) or not argument or "\x00" in argument
                for argument in training_argv
            )
            or config.get(_TRAINING_ARGV_SHA256_KEY)
            != hashlib.sha256(canonical_json_bytes(training_argv)).hexdigest()
        ):
            _fail("production_factory_training_argv_invalid")
        jit = config.get("jit_plan")
        if not isinstance(jit, Mapping) or set(jit) != _JIT_CONFIG_KEYS:
            _fail("production_factory_jit_config_invalid")
        sampling = _sampling_config_from_contract(jit["sampling_config"])
        if jit.get("reward_config_sha256") != _digest(TransitionRewardConfig().to_dict()):
            _fail("production_factory_reward_config_mismatch")
        branch_count = _integer(
            jit.get("branch_count"), role="production_factory_branch_count", minimum=1
        )
        if branch_count > 256:
            _fail("production_factory_branch_count_invalid")
        if not isinstance(task_issuer_signer_broker, CommandRoleSignerBroker) or not isinstance(
            evidence_verifier_signer_broker, CommandRoleSignerBroker
        ):
            _fail("production_factory_external_command_signers_required")
        if (
            task_issuer_signer_broker is evidence_verifier_signer_broker
            or task_issuer_signer_broker.identity == evidence_verifier_signer_broker.identity
            or task_issuer_signer_broker.custody_evidence_sha256
            == evidence_verifier_signer_broker.custody_evidence_sha256
        ):
            _fail("production_factory_signer_role_separation_required")
        if (
            jit.get("schema") != JIT_PROVIDER_CONFIG_SCHEMA
            or jit.get("signer_broker_identity") != task_issuer_signer_broker.identity
            or jit.get("signer_broker_source_sha256") != task_issuer_signer_broker.source_sha256
        ):
            _fail("production_factory_signer_broker_mismatch")
        issuer_pin = campaign_trust_policy.role_pin(TASK_ISSUER)
        if (
            issuer_pin["implementation_sha256"] != task_issuer_signer_broker.implementation_sha256
            or issuer_pin["release_sha256"] != task_issuer_signer_broker.release_sha256
            or issuer_pin["custody_evidence_sha256"]
            != task_issuer_signer_broker.custody_evidence_sha256
            or issuer_pin["custody_class"]
            not in {"host_isolated_service", "external_service", "remote_hsm"}
        ):
            _fail("production_factory_signer_custody_pin_mismatch")
        verifier_pin = campaign_trust_policy.role_pin(EVIDENCE_VERIFIER)
        if (
            verifier_pin["implementation_sha256"]
            != evidence_verifier_signer_broker.implementation_sha256
            or verifier_pin["release_sha256"] != evidence_verifier_signer_broker.release_sha256
            or verifier_pin["custody_evidence_sha256"]
            != evidence_verifier_signer_broker.custody_evidence_sha256
            or verifier_pin["custody_class"]
            not in {"host_isolated_service", "external_service", "remote_hsm"}
        ):
            _fail("production_factory_verifier_custody_pin_mismatch")
        replay_root = Path(frozen["ledger_roots"]["replay_artifacts"])
        expected_plan_root = str((replay_root / "jit-plans").resolve(strict=False))
        if jit.get("plan_store_root") != expected_plan_root:
            _fail("production_factory_plan_store_root_mismatch")
        output_root = Path(str(jit.get("trainer_output_root"))).expanduser()
        transaction_root = Path(str(jit.get("transaction_root"))).expanduser()
        if (
            not output_root.is_absolute()
            or not transaction_root.is_absolute()
            or str(output_root.resolve(strict=False)) != jit.get("trainer_output_root")
            or str(transaction_root.resolve(strict=False)) != jit.get("transaction_root")
            or transaction_root != output_root / "verified-transition-transactions"
        ):
            _fail("production_factory_trainer_roots_invalid")
        commitments = {
            _identifier(task_id, role="production_factory_task_id"): cast(
                dict[str, Any], _clone(document, role="production_factory_task")
            )
            for task_id, document in task_commitments.items()
        }
        nonces: dict[str, bytes] = {}
        for task_id, nonce in task_answer_nonces.items():
            normalized_task_id = _identifier(
                task_id, role="production_factory_answer_nonce_task_id"
            )
            if not isinstance(nonce, bytes):
                _fail("production_factory_answer_nonce_invalid")
            nonces[normalized_task_id] = bytes(nonce)
        for schedule in frozen["task_schedule"]:
            document = commitments.get(schedule["task_id"])
            try:
                validated = (
                    validate_public_training_task(document) if document is not None else None
                )
            except VerifiedTrainingTaskError as exc:
                raise VerifiedTransitionProductionFactoryError(
                    "production_factory_task_commitment_invalid"
                ) from exc
            if (
                validated is None
                or validated != document
                or _digest(validated) != schedule["immutable_task_sha256"]
                or schedule["task_id"] not in nonces
                or len(schedule["sample_seeds"]) != branch_count
            ):
                _fail("production_factory_task_commitment_mismatch")
        required_task_ids = {row["task_id"] for row in frozen["task_schedule"]}
        if set(commitments) != required_task_ids or set(nonces) != required_task_ids:
            _fail("production_factory_task_material_scope_mismatch")
        self._contract = frozen
        self._config = config
        self._training_argv = tuple(cast(list[str], training_argv))
        self._jit = dict(jit)
        self._sampling = sampling
        self._branch_count = branch_count
        self._ledger = campaign_ledger
        self._policy = campaign_trust_policy
        self._producer = evidence_producer
        self._producer_identity = evidence_producer_identity
        self._loader = durable_artifact_loader
        self._loader_identity = durable_artifact_loader_identity
        self._finalizer = campaign_finalizer
        self._finalizer_identity = campaign_finalizer_identity
        self._scorer = independent_scorer
        self._scorer_identity = scorer_identity
        self._encoder = token_encoder
        self._decoder = token_decoder
        self._codec_identity = token_codec_identity
        self._task_issuer_broker = task_issuer_signer_broker
        self._evidence_verifier_broker = evidence_verifier_signer_broker
        self._commitments = commitments
        self._answer_nonces = nonces
        self._initial_policy_state_custody = initial_policy_state_custody
        self._created = False
        self._lock = threading.RLock()

    @property
    def contract_sha256(self) -> str:
        return cast(str, self._contract["contract_sha256"])

    @property
    def ledger_roots(self) -> dict[str, str]:
        """Return the frozen absolute proof roots used for recovery."""

        return cast(
            dict[str, str],
            _clone(
                self._contract["ledger_roots"],
                role="production_factory_ledger_roots",
            ),
        )

    @property
    def training_argv(self) -> tuple[str, ...]:
        """Return the exact externally frozen trainer invocation."""

        return self._training_argv

    @property
    def initial_policy_state_custody(
        self,
    ) -> dict[str, Any] | None:
        """Return the reopened CP420Q state contract for trainer custody."""

        if self._initial_policy_state_custody is None:
            return None
        return cast(
            dict[str, Any],
            _clone(
                self._initial_policy_state_custody,
                role="initial_policy_state_custody",
            ),
        )

    def bind_training_tasks(self, tasks: Sequence[Any]) -> Sequence[Any]:
        bound = []
        observed: set[str] = set()
        for task in tasks:
            task_id = getattr(task, "task_id", None)
            if task_id in observed:
                _fail("production_factory_duplicate_training_task")
            commitment = self._commitments.get(task_id)
            if commitment is None:
                bound.append(task)
                continue
            try:
                public, _sealed = build_verified_training_task(
                    task,
                    answer_nonce=self._answer_nonces[cast(str, task_id)],
                )
            except VerifiedTrainingTaskError as exc:
                raise VerifiedTransitionProductionFactoryError(
                    "production_factory_runtime_task_unsupported"
                ) from exc
            if public.to_dict() != commitment:
                _fail("production_factory_runtime_task_commitment_mismatch")
            bound.append(
                ProviderBoundTrainingTask(
                    source_task=task,
                    task_commitment=commitment,
                )
            )
            observed.add(cast(str, task_id))
        required = {row["task_id"] for row in self._contract["task_schedule"]}
        if not required.issubset(observed):
            _fail("production_factory_scheduled_task_missing")
        return tuple(bound)

    def create(self, runtime: VerifiedTransitionProviderRuntime) -> VerifiedTransitionGroupProvider:
        with self._lock:
            if self._created:
                _fail("production_factory_already_created")
            if (
                runtime.execution_spec.sha256
                != self._contract["task_schedule"][0]["recurrent_execution_spec_sha256"]
                or len(runtime.execution_spec.branch_roles) != self._branch_count
                or runtime.sampling_max_tokens != self._sampling.max_tokens
                or runtime.dataset_sha256 != self._contract["dataset_sha256"]
                or runtime.group_size != self._branch_count
                or validate_tokenizer_bundle_identity(
                    runtime.tokenizer_trace_adapter.bundle_identity
                )
                != self._contract["tokenizer_bundle"]
                or runtime.output_directory.resolve(strict=False)
                != Path(self._jit["trainer_output_root"])
                or runtime.transaction_root.resolve(strict=False)
                != Path(self._jit["transaction_root"])
            ):
                _fail("production_factory_runtime_graph_mismatch")
            task_ids = {getattr(task, "task_id", None) for task in runtime.training_tasks}
            if any(row["task_id"] not in task_ids for row in self._contract["task_schedule"]):
                _fail("production_factory_runtime_task_missing")
            initial_policy = recurrent_policy_sha256(runtime.model, runtime.execution_spec)
            if initial_policy != self._contract["initial_policy_sha256"]:
                _fail("production_factory_initial_policy_mismatch")
            if self._initial_policy_state_custody is not None:
                try:
                    adapter_artifact = inspect_initial_adapter_snapshot(
                        self._initial_policy_state_custody["initial_adapter_path"],
                        execution_spec_sha256=runtime.execution_spec.sha256,
                    )
                    optimizer_artifact = inspect_initial_optimizer_snapshot(
                        self._initial_policy_state_custody["initial_optimizer_path"]
                    )
                except Exception as exc:
                    raise VerifiedTransitionProductionFactoryError(
                        "production_factory_initial_state_custody_unavailable"
                    ) from exc
                if (
                    adapter_artifact
                    != self._initial_policy_state_custody["initial_adapter_artifact"]
                    or optimizer_artifact
                    != self._initial_policy_state_custody["initial_optimizer_artifact"]
                ):
                    _fail("production_factory_initial_state_custody_mismatch")
            provider = ProductionVerifiedTransitionGroupProvider(
                contract=self._contract,
                provider_config=self._config,
                campaign_ledger=self._ledger,
                campaign_trust_policy=self._policy,
                evidence_producer=self._producer,
                evidence_producer_identity=self._producer_identity,
                durable_artifact_loader=self._loader,
                durable_artifact_loader_identity=self._loader_identity,
                campaign_finalizer=self._finalizer,
                campaign_finalizer_identity=self._finalizer_identity,
                independent_scorer=self._scorer,
                scorer_identity=self._scorer_identity,
                token_encoder=self._encoder,
                token_decoder=self._decoder,
                token_codec_identity=self._codec_identity,
                tokenizer_trace_adapter=runtime.tokenizer_trace_adapter,
                training_tasks=runtime.training_tasks,
                evidence_verifier_signer=self._evidence_verifier_broker,
            )
            wrapped = JITAdmittingVerifiedTransitionGroupProvider(
                provider=provider,
                policy=self._policy,
                signer_broker=self._task_issuer_broker,
                plan_store=JITVerifiedTransitionPlanStore(
                    self._jit["plan_store_root"],
                    contract_sha256=self.contract_sha256,
                ),
                sampling_config=self._sampling,
                branch_count=self._branch_count,
                reward_config_sha256=self._jit["reward_config_sha256"],
            )
            self._created = True
            return cast(VerifiedTransitionGroupProvider, wrapped)


__all__ = [
    "COMMAND_EVIDENCE_VERIFIER_REQUEST_SCHEMA",
    "COMMAND_EVIDENCE_VERIFIER_RESPONSE_SCHEMA",
    "COMMAND_SIGNER_REQUEST_SCHEMA",
    "COMMAND_SIGNER_RESPONSE_SCHEMA",
    "CommandRoleSignerBroker",
    "ExternalRoleSignerBroker",
    "JITAdmittingVerifiedTransitionGroupProvider",
    "JIT_PLAN_PACKAGE_SCHEMA",
    "JIT_PROVIDER_CONFIG_SCHEMA",
    "JITVerifiedTransitionPlanStore",
    "ProductionVerifiedTransitionProviderFactory",
    "ProviderBoundTrainingTask",
    "SAMPLING_CONFIG_CONTRACT_SCHEMA",
    "VerifiedTransitionProductionFactoryError",
    "sampling_config_contract_document",
]
