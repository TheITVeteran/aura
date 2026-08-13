#!/usr/bin/env python3
"""Independently evaluate a unified recurrence checkpoint on fresh tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx  # noqa: E402
from mlx.utils import tree_unflatten  # noqa: E402

from core.brain.canonical_json import canonical_json_bytes  # noqa: E402
from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (  # noqa: E402
    _runtime_semantic_identity as _dependency_runtime_semantic_identity,
)
from core.learning.intrinsic_recurrence_objective import (  # noqa: E402
    answer_cross_entropy,
)
from core.learning.recurrent_answer_emission import (  # noqa: E402
    RecurrentAnswerEmissionContract,
    tokenizer_answer_emission_contract,
)
from core.learning.recurrent_literal_grounding import (  # noqa: E402
    LiteralObservationContract,
    tokenizer_digit_token_ids,
)
from core.learning.recurrent_opcode_grounding import (  # noqa: E402
    OpcodeObservationContract,
    tokenizer_opcode_contract,
)
from core.learning.unified_intrinsic_objective import (  # noqa: E402
    UnifiedIntrinsicTrainingSpec,
    readout_fingerprint,
    unified_answer_trajectory,
)
from core.learning.unified_intrinsic_recurrence import (  # noqa: E402
    UnifiedRecurrenceConfig,
    UnifiedRecurrentController,
)
from core.runtime.atomic_writer import atomic_write_bytes  # noqa: E402
from core.runtime.mlx_memory_guard import host_pressure, mlx_memory_envelope  # noqa: E402
from core.runtime.model_lane_control import standalone_model_lane  # noqa: E402
from tools.python_source_closure import local_python_source_sha256s  # noqa: E402
from tools.resident_recurrent_sft_bootstrap_identity import (  # noqa: E402
    resident_bootstrap_tokenizer_identity,
)
from tools.train_intrinsic_recurrence import encode_example  # noqa: E402
from tools.train_unified_intrinsic_recurrence import (  # noqa: E402
    TRAINING_SOURCE_FILES,
    UnifiedTrainingBundle,
    _await_resource_guard,
    _canonical_sha256,
    _configure_window_tissue,
    _ground_state_value_embeddings,
    _model_identity,
    _runtime_identity,
    _trainable,
)
from tools.unified_intrinsic_checkpoint import (  # noqa: E402
    UnifiedCheckpointError,
    resolve_checkpoint_generation,
)
from tools.unified_intrinsic_preload_barrier import verify_release  # noqa: E402
from tools.unified_intrinsic_tokenization_contract import (  # noqa: E402
    TOKENIZED_DATASET_FILENAME,
    load_source_dataset,
    verify_tokenized_dataset,
)

EVALUATION_SCHEMA = "aura.unified_intrinsic_independent_evaluation.v1"
EVALUATION_SOURCE_FILES = (
    "core/brain/llm/latent_cortex/recurrence_adapter_identity_v2.py",
    "tools/evaluate_unified_intrinsic_checkpoint.py",
    "tools/evaluate_unified_intrinsic_decoding.py",
    "tools/unified_intrinsic_decode_journal.py",
)
ROOT_CONTROL_SCHEMA = "aura.unified_intrinsic.root_control_binding.v1"
MATCHED_CONTROL_SCHEMA = "aura.unified_intrinsic.matched_control_binding.v1"

_CONTROL_COMPATIBILITY_FIELDS = (
    "model",
    "runtime",
    "tokenizer",
    "spec",
    "window_geometry",
    "families",
    "task_depths",
    "init_seed",
    "bridge",
    "window_tissue_mode",
    "lora_rank",
    "controller_rank",
    "state_weight",
    "stutter_weight",
    "state_codebook_sha256",
    "state_codebook_grounding",
    "literal_observation_contract",
    "opcode_observation_contract",
    "answer_emission_contract",
    "depth_basis_size",
    "lora_targets",
    "readout_sha256",
)


@dataclass(frozen=True, slots=True)
class EvaluationLayout:
    """Frozen locations needed to evaluate either campaign layout."""

    checkpoint_dir: Path
    dataset_path: Path
    tokenized_dataset_path: Path
    bootstrap_output_dir: Path | None


def _load_resident_campaign_config(path: Path) -> dict[str, Any]:
    # Imported lazily so the legacy standalone evaluator remains lightweight.
    from tools.run_unified_intrinsic_resident_campaign import _load_config

    return _load_config(path)


def _evaluation_layout(campaign_dir: Path) -> EvaluationLayout:
    """Resolve legacy co-located or resident split-output campaign storage."""

    root = campaign_dir.expanduser().resolve(strict=True)
    config_path = root / "campaign.json"
    if not config_path.exists():
        return EvaluationLayout(
            checkpoint_dir=root,
            dataset_path=root / "dataset.json",
            tokenized_dataset_path=root / TOKENIZED_DATASET_FILENAME,
            bootstrap_output_dir=None,
        )
    config = _load_resident_campaign_config(config_path)
    paths = config.get("paths")
    if not isinstance(paths, dict):  # pragma: no cover - validated by _load_config
        raise RuntimeError("resident campaign paths are unavailable")
    configured_root = Path(str(paths["campaign_root"])).resolve(strict=True)
    if configured_root != root:
        raise RuntimeError("resident campaign root differs from evaluation root")
    return EvaluationLayout(
        checkpoint_dir=Path(str(paths["training_output"])).resolve(strict=True),
        dataset_path=Path(str(paths["dataset"])).resolve(strict=True),
        tokenized_dataset_path=Path(str(paths["tokenized_dataset"])).resolve(strict=True),
        bootstrap_output_dir=(
            Path(str(paths["bootstrap_output"])).resolve(strict=True)
            if paths.get("bootstrap_output") is not None
            else None
        ),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_semantic_identity(value: Any) -> dict[str, Any]:
    """Validate a campaign runtime while excluding only ephemeral bytecode caches."""

    if not isinstance(value, dict) or set(value) != {
        "environment",
        "interpreter",
        "identity_sha256",
    }:
        raise RuntimeError("unified campaign runtime identity is malformed")
    body = {key: value[key] for key in ("environment", "interpreter")}
    if value.get("identity_sha256") != _canonical_sha256(body):
        raise RuntimeError("unified campaign runtime commitment differs")
    interpreter = value.get("interpreter")
    if not isinstance(interpreter, dict):
        raise RuntimeError("unified campaign interpreter identity is malformed")
    return {
        "environment": _dependency_runtime_semantic_identity(value["environment"]),
        "interpreter": interpreter,
    }


def evaluation_source_sha256s(source_root: Path = REPO_ROOT) -> dict[str, str]:
    """Bind the evaluator entry points and every reachable local dependency."""

    return local_python_source_sha256s(source_root, EVALUATION_SOURCE_FILES)


def _evaluation_source_sha256s() -> dict[str, str]:
    return evaluation_source_sha256s()


def _evaluation_preload_evidence(
    *,
    resource_enabled: bool,
    preload_ready_path: Path | None,
    preload_release_path: Path | None,
    preload_key_path: Path | None,
    preload_config_sha256: str | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Resolve pressure from the live probe or a signed external handoff."""

    preload_values = (
        preload_ready_path,
        preload_release_path,
        preload_key_path,
        preload_config_sha256,
    )
    preload_enabled = all(value is not None for value in preload_values)
    if any(value is not None for value in preload_values) != preload_enabled:
        raise ValueError("evaluation preload arguments must be supplied together")
    if resource_enabled and not preload_enabled:
        raise ValueError("external evaluation resource guard requires signed preload")
    if preload_enabled and not resource_enabled:
        raise ValueError("evaluation signed preload requires external resource guard")
    if preload_enabled:
        release = verify_release(
            preload_release_path.expanduser(),
            ready_path=preload_ready_path.expanduser(),
            key_path=preload_key_path.expanduser(),
            config_sha256=str(preload_config_sha256),
            require_live_evidence=True,
        )
        pressure = dict(release["host_pressure"])
    else:
        release = None
        pressure = host_pressure()
    if pressure.get("available") is not True or pressure.get("under_pressure") is not False:
        raise RuntimeError("unified evaluation refused unavailable or pressured host")
    return pressure, release


def _controller_config(
    model: Any,
    identity: dict[str, Any],
    literal_contract: LiteralObservationContract,
    opcode_contract: OpcodeObservationContract,
) -> UnifiedRecurrenceConfig:
    return UnifiedRecurrenceConfig(
        hidden_size=int(model.model.layers[0].input_layernorm.weight.shape[0]),
        correction_rank=int(identity["controller_rank"]),
        depth_basis_size=int(identity["depth_basis_size"]),
        minimum_iterations=1,
        initialization_seed=int(identity["init_seed"]),
        literal_digit_token_ids=literal_contract.digit_token_ids,
        opcode_token_patterns=opcode_contract.patterns,
        opcode_context_patterns=opcode_contract.contexts,
    )


def _initial_controller(
    model: Any,
    tokenizer: Any,
    spec: UnifiedIntrinsicTrainingSpec,
    identity: dict[str, Any],
    literal_contract: LiteralObservationContract,
    opcode_contract: OpcodeObservationContract,
) -> UnifiedRecurrentController:
    """Reconstruct the exact pre-training controller for a matched-compute arm."""

    bootstrap = identity.get("bootstrap")
    if bootstrap is not None:
        raise RuntimeError("bootstrapped unified tissue requires its committed parent checkpoint")
    controller = UnifiedRecurrentController(
        _controller_config(model, identity, literal_contract, opcode_contract)
    )
    expected_grounding = identity.get("state_codebook_grounding")
    if not isinstance(expected_grounding, dict):
        raise RuntimeError("unified checkpoint state codebook grounding is absent")
    observed_grounding = _ground_state_value_embeddings(
        model,
        tokenizer,
        controller,
        prelude_end=spec.prelude_end,
        batch_size=int(expected_grounding.get("batch_size", 0)),
    )
    if observed_grounding != expected_grounding:
        raise RuntimeError("unified checkpoint initial grounding differs")
    expected_sha256 = identity.get("initial_controller_sha256")
    if not isinstance(expected_sha256, str) or controller.parameter_sha256() != expected_sha256:
        raise RuntimeError("unified checkpoint initial controller differs")
    return controller


def _root_control_receipt(
    campaign_dir: Path,
    *,
    stem: str,
    target_identity: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reopen a true pre-training root and bind it independently.

    Recovery and source-repair checkpoints may have several immediate parents.
    A root control is admissible only from a campaign whose own tissue was not
    bootstrapped, so an intermediate trained checkpoint cannot be mislabeled as
    untrained merely because it is convenient to load.
    """

    root = campaign_dir.expanduser().resolve(strict=True)
    layout = _evaluation_layout(root)
    try:
        resolved = resolve_checkpoint_generation(
            layout.checkpoint_dir,
            stem=stem,
            required=True,
        )
    except (OSError, UnifiedCheckpointError, ValueError) as exc:
        raise RuntimeError("unified root control checkpoint is invalid") from exc
    if resolved is None:  # pragma: no cover - required=True is exhaustive
        raise RuntimeError("unified root control checkpoint is unavailable")
    receipt = resolved.receipt
    receipt_body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    identity = receipt.get("identity")
    identity_body = (
        {key: value for key, value in identity.items() if key != "identity_sha256"}
        if isinstance(identity, dict)
        else {}
    )
    if (
        not isinstance(identity, dict)
        or receipt.get("receipt_sha256") != _canonical_sha256(receipt_body)
        or receipt.get("checkpoint_sha256") != _file_sha256(resolved.weights_path)
        or identity.get("identity_sha256") != _canonical_sha256(identity_body)
    ):
        raise RuntimeError("unified root control commitment differs")
    if identity.get("bootstrap") is not None:
        raise RuntimeError("unified root control is itself bootstrapped")
    initial_sha256 = identity.get("initial_controller_sha256")
    if not isinstance(initial_sha256, str) or len(initial_sha256) != 64:
        raise RuntimeError("unified root control identity is incomplete")
    if target_identity is not None:
        mismatches = [
            field
            for field in _CONTROL_COMPATIBILITY_FIELDS
            if _canonical_sha256(identity.get(field))
            != _canonical_sha256(target_identity.get(field))
        ]
        if mismatches:
            raise RuntimeError("unified root control topology differs: " + ",".join(mismatches))
    body = {
        "schema": ROOT_CONTROL_SCHEMA,
        "mode": "deterministic_pretraining_root",
        "campaign_root": str(root),
        "stem": stem,
        "checkpoint_step": receipt.get("step"),
        "checkpoint_sha256": receipt.get("checkpoint_sha256"),
        "checkpoint_receipt_sha256": receipt.get("receipt_sha256"),
        "campaign_identity_sha256": identity.get("identity_sha256"),
        "controller_sha256": initial_sha256,
    }
    return identity, {**body, "binding_sha256": _canonical_sha256(body)}


def root_control_binding(
    campaign_dir: Path,
    *,
    stem: str,
    target_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the immutable public binding for an external root control."""

    _identity, binding = _root_control_receipt(
        campaign_dir,
        stem=stem,
        target_identity=target_identity,
    )
    return binding


def campaign_initial_control_binding(
    campaign_dir: Path,
    *,
    identity: dict[str, Any],
) -> dict[str, Any]:
    """Bind the initialization reconstructed from the evaluated campaign itself."""

    root = campaign_dir.expanduser().resolve(strict=True)
    controller_sha256 = identity.get("initial_controller_sha256")
    identity_sha256 = identity.get("identity_sha256")
    if (
        not isinstance(controller_sha256, str)
        or len(controller_sha256) != 64
        or not isinstance(identity_sha256, str)
        or len(identity_sha256) != 64
    ):
        raise RuntimeError("unified matched control identity is incomplete")
    body = {
        "schema": MATCHED_CONTROL_SCHEMA,
        "mode": "campaign_episode_initial",
        "campaign_root": str(root),
        "campaign_identity_sha256": identity_sha256,
        "controller_sha256": controller_sha256,
    }
    return {**body, "binding_sha256": _canonical_sha256(body)}


def load_root_initial_controller(
    campaign_dir: Path,
    *,
    stem: str,
    model: Any,
    tokenizer: Any,
    spec: UnifiedIntrinsicTrainingSpec,
    target_identity: dict[str, Any],
    literal_contract: LiteralObservationContract,
    opcode_contract: OpcodeObservationContract,
) -> tuple[UnifiedRecurrentController, dict[str, Any]]:
    """Materialize a root initialization against an already loaded model."""

    identity, binding = _root_control_receipt(
        campaign_dir,
        stem=stem,
        target_identity=target_identity,
    )
    controller = _initial_controller(
        model,
        tokenizer,
        spec,
        identity,
        literal_contract,
        opcode_contract,
    )
    if controller.parameter_sha256() != binding["controller_sha256"]:
        raise RuntimeError("unified root control reconstruction differs")
    return controller, binding


def _bootstrap_initial_controller(
    layout: EvaluationLayout,
    model: Any,
    identity: dict[str, Any],
    literal_contract: LiteralObservationContract,
    opcode_contract: OpcodeObservationContract,
) -> UnifiedRecurrentController | None:
    """Load the exact tissue inherited at step zero of a child campaign."""

    bootstrap = identity.get("bootstrap")
    if bootstrap is None:
        return None
    if (
        not isinstance(bootstrap, dict)
        or bootstrap.get("schema") != "aura.unified_intrinsic.bootstrap_tissue.v1"
    ):
        raise RuntimeError("unified checkpoint bootstrap identity differs")
    if identity.get("window_tissue_mode") != "controller_only":
        raise RuntimeError("matched bootstrap reconstruction supports controller-only tissue")
    if layout.bootstrap_output_dir is None:
        raise RuntimeError("unified checkpoint bootstrap output is unavailable")
    stem = bootstrap.get("stem")
    if not isinstance(stem, str) or not stem:
        raise RuntimeError("unified checkpoint bootstrap stem differs")
    try:
        resolved = resolve_checkpoint_generation(
            layout.bootstrap_output_dir,
            stem=stem,
            required=True,
        )
    except UnifiedCheckpointError as exc:
        raise RuntimeError("unified checkpoint bootstrap parent is invalid") from exc
    if resolved is None:  # pragma: no cover - required=True is exhaustive
        raise RuntimeError("unified checkpoint bootstrap parent is unavailable")
    receipt = resolved.receipt
    parent_identity = receipt.get("identity")
    receipt_body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    parent_identity_body = (
        {key: value for key, value in parent_identity.items() if key != "identity_sha256"}
        if isinstance(parent_identity, dict)
        else {}
    )
    commitments = {
        "parent_step": receipt.get("step"),
        "parent_checkpoint_sha256": receipt.get("checkpoint_sha256"),
        "parent_receipt_sha256": receipt.get("receipt_sha256"),
        "parent_identity_sha256": (
            parent_identity.get("identity_sha256") if isinstance(parent_identity, dict) else None
        ),
    }
    if (
        not isinstance(parent_identity, dict)
        or any(bootstrap.get(key) != value for key, value in commitments.items())
        or receipt.get("receipt_sha256") != _canonical_sha256(receipt_body)
        or receipt.get("checkpoint_sha256") != _file_sha256(resolved.weights_path)
        or parent_identity.get("identity_sha256") != _canonical_sha256(parent_identity_body)
    ):
        raise RuntimeError("unified checkpoint bootstrap commitment differs")

    compatibility_fields = (
        "model",
        "runtime",
        "tokenizer",
        "spec",
        "window_geometry",
        "families",
        "task_depths",
        "init_seed",
        "bridge",
        "window_tissue_mode",
        "lora_rank",
        "controller_rank",
        "state_weight",
        "stutter_weight",
        "state_codebook_sha256",
        "state_codebook_grounding",
        "literal_observation_contract",
        "opcode_observation_contract",
        "answer_emission_contract",
        "depth_basis_size",
        "lora_targets",
        "readout_sha256",
    )
    mismatches = [
        field
        for field in compatibility_fields
        if _canonical_sha256(parent_identity.get(field)) != _canonical_sha256(identity.get(field))
    ]
    if mismatches:
        raise RuntimeError("unified checkpoint bootstrap topology differs: " + ",".join(mismatches))

    controller = UnifiedRecurrentController(
        _controller_config(model, identity, literal_contract, opcode_contract)
    )
    parent_bundle = UnifiedTrainingBundle(model, controller)
    tensors = mx.load(str(resolved.weights_path))
    trainable = {
        name.removeprefix("bundle."): value
        for name, value in tensors.items()
        if name.startswith("bundle.")
    }
    if set(trainable) != set(_trainable(parent_bundle)):
        raise RuntimeError("unified checkpoint bootstrap tensor inventory differs")
    parent_bundle.update(tree_unflatten(list(trainable.items())))
    mx.eval(parent_bundle.parameters())
    expected_sha256 = identity.get("initial_controller_sha256")
    if not isinstance(expected_sha256, str) or controller.parameter_sha256() != expected_sha256:
        raise RuntimeError("unified checkpoint bootstrap controller differs")
    return controller


def _load_checkpoint(
    campaign_dir: Path,
    *,
    stem: str,
) -> tuple[
    UnifiedTrainingBundle,
    UnifiedRecurrentController,
    Any,
    UnifiedIntrinsicTrainingSpec,
    dict[str, Any],
]:
    layout = _evaluation_layout(campaign_dir)
    try:
        resolved = resolve_checkpoint_generation(
            layout.checkpoint_dir,
            stem=stem,
            required=True,
        )
    except UnifiedCheckpointError as exc:
        raise RuntimeError(str(exc)) from exc
    if resolved is None:  # pragma: no cover - required=True is exhaustive
        raise RuntimeError("unified checkpoint is unavailable")
    receipt = resolved.receipt
    weights_path = resolved.weights_path
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    identity = receipt.get("identity")
    if (
        receipt.get("receipt_sha256") != _canonical_sha256(body)
        or not isinstance(identity, dict)
        or receipt.get("checkpoint_sha256") != _file_sha256(weights_path)
    ):
        raise RuntimeError("unified checkpoint commitment differs")
    identity_body = {key: value for key, value in identity.items() if key != "identity_sha256"}
    if identity.get("identity_sha256") != _canonical_sha256(identity_body):
        raise RuntimeError("unified campaign identity differs")
    source_sha256s = identity.get("source_sha256s")
    training_runtime_files = set(TRAINING_SOURCE_FILES) - set(EVALUATION_SOURCE_FILES)
    if (
        not isinstance(source_sha256s, dict)
        or set(source_sha256s) != set(TRAINING_SOURCE_FILES)
        or any(
            source_sha256s[relative] != _file_sha256(REPO_ROOT / relative)
            for relative in training_runtime_files
        )
    ):
        raise RuntimeError("unified campaign source differs")
    model_identity = identity.get("model")
    if (
        not isinstance(model_identity, dict)
        or _model_identity(str(model_identity.get("canonical_path"))) != model_identity
    ):
        raise RuntimeError("unified campaign model differs")
    runtime_identity = identity.get("runtime")
    observed_runtime_identity = _runtime_identity()
    if not isinstance(runtime_identity, dict) or _runtime_semantic_identity(
        runtime_identity
    ) != _runtime_semantic_identity(observed_runtime_identity):
        raise RuntimeError("unified campaign runtime differs")
    dataset_identity = identity.get("dataset")
    dataset_name = dataset_identity.get("path") if isinstance(dataset_identity, dict) else None
    if (
        not isinstance(dataset_name, str)
        or Path(dataset_name).name != dataset_name
        or not isinstance(dataset_identity, dict)
    ):
        raise RuntimeError("unified campaign dataset identity differs")
    dataset_path = layout.dataset_path
    if dataset_path.name != dataset_name:
        raise RuntimeError("unified campaign dataset path differs")
    try:
        dataset_size = dataset_path.stat().st_size
        dataset_sha256 = _file_sha256(dataset_path)
    except OSError as exc:
        raise RuntimeError("unified campaign dataset is unavailable") from exc
    if dataset_size != dataset_identity.get("size_bytes") or dataset_sha256 != dataset_identity.get(
        "sha256"
    ):
        raise RuntimeError("unified campaign dataset differs")
    train_tasks, holdout_tasks = load_source_dataset(dataset_path)

    from mlx_lm import load

    mx.random.seed(int(identity["init_seed"]))
    model, tokenizer = load(model_identity["canonical_path"])
    model.freeze()
    expected_tokenizer_identity = identity.get("tokenizer")
    observed_tokenizer_identity = resident_bootstrap_tokenizer_identity(
        Path(model_identity["canonical_path"]),
        tokenizer,
    )
    if (
        not isinstance(expected_tokenizer_identity, dict)
        or observed_tokenizer_identity != expected_tokenizer_identity
    ):
        raise RuntimeError("unified campaign tokenizer differs")
    tokenized_identity = identity.get("tokenized_dataset")
    tokenized_name = (
        tokenized_identity.get("path") if isinstance(tokenized_identity, dict) else None
    )
    if tokenized_name != TOKENIZED_DATASET_FILENAME:
        raise RuntimeError("unified campaign tokenized dataset identity differs")
    bridge = {"assistant_answer": "\n\nFINAL_ANSWER: "}.get(
        identity["bridge"],
        identity["bridge"],
    )
    observed_tokenized_identity = verify_tokenized_dataset(
        layout.tokenized_dataset_path,
        tokenizer,
        train_tasks,
        holdout_tasks,
        bridge=bridge,
        dataset_identity=dataset_identity,
        tokenizer_identity_sha256=observed_tokenizer_identity["identity_sha256"],
    )
    if observed_tokenized_identity != tokenized_identity:
        raise RuntimeError("unified campaign tokenized dataset differs")
    literal_identity = identity.get("literal_observation_contract")
    if not isinstance(literal_identity, dict):
        raise RuntimeError("unified checkpoint literal contract is absent")
    literal_contract = LiteralObservationContract(
        tuple(literal_identity.get("digit_token_ids", ())),
        max_value=literal_identity.get("max_value"),
        schema=literal_identity.get("schema"),
    )
    if literal_contract.contract_sha256 != literal_identity.get(
        "contract_sha256"
    ) or literal_contract.digit_token_ids != tokenizer_digit_token_ids(tokenizer):
        raise RuntimeError("unified checkpoint literal contract differs")
    opcode_identity = identity.get("opcode_observation_contract")
    if not isinstance(opcode_identity, dict):
        raise RuntimeError("unified checkpoint opcode contract is absent")
    opcode_contract = OpcodeObservationContract(
        tuple(
            (
                row.get("opcode"),
                tuple(row.get("token_ids", ())),
            )
            for row in opcode_identity.get("patterns", ())
            if isinstance(row, dict)
        ),
        tuple(
            (
                row.get("name"),
                tuple(row.get("token_ids", ())),
            )
            for row in opcode_identity.get("contexts", ())
            if isinstance(row, dict)
        ),
        schema=opcode_identity.get("schema"),
    )
    tokenizer_contract = tokenizer_opcode_contract(tokenizer)
    if (
        opcode_contract.contract_sha256 != opcode_identity.get("contract_sha256")
        or opcode_contract != tokenizer_contract
    ):
        raise RuntimeError("unified checkpoint opcode contract differs")
    answer_identity = identity.get("answer_emission_contract")
    if not isinstance(answer_identity, dict):
        raise RuntimeError("unified checkpoint answer emission contract is absent")
    answer_contract = RecurrentAnswerEmissionContract(
        digit_token_ids=tuple(answer_identity.get("digit_token_ids", ())),
        eos_token_id=answer_identity.get("eos_token_id"),
        family_markers=tuple(
            (row.get("family"), tuple(row.get("token_ids", ())))
            for row in answer_identity.get("family_markers", ())
            if isinstance(row, dict)
        ),
        syntax=tuple(
            (row.get("name"), tuple(row.get("token_ids", ())))
            for row in answer_identity.get("syntax", ())
            if isinstance(row, dict)
        ),
        schema=answer_identity.get("schema"),
    )
    tokenizer_answer_contract = tokenizer_answer_emission_contract(
        tokenizer,
        tokenizer_contract,
    )
    if (
        answer_contract.contract_sha256 != answer_identity.get("contract_sha256")
        or answer_contract != tokenizer_answer_contract
    ):
        raise RuntimeError("unified checkpoint answer emission contract differs")
    spec = UnifiedIntrinsicTrainingSpec(**identity["spec"])
    wiring = _configure_window_tissue(
        model,
        spec,
        mode=str(identity.get("window_tissue_mode", "scoped_lora")),
        rank=int(identity["lora_rank"]),
        targets=tuple(identity["lora_targets"]),
        depth_basis_size=int(identity["depth_basis_size"]),
    )
    if wiring != identity["wiring"]:
        raise RuntimeError("unified checkpoint wiring differs")
    initial_controller = _bootstrap_initial_controller(
        layout,
        model,
        identity,
        literal_contract,
        opcode_contract,
    ) or _initial_controller(
        model,
        tokenizer,
        spec,
        identity,
        literal_contract,
        opcode_contract,
    )
    controller = UnifiedRecurrentController(
        _controller_config(model, identity, literal_contract, opcode_contract)
    )
    bundle = UnifiedTrainingBundle(model, controller)
    tensors = mx.load(str(weights_path))
    trainable = {
        name.removeprefix("bundle."): value
        for name, value in tensors.items()
        if name.startswith("bundle.")
    }
    if set(trainable) != set(_trainable(bundle)):
        raise RuntimeError("unified checkpoint tensor inventory differs")
    bundle.update(tree_unflatten(list(trainable.items())))
    mx.eval(bundle.parameters())
    if readout_fingerprint(model, spec.coda_start) != identity["readout_sha256"]:
        raise RuntimeError("unified checkpoint readout differs")
    return bundle, initial_controller, tokenizer, spec, identity


@contextmanager
def unified_evaluation_context(
    campaign_dir: Path,
    *,
    stem: str,
    memory_limit_gb: float = 40.0,
    cache_limit_gb: float = 2.0,
    wired_limit_gb: float = 48.0,
    resource_stage_path: Path | None = None,
    resource_startup_lethal_mb: float | None = None,
    resource_steady_lethal_mb: float | None = None,
    preload_ready_path: Path | None = None,
    preload_release_path: Path | None = None,
    preload_key_path: Path | None = None,
    preload_config_sha256: str | None = None,
) -> Iterator[
    tuple[
        UnifiedTrainingBundle,
        UnifiedRecurrentController,
        Any,
        UnifiedIntrinsicTrainingSpec,
        dict[str, Any],
        Any,
        dict[str, Any] | None,
    ]
]:
    """Own the model lane and memory envelope for a complete evaluation."""

    campaign_dir = campaign_dir.expanduser().resolve(strict=True)
    layout = _evaluation_layout(campaign_dir)
    resolved = resolve_checkpoint_generation(
        layout.checkpoint_dir,
        stem=stem,
        required=True,
    )
    if resolved is None:  # pragma: no cover - required=True is exhaustive
        raise RuntimeError("unified checkpoint is unavailable")
    identity = resolved.receipt.get("identity")
    model_identity = identity.get("model") if isinstance(identity, dict) else None
    model_path = model_identity.get("canonical_path") if isinstance(model_identity, dict) else None
    if not isinstance(model_path, str) or not model_path:
        raise RuntimeError("unified campaign model identity differs")
    resource_values = (
        resource_stage_path,
        resource_startup_lethal_mb,
        resource_steady_lethal_mb,
    )
    resource_enabled = all(value is not None for value in resource_values)
    if any(value is not None for value in resource_values) != resource_enabled:
        raise ValueError("evaluation resource guard arguments must be supplied together")
    _pressure, preload_release = _evaluation_preload_evidence(
        resource_enabled=resource_enabled,
        preload_ready_path=preload_ready_path,
        preload_release_path=preload_release_path,
        preload_key_path=preload_key_path,
        preload_config_sha256=preload_config_sha256,
    )
    with (
        standalone_model_lane(
            owner_id=f"evaluate-unified-intrinsic:{campaign_dir.parent.name}",
            model_path=model_path,
            purpose="benchmark",
            preemptible=False,
            allow_owner_eviction=False,
            metadata={
                "tool": "evaluate_unified_intrinsic_checkpoint",
                "operator_launched": True,
                "memory_envelope_gb": memory_limit_gb,
            },
        ),
        mlx_memory_envelope(
            memory_gb=memory_limit_gb,
            cache_gb=cache_limit_gb,
            wired_gb=wired_limit_gb,
            restore_limits_on_exit=False,
        ) as envelope,
    ):
        bundle, initial_controller, tokenizer, spec, identity = _load_checkpoint(
            campaign_dir,
            stem=stem,
        )
        resource_receipt: dict[str, Any] | None = None
        if resource_enabled:
            runtime_guard = _await_resource_guard(
                resource_stage_path.expanduser(),
                trainer_sha256=_file_sha256(Path(__file__).resolve(strict=True)),
                startup_lethal_mb=float(resource_startup_lethal_mb),
                steady_lethal_mb=float(resource_steady_lethal_mb),
                timeout_s=120.0,
            )
            resource_receipt = {
                "preload_release": preload_release,
                "runtime_guard": runtime_guard,
            }
        yield (
            bundle,
            initial_controller,
            tokenizer,
            spec,
            identity,
            envelope,
            resource_receipt,
        )


def _sign_test_p_value(differences: list[float]) -> float | None:
    signs = [value for value in differences if abs(value) > 1e-12]
    if not signs:
        return None
    wins = sum(value > 0.0 for value in signs)
    tail = min(wins, len(signs) - wins)
    probability = sum(math.comb(len(signs), k) for k in range(tail + 1)) / (2 ** len(signs))
    return min(1.0, 2.0 * probability)


def _fresh_tasks(
    identity: dict[str, Any],
    *,
    per_cell: int,
    seed: int,
    task_depth: int | None = None,
) -> list[Any]:
    from core.learning import recurrence_curriculum as curriculum

    families = tuple(identity["families"])
    campaign_depths = tuple(
        int(value) for value in identity.get("task_depths", (identity.get("task_depth"),))
    )
    if not campaign_depths or any(depth < 1 for depth in campaign_depths):
        raise RuntimeError("unified campaign task depths are invalid")
    train = curriculum.task_battery(
        families,
        campaign_depths,
        int(identity["per_cell"]),
        seed=int(identity["seed"]),
    )
    selected = curriculum.task_battery(
        families,
        campaign_depths,
        int(identity["holdout_per_cell"]),
        seed=int(identity["seed"]) + 9_973,
    )
    excluded = {task.prompt for task in (*train, *selected)}
    evaluation_depths = campaign_depths if task_depth is None else (int(task_depth),)
    fresh = curriculum.task_battery(
        families,
        evaluation_depths,
        per_cell,
        seed=seed,
    )
    result = [task for task in fresh if task.prompt not in excluded]
    if len(result) != len(fresh):
        raise RuntimeError("independent task battery overlaps campaign data")
    return result


def _evaluate_loaded_checkpoint(
    campaign_dir: Path,
    *,
    bundle: UnifiedTrainingBundle,
    tokenizer: Any,
    spec: UnifiedIntrinsicTrainingSpec,
    identity: dict[str, Any],
    envelope: Any,
    stem: str,
    per_cell: int,
    evaluation_seed: int,
) -> dict[str, Any]:
    tasks = _fresh_tasks(identity, per_cell=per_cell, seed=evaluation_seed)
    bridge = {"assistant_answer": "\n\nFINAL_ANSWER: "}.get(
        identity["bridge"],
        identity["bridge"],
    )
    t1 = spec.plan_at(1)
    deepest = max(spec.heldout_depths)
    deep_plan = spec.plan_at(deepest)
    rows = []
    from core.brain.llm.latent_cortex.recurrence_adapter import (
        recurrence_adapter_scope,
    )

    for task in tasks:
        prompt, answer = encode_example(tokenizer, task, bridge)
        base_loss, _ = answer_cross_entropy(bundle.model, prompt, answer, t1)
        with recurrence_adapter_scope(start=None, stop=None):
            _states, shallow_losses = unified_answer_trajectory(
                bundle.model,
                prompt,
                answer,
                t1,
                bundle.controller,
                use_state_slots=True,
            )
            _states, deep_losses = unified_answer_trajectory(
                bundle.model,
                prompt,
                answer,
                deep_plan,
                bundle.controller,
                use_state_slots=True,
            )
        shallow = float(shallow_losses[-1].item())
        deep = float(deep_losses[-1].item())
        rows.append(
            {
                "family": task.family,
                "prompt_sha256": hashlib.sha256(task.prompt.encode()).hexdigest(),
                "base_t1_ce": float(base_loss.item()),
                "trained_t1_ce": shallow,
                f"trained_t{deepest}_ce": deep,
                "depth_ce_gain": shallow - deep,
                "base_ce_gain": float(base_loss.item()) - deep,
            }
        )
        envelope.reclaim(force=True)
    differences = [row["depth_ce_gain"] for row in rows]
    base_differences = [row["base_ce_gain"] for row in rows]
    family_rows = {}
    for family in identity["families"]:
        selected = [row for row in rows if row["family"] == family]
        family_rows[family] = {
            "tasks": len(selected),
            "mean_depth_ce_gain": sum(row["depth_ce_gain"] for row in selected) / len(selected),
            "depth_wins": sum(row["depth_ce_gain"] > 0.0 for row in selected),
        }
    layout = _evaluation_layout(campaign_dir)
    resolved = resolve_checkpoint_generation(
        layout.checkpoint_dir,
        stem=stem,
        required=True,
    )
    if resolved is None:  # pragma: no cover - required=True is exhaustive
        raise RuntimeError("unified checkpoint is unavailable")
    body = {
        "schema": EVALUATION_SCHEMA,
        "campaign_identity_sha256": identity["identity_sha256"],
        "checkpoint_sha256": _file_sha256(resolved.weights_path),
        "evaluation_source_sha256s": _evaluation_source_sha256s(),
        "evaluation_seed": evaluation_seed,
        "per_cell": per_cell,
        "task_count": len(rows),
        "train_depths": list(spec.train_depths),
        "heldout_depth": deepest,
        "mean_depth_ce_gain": sum(differences) / len(differences),
        "relative_depth_ce_gain": sum(differences)
        / max(sum(row["trained_t1_ce"] for row in rows), 1e-9),
        "depth_wins": sum(value > 0.0 for value in differences),
        "depth_losses": sum(value < 0.0 for value in differences),
        "depth_sign_test_p_value": _sign_test_p_value(differences),
        "mean_base_ce_gain": sum(base_differences) / len(base_differences),
        "base_wins": sum(value > 0.0 for value in base_differences),
        "family_results": family_rows,
        "rows": rows,
        "claim_boundary": (
            "teacher-forced answer cross-entropy on fresh formal tasks; not decoded "
            "accuracy, broad reasoning, resident-32B evidence, or a WOW Signal"
        ),
    }
    return {**body, "report_sha256": _canonical_sha256(body)}


def evaluate_checkpoint(
    campaign_dir: Path,
    *,
    stem: str,
    per_cell: int,
    evaluation_seed: int,
    memory_limit_gb: float = 40.0,
    cache_limit_gb: float = 2.0,
    wired_limit_gb: float = 48.0,
    resource_stage_path: Path | None = None,
    resource_startup_lethal_mb: float | None = None,
    resource_steady_lethal_mb: float | None = None,
    preload_ready_path: Path | None = None,
    preload_release_path: Path | None = None,
    preload_key_path: Path | None = None,
    preload_config_sha256: str | None = None,
) -> dict[str, Any]:
    with unified_evaluation_context(
        campaign_dir,
        stem=stem,
        memory_limit_gb=memory_limit_gb,
        cache_limit_gb=cache_limit_gb,
        wired_limit_gb=wired_limit_gb,
        resource_stage_path=resource_stage_path,
        resource_startup_lethal_mb=resource_startup_lethal_mb,
        resource_steady_lethal_mb=resource_steady_lethal_mb,
        preload_ready_path=preload_ready_path,
        preload_release_path=preload_release_path,
        preload_key_path=preload_key_path,
        preload_config_sha256=preload_config_sha256,
    ) as loaded:
        (
            bundle,
            _initial_controller_unused,
            tokenizer,
            spec,
            identity,
            envelope,
            resource_receipt,
        ) = loaded
        report = _evaluate_loaded_checkpoint(
            campaign_dir,
            bundle=bundle,
            tokenizer=tokenizer,
            spec=spec,
            identity=identity,
            envelope=envelope,
            stem=stem,
            per_cell=per_cell,
            evaluation_seed=evaluation_seed,
        )
        body = {
            **{key: value for key, value in report.items() if key != "report_sha256"},
            "resource_envelope": envelope.to_receipt(),
            "resource_guard": resource_receipt,
        }
        return {**body, "report_sha256": _canonical_sha256(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--stem", default="checkpoint_best_heldout")
    parser.add_argument("--per-cell", type=int, default=8)
    parser.add_argument("--evaluation-seed", type=int, default=20260810203)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--memory-limit-gb", type=float, default=40.0)
    parser.add_argument("--cache-limit-gb", type=float, default=2.0)
    parser.add_argument("--wired-limit-gb", type=float, default=48.0)
    parser.add_argument("--resource-stage-path", type=Path)
    parser.add_argument("--resource-startup-lethal-mb", type=float)
    parser.add_argument("--resource-steady-lethal-mb", type=float)
    parser.add_argument("--preload-ready-path", type=Path)
    parser.add_argument("--preload-release-path", type=Path)
    parser.add_argument("--preload-key-path", type=Path)
    parser.add_argument("--preload-config-sha256")
    args = parser.parse_args()
    report = evaluate_checkpoint(
        args.campaign.expanduser().resolve(strict=True),
        stem=args.stem,
        per_cell=args.per_cell,
        evaluation_seed=args.evaluation_seed,
        memory_limit_gb=args.memory_limit_gb,
        cache_limit_gb=args.cache_limit_gb,
        wired_limit_gb=args.wired_limit_gb,
        resource_stage_path=args.resource_stage_path,
        resource_startup_lethal_mb=args.resource_startup_lethal_mb,
        resource_steady_lethal_mb=args.resource_steady_lethal_mb,
        preload_ready_path=args.preload_ready_path,
        preload_release_path=args.preload_release_path,
        preload_key_path=args.preload_key_path,
        preload_config_sha256=args.preload_config_sha256,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        target = args.report.expanduser().resolve()
        atomic_write_bytes(target, canonical_json_bytes(report) + b"\n")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
