#!/usr/bin/env python3
"""Independently evaluate a unified recurrence checkpoint on fresh tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx  # noqa: E402
from mlx.utils import tree_unflatten  # noqa: E402

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
from core.runtime.mlx_memory_guard import host_pressure, mlx_memory_envelope  # noqa: E402
from core.runtime.model_lane_control import standalone_model_lane  # noqa: E402
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
from tools.unified_intrinsic_tokenization_contract import (  # noqa: E402
    TOKENIZED_DATASET_FILENAME,
    load_source_dataset,
    verify_tokenized_dataset,
)

EVALUATION_SCHEMA = "aura.unified_intrinsic_independent_evaluation.v1"
EVALUATION_SOURCE_FILES = (
    "tools/evaluate_unified_intrinsic_checkpoint.py",
    "tools/evaluate_unified_intrinsic_decoding.py",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluation_source_sha256s() -> dict[str, str]:
    return {
        relative: _file_sha256(REPO_ROOT / relative)
        for relative in EVALUATION_SOURCE_FILES
    }


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
    if (
        not isinstance(expected_sha256, str)
        or controller.parameter_sha256() != expected_sha256
    ):
        raise RuntimeError("unified checkpoint initial controller differs")
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
    try:
        resolved = resolve_checkpoint_generation(
            campaign_dir,
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
    identity_body = {
        key: value for key, value in identity.items() if key != "identity_sha256"
    }
    if identity.get("identity_sha256") != _canonical_sha256(identity_body):
        raise RuntimeError("unified campaign identity differs")
    source_sha256s = identity.get("source_sha256s")
    if not isinstance(source_sha256s, dict) or set(source_sha256s) != set(
        TRAINING_SOURCE_FILES
    ) or any(
        source_sha256s[relative] != _file_sha256(REPO_ROOT / relative)
        for relative in TRAINING_SOURCE_FILES
    ):
        raise RuntimeError("unified campaign source differs")
    model_identity = identity.get("model")
    if (
        not isinstance(model_identity, dict)
        or _model_identity(str(model_identity.get("canonical_path"))) != model_identity
    ):
        raise RuntimeError("unified campaign model differs")
    runtime_identity = identity.get("runtime")
    if not isinstance(runtime_identity, dict) or _runtime_identity() != runtime_identity:
        raise RuntimeError("unified campaign runtime differs")
    dataset_identity = identity.get("dataset")
    dataset_name = (
        dataset_identity.get("path") if isinstance(dataset_identity, dict) else None
    )
    if (
        not isinstance(dataset_name, str)
        or Path(dataset_name).name != dataset_name
        or not isinstance(dataset_identity, dict)
    ):
        raise RuntimeError("unified campaign dataset identity differs")
    dataset_path = campaign_dir / dataset_name
    try:
        dataset_size = dataset_path.stat().st_size
        dataset_sha256 = _file_sha256(dataset_path)
    except OSError as exc:
        raise RuntimeError("unified campaign dataset is unavailable") from exc
    if (
        dataset_size != dataset_identity.get("size_bytes")
        or dataset_sha256 != dataset_identity.get("sha256")
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
        campaign_dir / tokenized_name,
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
    if (
        literal_contract.contract_sha256 != literal_identity.get("contract_sha256")
        or literal_contract.digit_token_ids != tokenizer_digit_token_ids(tokenizer)
    ):
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
        answer_contract.contract_sha256
        != answer_identity.get("contract_sha256")
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
    initial_controller = _initial_controller(
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
    resolved = resolve_checkpoint_generation(campaign_dir, stem=stem, required=True)
    if resolved is None:  # pragma: no cover - required=True is exhaustive
        raise RuntimeError("unified checkpoint is unavailable")
    identity = resolved.receipt.get("identity")
    model_identity = identity.get("model") if isinstance(identity, dict) else None
    model_path = (
        model_identity.get("canonical_path")
        if isinstance(model_identity, dict)
        else None
    )
    if not isinstance(model_path, str) or not model_path:
        raise RuntimeError("unified campaign model identity differs")
    pressure = host_pressure()
    if pressure.get("available") is not True or pressure.get("under_pressure") is not False:
        raise RuntimeError("unified evaluation refused unavailable or pressured host")
    resource_values = (
        resource_stage_path,
        resource_startup_lethal_mb,
        resource_steady_lethal_mb,
    )
    resource_enabled = all(value is not None for value in resource_values)
    if any(value is not None for value in resource_values) != resource_enabled:
        raise ValueError("evaluation resource guard arguments must be supplied together")
    with standalone_model_lane(
        owner_id=f"evaluate-unified-intrinsic:{campaign_dir.parent.name}",
        model_path=model_path,
        purpose="benchmark",
        request_gb=memory_limit_gb,
        preemptible=False,
        allow_owner_eviction=False,
        metadata={
            "tool": "evaluate_unified_intrinsic_checkpoint",
            "operator_launched": True,
        },
    ), mlx_memory_envelope(
        memory_gb=memory_limit_gb,
        cache_gb=cache_limit_gb,
        wired_gb=wired_limit_gb,
        restore_limits_on_exit=False,
    ) as envelope:
        bundle, initial_controller, tokenizer, spec, identity = _load_checkpoint(
            campaign_dir,
            stem=stem,
        )
        resource_receipt: dict[str, Any] | None = None
        if resource_enabled:
            resource_receipt = _await_resource_guard(
                resource_stage_path.expanduser(),
                trainer_sha256=_file_sha256(Path(__file__).resolve(strict=True)),
                startup_lethal_mb=float(resource_startup_lethal_mb),
                steady_lethal_mb=float(resource_steady_lethal_mb),
                timeout_s=120.0,
            )
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
    probability = sum(math.comb(len(signs), k) for k in range(tail + 1)) / (
        2 ** len(signs)
    )
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
        int(value)
        for value in identity.get("task_depths", (identity.get("task_depth"),))
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
    evaluation_depths = (
        campaign_depths if task_depth is None else (int(task_depth),)
    )
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
            "mean_depth_ce_gain": sum(row["depth_ce_gain"] for row in selected)
            / len(selected),
            "depth_wins": sum(row["depth_ce_gain"] > 0.0 for row in selected),
        }
    resolved = resolve_checkpoint_generation(campaign_dir, stem=stem, required=True)
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
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        target = args.report.expanduser().resolve()
        scratch = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        scratch.write_text(encoded, encoding="utf-8")
        with scratch.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(scratch, target)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
