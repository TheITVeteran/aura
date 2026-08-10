#!/usr/bin/env python3
"""Distill verified RLC state corrections into persistent recurrent tissue.

This is the first persistent-transfer discriminator for the causal CP144
mechanism.  Private exact solutions supervise only internal activation
differences on training tasks.  Held-out generation runs with both the exact
objective producer and the per-query verified teacher disabled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec  # noqa: E402
from core.brain.llm.latent_cortex.fast_weights import EpisodicFastWeights  # noqa: E402
from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (  # noqa: E402
    full_weight_checkpoint_identity,
)
from core.brain.llm.latent_cortex.types import FastWeightsConfig  # noqa: E402
from core.learning.recurrence_curriculum import task_battery  # noqa: E402
from core.learning.recurrence_native_objective_v2 import (  # noqa: E402
    generate_cached_live_path_rollin,
)
from core.learning.recurrent_behavioral_probe import (  # noqa: E402
    build_paired_full_engine_probe_reports,
    paired_generation_seed,
    tokenize_task,
)
from core.learning.recurrent_checkpoint_admission import (  # noqa: E402
    build_recurrence_task_manifest,
)
from core.learning.recurrent_grpo import (  # noqa: E402
    attach_coda_policy_adapters_at_sites,
    attach_recurrent_policy_adapters,
)
from core.learning.recurrent_sft_execution import (  # noqa: E402
    adapter_tensor_dict,
    adapter_tensor_fingerprint,
)
from core.learning.verified_trajectory_distillation import (  # noqa: E402
    evaluate_verified_trajectory_transfer,
    fit_verified_trajectory_inventory,
    install_verified_trajectory_inventory,
)
from core.runtime.atomic_writer import atomic_append_text, atomic_write_bytes  # noqa: E402
from core.runtime.mlx_memory_guard import mlx_memory_envelope  # noqa: E402
from core.runtime.model_lane_control import standalone_model_lane  # noqa: E402

CANARY_SCHEMA: Final = "aura.verified_trajectory_distillation_canary.v1"
PROGRESS_SCHEMA: Final = "aura.verified_trajectory_distillation.progress.v1"
SOURCE_PATHS: Final = (
    "core/brain/llm/latent_cortex/engine.py",
    "core/brain/llm/latent_cortex/fast_weights.py",
    "core/brain/llm/latent_cortex/recurrence_adapter.py",
    "core/learning/recurrence_curriculum.py",
    "core/learning/recurrence_native_objective_v2.py",
    "core/learning/recurrent_behavioral_probe.py",
    "core/learning/recurrent_grpo.py",
    "core/learning/verified_trajectory_distillation.py",
    "tools/run_verified_trajectory_distillation_canary.py",
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class ProgressLedger:
    def __init__(self, out_dir: Path, *, source_commit: str) -> None:
        out_dir.mkdir(parents=True, exist_ok=False)
        self.out_dir = out_dir
        self.source_commit = source_commit
        self.started = time.time()
        self.sequence = 0

    def emit(self, phase: str, **details: Any) -> dict[str, Any]:
        self.sequence += 1
        body = {
            "schema": PROGRESS_SCHEMA,
            "sequence": self.sequence,
            "phase": phase,
            "pid": os.getpid(),
            "source_commit": self.source_commit,
            "elapsed_s": time.time() - self.started,
            "details": details,
        }
        event = {**body, "event_sha256": _sha256_bytes(_canonical_bytes(body))}
        payload = _canonical_bytes(event)
        atomic_append_text(self.out_dir / "progress.jsonl", payload.decode() + "\n")
        atomic_write_bytes(self.out_dir / "progress.json", payload, mode=0o600)
        detail = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
        print(f"[trajectory] {phase} {detail}".rstrip(), file=sys.stderr, flush=True)
        return event


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_state() -> tuple[str, dict[str, dict[str, Any]]]:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("trajectory canary requires a clean source worktree")
    head = _git("rev-parse", "HEAD")
    if head != _git("rev-parse", "origin/main"):
        raise RuntimeError("trajectory canary source is not published on origin/main")
    bindings = {}
    for relative in SOURCE_PATHS:
        payload = (REPO_ROOT / relative).read_bytes()
        committed = subprocess.run(
            ["git", "show", f"{head}:{relative}"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        if payload != committed:
            raise RuntimeError(f"trajectory source differs from commit: {relative}")
        bindings[relative] = {
            "sha256": _sha256_bytes(payload),
            "size_bytes": len(payload),
        }
    return head, bindings


def _trajectory_io_features(
    model: Any,
    fast_weights: EpisodicFastWeights,
    context_tokens: Sequence[int],
    *,
    token_start: int,
) -> tuple[dict[int, Any], dict[int, Any]]:
    import mlx.core as mx

    tokens = [int(token) for token in context_tokens]
    if not tokens or not 0 <= token_start < len(tokens):
        raise ValueError("trajectory context boundary is invalid")

    def forward() -> Any:
        hidden = model.model.embed_tokens(mx.array([tokens]))
        for layer in model.model.layers:
            hidden = layer(hidden, None, None)
        mx.eval(hidden)
        return hidden

    return fast_weights.capture_io_features(forward, token_start=token_start)


def _capture_training_inventory(
    model: Any,
    tokenizer: Any,
    tasks: Sequence[Any],
    *,
    spec: RLCExecutionSpec,
    rank: int,
    layers: int,
    seed: int,
    progress: ProgressLedger,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[dict[str, Any]]]:
    import mlx.core as mx

    prelude_end = max(1, int(len(model.model.layers) * spec.prelude_frac))
    coda_start = min(
        len(model.model.layers) - 1,
        len(model.model.layers) - max(1, int(len(model.model.layers) * spec.coda_frac)),
    )
    fast_weights = EpisodicFastWeights(
        FastWeightsConfig(
            enabled=True,
            rank=rank,
            scale=1.0,
            target="o_proj",
            layer_placement="early",
            max_wrapped_layers=layers,
            opt_steps=0,
            canary_enabled=False,
            canary_generated_enabled=False,
        )
    )
    fast_weights.attach(
        model.model,
        (prelude_end, coda_start),
        seed_stat=1.0,
        episode_id=f"trajectory-distillation-{seed}",
    )
    inputs: dict[str, list[np.ndarray]] = defaultdict(list)
    corrections: dict[str, list[np.ndarray]] = defaultdict(list)
    manifest: list[dict[str, Any]] = []
    try:
        for task_ordinal, task in enumerate(tasks):
            prompt_tokens, target_tokens = tokenize_task(
                tokenizer,
                task.prompt,
                task.training_target,
            )
            eos = getattr(tokenizer, "eos_token_id", None)
            if eos is not None and target_tokens and target_tokens[-1] == int(eos):
                target_tokens = target_tokens[:-1]
            for branch_index in range(len(spec.branch_roles)):
                sample_seed = paired_generation_seed(
                    seed,
                    task_ordinal,
                    task.task_id,
                    branch_index + 1,
                )
                incumbent = generate_cached_live_path_rollin(
                    model,
                    prompt_tokens,
                    spec=spec,
                    branch_index=branch_index,
                    token_count=len(target_tokens),
                    seed=sample_seed,
                    temperature=0.0,
                )
                _teacher_inputs, teacher_outputs = _trajectory_io_features(
                    model,
                    fast_weights,
                    [*prompt_tokens, *target_tokens],
                    token_start=len(prompt_tokens),
                )
                incumbent_inputs, incumbent_outputs = _trajectory_io_features(
                    model,
                    fast_weights,
                    [*prompt_tokens, *incumbent.tokens],
                    token_start=len(prompt_tokens),
                )
                if set(teacher_outputs) != set(incumbent_outputs) or set(
                    incumbent_inputs
                ) != set(incumbent_outputs):
                    raise RuntimeError("trajectory decode I/O inventories differ")
                for layer_index in sorted(incumbent_outputs):
                    site = f"model.layers.{layer_index}.self_attn.o_proj"
                    layer_inputs = incumbent_inputs[layer_index]
                    layer_corrections = (
                        teacher_outputs[layer_index] - incumbent_outputs[layer_index]
                    )
                    mx.eval(layer_inputs, layer_corrections)
                    if int(layer_inputs.shape[0]) != int(layer_corrections.shape[0]):
                        raise RuntimeError("trajectory decode I/O row counts differ")
                    for position in range(int(layer_inputs.shape[0])):
                        inputs[site].append(
                            np.asarray(layer_inputs[position]).astype(np.float64)
                        )
                        corrections[site].append(
                            np.asarray(layer_corrections[position]).astype(np.float64)
                        )
                manifest.append(
                    {
                        "task_id": task.task_id,
                        "branch_index": branch_index,
                        "prompt_sha256": _sha256_bytes(task.prompt.encode()),
                        "private_target_sha256": _sha256_bytes(
                            task.training_target.encode()
                        ),
                        "incumbent_tokens_sha256": _sha256_bytes(
                            _canonical_bytes(list(incumbent.tokens))
                        ),
                    }
                )
                progress.emit(
                    "trajectory_pair_captured",
                    task=task_ordinal + 1,
                    total_tasks=len(tasks),
                    branch=branch_index,
                    sites=len(incumbent_outputs),
                )
                mx.clear_cache()
    finally:
        fast_weights.detach()
    inventory = {
        site: (np.stack(inputs[site]), np.stack(corrections[site]))
        for site in sorted(inputs)
    }
    if len(inventory) != layers:
        raise RuntimeError("captured trajectory site inventory differs from topology")
    return inventory, manifest


def _report_score(report: Mapping[str, Any]) -> int:
    return int(report["total_correct"])


@contextmanager
def _zeroed_recurrence_adapter(model: Any) -> Iterator[None]:
    import mlx.core as mx

    from core.brain.llm.latent_cortex.recurrence_adapter import (
        ScopedCodaLoRALinear,
        ScopedLoRALinear,
    )

    snapshots = []
    for layer in model.model.layers:
        projection = getattr(getattr(layer, "self_attn", None), "o_proj", None)
        if isinstance(projection, (ScopedLoRALinear, ScopedCodaLoRALinear)):
            snapshots.append((projection, projection.lora_b))
            projection.lora_b = mx.zeros_like(projection.lora_b)
    if not snapshots:
        raise RuntimeError("trajectory lesion found no scoped adapter")
    mx.eval(*(projection.lora_b for projection, _ in snapshots))
    try:
        yield
    finally:
        for projection, value in snapshots:
            projection.lora_b = value
        mx.eval(*(projection.lora_b for projection, _ in snapshots))


@contextmanager
def _permuted_recurrence_adapter(model: Any) -> Iterator[None]:
    import mlx.core as mx

    from core.brain.llm.latent_cortex.recurrence_adapter import (
        ScopedCodaLoRALinear,
        ScopedLoRALinear,
    )

    snapshots = []
    for layer in model.model.layers:
        projection = getattr(getattr(layer, "self_attn", None), "o_proj", None)
        if isinstance(projection, (ScopedLoRALinear, ScopedCodaLoRALinear)):
            snapshots.append((projection, projection.lora_b))
            width = int(projection.lora_b.shape[1])
            permutation = mx.array(list(range(1, width)) + [0])
            projection.lora_b = projection.lora_b[:, permutation]
    if not snapshots:
        raise RuntimeError("trajectory sham found no scoped adapter")
    mx.eval(*(projection.lora_b for projection, _ in snapshots))
    try:
        yield
    finally:
        for projection, value in snapshots:
            projection.lora_b = value
        mx.eval(*(projection.lora_b for projection, _ in snapshots))


def run_canary(
    *,
    model_path: Path,
    out_dir: Path,
    seed: int,
    memory_fraction: float,
    training_families: Sequence[str],
    training_depths: Sequence[int],
    training_per_cell: int,
    validation_per_cell: int,
    proxy_per_cell: int,
    lora_rank: int,
    lora_layers: int,
    regularization: float,
    gain: float,
    stop_after_transfer_diagnostic: bool = False,
) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm import load

    source_commit, source_bindings = _source_state()
    progress = ProgressLedger(out_dir, source_commit=source_commit)
    progress.emit("source_bound", model_path=str(model_path))
    base_before = full_weight_checkpoint_identity(model_path)
    spec = RLCExecutionSpec(
        n_slots=4,
        branch_roles=("constructive_solution", "critical_audit"),
        recurrent_steps=2,
        exchange_interval=1,
    )
    with (
        standalone_model_lane(
            owner_id=f"verified-trajectory:{out_dir.name}",
            model_path=str(model_path),
            purpose="training",
            preemptible=False,
            metadata={"tool": Path(__file__).name, "source_commit": source_commit},
        ),
        mlx_memory_envelope(fraction=memory_fraction, restore_limits_on_exit=True),
    ):
        progress.emit("model_load")
        model, tokenizer = load(str(model_path))
        sites = attach_recurrent_policy_adapters(
            model,
            spec,
            lora_rank=lora_rank,
            lora_layers=lora_layers,
            lora_targets=("o_proj",),
            initialization_seed=(seed ^ 0x51F7A11) & 0xFFFFFFFF,
            lora_scale=1.0,
            lora_layer_placement="early",
        )
        adapter_before = adapter_tensor_fingerprint(adapter_tensor_dict(model))
        training_tasks = task_battery(
            list(training_families),
            list(training_depths),
            training_per_cell,
            seed=seed,
        )
        validation_tasks = task_battery(
            list(training_families),
            list(training_depths),
            validation_per_cell,
            seed=seed + 3_959,
            excluded_prompts=tuple(task.prompt for task in training_tasks),
            excluded_task_ids=tuple(task.task_id for task in training_tasks),
        )
        proxy_tasks = task_battery(
            list(training_families),
            list(training_depths),
            proxy_per_cell,
            seed=seed + 7_919,
            excluded_prompts=tuple(
                task.prompt for task in (*training_tasks, *validation_tasks)
            ),
            excluded_task_ids=tuple(
                task.task_id for task in (*training_tasks, *validation_tasks)
            ),
        )
        proxy_manifest, proxy_manifest_sha256 = build_recurrence_task_manifest(proxy_tasks)

        def probe_progress(event: dict[str, Any]) -> None:
            progress.emit("behavioral_probe", **event)

        teaching_pairs, teaching_manifest = _capture_training_inventory(
            model,
            tokenizer,
            training_tasks,
            spec=spec,
            rank=min(2, lora_rank),
            layers=lora_layers,
            seed=seed,
            progress=progress,
        )
        validation_pairs, validation_manifest = _capture_training_inventory(
            model,
            tokenizer,
            validation_tasks,
            spec=spec,
            rank=min(2, lora_rank),
            layers=lora_layers,
            seed=seed + 3_959,
            progress=progress,
        )
        decode_phases = {site: "decode" for site in teaching_pairs}
        fitted = fit_verified_trajectory_inventory(
            teaching_pairs,
            rank=lora_rank,
            regularization=regularization,
            gain=gain,
            adapter_scale=1.0,
            site_phases=decode_phases,
            normalize_corrections=False,
        )
        transfer_diagnostic = evaluate_verified_trajectory_transfer(
            fitted,
            validation_pairs,
            training_pairs=teaching_pairs,
        )
        progress.emit(
            "transfer_diagnostic",
            **transfer_diagnostic["aggregate"],
        )
        teaching_pairs_path = out_dir / "private_teaching_pairs.npz"
        np.savez_compressed(
            teaching_pairs_path,
            **{
                f"training__{site.replace('.', '__')}__{kind}": matrix
                for site, pair in teaching_pairs.items()
                for kind, matrix in (("inputs", pair[0]), ("corrections", pair[1]))
            },
            **{
                f"validation__{site.replace('.', '__')}__{kind}": matrix
                for site, pair in validation_pairs.items()
                for kind, matrix in (("inputs", pair[0]), ("corrections", pair[1]))
            },
        )
        teaching_pairs_path.chmod(0o600)
        teaching_pairs_payload = teaching_pairs_path.read_bytes()
        teaching_pairs_artifact = {
            "sha256": _sha256_bytes(teaching_pairs_payload),
            "size_bytes": len(teaching_pairs_payload),
            "mode": "0600",
        }
        if stop_after_transfer_diagnostic:
            preflight_gates = {
                "base_checkpoint_immutable": (
                    base_before == full_weight_checkpoint_identity(model_path)
                ),
                "heldout_operator_better_than_zero": bool(
                    transfer_diagnostic["aggregate"]["better_than_zero_operator"]
                ),
                "heldout_operator_direction_positive": (
                    float(transfer_diagnostic["aggregate"]["cosine"]) > 0.0
                ),
            }
            preflight_body = {
                "schema": CANARY_SCHEMA,
                "mode": "transfer_diagnostic_only",
                "source_commit": source_commit,
                "source_bindings": source_bindings,
                "model_path": str(model_path),
                "model_identity": base_before,
                "execution_spec": spec.to_dict(),
                "execution_spec_sha256": spec.sha256,
                "private_teaching_manifest": teaching_manifest,
                "private_validation_manifest": validation_manifest,
                "private_teaching_pairs_artifact": teaching_pairs_artifact,
                "factor_receipts": {
                    site: dict(value.receipt) for site, value in fitted.items()
                },
                "transfer_diagnostic": transfer_diagnostic,
                "gates": preflight_gates,
                "preflight_admitted": all(preflight_gates.values()),
                "admitted": False,
                "claim_boundary": (
                    "heldout_internal_operator_transfer_only_not_behavioral_gain"
                ),
            }
            preflight_receipt = {
                **preflight_body,
                "receipt_sha256": _sha256_bytes(_canonical_bytes(preflight_body)),
            }
            atomic_write_bytes(
                out_dir / "receipt.json",
                _canonical_bytes(preflight_receipt),
                mode=0o600,
            )
            progress.emit(
                "complete",
                preflight_admitted=preflight_receipt["preflight_admitted"],
                **preflight_gates,
            )
            return preflight_receipt
        ordinary_before, untreated = build_paired_full_engine_probe_reports(
            model,
            tokenizer,
            proxy_tasks,
            model_path=model_path,
            spec=spec,
            adapter_sha256=adapter_before,
            task_manifest_sha256=proxy_manifest_sha256,
            seed=seed,
            objective_program_enabled=False,
            verified_objective_teacher_enabled=False,
            progress_callback=probe_progress,
        )
        progress.emit(
            "baseline_complete",
            ordinary=_report_score(ordinary_before),
            untreated=_report_score(untreated),
            observations=untreated["total_observations"],
        )
        del model
        mx.synchronize()
        mx.clear_cache()
        model, tokenizer = load(str(model_path))
        coda_sites = attach_coda_policy_adapters_at_sites(
            model,
            tuple(fitted),
            lora_rank=lora_rank,
            initialization_seed=(seed ^ 0xC0DA) & 0xFFFFFFFF,
            lora_scale=1.0,
        )
        if set(coda_sites) != set(sites):
            raise RuntimeError("trajectory coda topology differs from capture topology")
        installation = install_verified_trajectory_inventory(
            model,
            fitted,
            expected_sites=coda_sites,
        )
        adapter_after = adapter_tensor_fingerprint(adapter_tensor_dict(model))
        progress.emit("trajectory_installed", adapter_sha256=adapter_after)
        ordinary_after, treatment = build_paired_full_engine_probe_reports(
            model,
            tokenizer,
            proxy_tasks,
            model_path=model_path,
            spec=spec,
            adapter_sha256=adapter_after,
            task_manifest_sha256=proxy_manifest_sha256,
            seed=seed,
            objective_program_enabled=False,
            verified_objective_teacher_enabled=False,
            progress_callback=probe_progress,
        )
        treatment_gain = _report_score(treatment) > max(
            _report_score(untreated),
            _report_score(ordinary_after),
        )
        progress.emit(
            "treatment_complete",
            ordinary=_report_score(ordinary_after),
            untreated=_report_score(untreated),
            treatment=_report_score(treatment),
            strict_gain=treatment_gain,
        )
        lesion = None
        sham = None
        restored_after_lesion = True
        restored_after_sham = True
        if treatment_gain:
            with _zeroed_recurrence_adapter(model):
                lesion_sha256 = adapter_tensor_fingerprint(adapter_tensor_dict(model))
                _ordinary_lesion, lesion = build_paired_full_engine_probe_reports(
                    model,
                    tokenizer,
                    proxy_tasks,
                    model_path=model_path,
                    spec=spec,
                    adapter_sha256=lesion_sha256,
                    task_manifest_sha256=proxy_manifest_sha256,
                    seed=seed,
                    objective_program_enabled=False,
                    verified_objective_teacher_enabled=False,
                    progress_callback=probe_progress,
                )
            restored_after_lesion = (
                adapter_tensor_fingerprint(adapter_tensor_dict(model)) == adapter_after
            )
            with _permuted_recurrence_adapter(model):
                sham_sha256 = adapter_tensor_fingerprint(adapter_tensor_dict(model))
                _ordinary_sham, sham = build_paired_full_engine_probe_reports(
                    model,
                    tokenizer,
                    proxy_tasks,
                    model_path=model_path,
                    spec=spec,
                    adapter_sha256=sham_sha256,
                    task_manifest_sha256=proxy_manifest_sha256,
                    seed=seed,
                    objective_program_enabled=False,
                    verified_objective_teacher_enabled=False,
                    progress_callback=probe_progress,
                )
            restored_after_sham = (
                adapter_tensor_fingerprint(adapter_tensor_dict(model)) == adapter_after
            )
        mx.save_safetensors(str(out_dir / "adapter.safetensors"), adapter_tensor_dict(model))

    gates = {
        "base_checkpoint_immutable": base_before == full_weight_checkpoint_identity(model_path),
        "adapter_mutated": adapter_before != adapter_after,
        "heldout_operator_better_than_zero": bool(
            transfer_diagnostic["aggregate"]["better_than_zero_operator"]
        ),
        "heldout_operator_direction_positive": (
            float(transfer_diagnostic["aggregate"]["cosine"]) > 0.0
        ),
        "ordinary_control_stable": _report_score(ordinary_before)
        == _report_score(ordinary_after),
        "teacher_free_treatment_strict_gain": treatment_gain,
        "treatment_beats_lesion": bool(
            lesion is not None and _report_score(treatment) > _report_score(lesion)
        ),
        "treatment_beats_norm_preserving_sham": bool(
            sham is not None and _report_score(treatment) > _report_score(sham)
        ),
        "lesion_restored_exactly": restored_after_lesion,
        "sham_restored_exactly": restored_after_sham,
    }
    body = {
        "schema": CANARY_SCHEMA,
        "source_commit": source_commit,
        "source_bindings": source_bindings,
        "model_path": str(model_path),
        "model_identity": base_before,
        "execution_spec": spec.to_dict(),
        "execution_spec_sha256": spec.sha256,
        "configuration": {
            "seed": seed,
            "training_families": list(training_families),
            "training_depths": list(training_depths),
            "training_per_cell": training_per_cell,
            "validation_per_cell": validation_per_cell,
            "proxy_per_cell": proxy_per_cell,
            "lora_rank": lora_rank,
            "lora_layers": lora_layers,
            "regularization": regularization,
            "gain": gain,
        },
        "private_teaching_manifest": teaching_manifest,
        "private_validation_manifest": validation_manifest,
        "private_teaching_pairs_artifact": teaching_pairs_artifact,
        "transfer_diagnostic": transfer_diagnostic,
        "factor_receipts": {site: dict(value.receipt) for site, value in fitted.items()},
        "installation": installation,
        "proxy_manifest": proxy_manifest,
        "reports": {
            "ordinary_before": ordinary_before,
            "untreated": untreated,
            "ordinary_after": ordinary_after,
            "treatment": treatment,
            "lesion": lesion,
            "sham": sham,
        },
        "gates": gates,
        "admitted": all(gates.values()),
        "claim_boundary": (
            "bounded_teacher_free_persistent_gain_only_not_general_or_frontier_gain"
        ),
    }
    receipt = {**body, "receipt_sha256": _sha256_bytes(_canonical_bytes(body))}
    atomic_write_bytes(out_dir / "receipt.json", _canonical_bytes(receipt), mode=0o600)
    progress.emit("complete", admitted=receipt["admitted"], **gates)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026081003)
    parser.add_argument("--memory-fraction", type=float, default=0.35)
    parser.add_argument("--training-families", default="boolean,modular")
    parser.add_argument("--training-depths", default="2,3")
    parser.add_argument("--training-per-cell", type=int, default=2)
    parser.add_argument("--validation-per-cell", type=int, default=1)
    parser.add_argument("--proxy-per-cell", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-layers", type=int, default=8)
    parser.add_argument("--regularization", type=float, default=1e-3)
    parser.add_argument("--gain", type=float, default=0.25)
    parser.add_argument("--stop-after-transfer-diagnostic", action="store_true")
    args = parser.parse_args()
    families = tuple(part.strip() for part in args.training_families.split(",") if part.strip())
    try:
        depths = tuple(int(part.strip()) for part in args.training_depths.split(",") if part.strip())
        receipt = run_canary(
            model_path=args.model.expanduser().resolve(strict=True),
            out_dir=args.out_dir.expanduser().resolve(strict=False),
            seed=args.seed,
            memory_fraction=args.memory_fraction,
            training_families=families,
            training_depths=depths,
            training_per_cell=args.training_per_cell,
            validation_per_cell=args.validation_per_cell,
            proxy_per_cell=args.proxy_per_cell,
            lora_rank=args.lora_rank,
            lora_layers=args.lora_layers,
            regularization=args.regularization,
            gain=args.gain,
            stop_after_transfer_diagnostic=args.stop_after_transfer_diagnostic,
        )
    except BaseException as exc:  # noqa: BLE001 - preserve diagnostic process status
        print(f"trajectory canary failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True, indent=2))
    accepted = (
        receipt.get("preflight_admitted")
        if receipt.get("mode") == "transfer_diagnostic_only"
        else receipt["admitted"]
    )
    return 0 if accepted else 3


if __name__ == "__main__":
    raise SystemExit(main())
