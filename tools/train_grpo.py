#!/usr/bin/env python3
"""Verifier-driven RL on the resident cortex (CP233).

The training loop for CP229. Anima Rationis line 511 records the existence
proof: QwQ-32B reached DeepSeek-R1-comparable reasoning through RL over a
32B foundation with correctness verifiers for mathematics and execution
feedback for code -- the same parameter class as Aura's cortex.

Per step:

    1. sample K completions for one prompt at temperature
    2. grade each with a PROGRAM (never the model's own opinion)
    3. advantage_i = (r_i - mean r) / std r      -- the group is the baseline
    4. loss = -mean(advantage_i * logprob_i) + beta * KL(policy || reference)

The reference policy is this same model with the adapter scope disabled,
so the KL leash is measured against the true pre-RL behaviour rather than
a stale copy -- and it costs no extra memory, which matters on a host that
has already been taken down once by an unbounded run.

What this run refuses to do:

* **Report a loss curve as progress.** If every completion in a group earns
  the same grade, the advantages are all zero and the step taught nothing.
  Those groups are counted, and a run made mostly of them is declared to
  have no learning signal regardless of how tidy its loss looks.
* **Score itself on its training set.** Held-out tasks come from a
  separate seed with proven-disjoint prompts, and the verdict is the
  held-out number.
* **Claim a gain from format compliance.** Reward is correctness; format
  credit is capped, because formatting is far easier to learn than
  reasoning and a model that learns it looks like it is improving.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import signal
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.learning.grpo import (  # noqa: E402
    GRPOConfig,
    GRPOTelemetry,
    group_advantages,
    grpo_loss,
    reward_from_verdict,
    sequence_token_logprobs,
)
from core.learning.grpo_training_state import (  # noqa: E402
    GRPOCheckpointError,
    canonical_json_bytes,
    load_grpo_checkpoint,
    save_grpo_checkpoint,
    sha256_bytes,
)
from core.learning.verifiable_tasks import (  # noqa: E402
    disjoint_split,
    scaling_report,
)
from core.runtime.atomic_writer import (  # noqa: E402
    atomic_write_bytes,
    atomic_write_bytes_if_absent,
    ensure_private_directory,
    interprocess_file_lock,
)
from core.runtime.mlx_memory_guard import mlx_memory_envelope  # noqa: E402

GRPO_TRAIN_SCHEMA = "aura.grpo_training.v4"
GRPO_DATASET_SCHEMA = "aura.grpo_dataset.v1"
GRPO_PROTOCOL_SCHEMA = "aura.grpo_protocol.v3"
RNG_STRATEGY = "stateless_sha256_step_seeded_v1"
EXECUTION_MODES = ("standard", "recurrent")
TASK_SOURCES = ("verifiable", "recurrence_curriculum")
_ADAPTER_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


# Set by main() from --cot. Reasoning room is the fix the CP238 finding
# pointed at: the model failed program_trace at 0.05 because the terse
# FINAL_ANSWER format denied it chain-of-thought. This invites the
# token-level deliberation that actually makes models reason.
_COT_PREAMBLE = ""


def _stable_seed(base_seed: int, *parts: Any) -> int:
    """Process-independent seed for one named training decision."""
    payload = canonical_json_bytes([int(base_seed), *parts])
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _source_binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    before = resolved.stat()
    payload = resolved.read_bytes()
    after = resolved.stat()
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise RuntimeError(f"training source changed while hashing: {resolved}")
    return {
        "path": str(resolved.relative_to(REPO_ROOT)),
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
    }


def _task_record(task: Any) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "prompt": task.prompt,
        "domain": task.domain,
        "depth": task.depth,
        "knowledge": task.knowledge,
        "grader": task.grader,
        "expected": task.expected,
        "metadata": task.metadata,
    }


def _dataset_payload(
    train_tasks: Sequence[Any], holdout_tasks: Sequence[Any], *, seed: int
) -> dict[str, Any]:
    return {
        "schema": GRPO_DATASET_SCHEMA,
        "seed": int(seed),
        "train": [_task_record(task) for task in train_tasks],
        "holdout": [_task_record(task) for task in holdout_tasks],
    }


def _assert_exact_adapter_keys(
    expected: Mapping[str, Any], loaded: Mapping[str, Any]
) -> None:
    expected_keys = set(expected)
    loaded_keys = set(loaded)
    if loaded_keys == expected_keys:
        return
    missing = sorted(expected_keys - loaded_keys)
    unexpected = sorted(loaded_keys - expected_keys)
    raise GRPOCheckpointError(
        "checkpoint adapter keyset differs "
        f"(missing={missing[:5]}, unexpected={unexpected[:5]})"
    )


def _point_estimate_delta(
    baseline: Mapping[str, Any] | None, final: Mapping[str, Any] | None
) -> float | None:
    if baseline is None or final is None:
        return None
    return round(float(final["overall"]) - float(baseline["overall"]), 6)


def _calibration_token_budget(max_tokens: int, requested: int) -> int:
    if requested not in (0, max_tokens):
        raise ValueError(
            "calibration tokens must equal training max tokens; a shorter "
            "probe truncates reasoning and corrupts learnability"
        )
    return max_tokens


def _build_task_split(
    *,
    task_source: str,
    domains: list[str],
    depths: list[int],
    train_per_cell: int,
    holdout_per_cell: int,
    seed: int,
) -> tuple[list[Any], list[Any], Path]:
    """Build one source-bound split without mixing training registries."""
    if task_source == "verifiable":
        train, holdout = disjoint_split(
            domains=domains,
            depths=depths,
            train_per_cell=train_per_cell,
            holdout_per_cell=holdout_per_cell,
            seed=seed,
        )
        source = REPO_ROOT / "core/learning/verifiable_tasks.py"
    elif task_source == "recurrence_curriculum":
        from core.learning.recurrence_curriculum import disjoint_task_split

        train, holdout = disjoint_task_split(
            families=domains,
            depths=depths,
            train_per_cell=train_per_cell,
            holdout_per_cell=holdout_per_cell,
            seed=seed,
        )
        source = REPO_ROOT / "core/learning/recurrence_curriculum.py"
    else:
        raise ValueError(f"unsupported task source: {task_source}")
    return list(train), list(holdout), source


def _publish_adapter_snapshot(path: Path, tensors: Mapping[str, Any]) -> None:
    import mlx.core as mx

    scratch = path.parent / f".{path.stem}.{time.time_ns()}.tmp.safetensors"
    try:
        mx.save_safetensors(str(scratch), dict(tensors))
        atomic_write_bytes(path, scratch.read_bytes(), mode=0o600)
    finally:
        scratch.unlink(missing_ok=True)


def _publish_immutable_bytes(path: Path, payload: bytes, *, role: str) -> None:
    if path.is_symlink():
        raise GRPOCheckpointError(f"{role} symlink is forbidden")
    if path.exists():
        try:
            existing = path.read_bytes()
        except OSError as exc:
            raise GRPOCheckpointError(f"{role} is unreadable") from exc
        if existing != payload:
            raise GRPOCheckpointError(f"{role} differs from the frozen run")
        return
    if not atomic_write_bytes_if_absent(path, payload, mode=0o600):
        if path.is_symlink() or path.read_bytes() != payload:
            raise GRPOCheckpointError(f"{role} publication raced with different bytes")


def _artifact_binding(relative: str, payload: bytes) -> dict[str, Any]:
    return {
        "path": relative,
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
    }


def _read_recurrent_bundle_artifacts(
    out_dir: Path, manifest: Mapping[str, Any]
) -> dict[str, bytes]:
    from core.brain.llm.latent_cortex.recurrent_grpo_adapter_identity import (
        declared_bindings,
    )

    root = out_dir.resolve(strict=True)
    artifacts: dict[str, bytes] = {}
    for _role, binding in declared_bindings(manifest):
        path = (root / binding["path"]).resolve(strict=True)
        if path.parent != root and root not in path.parents:
            raise GRPOCheckpointError("recurrent adapter artifact escapes run root")
        artifacts[binding["path"]] = path.read_bytes()
    artifacts["training_completion.json"] = (
        root / "training_completion.json"
    ).read_bytes()
    return artifacts


def _validate_published_recurrent_bundle(
    out_dir: Path,
    *,
    adapter_id: str,
    base_identity: Mapping[str, Any],
    behavior_identity: Mapping[str, Any],
    personality_identity: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
) -> dict[str, Any]:
    from core.brain.llm.latent_cortex.adapter_identity import (
        inspect_mlx_tensor_metadata,
    )
    from core.brain.llm.latent_cortex.recurrent_grpo_adapter_identity import (
        MANIFEST_FILE,
        strict_json_loads,
        validate_recurrent_grpo_adapter_identity,
    )

    manifest_bytes = (out_dir / MANIFEST_FILE).read_bytes()
    manifest = strict_json_loads(manifest_bytes, role="published_manifest")
    adapter_path = out_dir / manifest["adapter"]["path"]
    return validate_recurrent_grpo_adapter_identity(
        manifest_bytes,
        adapter_id=adapter_id,
        actual_base_checkpoint=base_identity,
        actual_model_behavior_bundle=behavior_identity,
        actual_personality_adapter=personality_identity,
        actual_runtime_environment=runtime_identity,
        artifacts=_read_recurrent_bundle_artifacts(out_dir, manifest),
        tensor_metadata=inspect_mlx_tensor_metadata(adapter_path),
    )


def _publish_recurrent_adapter_bundle(
    out_dir: Path,
    *,
    adapter_id: str,
    protocol: Mapping[str, Any],
    protocol_bytes: bytes,
    dataset_bytes: bytes,
    receipt: Mapping[str, Any],
    receipt_bytes: bytes,
    execution_spec: Any,
    source_roles: Mapping[str, Path],
) -> dict[str, Any]:
    """Publish and immediately revalidate a campaign-loadable GRPO identity."""

    from core.brain.llm.latent_cortex.adapter_identity import (
        inspect_mlx_tensor_metadata,
    )
    from core.brain.llm.latent_cortex.recurrent_grpo_adapter_identity import (
        COMPLETION_SCHEMA,
        LOADER_CONFIG_SCHEMA,
        MANIFEST_FILE,
        MANIFEST_SCHEMA,
        REQUIRED_SOURCE_ROLES,
        TRAINING_METHOD,
        declared_bindings,
        validate_recurrent_grpo_adapter_identity,
    )

    completion_path = out_dir / "training_completion.json"
    if completion_path.exists():
        return _validate_published_recurrent_bundle(
            out_dir,
            adapter_id=adapter_id,
            base_identity=protocol["base_checkpoint"],
            behavior_identity=protocol["model_behavior"],
            personality_identity=protocol["personality_adapter"],
            runtime_identity=protocol["runtime"],
        )
    if set(source_roles) != REQUIRED_SOURCE_ROLES:
        raise GRPOCheckpointError("recurrent GRPO source inventory is incomplete")
    if receipt.get("execution_mode") != "recurrent":
        raise GRPOCheckpointError("only recurrent GRPO can publish this identity")
    termination = receipt.get("termination")
    if (
        not isinstance(termination, Mapping)
        or termination.get("reason") != "max_steps"
        or termination.get("completed_budget") is not True
        or termination.get("signal") is not None
    ):
        raise GRPOCheckpointError("recurrent GRPO training is not complete")

    campaign_dir = ensure_private_directory(out_dir / "campaign_adapter")
    source_adapter = out_dir / "grpo_adapters.safetensors"
    adapter_bytes = source_adapter.read_bytes()
    documents = {
        "campaign_adapter/adapters.safetensors": adapter_bytes,
        "campaign_adapter/adapter_final.safetensors": adapter_bytes,
        "campaign_adapter/grpo_receipt.json": receipt_bytes,
        "campaign_adapter/training_protocol.json": protocol_bytes,
        "campaign_adapter/dataset_manifest.json": dataset_bytes,
        "campaign_adapter/execution_spec.json": canonical_json_bytes(
            execution_spec.to_dict()
        ),
    }
    for relative, payload in documents.items():
        _publish_immutable_bytes(
            out_dir / relative,
            payload,
            role=relative.replace("/", " "),
        )

    tensor_metadata = inspect_mlx_tensor_metadata(
        campaign_dir / "adapters.safetensors"
    )
    tensor_records = [record.to_dict() for record in tensor_metadata]
    projection_paths = sorted(
        {
            record["key"].removesuffix(".lora_a").removesuffix(".lora_b")
            for record in tensor_records
        }
    )
    targets = [part.strip() for part in protocol["training"]["lora_targets"].split(",")]
    trainable_params = sum(
        math.prod(record["shape"]) for record in tensor_records
    )
    unique_layers = {int(path.split(".")[2]) for path in projection_paths}
    loader_config = {
        "schema": LOADER_CONFIG_SCHEMA,
        "fine_tune_type": "recurrent_grpo_scoped_lora",
        "loader": "aura_custom_loader_required",
        "model": protocol["model_path"],
        "num_layers": len(unique_layers),
        "wrapped_projection_count": len(projection_paths),
        "lora_parameters": {
            "rank": protocol["training"]["lora_rank"],
            "scale": 20.0,
            "dropout": 0.0,
            "keys": targets,
        },
        "execution_spec_sha256": execution_spec.sha256,
        "training_method": TRAINING_METHOD,
    }
    loader_bytes = canonical_json_bytes(loader_config)
    _publish_immutable_bytes(
        campaign_dir / "adapter_config.json",
        loader_bytes,
        role="campaign adapter loader config",
    )

    sources: dict[str, dict[str, Any]] = {}
    for role in sorted(REQUIRED_SOURCE_ROLES):
        protocol_binding = protocol["sources"][role]
        snapshot_relative = f"source_snapshots/{role}.py"
        snapshot_bytes = (out_dir / snapshot_relative).read_bytes()
        if (
            len(snapshot_bytes) != protocol_binding["size_bytes"]
            or sha256_bytes(snapshot_bytes) != protocol_binding["sha256"]
        ):
            raise GRPOCheckpointError(f"frozen recurrent source differs: {role}")
        sources[role] = {
            "origin_path": protocol_binding["path"],
            "snapshot_path": snapshot_relative,
            "sha256": protocol_binding["sha256"],
            "size_bytes": protocol_binding["size_bytes"],
        }

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "adapter_id": adapter_id,
        "training_method": TRAINING_METHOD,
        "base_checkpoint": protocol["base_checkpoint"],
        "model_behavior_bundle": protocol["model_behavior"],
        "personality_adapter": protocol["personality_adapter"],
        "training_runtime": protocol["runtime"],
        "adapter": _artifact_binding(
            "campaign_adapter/adapters.safetensors", adapter_bytes
        ),
        "adapter_alias": _artifact_binding(
            "campaign_adapter/adapter_final.safetensors", adapter_bytes
        ),
        "loader_config": _artifact_binding(
            "campaign_adapter/adapter_config.json", loader_bytes
        ),
        "training_receipt": _artifact_binding(
            "campaign_adapter/grpo_receipt.json", receipt_bytes
        ),
        "training_protocol": _artifact_binding(
            "campaign_adapter/training_protocol.json", protocol_bytes
        ),
        "dataset_manifest": _artifact_binding(
            "campaign_adapter/dataset_manifest.json", dataset_bytes
        ),
        "execution_spec": _artifact_binding(
            "campaign_adapter/execution_spec.json",
            documents["campaign_adapter/execution_spec.json"],
        ),
        "protocol_sha256": sha256_bytes(protocol_bytes),
        "dataset_sha256": sha256_bytes(dataset_bytes),
        "execution_spec_sha256": execution_spec.sha256,
        "sources": sources,
        "lora": {
            "rank": protocol["training"]["lora_rank"],
            "targets": targets,
            "wrapped_projections": len(projection_paths),
            "projection_paths": projection_paths,
            "trainable_params": trainable_params,
        },
        "tensors": tensor_records,
    }
    manifest_bytes = canonical_json_bytes(manifest)
    _publish_immutable_bytes(
        out_dir / MANIFEST_FILE,
        manifest_bytes,
        role="recurrent GRPO adapter manifest",
    )
    completion = {
        "schema": COMPLETION_SCHEMA,
        "complete": True,
        "halt_reason": "max_steps",
        "step": receipt["steps"],
        "optimizer_updates": receipt["optimizer_updates"],
        "adapter_sha256": manifest["adapter"]["sha256"],
        "receipt_sha256": manifest["training_receipt"]["sha256"],
        "protocol_sha256": manifest["protocol_sha256"],
        "execution_spec_sha256": execution_spec.sha256,
        "manifest_sha256": sha256_bytes(manifest_bytes),
    }
    completion_bytes = canonical_json_bytes(completion)
    preflight_artifacts: dict[str, bytes] = {}
    for _role, binding in declared_bindings(manifest):
        preflight_artifacts[binding["path"]] = (out_dir / binding["path"]).read_bytes()
    preflight_artifacts["training_completion.json"] = completion_bytes
    preflight_identity = validate_recurrent_grpo_adapter_identity(
        manifest_bytes,
        adapter_id=adapter_id,
        actual_base_checkpoint=protocol["base_checkpoint"],
        actual_model_behavior_bundle=protocol["model_behavior"],
        actual_personality_adapter=protocol["personality_adapter"],
        actual_runtime_environment=protocol["runtime"],
        artifacts=preflight_artifacts,
        tensor_metadata=tensor_metadata,
    )
    _publish_immutable_bytes(
        completion_path,
        completion_bytes,
        role="recurrent GRPO training completion",
    )
    published_identity = _validate_published_recurrent_bundle(
        out_dir,
        adapter_id=adapter_id,
        base_identity=protocol["base_checkpoint"],
        behavior_identity=protocol["model_behavior"],
        personality_identity=protocol["personality_adapter"],
        runtime_identity=protocol["runtime"],
    )
    if published_identity != preflight_identity:
        raise GRPOCheckpointError("published recurrent identity differs from preflight")
    return published_identity


def _render(tokenizer, task) -> str:
    content = task.prompt
    if _COT_PREAMBLE:
        content = _COT_PREAMBLE + "\n\n" + content
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        tokenize=False,
    )


def _load_execution_spec(mode: str, path: str | None):
    if mode not in EXECUTION_MODES:
        raise ValueError(f"unsupported execution mode: {mode}")
    if mode == "standard":
        if path:
            raise ValueError("--execution-spec only applies to recurrent mode")
        return None
    if not path:
        raise ValueError("recurrent mode requires --execution-spec")
    from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec

    spec_path = Path(path).expanduser().resolve(strict=True)
    if not spec_path.is_file():
        raise ValueError("execution spec must be a regular file")
    try:
        payload = json.loads(spec_path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("execution spec is not readable canonical JSON") from exc
    return RLCExecutionSpec.from_dict(payload)


def _task_prompt_tokens(tokenizer, task) -> list[int]:
    tokens = list(tokenizer.encode(_render(tokenizer, task)))
    if not tokens or any(type(token) is not int or token < 0 for token in tokens):
        raise RuntimeError("rendered task produced invalid prompt tokens")
    return tokens


def sample_recurrent_group(
    model,
    tokenizer,
    task,
    *,
    spec,
    size: int,
    max_tokens: int,
    seed: int,
):
    """Bounded behavior-policy completions from the fixed recurrent graph."""

    from core.learning.recurrent_grpo import (
        RecurrentSamplingAdmissionError,
        RecurrentSamplingConfig,
        sample_recurrent_completion,
    )

    prompt_tokens = _task_prompt_tokens(tokenizer, task)
    sampling = RecurrentSamplingConfig(max_tokens=max_tokens)
    samples = []
    completions: list[str] = []
    rejected_receipts: list[dict[str, Any]] = []
    max_attempts = max(size + 2, size * 4)
    for attempt in range(max_attempts):
        if len(samples) >= size:
            break
        try:
            sample = sample_recurrent_completion(
                model,
                prompt_tokens,
                spec=spec,
                seed=_stable_seed(seed, "recurrent_completion", attempt),
                sampling=sampling,
                tokenizer=tokenizer,
            )
        except RecurrentSamplingAdmissionError as exc:
            receipt = exc.sample.receipt()
            receipt["rejected_attempt"] = attempt
            rejected_receipts.append(receipt)
            continue
        samples.append(sample)
        completions.append(tokenizer.decode(list(sample.tokens)))
    if len(samples) < size:
        payload = {
            "schema": "aura.recurrent_group_sampling_exhausted.v1",
            "requested": int(size),
            "admitted": len(samples),
            "attempts": int(max_attempts),
            "rejected": rejected_receipts,
        }
        raise RuntimeError(
            "recurrent group sampling exhausted admissible cached completions: "
            + json.dumps(payload, separators=(",", ":"), sort_keys=True)[:2000]
        )
    return prompt_tokens, samples, completions


def _record_recurrent_step_failure(
    out_dir: Path,
    *,
    protocol_sha256: str,
    dataset_sha256: str,
    execution_spec_sha256: str,
    attempted_step: int,
    last_durable_step: int,
    phase: str,
    task_id: str | None,
    sample_seed: int | None,
    samples: Sequence[Any],
    error: Exception,
) -> Path:
    """Durably bind a failed recurrent attempt without mutating its checkpoint."""

    sample_receipts = [sample.receipt() for sample in samples]
    rejected_sample = getattr(error, "sample", None)
    payload = {
        "schema": "aura.grpo_recurrent_failure.v1",
        "protocol_sha256": protocol_sha256,
        "dataset_sha256": dataset_sha256,
        "execution_spec_sha256": execution_spec_sha256,
        "attempted_step": int(attempted_step),
        "last_durable_step": int(last_durable_step),
        "volatile_completed_steps": max(
            0, int(attempted_step) - 1 - int(last_durable_step)
        ),
        "phase": str(phase),
        "task_id": task_id,
        "sample_seed": sample_seed,
        "samples": sample_receipts,
        "rejected_sample": (
            rejected_sample.receipt()
            if rejected_sample is not None
            and callable(getattr(rejected_sample, "receipt", None))
            else None
        ),
        "error": {
            "type": type(error).__name__,
            "message": str(error)[:2000],
        },
        "recorded_at_ns": time.time_ns(),
    }
    encoded = canonical_json_bytes(payload)
    incident = sha256_bytes(encoded)[:16]
    failures = ensure_private_directory(out_dir / "failures")
    path = failures / f"step-{attempted_step:06d}-{incident}.json"
    if not atomic_write_bytes_if_absent(path, encoded, mode=0o600):
        if path.read_bytes() != encoded:
            raise GRPOCheckpointError("recurrent failure receipt publication raced")
    latest = canonical_json_bytes(
        {
            "schema": "aura.grpo_recurrent_failure_pointer.v1",
            "receipt": str(path.relative_to(out_dir)),
            "receipt_sha256": sha256_bytes(encoded),
        }
    )
    atomic_write_bytes(out_dir / "latest_failure.json", latest, mode=0o600)
    return path


def sample_group(model, tokenizer, task, *, size, max_tokens, temperature, seed):
    """K completions for one prompt. Diversity is the mechanism."""
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    prompt = _render(tokenizer, task)
    completions: list[str] = []
    for index in range(size):
        mx.random.seed(seed * 1000 + index)
        pieces: list[str] = []
        for response in stream_generate(
            model, tokenizer, prompt=prompt, max_tokens=max_tokens,
            sampler=make_sampler(temp=temperature, top_p=0.95),
        ):
            pieces.append(response.text)
        completions.append("".join(pieces))
    return prompt, completions


def completion_logprob(model, tokenizer, prompt, completion, *, adapters_on):
    """Log-probability of a completion, with adapters on or off.

    Adapters off gives the reference policy for the KL term at zero extra
    memory -- a second resident copy of a 32B is exactly the kind of thing
    that took this host down.
    """
    import mlx.core as mx

    from core.brain.llm.latent_cortex.recurrence_adapter import (
        recurrence_adapter_scope,
    )

    prompt_ids = tokenizer.encode(prompt)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    if not completion_ids:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if type(eos_token_id) is not int or eos_token_id < 0:
            raise RuntimeError("empty completion has no EOS token for policy credit")
        completion_ids = [eos_token_id]
    full = mx.array([prompt_ids + completion_ids])
    count = len(completion_ids)

    def forward():
        logits = model(full)
        start = full.shape[1] - count - 1
        return sequence_token_logprobs(
            logits[:, start : start + count, :], mx.array([completion_ids])
        )

    if adapters_on:
        with recurrence_adapter_scope(start=None, stop=None):
            return forward()
    return forward()  # no scope => ScopedLoRALinear passes through


def evaluate_heldout(
    model,
    tokenizer,
    tasks,
    *,
    max_tokens,
    envelope,
    adapters_on: bool,
    progress_label: str = "",
    progress_every: int = 4,
):
    """Greedy held-out accuracy by depth with explicit adapter exposure."""
    from mlx_lm import stream_generate

    from core.brain.llm.latent_cortex.recurrence_adapter import (
        recurrence_adapter_scope,
    )

    results = []
    scope = (
        recurrence_adapter_scope(start=None, stop=None)
        if adapters_on
        else nullcontext()
    )
    total = len(tasks)
    correct_so_far = 0
    with scope:
        for index, task in enumerate(tasks, start=1):
            pieces: list[str] = []
            for response in stream_generate(
                model,
                tokenizer,
                prompt=_render(tokenizer, task),
                max_tokens=max_tokens,
            ):
                pieces.append(response.text)
            verdict = task.grade("".join(pieces))
            correct = bool(verdict["correct"])
            results.append((task, correct))
            correct_so_far += int(correct)
            if envelope is not None:
                envelope.reclaim(force=True)
            if progress_label and (
                index == total
                or index == 1
                or index % max(1, progress_every) == 0
            ):
                print(
                    f"[{progress_label}] {index}/{total} "
                    f"running={correct_so_far / max(1, index):.3f}",
                    flush=True,
                )
    report = scaling_report(results)
    report["adapters_on"] = adapters_on
    report["execution_mode"] = "standard"
    return report


def evaluate_recurrent_heldout(
    model,
    tokenizer,
    tasks,
    *,
    spec,
    max_tokens: int,
    envelope,
    adapters_on: bool,
    seed: int,
    progress_label: str = "",
    progress_every: int = 4,
):
    """Greedy held-out accuracy through the exact fixed RLC graph."""

    import mlx.core as mx

    from core.brain.llm.latent_cortex.engine import LatentCortexEngine
    from core.brain.llm.latent_cortex.recurrence_adapter import (
        recurrence_adapter_disabled,
    )
    from core.learning.recurrent_grpo import (
        RecurrentSamplingConfig,
        cortex_config_from_execution_spec,
    )

    config = cortex_config_from_execution_spec(
        spec,
        sampling=RecurrentSamplingConfig(max_tokens=max_tokens),
    )
    config.decode_temperature = 0.0
    engine = LatentCortexEngine(
        model,
        tokenizer=tokenizer,
        config=config,
        schedule_library=None,
    )
    results = []
    receipts: list[dict[str, Any]] = []
    total = len(tasks)
    correct_so_far = 0
    for index, task in enumerate(tasks):
        mx.random.seed(_stable_seed(seed, "recurrent_eval", index, task.task_id))
        scope = nullcontext() if adapters_on else recurrence_adapter_disabled()
        with scope:
            result = engine.reason(
                token_ids=_task_prompt_tokens(tokenizer, task),
                decode_max_tokens=max_tokens,
                decode_sentence_grace_tokens=0,
            )
        if not result.ok:
            raise RuntimeError(
                f"recurrent held-out task {task.task_id} failed: {result.reason}"
            )
        verdict = task.grade(result.text)
        correct = bool(verdict["correct"])
        results.append((task, correct))
        correct_so_far += int(correct)
        receipts.append(
            {
                "task_id": task.task_id,
                "selected_branch": result.receipt.selected_branch,
                "steps_taken": result.receipt.steps_taken,
                "decode_termination": result.receipt.decode_termination,
                "output_tokens": len(result.tokens),
                "correct": bool(verdict["correct"]),
            }
        )
        if envelope is not None:
            envelope.reclaim(force=True)
        completed = index + 1
        if progress_label and (
            completed == total
            or completed == 1
            or completed % max(1, progress_every) == 0
        ):
            print(
                f"[{progress_label}] {completed}/{total} "
                f"running={correct_so_far / max(1, completed):.3f}",
                flush=True,
            )
    report = scaling_report(results)
    report["adapters_on"] = adapters_on
    report["execution_mode"] = "recurrent"
    report["execution_spec_sha256"] = spec.sha256
    report["episode_receipts"] = receipts
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--adapter-id",
        default="recurrent-grpo",
        help="stable adapter identity recorded in recurrent campaign bundles",
    )
    parser.add_argument(
        "--execution-mode",
        choices=EXECUTION_MODES,
        default="standard",
    )
    parser.add_argument(
        "--execution-spec",
        help="strict RLCExecutionSpec JSON required by recurrent mode",
    )
    parser.add_argument(
        "--task-source",
        choices=TASK_SOURCES,
        default="verifiable",
        help="immutable programmatic training registry; frontier tasks stay evaluation-only",
    )
    parser.add_argument("--domains", default="arithmetic_chain,program_trace,constraint_order")
    parser.add_argument("--depths", default="2,4,8")
    parser.add_argument("--train-per-cell", type=int, default=32)
    parser.add_argument("--holdout-per-cell", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--kl-coefficient", type=float, default=0.04)
    parser.add_argument("--format-credit", type=float, default=0.0)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-targets", default="o_proj,v_proj,q_proj")
    parser.add_argument("--lora-layers", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--checkpoint-keep", type=int, default=3)
    parser.add_argument("--calibrate", action="store_true",
                        help="measure pass rates before training to skip dead cells")
    parser.add_argument("--calibrate-samples", type=int, default=2)
    parser.add_argument("--calibrate-group", type=int, default=4,
                        help="completions per calibration probe (cheaper than the train group)")
    parser.add_argument("--calibrate-tokens", type=int, default=0,
                        help="max tokens per calibration probe; 0 = match --max-tokens "
                             "(reasoning tasks need room to finish, or the probe "
                             "underestimates pass rate and mislabels learnable cells)")
    parser.add_argument("--calibrate-minutes", type=float, default=15.0,
                        help="wall-clock cap on the whole calibration phase")
    parser.add_argument("--cot", action="store_true",
                        help="invite step-by-step reasoning before the answer")
    parser.add_argument("--max-minutes", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--memory-fraction", type=float, default=0.55)
    args = parser.parse_args()

    for name in (
        "train_per_cell",
        "holdout_per_cell",
        "group_size",
        "max_tokens",
        "lora_rank",
        "lora_layers",
        "max_steps",
        "eval_every",
        "checkpoint_every",
        "checkpoint_keep",
        "calibrate_samples",
        "calibrate_group",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_minutes <= 0.0 or args.calibrate_minutes <= 0.0:
        parser.error("time budgets must be positive")
    if not 0.0 < args.memory_fraction <= 0.9:
        parser.error("--memory-fraction must be inside (0, 0.9]")
    try:
        _calibration_token_budget(args.max_tokens, args.calibrate_tokens)
    except ValueError as exc:
        parser.error(str(exc))
    if not 0.0 < args.temperature <= 2.0:
        parser.error("--temperature must be inside (0, 2]")
    if not 0.0 < args.learning_rate <= 1.0:
        parser.error("--learning-rate must be inside (0, 1]")
    if not 0.0 <= args.format_credit <= 0.2:
        parser.error("--format-credit must be inside [0, 0.2]")
    if _ADAPTER_ID_RE.fullmatch(args.adapter_id) is None:
        parser.error("--adapter-id must be a stable identifier")
    try:
        execution_spec = _load_execution_spec(
            args.execution_mode, args.execution_spec
        )
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    if args.execution_mode == "recurrent" and args.temperature != 1.0:
        parser.error("recurrent mode requires --temperature 1")

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    global _COT_PREAMBLE
    if args.cot:
        _COT_PREAMBLE = (
            "Work through this step by step, then end with your answer on "
            "its own line."
        )

    config = GRPOConfig(
        group_size=args.group_size, kl_coefficient=args.kl_coefficient
    )
    recurrent_config = None
    if execution_spec is not None:
        from core.learning.recurrent_grpo import RecurrentGRPOConfig

        recurrent_config = RecurrentGRPOConfig(
            kl_coefficient=args.kl_coefficient,
            advantage_clip=config.advantage_clip,
        )
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    depths = [int(d) for d in args.depths.split(",") if d.strip()]
    if not domains or not depths or any(depth <= 0 for depth in depths):
        parser.error("--domains and positive --depths are required")
    train_tasks, holdout, task_source_path = _build_task_split(
        task_source=args.task_source,
        domains=domains,
        depths=depths,
        train_per_cell=args.train_per_cell,
        holdout_per_cell=args.holdout_per_cell,
        seed=args.seed,
    )
    print(
        f"[tasks] {len(train_tasks)} train / {len(holdout)} held-out "
        f"from {args.task_source} (disjoint prompts and identities verified)",
        flush=True,
    )

    out_dir = ensure_private_directory(Path(args.out_dir).expanduser().resolve())
    dataset = _dataset_payload(train_tasks, holdout, seed=args.seed)
    dataset_bytes = canonical_json_bytes(dataset)
    dataset_sha256 = sha256_bytes(dataset_bytes)

    from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (
        full_weight_checkpoint_identity,
        model_behavior_bundle_identity,
        personality_bundle_identity,
        runtime_environment_identity,
    )

    model_path = str(Path(args.model).expanduser().resolve(strict=True))
    source_files = {
        "trainer": Path(__file__),
        "grpo": REPO_ROOT / "core/learning/grpo.py",
        "curriculum": REPO_ROOT / "core/learning/adaptive_curriculum.py",
        "tasks": task_source_path,
        "checkpoint": REPO_ROOT / "core/learning/grpo_training_state.py",
        "adapter": (
            REPO_ROOT
            / "core/brain/llm/latent_cortex/recurrence_adapter.py"
        ),
    }
    if execution_spec is not None:
        source_files.update(
            {
                "recurrent_grpo": REPO_ROOT / "core/learning/recurrent_grpo.py",
                "recurrent_objective": (
                    REPO_ROOT / "core/learning/recurrence_native_objective_v2.py"
                ),
                "execution_spec": (
                    REPO_ROOT
                    / "core/brain/llm/latent_cortex/execution_spec.py"
                ),
                "latent_engine": (
                    REPO_ROOT / "core/brain/llm/latent_cortex/engine.py"
                ),
                "recurrence": (
                    REPO_ROOT / "core/brain/llm/latent_cortex/recurrence.py"
                ),
            }
        )
    sources = {role: _source_binding(path) for role, path in source_files.items()}
    base_identity = full_weight_checkpoint_identity(model_path)
    behavior_identity = model_behavior_bundle_identity(model_path)
    personality_identity = personality_bundle_identity(None)
    runtime_identity = runtime_environment_identity()
    protocol = {
        "schema": GRPO_PROTOCOL_SCHEMA,
        "adapter_id": args.adapter_id,
        "model_path": model_path,
        "base_checkpoint": base_identity,
        "model_behavior": behavior_identity,
        "personality_adapter": personality_identity,
        "runtime": runtime_identity,
        "dataset_sha256": dataset_sha256,
        "sources": sources,
        "training": {
            "execution_mode": args.execution_mode,
            "execution_spec": (
                execution_spec.to_dict() if execution_spec is not None else None
            ),
            "execution_spec_sha256": (
                execution_spec.sha256 if execution_spec is not None else None
            ),
            "domains": domains,
            "depths": depths,
            "train_per_cell": args.train_per_cell,
            "holdout_per_cell": args.holdout_per_cell,
            "group_size": args.group_size,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "kl_coefficient": args.kl_coefficient,
            "format_credit": args.format_credit,
            "lora_rank": args.lora_rank,
            "lora_targets": args.lora_targets,
            "lora_layers": args.lora_layers,
            "learning_rate": args.learning_rate,
            "max_steps": args.max_steps,
            "eval_every": args.eval_every,
            "checkpoint_every": args.checkpoint_every,
            "calibrate": args.calibrate,
            "calibrate_samples": args.calibrate_samples,
            "calibrate_group": args.calibrate_group,
            "calibrate_tokens": args.calibrate_tokens,
            "calibrate_minutes": args.calibrate_minutes,
            "cot": args.cot,
            "seed": args.seed,
            "memory_fraction": args.memory_fraction,
            "rng_strategy": RNG_STRATEGY,
        },
    }
    protocol_bytes = canonical_json_bytes(protocol)
    protocol_sha256 = sha256_bytes(protocol_bytes)
    with interprocess_file_lock(out_dir / ".checkpoint.lock"):
        _publish_immutable_bytes(
            out_dir / "dataset_manifest.json", dataset_bytes, role="dataset manifest"
        )
        _publish_immutable_bytes(
            out_dir / "training_protocol.json", protocol_bytes, role="training protocol"
        )
        source_snapshot_dir = ensure_private_directory(out_dir / "source_snapshots")
        for role, source_path in source_files.items():
            source_bytes = source_path.resolve(strict=True).read_bytes()
            binding = sources[role]
            if (
                len(source_bytes) != binding["size_bytes"]
                or sha256_bytes(source_bytes) != binding["sha256"]
            ):
                raise RuntimeError(f"training source changed while snapshotting: {role}")
            _publish_immutable_bytes(
                source_snapshot_dir / f"{role}.py",
                source_bytes,
                role=f"{role} source snapshot",
            )

    started_wall = time.time()
    started_monotonic = time.monotonic()
    deadline = started_monotonic + args.max_minutes * 60.0

    from mlx_lm import load
    from core.runtime.model_lane_control import standalone_model_lane

    with standalone_model_lane(
        owner_id=f"train-grpo:{Path(args.out_dir).name}",
        model_path=args.model,
        purpose="training",
        preemptible=False,
        metadata={"tool": "train_grpo", "operator_launched": True},
    ), mlx_memory_envelope(fraction=args.memory_fraction) as envelope:
        print(f"[envelope] {envelope.to_receipt()}", flush=True)
        model, tokenizer = load(args.model)
        model.freeze()

        from core.brain.llm.latent_cortex.recurrence_adapter import (
            ScopedLoRALinear,
            recurrence_adapter_scope,
        )

        total_layers = len(model.model.layers)
        targets = tuple(t.strip() for t in args.lora_targets.split(","))
        attached = 0
        if execution_spec is None:
            adapted_indices = range(
                max(0, total_layers - args.lora_layers), total_layers
            )
        else:
            prelude_end = max(
                1, int(total_layers * execution_spec.prelude_frac)
            )
            coda_start = min(
                total_layers - 1,
                total_layers
                - max(1, int(total_layers * execution_spec.coda_frac)),
            )
            adapted_indices = range(
                max(prelude_end, coda_start - args.lora_layers),
                coda_start,
            )
        for index in adapted_indices:
            layer = model.model.layers[index]
            for parent_name in ("self_attn", "mlp"):
                parent = getattr(layer, parent_name, None)
                if parent is None:
                    continue
                for target in targets:
                    projection = getattr(parent, target, None)
                    if projection is not None and not isinstance(
                        projection, ScopedLoRALinear
                    ):
                        setattr(
                            parent, target,
                            ScopedLoRALinear.from_base(projection, r=args.lora_rank),
                        )
                        attached += 1
        if not attached:
            raise RuntimeError("no projections adapted; check --lora-targets")
        print(f"[wiring] {attached} projections adapted", flush=True)

        from mlx.utils import tree_flatten

        from core.learning.adaptive_curriculum import (
            AdaptiveCurriculum,
            warm_start_pass_rates,
        )

        optimizer = optim.Adam(learning_rate=args.learning_rate)
        optimizer.init(model.trainable_parameters())
        telemetry = GRPOTelemetry()
        history: list[dict[str, Any]] = []
        step_receipts: list[dict[str, Any]] = []
        baseline_eval: dict[str, Any] | None = None
        calibration: dict[str, Any] | None = None
        step = 0
        optimizer_updates = 0
        last_step_kind = "initial"
        prior_elapsed_s = 0.0
        invocation_count = 1

        by_cell: dict[tuple[str, int], list[Any]] = {}
        for task in train_tasks:
            by_cell.setdefault((task.domain, task.depth), []).append(task)
        curriculum = AdaptiveCurriculum.over(
            sorted({domain for domain, _depth in by_cell}),
            sorted({depth for _domain, depth in by_cell}),
        )

        expected_adapters = dict(tree_flatten(model.trainable_parameters()))
        if not expected_adapters or any("lora" not in key for key in expected_adapters):
            raise RuntimeError("trainable tree contains non-LoRA parameters")

        resumed = None
        if (out_dir / "latest.json").exists():
            resumed = load_grpo_checkpoint(
                out_dir,
                expected_protocol_sha256=protocol_sha256,
                expected_dataset_sha256=dataset_sha256,
            )
            _assert_exact_adapter_keys(expected_adapters, resumed.adapter_tensors)
            model.load_weights(list(resumed.adapter_tensors.items()), strict=False)
            optimizer.state = resumed.optimizer_state
            optimizer.init(model.trainable_parameters())
            state = resumed.state
            step = int(state["step"])
            optimizer_updates = int(state["optimizer_updates"])
            last_step_kind = str(state["last_step_kind"])
            curriculum = AdaptiveCurriculum.from_state(state["curriculum"])
            telemetry = GRPOTelemetry.from_state(state["telemetry"])
            history = list(state["history"])
            raw_step_receipts = state.get("step_receipts", [])
            if not isinstance(raw_step_receipts, list) or any(
                not isinstance(entry, dict) for entry in raw_step_receipts
            ):
                raise GRPOCheckpointError("checkpoint step receipts are invalid")
            step_receipts = list(raw_step_receipts)
            if state.get("execution_mode", "standard") != args.execution_mode:
                raise GRPOCheckpointError("checkpoint execution mode differs")
            if state.get("execution_spec_sha256") != (
                execution_spec.sha256 if execution_spec is not None else None
            ):
                raise GRPOCheckpointError("checkpoint execution spec differs")
            if execution_spec is not None and len(step_receipts) != step:
                raise GRPOCheckpointError(
                    "recurrent checkpoint does not receipt every committed step"
                )
            if execution_spec is None and step_receipts:
                raise GRPOCheckpointError(
                    "standard checkpoint contains recurrent step receipts"
                )
            baseline_eval = state["baseline_eval"]
            calibration = state["calibration"]
            prior_elapsed_s = float(state["elapsed_training_s"])
            invocation_count = int(state["invocation_count"]) + 1
            print(
                f"[resume] exact step={step} optimizer_updates={optimizer_updates} "
                f"checkpoint={resumed.checkpoint_dir.name}",
                flush=True,
            )
        elif (out_dir / "checkpoints" / "checkpoint_manifest.json").exists():
            raise GRPOCheckpointError(
                "legacy GRPO checkpoint lacks optimizer/protocol state; use a fresh "
                "output directory instead of claiming exact resume"
            )

        def elapsed_training_s() -> float:
            return prior_elapsed_s + (time.monotonic() - started_monotonic)

        def adapter_tensors() -> dict[str, Any]:
            tensors = dict(tree_flatten(model.trainable_parameters()))
            _assert_exact_adapter_keys(expected_adapters, tensors)
            return tensors

        last_durable_step = step

        def checkpoint_now() -> Path:
            nonlocal last_durable_step
            optimizer_tensors = dict(tree_flatten(optimizer.state))
            if not optimizer_tensors:
                raise GRPOCheckpointError("optimizer state is empty")
            path = save_grpo_checkpoint(
                out_dir,
                adapter_tensors=adapter_tensors(),
                optimizer_tensors=optimizer_tensors,
                state={
                    "protocol_sha256": protocol_sha256,
                    "dataset_sha256": dataset_sha256,
                    "step": step,
                    "curriculum": curriculum.state(),
                    "telemetry": telemetry.state(),
                    "history": history,
                    "step_receipts": step_receipts,
                    "baseline_eval": baseline_eval,
                    "calibration": calibration,
                    "elapsed_training_s": elapsed_training_s(),
                    "invocation_count": invocation_count,
                    "rng_strategy": RNG_STRATEGY,
                    "optimizer_updates": optimizer_updates,
                    "last_step_kind": last_step_kind,
                    "last_step_committed": True,
                    "execution_mode": args.execution_mode,
                    "execution_spec_sha256": (
                        execution_spec.sha256
                        if execution_spec is not None
                        else None
                    ),
                },
                keep=args.checkpoint_keep,
            )
            last_durable_step = step
            return path

        if resumed is None:
            if execution_spec is None:
                baseline_eval = evaluate_heldout(
                    model,
                    tokenizer,
                    holdout,
                    max_tokens=args.max_tokens,
                    envelope=envelope,
                    adapters_on=False,
                    progress_label="baseline-standard",
                )
                baseline_role = "frozen_pretraining_baseline"
            else:
                baseline_eval = evaluate_recurrent_heldout(
                    model,
                    tokenizer,
                    holdout,
                    spec=execution_spec,
                    max_tokens=args.max_tokens,
                    envelope=envelope,
                    adapters_on=False,
                    seed=_stable_seed(args.seed, "baseline"),
                    progress_label="baseline-recurrent",
                )
                baseline_role = "frozen_base_recurrent_baseline"
            baseline_eval["step"] = 0
            baseline_eval["role"] = baseline_role
            print(
                f"[baseline 0] overall={baseline_eval['overall']:.3f} "
                f"by_depth={baseline_eval['accuracy_by_depth']}",
                flush=True,
            )

        training_allowed = True
        if args.calibrate and resumed is None:
            cal_group = min(config.group_size, args.calibrate_group)
            cal_tokens = _calibration_token_budget(
                args.max_tokens, args.calibrate_tokens
            )
            cal_deadline = time.monotonic() + args.calibrate_minutes * 60.0
            cells_sorted = sorted(by_cell)
            probe_counts: dict[tuple[str, int], int] = {}
            probes: list[dict[str, Any]] = []
            print(
                f"[calibrate] {len(cells_sorted)} cells x {cal_group} completions "
                f"x {cal_tokens} tokens, cap {args.calibrate_minutes}m",
                flush=True,
            )

            def _measure(family: str, difficulty: int) -> float | None:
                key = (family, difficulty)
                pool = by_cell.get(key)
                probe_index = probe_counts.get(key, 0)
                probe_counts[key] = probe_index + 1
                if not pool or time.monotonic() >= cal_deadline:
                    return None
                decision_seed = _stable_seed(
                    args.seed, "calibration", family, difficulty, probe_index
                )
                probe = pool[decision_seed % len(pool)]
                if execution_spec is None:
                    with recurrence_adapter_scope(start=None, stop=None):
                        _, completions = sample_group(
                            model,
                            tokenizer,
                            probe,
                            size=cal_group,
                            max_tokens=cal_tokens,
                            temperature=args.temperature,
                            seed=decision_seed,
                        )
                else:
                    _, _samples, completions = sample_recurrent_group(
                        model,
                        tokenizer,
                        probe,
                        spec=execution_spec,
                        size=cal_group,
                        max_tokens=cal_tokens,
                        seed=decision_seed,
                    )
                rate = sum(
                    int(bool(probe.grade(completion)["correct"]))
                    for completion in completions
                ) / len(completions)
                probes.append(
                    {
                        "family": family,
                        "difficulty": difficulty,
                        "probe_index": probe_index,
                        "task_id": probe.task_id,
                        "seed": decision_seed,
                        "pass_rate": round(rate, 6),
                    }
                )
                print(
                    f"[calibrate] {family}@{difficulty} pass={rate:.2f} "
                    f"({elapsed_training_s() / 60.0:.1f}m)",
                    flush=True,
                )
                return rate

            curriculum = warm_start_pass_rates(
                sorted({domain for domain, _depth in by_cell}),
                sorted({depth for _domain, depth in by_cell}),
                _measure,
                samples_per_cell=args.calibrate_samples,
            )
            curriculum_report = curriculum.report()
            expected_probes = len(cells_sorted) * max(2, args.calibrate_samples)
            calibration = {
                **curriculum_report,
                "max_tokens": cal_tokens,
                "group_size": cal_group,
                "probes": probes,
                "expected_probes": expected_probes,
                "partial": len(probes) < expected_probes,
            }
            print(f"[calibrate] {calibration}", flush=True)
            training_allowed = bool(
                calibration.get("learnable") or calibration.get("unexplored")
            )

        # Step zero is durable only after the true frozen baseline and any
        # calibration are complete. A restart cannot silently recompute them
        # under different random process state.
        checkpoint_path = checkpoint_now()

        requested_signal: int | None = None

        def request_stop(signum: int, _frame: Any) -> None:
            nonlocal requested_signal
            if requested_signal is None:
                requested_signal = int(signum)
                print(
                    f"[signal] {signal.Signals(signum).name}; stopping after "
                    "the current committed step",
                    flush=True,
                )

        previous_handlers = {
            signum: signal.getsignal(signum)
            for signum in (signal.SIGINT, signal.SIGTERM)
        }
        for signum in previous_handlers:
            signal.signal(signum, request_stop)

        halt_reason = "no_reachable_frontier" if not training_allowed else "max_steps"
        active_recurrent_step: dict[str, Any] | None = None
        try:
            while training_allowed and step < args.max_steps:
                if requested_signal is not None:
                    halt_reason = "interrupted"
                    break
                if time.monotonic() >= deadline:
                    halt_reason = "wall_clock_budget"
                    break

                step_number = step + 1
                decision_rng = random.Random(
                    _stable_seed(args.seed, "curriculum", step_number)
                )
                cell = curriculum.sample(decision_rng)
                pool = by_cell.get(cell) or train_tasks
                task_rng = random.Random(_stable_seed(args.seed, "task", step_number))
                task = pool[task_rng.randrange(len(pool))]
                sample_seed = _stable_seed(args.seed, "group", step_number, task.task_id)
                recurrent_samples = None
                if execution_spec is not None:
                    active_recurrent_step = {
                        "attempted_step": step_number,
                        "phase": "sampling",
                        "task_id": task.task_id,
                        "sample_seed": sample_seed,
                        "samples": (),
                    }
                if execution_spec is None:
                    with recurrence_adapter_scope(start=None, stop=None):
                        prompt, completions = sample_group(
                            model,
                            tokenizer,
                            task,
                            size=config.group_size,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            seed=sample_seed,
                        )
                else:
                    prompt, recurrent_samples, completions = (
                        sample_recurrent_group(
                            model,
                            tokenizer,
                            task,
                            spec=execution_spec,
                            size=config.group_size,
                            max_tokens=args.max_tokens,
                            seed=sample_seed,
                        )
                    )
                    active_recurrent_step["samples"] = tuple(recurrent_samples)
                    active_recurrent_step["phase"] = "grading"
                rewards = [
                    reward_from_verdict(
                        task.grade(text), format_credit=args.format_credit
                    )
                    for text in completions
                ]
                advantage_report = group_advantages(
                    rewards, clip=config.advantage_clip
                )
                loss_value: float | None = None
                step_kind = "degenerate_group"
                update_receipt: dict[str, Any] | None = None

                if not advantage_report["degenerate"]:
                    if execution_spec is None:
                        reference = [
                            mx.stop_gradient(
                                completion_logprob(
                                    model,
                                    tokenizer,
                                    prompt,
                                    text,
                                    adapters_on=False,
                                )
                            )
                            for text in completions
                        ]

                        def loss_fn(
                            _model,
                            *,
                            _prompt=prompt,
                            _completions=tuple(completions),
                            _advantages=tuple(advantage_report["advantages"]),
                            _reference=tuple(reference),
                        ):
                            policy = [
                                completion_logprob(
                                    _model,
                                    tokenizer,
                                    _prompt,
                                    text,
                                    adapters_on=True,
                                )
                                for text in _completions
                            ]
                            loss, _report = grpo_loss(
                                policy,
                                _advantages,
                                reference_logprobs=_reference,
                                kl_coefficient=config.kl_coefficient,
                            )
                            return loss

                        loss, grads = nn.value_and_grad(model, loss_fn)(model)
                        loss_value = float(loss)
                    else:
                        from core.learning.recurrent_grpo import (
                            exact_adjoint_sampled_group_value_and_grad,
                        )

                        if recurrent_samples is None or recurrent_config is None:
                            raise RuntimeError("recurrent training state is missing")
                        active_recurrent_step["phase"] = "exact_adjoint"
                        recurrent_result = (
                            exact_adjoint_sampled_group_value_and_grad(
                                model,
                                prompt,
                                recurrent_samples,
                                rewards,
                                spec=execution_spec,
                                config=recurrent_config,
                            )
                        )
                        if recurrent_result.gradients is None:
                            raise RuntimeError(
                                "non-degenerate recurrent group has no gradient"
                            )
                        grads = recurrent_result.gradients
                        loss_value = float(
                            recurrent_result.gradient_surrogate_value
                        )
                        update_receipt = recurrent_result.receipt()
                        active_recurrent_step["phase"] = "optimizer_update"
                    optimizer.update(model, grads)
                    mx.eval(model.parameters(), optimizer.state)
                    optimizer_updates += 1
                    step_kind = "optimizer_update"
                    del grads
                    envelope.reclaim(force=True)

                if recurrent_samples is not None:
                    from core.learning.recurrent_grpo import (
                        recurrent_policy_sha256,
                    )

                    step_receipts.append(
                        {
                            "step": step_number,
                            "task_id": task.task_id,
                            "sample_seed": sample_seed,
                            "execution_spec_sha256": execution_spec.sha256,
                            "samples": [
                                sample.receipt() for sample in recurrent_samples
                            ],
                            "rewards": [float(value) for value in rewards],
                            "advantage_report": advantage_report,
                            "step_kind": step_kind,
                            "update": update_receipt,
                            "policy_after_sha256": recurrent_policy_sha256(
                                model, execution_spec
                            ),
                        }
                    )

                # State mutates only after a complete optimizer update or a
                # fully graded degenerate group. The durable step is therefore
                # always replay-safe.
                telemetry.observe(advantage_report)
                curriculum.observe(
                    task.domain,
                    task.depth,
                    advantage_report["mean_reward"],
                    degenerate=advantage_report["degenerate"],
                )
                step = step_number
                last_step_kind = step_kind
                if active_recurrent_step is not None:
                    active_recurrent_step["phase"] = "post_update_evaluation"

                if step % 10 == 0:
                    detail = (
                        f"loss={loss_value:.4f}"
                        if loss_value is not None
                        else "degenerate"
                    )
                    print(
                        f"[step {step}] {detail} "
                        f"mean_r={advantage_report['mean_reward']:.2f} "
                        f"({elapsed_training_s() / 60.0:.1f}m)",
                        flush=True,
                    )

                if step % args.eval_every == 0:
                    if execution_spec is None:
                        report = evaluate_heldout(
                            model,
                            tokenizer,
                            holdout,
                            max_tokens=args.max_tokens,
                            envelope=envelope,
                            adapters_on=True,
                            progress_label=f"eval-standard-{step}",
                        )
                        report_role = "adapter_standard_decode"
                    else:
                        report = evaluate_recurrent_heldout(
                            model,
                            tokenizer,
                            holdout,
                            spec=execution_spec,
                            max_tokens=args.max_tokens,
                            envelope=envelope,
                            adapters_on=True,
                            seed=_stable_seed(args.seed, "eval", step),
                            progress_label=f"eval-recurrent-{step}",
                        )
                        report_role = "adapter_recurrent_decode"
                    report["step"] = step
                    report["role"] = report_role
                    history.append(report)
                    print(
                        f"[eval {step}] overall={report['overall']:.3f} "
                        f"delta={_point_estimate_delta(baseline_eval, report)} "
                        f"by_depth={report['accuracy_by_depth']}",
                        flush=True,
                    )
                if step % args.checkpoint_every == 0:
                    checkpoint_path = checkpoint_now()
                if not curriculum.report()["has_reachable_frontier"]:
                    training_allowed = False
                    halt_reason = "frontier_exhausted"
                    print(
                        "[halt] every measured curriculum cell is saturated "
                        "or hopeless",
                        flush=True,
                    )
                active_recurrent_step = None

            if step >= args.max_steps:
                halt_reason = "max_steps"
            if (
                requested_signal is None
                and training_allowed
                and (not history or history[-1].get("step") != step)
            ):
                if execution_spec is None:
                    report = evaluate_heldout(
                        model,
                        tokenizer,
                        holdout,
                        max_tokens=args.max_tokens,
                        envelope=envelope,
                        adapters_on=True,
                    )
                    report_role = "adapter_standard_decode"
                else:
                    report = evaluate_recurrent_heldout(
                        model,
                        tokenizer,
                        holdout,
                        spec=execution_spec,
                        max_tokens=args.max_tokens,
                        envelope=envelope,
                        adapters_on=True,
                        seed=_stable_seed(args.seed, "eval", step),
                    )
                    report_role = "adapter_recurrent_decode"
                report["step"] = step
                report["role"] = report_role
                history.append(report)
            checkpoint_path = checkpoint_now()
        except Exception as exc:
            if execution_spec is not None:
                context = active_recurrent_step or {
                    "attempted_step": step + 1,
                    "phase": "between_steps",
                    "task_id": None,
                    "sample_seed": None,
                    "samples": (),
                }
                failure_path = _record_recurrent_step_failure(
                    out_dir,
                    protocol_sha256=protocol_sha256,
                    dataset_sha256=dataset_sha256,
                    execution_spec_sha256=execution_spec.sha256,
                    attempted_step=context["attempted_step"],
                    last_durable_step=last_durable_step,
                    phase=context["phase"],
                    task_id=context["task_id"],
                    sample_seed=context["sample_seed"],
                    samples=context["samples"],
                    error=exc,
                )
                print(f"[failure-receipt] {failure_path}", flush=True)
            raise
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)

        adapters = adapter_tensors()
        _publish_adapter_snapshot(out_dir / "grpo_adapters.safetensors", adapters)
        curriculum_report = curriculum.report()
        print(f"[curriculum] {curriculum_report}", flush=True)

    learning_signal = telemetry.verdict(config)
    final = history[-1] if history else None
    delta = _point_estimate_delta(baseline_eval, final)
    completed = halt_reason in {"max_steps", "wall_clock_budget"}
    receipt = {
        "schema": GRPO_TRAIN_SCHEMA,
        "adapter_id": args.adapter_id,
        "protocol_sha256": protocol_sha256,
        "dataset_sha256": dataset_sha256,
        "model": {
            "path": model_path,
            "base_checkpoint": base_identity,
            "behavior": behavior_identity,
        },
        "config": config.to_receipt(),
        "execution_mode": args.execution_mode,
        "execution_spec": (
            execution_spec.to_dict() if execution_spec is not None else None
        ),
        "execution_spec_sha256": (
            execution_spec.sha256 if execution_spec is not None else None
        ),
        "domains": domains,
        "depths": depths,
        "train_tasks": len(train_tasks),
        "holdout_tasks": len(holdout),
        "steps": step,
        "optimizer_updates": optimizer_updates,
        "invocation_count": invocation_count,
        "termination": {
            "reason": halt_reason,
            "completed_budget": completed,
            "signal": requested_signal,
        },
        "learning_signal": learning_signal,
        "curriculum": curriculum_report,
        "calibration": calibration,
        "baseline": baseline_eval,
        "history": history,
        "step_receipts": step_receipts,
        "final": final,
        "adapter_decode_delta": delta,
        "adapter_standard_decode_delta": (
            delta if execution_spec is None else None
        ),
        "adapter_recurrent_decode_delta": (
            delta if execution_spec is not None else None
        ),
        "checkpoint": str(checkpoint_path.relative_to(out_dir)),
        "verdict": {
            "had_signal": bool(learning_signal["learning_signal"]),
            "point_estimate_improved": bool(delta is not None and delta > 0.0),
            "causal_gain_proven": False,
            "causal_gain_blocker": (
                "requires fresh powered base/adapter x standard/RLC factorial gate"
            ),
            "diagnosis": learning_signal["diagnosis"],
        },
        "elapsed_minutes": round((time.time() - started_wall) / 60.0, 2),
    }
    receipt_bytes = canonical_json_bytes(receipt)
    atomic_write_bytes(out_dir / "grpo_receipt.json", receipt_bytes, mode=0o600)
    if execution_spec is not None and halt_reason == "max_steps":
        identity = _publish_recurrent_adapter_bundle(
            out_dir,
            adapter_id=args.adapter_id,
            protocol=protocol,
            protocol_bytes=protocol_bytes,
            dataset_bytes=dataset_bytes,
            receipt=receipt,
            receipt_bytes=receipt_bytes,
            execution_spec=execution_spec,
            source_roles=source_files,
        )
        print(
            "[campaign-adapter] "
            f"identity={identity['composite_identity_sha256']} "
            f"adapter={identity['adapter_sha256']}",
            flush=True,
        )
    print(f"[verdict] {receipt['verdict']}", flush=True)
    print(f"[receipt] {out_dir / 'grpo_receipt.json'}", flush=True)
    if requested_signal is not None:
        return 128 + requested_signal
    if not training_allowed:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
