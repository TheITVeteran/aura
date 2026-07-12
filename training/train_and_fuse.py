#!/usr/bin/env python3
"""End-to-end LoRA training + fuse + auto-pickup pipeline.

What this script does, in one run:

1. Optionally rebuild the training dataset (build_dataset_v3) so the
   personality + architecture corpus is fresh.
2. Run the LoRA fine-tune (mlx_lm.lora) with the existing
   training/finetune_lora.py hyperparameters.
3. Fuse the resulting adapter into the base model with mlx_lm.fuse,
   producing a new versioned directory under training/fused-model/.
4. Write training/fused-model/active.json — a small manifest that Aura's
   model_registry reads on boot to pick up the newest fused model
   automatically. No .env edit required.
5. Verify the new model loads, then atomically swap the manifest.

After this script finishes, restarting Aura will use the new weights.
The previous fused model directory is kept (under a versioned name) so
you can roll back by editing active.json or pointing AURA_LLM__MLX_MODEL_PATH.

Usage:
    python training/train_and_fuse.py
    python training/train_and_fuse.py --skip-dataset      # reuse existing data
    python training/train_and_fuse.py --skip-train        # only fuse + publish
    python training/train_and_fuse.py --tag mythos-v1     # name this run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError:  # pragma: no cover - production requirements include psutil.
    psutil = None  # type: ignore[assignment]

TRAINING_DIR = Path(__file__).parent
REPO_DIR = TRAINING_DIR.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from core.runtime.model_lane_control import (  # noqa: E402
    LaneClaim,
    estimate_model_job_footprint_gb,
)
from core.runtime.subprocess_gateway import get_subprocess_gateway  # noqa: E402

DATA_DIR = TRAINING_DIR / "data"
ADAPTER_DIR = TRAINING_DIR / "adapters" / "aura-personality"
CRSM_DELTA_DATA_DIR = DATA_DIR / "crsm_delta"
CRSM_DELTA_MANIFEST = DATA_DIR / "crsm_delta_manifest.json"
FUSED_BASE_DIR = TRAINING_DIR / "fused-model"
ACTIVE_MANIFEST = FUSED_BASE_DIR / "active.json"
CRSM_DATASET = REPO_DIR / "data" / "synthetic_training" / "lora_dataset.jsonl"
CRSM_INTEGRATION_MANIFEST = DATA_DIR / "crsm_integration_manifest.json"

DEFAULT_BASE_MODEL = REPO_DIR / "models" / "Qwen2.5-32B-Instruct-4bit"
TRAINING_COMMAND_TIMEOUT_S = float(os.environ.get("AURA_TRAINING_COMMAND_TIMEOUT_S", "86400"))
_GIB = 1024**3
_LIVE_AURA_CMD_MARKERS = (
    "aura_main.py",
    "interface/server.py",
    "core/brain/llm/mlx_worker.py",
    "tools/live_boot_proof.py",
    "tools/visible_journal_demo_proof.py",
)


def _run(
    cmd: list[str],
    *,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
    model_job: bool = False,
    model_lane_claim: LaneClaim | None = None,
    source: str = "training_tooling:train_and_fuse",
) -> int:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    gateway = get_subprocess_gateway()
    run_kwargs = {
        "cwd": REPO_DIR,
        "env": env,
        "timeout": timeout if timeout is not None else TRAINING_COMMAND_TIMEOUT_S,
        "capture_output": False,
        "offline_tooling": True,
        "source": source,
    }
    if model_job or model_lane_claim is not None:
        result = gateway.run_model_blocking(
            cmd,
            **run_kwargs,
            model_lane_claim=model_lane_claim,
        )
    else:
        result = gateway.run(cmd, **run_kwargs)
    return result.returncode


def build_dataset() -> None:
    builder = TRAINING_DIR / "build_dataset_v3.py"
    if not builder.exists():
        print(f"  Dataset builder not found at {builder}; skipping.")
        return
    rc = _run([sys.executable, str(builder)], source="training_tooling:build_dataset")
    if rc != 0:
        sys.exit(f"Dataset build failed (exit {rc}).")


def train_lora(*, base_model: Path, resume: bool = False) -> None:
    finetune = TRAINING_DIR / "finetune_lora.py"
    if not finetune.exists():
        sys.exit(f"finetune_lora.py not found at {finetune}.")
    # Pass base_model through env so finetune_lora's find_base_model() picks
    # the right size — same script supports 32B, 72B, 14B, 7B, etc.
    env = os.environ.copy()
    env["AURA_LORA_BASE_MODEL"] = str(base_model)
    cmd = [sys.executable, str(finetune)]
    if resume:
        cmd.append("--resume")
    print(f"  AURA_LORA_BASE_MODEL={base_model}", flush=True)
    rc = _run(
        cmd,
        env=env,
        source="training_tooling:train_lora",
    )
    if rc != 0:
        sys.exit(f"LoRA fine-tune failed (exit {rc}).")


def _env_int(name: str, default: int, *, minimum: int = 0, maximum: int | None = None) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _jsonl_file_stats(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    lines = 0
    with path.open("rb") as fh:
        for raw in fh:
            lines += 1
            digest.update(raw)
    stat = path.stat()
    return {
        "path": str(path),
        "lines": lines,
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "sha256": digest.hexdigest(),
    }


def _example_key(example: dict[str, Any]) -> str:
    try:
        messages = example.get("messages") if isinstance(example, dict) else None
        if not isinstance(messages, list):
            return ""
        parts: list[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "").strip().lower()
            content = " ".join(str(message.get("content") or "").split()).lower()
            parts.append(f"{role}:{content}")
        return "\n".join(parts)
    except (AttributeError, TypeError, ValueError):
        return ""


def _reservoir_sample_jsonl(
    path: Path,
    *,
    count: int,
    rng: random.Random,
    exclude_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Sample retention examples without loading the full corpus into memory."""
    if count <= 0 or not path.exists():
        return []
    exclude_keys = exclude_keys or set()
    sample: list[dict[str, Any]] = []
    seen = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                example = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(example, dict) or not isinstance(example.get("messages"), list):
                continue
            key = _example_key(example)
            if not key or key in exclude_keys:
                continue
            seen += 1
            if len(sample) < count:
                sample.append(example)
                continue
            idx = rng.randrange(seen)
            if idx < count:
                sample[idx] = example
    return sample


def _write_jsonl(path: Path, examples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for example in examples:
            fh.write(json.dumps(example, ensure_ascii=False) + "\n")


def _selection_digest(examples: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for example in examples:
        digest.update(json.dumps(example, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_crsm_delta_dataset(
    *,
    output_dir: Path = CRSM_DELTA_DATA_DIR,
    max_crsm_examples: int | None = None,
    retention_examples: int | None = None,
    valid_fraction: float = 0.1,
    seed: int = 20260628,
) -> dict[str, Any]:
    """Create a bounded, provenance-rich dataset for CRSM incremental LoRA.

    This is deliberately not a marker shortcut. It extracts the eligible CRSM
    captures through the same production gate as the full corpus, adds a small
    retention sample from the existing training data, writes a standalone
    MLX-compatible dataset, and records hashes proving exactly what trained.
    """
    from training.build_dataset_v3 import build_crsm_experience_examples

    max_crsm_examples = (
        _env_int("AURA_CRSM_DELTA_MAX_EXAMPLES", 600, minimum=1, maximum=5000)
        if max_crsm_examples is None
        else max(1, int(max_crsm_examples))
    )
    retention_examples = (
        _env_int("AURA_CRSM_DELTA_RETENTION_EXAMPLES", 512, minimum=0, maximum=5000)
        if retention_examples is None
        else max(0, int(retention_examples))
    )
    rng = random.Random(seed)

    crsm_examples, crsm_manifest = build_crsm_experience_examples(
        CRSM_DATASET,
        max_examples=max_crsm_examples,
    )
    if not crsm_examples:
        sys.exit("CRSM delta dataset build failed: no eligible CRSM captures after safety filtering.")

    crsm_keys = {_example_key(example) for example in crsm_examples}
    retention_pool = _reservoir_sample_jsonl(
        DATA_DIR / "train.jsonl",
        count=retention_examples,
        rng=rng,
        exclude_keys=crsm_keys,
    )
    if len(retention_pool) < retention_examples:
        retention_pool.extend(
            _reservoir_sample_jsonl(
                DATA_DIR / "valid.jsonl",
                count=retention_examples - len(retention_pool),
                rng=rng,
                exclude_keys=crsm_keys | {_example_key(example) for example in retention_pool},
            )
        )

    selected = [*crsm_examples, *retention_pool]
    rng.shuffle(selected)
    valid_count = max(1, min(len(selected) - 1, int(round(len(selected) * valid_fraction))))
    valid = selected[:valid_count]
    train = selected[valid_count:]

    if not train or not valid:
        sys.exit("CRSM delta dataset build failed: train/valid split would be empty.")

    train_path = output_dir / "train.jsonl"
    valid_path = output_dir / "valid.jsonl"
    _write_jsonl(train_path, train)
    _write_jsonl(valid_path, valid)

    manifest = {
        **crsm_manifest,
        "builder": "training/train_and_fuse.py:build_crsm_delta_dataset",
        "delta_mode": True,
        "seed": seed,
        "retention_examples": len(retention_pool),
        "selection_sha256": _selection_digest(selected),
        "output": {
            "builder": "training/train_and_fuse.py",
            "total_examples": len(selected),
            "crsm_examples": len(crsm_examples),
            "retention_examples": len(retention_pool),
            "train": _jsonl_file_stats(train_path),
            "valid": _jsonl_file_stats(valid_path),
        },
    }
    CRSM_DELTA_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    CRSM_DELTA_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print("\nBuilt CRSM delta dataset:")
    print(json.dumps(manifest["output"], indent=2, sort_keys=True))
    return manifest


def _latest_adapter_file(adapter_dir: Path = ADAPTER_DIR) -> Path | None:
    primary = adapter_dir / "adapters.safetensors"
    if primary.exists():
        return primary
    checkpoints = sorted(adapter_dir.glob("[0-9]*_adapters.safetensors"))
    return checkpoints[-1] if checkpoints else None


def build_crsm_delta_train_command(
    *,
    base_model: Path,
    data_dir: Path,
    adapter_dir: Path,
    resume_adapter_file: Path,
    iters: int,
    max_seq_length: int,
    lora_config_path: Path,
    save_every: int | None = None,
    steps_per_eval: int | None = None,
    steps_per_report: int = 10,
) -> list[str]:
    save_every = save_every or max(25, min(100, iters))
    steps_per_eval = steps_per_eval or max(25, min(100, iters))
    return [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        str(base_model),
        "--train",
        "--data",
        str(data_dir),
        "--adapter-path",
        str(adapter_dir),
        "--resume-adapter-file",
        str(resume_adapter_file),
        "--iters",
        str(iters),
        "--num-layers",
        "-1",
        "--batch-size",
        "1",
        "--learning-rate",
        "5e-6",
        "--save-every",
        str(save_every),
        "--steps-per-eval",
        str(steps_per_eval),
        "--steps-per-report",
        str(steps_per_report),
        "--max-seq-length",
        str(max_seq_length),
        "--grad-checkpoint",
        "-c",
        str(lora_config_path),
    ]


def train_crsm_delta_lora(
    *,
    base_model: Path,
    data_dir: Path = CRSM_DELTA_DATA_DIR,
    adapter_dir: Path | None = None,
    iters: int | None = None,
    max_seq_length: int | None = None,
) -> Path:
    """Run a real bounded LoRA update from current CRSM captures."""
    resume_adapter = _latest_adapter_file()
    if resume_adapter is None:
        sys.exit(f"CRSM delta training failed: no source adapter found under {ADAPTER_DIR}.")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    adapter_dir = adapter_dir or (ADAPTER_DIR.parent / f"aura-personality-crsm-delta-{timestamp}")
    adapter_dir.mkdir(parents=True, exist_ok=True)

    lora_config_path = ADAPTER_DIR / "lora_config.yaml"
    if not lora_config_path.exists():
        lora_config_path = ADAPTER_DIR / "lora_config.json"
    if not lora_config_path.exists():
        sys.exit(f"CRSM delta training failed: missing LoRA config under {ADAPTER_DIR}.")

    iters = (
        _env_int("AURA_CRSM_DELTA_ITERS", 600, minimum=25, maximum=5000)
        if iters is None
        else max(1, int(iters))
    )
    max_seq_length = (
        _env_int("AURA_CRSM_DELTA_MAX_SEQ_LENGTH", 2048, minimum=512, maximum=4096)
        if max_seq_length is None
        else max(128, int(max_seq_length))
    )
    cmd = build_crsm_delta_train_command(
        base_model=base_model,
        data_dir=data_dir,
        adapter_dir=adapter_dir,
        resume_adapter_file=resume_adapter,
        iters=iters,
        max_seq_length=max_seq_length,
        lora_config_path=lora_config_path,
    )
    rc = _run(
        cmd,
        model_job=True,
        source="training_tooling:crsm_delta_lora",
    )
    if rc != 0:
        sys.exit(f"CRSM delta LoRA fine-tune failed (exit {rc}).")
    if not (adapter_dir / "adapters.safetensors").exists():
        sys.exit(f"CRSM delta LoRA fine-tune ended without {adapter_dir / 'adapters.safetensors'}.")
    return adapter_dir


def _model_size_tag(base_model: Path) -> str:
    """Derive a short size tag from the base-model directory name ('32B',
    '72B', '14B', '7B'). Falls back to 'model' when no size token matches."""
    name = base_model.name.lower()
    for size in ("72b", "32b", "14b", "8b", "7b", "3b", "1.5b", "0.5b"):
        if size in name:
            return size.upper().replace(".", "_")
    return "model"


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _default_training_headroom_gb(size_tag: str, *, skip_train: bool) -> tuple[float, float]:
    """Return minimum available RAM and free disk for train/fuse safety."""
    if size_tag == "72B":
        return (44.0, 220.0) if not skip_train else (32.0, 160.0)
    if size_tag == "32B":
        return (28.0, 110.0) if not skip_train else (20.0, 90.0)
    if size_tag in {"14B", "8B", "7B"}:
        return (18.0, 60.0) if not skip_train else (12.0, 40.0)
    return (12.0, 40.0) if not skip_train else (8.0, 25.0)


def _live_aura_processes() -> list[dict[str, Any]]:
    if psutil is None:
        return []
    current_pid = os.getpid()
    found: list[dict[str, Any]] = []
    try:
        iterator = psutil.process_iter(["pid", "name", "cmdline"])
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return found
    for proc in iterator:
        try:
            info = getattr(proc, "info", {}) or {}
            pid = int(info.get("pid") or proc.pid)
            if pid == current_pid:
                continue
            cmdline = info.get("cmdline") or []
            if isinstance(cmdline, str):
                cmdline = [cmdline]
            cmd = " ".join(str(part) for part in cmdline)
            if any(marker in cmd for marker in _LIVE_AURA_CMD_MARKERS):
                found.append({"pid": pid, "name": info.get("name"), "cmdline": cmd[:500]})
        except (psutil.Error, AttributeError, RuntimeError, TypeError, ValueError):
            continue
    return found


def training_preflight(*, base_model: Path, skip_train: bool, crsm_delta: bool = False) -> dict[str, Any]:
    size_tag = _model_size_tag(base_model)
    default_min_available_gb, default_min_free_disk_gb = _default_training_headroom_gb(
        size_tag,
        skip_train=skip_train,
    )
    min_available_gb = _env_float("AURA_TRAINING_MIN_AVAILABLE_GB", default_min_available_gb)
    min_free_disk_gb = _env_float("AURA_TRAINING_MIN_FREE_DISK_GB", default_min_free_disk_gb)
    max_memory_percent = _env_float("AURA_TRAINING_MAX_MEMORY_PERCENT", 82.0)

    blockers: list[str] = []
    memory: dict[str, Any] = {"available_gb": None, "percent": None}
    if psutil is None:
        blockers.append("psutil_unavailable")
    else:
        try:
            vm = psutil.virtual_memory()
            available_gb = float(getattr(vm, "available", 0) or 0) / _GIB
            percent = float(getattr(vm, "percent", 100.0) or 100.0)
            memory = {"available_gb": round(available_gb, 2), "percent": round(percent, 1)}
            if available_gb < min_available_gb:
                blockers.append(
                    f"available_memory:{available_gb:.1f}GB < required {min_available_gb:.1f}GB"
                )
            if percent > max_memory_percent:
                blockers.append(f"memory_pressure:{percent:.1f}% > {max_memory_percent:.1f}%")
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            blockers.append(f"memory_probe_failed:{type(exc).__name__}")

    disk_path = FUSED_BASE_DIR if FUSED_BASE_DIR.exists() else FUSED_BASE_DIR.parent
    disk_usage = shutil.disk_usage(disk_path)
    free_disk_gb = disk_usage.free / _GIB
    if free_disk_gb < min_free_disk_gb:
        blockers.append(f"free_disk:{free_disk_gb:.1f}GB < required {min_free_disk_gb:.1f}GB")

    live_processes = [] if _env_flag("AURA_TRAINING_ALLOW_LIVE_AURA") else _live_aura_processes()
    if live_processes:
        blockers.append(f"live_aura_processes:{len(live_processes)}")

    return {
        "passed": not blockers,
        "mode": (
            "crsm_delta_train_fuse_publish"
            if crsm_delta and not skip_train
            else "fuse_publish"
            if skip_train
            else "train_fuse_publish"
        ),
        "base_model": str(base_model),
        "size": size_tag,
        "requirements": {
            "min_available_gb": min_available_gb,
            "max_memory_percent": max_memory_percent,
            "min_free_disk_gb": min_free_disk_gb,
            "block_live_aura": not _env_flag("AURA_TRAINING_ALLOW_LIVE_AURA"),
        },
        "memory": memory,
        "disk": {"path": str(disk_path), "free_gb": round(free_disk_gb, 2)},
        "live_aura_processes": live_processes,
        "blockers": blockers,
    }


def enforce_training_preflight(*, base_model: Path, skip_train: bool, crsm_delta: bool = False) -> dict[str, Any]:
    report = training_preflight(base_model=base_model, skip_train=skip_train, crsm_delta=crsm_delta)
    print("\nTraining preflight:")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        sys.exit("Training preflight failed: " + "; ".join(report["blockers"]))
    return report


def fuse_adapter(*, base_model: Path, tag: str, adapter_dir: Path = ADAPTER_DIR) -> Path:
    """mlx_lm fuse base_model + adapter → versioned fused-model dir."""
    if not (adapter_dir / "adapters.safetensors").exists():
        sys.exit(
            f"No adapter found at {adapter_dir}/adapters.safetensors — "
            "run training first or pass --skip-train only after a previous train."
        )

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    size_tag = _model_size_tag(base_model)
    fused_name = (
        f"Aura-{size_tag}-{tag}-{timestamp}" if tag
        else f"Aura-{size_tag}-{timestamp}"
    )
    fused_path = FUSED_BASE_DIR / fused_name
    fused_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nFusing → {fused_path}")
    rc = _run(
        [
            sys.executable,
            "-m",
            "mlx_lm",
            "fuse",
            "--model",
            str(base_model),
            "--adapter-path",
            str(adapter_dir),
            "--save-path",
            str(fused_path),
        ],
        timeout=1800,
        model_job=True,
        source="training_tooling:fuse_adapter",
    )
    if rc != 0:
        sys.exit(f"Fuse failed (exit {rc}).")
    if not fused_path.exists() or not any(fused_path.iterdir()):
        sys.exit(f"Fuse claimed success but {fused_path} is empty.")
    return fused_path


def verify_load(fused_path: Path) -> None:
    """Smoke-test: tokenize one prompt to confirm the fused model is loadable."""
    print(f"\nVerifying fused model loads: {fused_path}")
    code = (
        "import sys\n"
        "from core.runtime.model_lane_control import standalone_model_lane\n"
        "from mlx_lm import load\n"
        "model_path = sys.argv[1]\n"
        "with standalone_model_lane(\n"
        "    owner_id='verify-fused-model',\n"
        "    model_path=model_path,\n"
        "    purpose='benchmark',\n"
        "    preemptible=True,\n"
        "):\n"
        "    model, tok = load(model_path)\n"
        "    ids = tok.encode('Hello')\n"
        "    print(f'OK: tokenized {len(ids)} tokens, vocab_size={tok.vocab_size}')\n"
    )
    timeout = 600.0
    claim = LaneClaim(
        owner_id=f"training:verify-fused:{os.getpid()}:{time.time_ns()}",
        model_path=str(fused_path),
        request_gb=estimate_model_job_footprint_gb(
            str(fused_path),
            purpose="benchmark",
        ),
        purpose="benchmark",
        priority=50,
        preemptible=True,
        reservation_ttl_s=timeout + 30.0,
        owner_lease_ttl_s=timeout + 30.0,
        metadata={"source": "training_tooling:verify_fused_model"},
    )
    rc = _run(
        [sys.executable, "-c", code, str(fused_path)],
        timeout=timeout,
        model_lane_claim=claim,
        source="training_tooling:verify_fused_model",
    )
    if rc != 0:
        sys.exit(f"Verification load failed (exit {rc}).")


def publish_manifest(fused_path: Path, *, tag: str, base_model: Path) -> None:
    """Atomically write active.json so Aura's next boot uses the new model.

    The manifest now includes the base-model size so downstream RAM-aware
    routing (model_registry, inference_gate) can branch on it without
    re-parsing the directory name."""
    FUSED_BASE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "active_model_path": str(fused_path),
        "fused_at": int(time.time()),
        "tag": tag or "",
        "size": _model_size_tag(base_model),
        "base_model": str(base_model),
        "schema_version": 2,
    }
    tmp = ACTIVE_MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(tmp, ACTIVE_MANIFEST)
    print(f"\nWrote active manifest: {ACTIVE_MANIFEST}")
    print(json.dumps(manifest, indent=2))
    print(
        "\nNext Aura boot will use this fused model automatically. "
        "If AURA_LLM__MLX_MODEL_PATH is set in .env it still wins — "
        "remove or update that line to let the manifest drive."
    )


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}


def mark_crsm_loop_consumed_after_training(
    fused_path: Path,
    *,
    manifest_path: Path | None = None,
    source: str = "training.train_and_fuse",
) -> None:
    """Close the CRSM→LoRA monitor only after real train/fuse evidence exists."""
    manifest_path = manifest_path or CRSM_INTEGRATION_MANIFEST
    if not CRSM_DATASET.exists():
        return
    manifest = _read_json(manifest_path)
    if not manifest:
        print(
            "\nCRSM loop not marked consumed: missing "
            f"{manifest_path}. Rebuild the dataset before training."
        )
        return

    try:
        dataset_stat = CRSM_DATASET.stat()
    except OSError:
        return

    source_lines = int(manifest.get("source_lines", 0) or 0)
    source_mtime = float(manifest.get("source_mtime", 0.0) or 0.0)
    accepted = int(manifest.get("accepted", 0) or 0)
    rejected = max(0, source_lines - accepted)

    if source_lines <= 0:
        print("\nCRSM loop not marked consumed: integration manifest saw no source captures.")
        return
    if source_mtime + 1.0 < float(dataset_stat.st_mtime):
        print("\nCRSM loop not marked consumed: capture dataset changed after integration manifest.")
        return

    from core.consciousness.crsm_loop_monitor import get_crsm_loop_monitor

    get_crsm_loop_monitor().mark_dataset_consumed(
        model_path=str(fused_path),
        lines_consumed=source_lines,
        accepted_lines=accepted,
        rejected_lines=rejected,
        manifest_path=str(manifest_path),
        source=source,
    )
    print(
        "\nMarked CRSM captures handled after successful train/fuse: "
        f"{accepted} trained, {rejected} retired."
    )


def record_crsm_delta_training_state(
    *,
    adapter_dir: Path,
    fused_path: Path,
    manifest_path: Path = CRSM_DELTA_MANIFEST,
    iters: int | None = None,
    max_seq_length: int | None = None,
) -> None:
    """Persist operator-visible evidence for the bounded CRSM delta run."""
    state_path = ADAPTER_DIR / "training_state.json"
    try:
        state = _read_json(state_path)
        manifest = _read_json(manifest_path)
        output = dict(manifest.get("output") or {})
        payload = {
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "adapter_path": str(adapter_dir),
            "fused_model_path": str(fused_path),
            "manifest_path": str(manifest_path),
            "source_lines": int(manifest.get("source_lines", 0) or 0),
            "accepted": int(manifest.get("accepted", 0) or 0),
            "rejected": max(0, int(manifest.get("source_lines", 0) or 0) - int(manifest.get("accepted", 0) or 0)),
            "retention_examples": int(output.get("retention_examples", 0) or 0),
            "train_sha256": (dict(output.get("train") or {})).get("sha256"),
            "valid_sha256": (dict(output.get("valid") or {})).get("sha256"),
            "iters": iters,
            "max_seq_length": max_seq_length,
            "status": "fused_published_consumed",
        }
        state["crsm_delta"] = payload
        tmp = state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, state_path)
    except (OSError, TypeError, ValueError) as exc:
        print(f"\nWarning: failed to record CRSM delta training state: {type(exc).__name__}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--mark-crsm-consumed",
        action="store_true",
        help=(
            "Mark CRSM captures consumed after fuse/publish even when --skip-train "
            "is used. Intended for run_unattended resume_training.py, where the "
            "training step happened in a separate process before this fuse call."
        ),
    )
    parser.add_argument(
        "--crsm-delta",
        action="store_true",
        help=(
            "Run a bounded real LoRA update over the current CRSM captures plus "
            "retention examples, fuse it, publish it, and then mark CRSM consumed."
        ),
    )
    parser.add_argument(
        "--crsm-delta-iters",
        type=int,
        default=None,
        help="Override AURA_CRSM_DELTA_ITERS for the bounded CRSM LoRA update.",
    )
    parser.add_argument(
        "--crsm-delta-max-examples",
        type=int,
        default=None,
        help="Maximum eligible CRSM examples to include in the bounded delta dataset.",
    )
    parser.add_argument(
        "--crsm-delta-retention-examples",
        type=int,
        default=None,
        help="Retention examples sampled from the existing corpus for the bounded delta dataset.",
    )
    parser.add_argument(
        "--crsm-delta-max-seq-length",
        type=int,
        default=None,
        help="Override AURA_CRSM_DELTA_MAX_SEQ_LENGTH for the bounded CRSM LoRA update.",
    )
    parser.add_argument(
        "--base-model",
        default=os.environ.get("AURA_LORA_BASE_MODEL", str(DEFAULT_BASE_MODEL)),
    )
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    base_model = Path(args.base_model)
    if not base_model.exists():
        sys.exit(f"Base model not found: {base_model}")

    print("=" * 60)
    print("  AURA TRAIN → FUSE → PUBLISH PIPELINE")
    print("=" * 60)
    print(f"  base_model: {base_model}")
    print(f"  adapter:    {ADAPTER_DIR}")
    print(f"  output dir: {FUSED_BASE_DIR}")
    print(f"  tag:        {args.tag or '(none)'}")
    print("=" * 60)

    if args.crsm_delta and args.skip_train:
        sys.exit("--crsm-delta requires a real training step; refusing --skip-train shortcut.")

    if not _env_flag("AURA_TRAINING_BYPASS_PREFLIGHT"):
        enforce_training_preflight(
            base_model=base_model,
            skip_train=args.skip_train,
            crsm_delta=args.crsm_delta,
        )
    else:
        print("\nTraining preflight bypassed by AURA_TRAINING_BYPASS_PREFLIGHT=1.")
    if args.preflight_only:
        print("\nPreflight-only mode complete; no dataset, training, fuse, or publish actions executed.")
        return

    adapter_dir = ADAPTER_DIR
    crsm_marker_manifest = CRSM_INTEGRATION_MANIFEST
    crsm_marker_source = "training.train_and_fuse"
    crsm_delta_adapter_dir: Path | None = None

    if args.crsm_delta:
        build_crsm_delta_dataset(
            max_crsm_examples=args.crsm_delta_max_examples,
            retention_examples=args.crsm_delta_retention_examples,
        )
        adapter_dir = train_crsm_delta_lora(
            base_model=base_model,
            data_dir=CRSM_DELTA_DATA_DIR,
            iters=args.crsm_delta_iters,
            max_seq_length=args.crsm_delta_max_seq_length,
        )
        crsm_delta_adapter_dir = adapter_dir
        crsm_marker_manifest = CRSM_DELTA_MANIFEST
        crsm_marker_source = "training.train_and_fuse.crsm_delta"
    elif not args.skip_dataset:
        build_dataset()
    if not args.crsm_delta and not args.skip_train:
        train_lora(base_model=base_model, resume=args.resume)
    fused_path = fuse_adapter(base_model=base_model, tag=args.tag, adapter_dir=adapter_dir)
    verify_load(fused_path)
    publish_manifest(fused_path, tag=args.tag, base_model=base_model)
    if args.crsm_delta or not args.skip_train or args.mark_crsm_consumed:
        mark_crsm_loop_consumed_after_training(
            fused_path,
            manifest_path=crsm_marker_manifest,
            source=crsm_marker_source,
        )
    if args.crsm_delta and crsm_delta_adapter_dir is not None:
        record_crsm_delta_training_state(
            adapter_dir=crsm_delta_adapter_dir,
            fused_path=fused_path,
            manifest_path=CRSM_DELTA_MANIFEST,
            iters=args.crsm_delta_iters or _env_int("AURA_CRSM_DELTA_ITERS", 600, minimum=25, maximum=5000),
            max_seq_length=(
                args.crsm_delta_max_seq_length
                or _env_int("AURA_CRSM_DELTA_MAX_SEQ_LENGTH", 2048, minimum=512, maximum=4096)
            ),
        )


if __name__ == "__main__":
    main()
