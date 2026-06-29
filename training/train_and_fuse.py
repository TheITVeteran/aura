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
import json
import os
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

from core.runtime.subprocess_gateway import get_subprocess_gateway  # noqa: E402

DATA_DIR = TRAINING_DIR / "data"
ADAPTER_DIR = TRAINING_DIR / "adapters" / "aura-personality"
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
    source: str = "training_tooling:train_and_fuse",
) -> int:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    result = get_subprocess_gateway().run(
        cmd,
        cwd=REPO_DIR,
        timeout=timeout if timeout is not None else TRAINING_COMMAND_TIMEOUT_S,
        capture_output=False,
        offline_tooling=True,
        source=source,
    )
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
    print(f"\n$ {' '.join(cmd)}  (AURA_LORA_BASE_MODEL={base_model})", flush=True)
    result = get_subprocess_gateway().run(
        cmd,
        cwd=REPO_DIR,
        env=env,
        timeout=TRAINING_COMMAND_TIMEOUT_S,
        capture_output=False,
        offline_tooling=True,
        source="training_tooling:train_lora",
    )
    if result.returncode != 0:
        sys.exit(f"LoRA fine-tune failed (exit {result.returncode}).")


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


def training_preflight(*, base_model: Path, skip_train: bool) -> dict[str, Any]:
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
        "mode": "fuse_publish" if skip_train else "train_fuse_publish",
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


def enforce_training_preflight(*, base_model: Path, skip_train: bool) -> dict[str, Any]:
    report = training_preflight(base_model=base_model, skip_train=skip_train)
    print("\nTraining preflight:")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        sys.exit("Training preflight failed: " + "; ".join(report["blockers"]))
    return report


def fuse_adapter(*, base_model: Path, tag: str) -> Path:
    """mlx_lm fuse base_model + adapter → versioned fused-model dir."""
    if not (ADAPTER_DIR / "adapters.safetensors").exists():
        sys.exit(
            f"No adapter found at {ADAPTER_DIR}/adapters.safetensors — "
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
            str(ADAPTER_DIR),
            "--save-path",
            str(fused_path),
        ],
        timeout=1800,
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
        "from mlx_lm import load\n"
        f"model, tok = load({str(fused_path)!r})\n"
        "ids = tok.encode('Hello')\n"
        "print(f'OK: tokenized {len(ids)} tokens, vocab_size={tok.vocab_size}')\n"
    )
    rc = _run([sys.executable, "-c", code], timeout=600, source="training_tooling:verify_fused_model")
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


def mark_crsm_loop_consumed_after_training(fused_path: Path) -> None:
    """Close the CRSM→LoRA monitor only after real train/fuse evidence exists."""
    if not CRSM_DATASET.exists():
        return
    manifest = _read_json(CRSM_INTEGRATION_MANIFEST)
    if not manifest:
        print(
            "\nCRSM loop not marked consumed: missing "
            f"{CRSM_INTEGRATION_MANIFEST}. Rebuild the dataset before training."
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
        manifest_path=str(CRSM_INTEGRATION_MANIFEST),
        source="training.train_and_fuse",
    )
    print(
        "\nMarked CRSM captures handled after successful train/fuse: "
        f"{accepted} trained, {rejected} retired."
    )


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

    if not _env_flag("AURA_TRAINING_BYPASS_PREFLIGHT"):
        enforce_training_preflight(base_model=base_model, skip_train=args.skip_train)
    else:
        print("\nTraining preflight bypassed by AURA_TRAINING_BYPASS_PREFLIGHT=1.")
    if args.preflight_only:
        print("\nPreflight-only mode complete; no dataset, training, fuse, or publish actions executed.")
        return

    if not args.skip_dataset:
        build_dataset()
    if not args.skip_train:
        train_lora(base_model=base_model, resume=args.resume)
    fused_path = fuse_adapter(base_model=base_model, tag=args.tag)
    verify_load(fused_path)
    publish_manifest(fused_path, tag=args.tag, base_model=base_model)
    if not args.skip_train or args.mark_crsm_consumed:
        mark_crsm_loop_consumed_after_training(fused_path)


if __name__ == "__main__":
    main()
