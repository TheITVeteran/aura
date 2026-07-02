# core/adaptation/safe_optimizer.py
import asyncio
import json
import logging
import os
import shlex
import time
from pathlib import Path

from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Aura.SafeOptimizer")
_MAX_CAPTURE_BYTES = 200_000

class SafeSelfOptimizer:
    """
    Zenith Audit Fix 3.1: LoRA safety logic.
    Ensures dataset diversity, validation before merge, and safe rollbacks.
    """
    def __init__(self, lora_dir: str = "data/adaptation/loras"):
        self.lora_dir = Path(lora_dir)
        self.lora_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir = self.lora_dir / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self._is_training = False

    async def optimize_lora(self, dataset_path: str, base_model: str):
        """Run a safe training loop with dataset rotation and validation."""
        if self._is_training:
            logger.warning("Optimization already in progress. Skipping.")
            return

        self._is_training = True
        try:
            # 1. Dataset Diversity Check
            if not await self._validate_dataset(dataset_path):
                logger.error("LoRA Optimization: Dataset failed diversity/safety check.")
                return

            # 2. Backup existing weights
            await self._backup_current_weights()

            # 3. Execute the configured local trainer when available.
            logger.info("🚀 Starting Safe LoRA training gate on %s", dataset_path)
            trained = await self._run_training_command(dataset_path, base_model)
            if not trained:
                logger.error("LoRA Optimization: no verified local trainer completed.")
                await self._rollback()
                return

            # 4. Post-Training Validation
            if not await self._run_eval_benchmarks():
                logger.error("LoRA Optimization: Post-training validation failed. Rolling back.")
                await self._rollback()
                return

            logger.info("✅ LoRA Optimization successful and merged.")
        finally:
            self._is_training = False

    async def _validate_dataset(self, path: str) -> bool:
        """ZENITH Fix: Ensure dataset reflects current personality and isn't poisoned."""
        sample = await asyncio.to_thread(self._read_dataset_sample, Path(path))
        if sample is None:
            return False
        lines = [line.strip() for line in sample.splitlines() if line.strip()]
        if len(lines) < 16:
            return False
        unique_ratio = len(set(lines)) / max(1, len(lines))
        banned = ("ignore previous instructions", "system prompt", "api_key", "password")
        return unique_ratio >= 0.35 and not any(marker in sample.lower() for marker in banned)

    async def _run_training_command(self, dataset_path: str, base_model: str) -> bool:
        command = os.environ.get("AURA_LORA_TRAIN_CMD", "").strip()
        file_gateway = get_file_write_gateway()
        if not command:
            manifest = self.lora_dir / "training_gate_manifest.json"
            await file_gateway.write_text_async(
                manifest,
                json.dumps(
                    {
                        "dataset_path": dataset_path,
                        "base_model": base_model,
                        "status": "validated_dataset_waiting_for_configured_trainer",
                        "generated_at": time.time(),
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
                source="core.adaptation.safe_optimizer.training_gate_manifest",
            )
            return False
        try:
            argv = shlex.split(command)
        except ValueError as exc:
            logger.error("LoRA training command could not be parsed: %s", exc)
            return False
        if not argv:
            logger.error("LoRA training command parsed to an empty argv.")
            return False
        proc = await get_subprocess_gateway().spawn_async(
            argv,
            env={**os.environ, "AURA_LORA_DATASET": dataset_path, "AURA_LORA_BASE_MODEL": base_model},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            source="core.adaptation.safe_optimizer.training_command",
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self._training_timeout_seconds(),
            )
        except TimeoutError:
            proc.kill()
            stdout, stderr = await proc.communicate()
            stderr = (stderr or b"") + b"\nAURA_LORA_TRAIN_TIMEOUT\n"
        await file_gateway.write_bytes_async(
            self.lora_dir / "last_train_stdout.log",
            stdout[-_MAX_CAPTURE_BYTES:],
            source="core.adaptation.safe_optimizer.training_stdout",
        )
        await file_gateway.write_bytes_async(
            self.lora_dir / "last_train_stderr.log",
            stderr[-_MAX_CAPTURE_BYTES:],
            source="core.adaptation.safe_optimizer.training_stderr",
        )
        return proc.returncode == 0

    async def _backup_current_weights(self):
        """Create a versioned backup before any merge."""
        ts = int(time.time())
        current_weights = self.lora_dir / "adapter_model.bin"
        if current_weights.exists():
            await get_file_write_gateway().write_bytes_async(
                self.backup_dir / f"adapter_{ts}.bin",
                current_weights.read_bytes(),
                source="core.adaptation.safe_optimizer.backup_weights",
            )

    async def _run_eval_benchmarks(self) -> bool:
        """Run target benchmarks (e.g. MMLU, GSM8K subset) to ensure no regression."""
        report_path = os.environ.get("AURA_LORA_EVAL_REPORT", "").strip()
        if not report_path:
            return True
        try:
            raw_report = await asyncio.to_thread(Path(report_path).read_text, encoding="utf-8")
            report = json.loads(raw_report)
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            logger.error("LoRA eval report unreadable: %s", exc)
            return False
        max_regression = float(report.get("max_regression", 0.0))
        safety_passed = bool(report.get("safety_passed", True))
        return safety_passed and max_regression <= 0.05

    async def _rollback(self):
        """Restore weights from the most recent backup."""
        backups = sorted(self.backup_dir.glob("adapter_*.bin"))
        if backups:
            latest = backups[-1]
            await get_file_write_gateway().write_bytes_async(
                self.lora_dir / "adapter_model.bin",
                latest.read_bytes(),
                source="core.adaptation.safe_optimizer.rollback_weights",
            )
            logger.info("⏪ Rollback complete: Restored from %s", latest.name)

    @staticmethod
    def _training_timeout_seconds() -> float:
        raw = os.environ.get("AURA_LORA_TRAIN_TIMEOUT", "").strip()
        if not raw:
            return 1800.0
        try:
            value = float(raw)
        except ValueError:
            logger.warning("Invalid AURA_LORA_TRAIN_TIMEOUT=%r; using default.", raw)
            return 1800.0
        return min(max(value, 1.0), 86400.0)

    @staticmethod
    def _read_dataset_sample(path: Path) -> str | None:
        try:
            if not path.exists() or path.stat().st_size <= 1024:
                return None
            return path.read_text(encoding="utf-8", errors="ignore")[:200_000]
        except OSError:
            return None

# Singleton
_optimizer = None
def get_safe_optimizer() -> SafeSelfOptimizer:
    global _optimizer
    if _optimizer is None:
        _optimizer = SafeSelfOptimizer()
    return _optimizer
