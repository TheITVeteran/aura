"""core/learning/lora_trainer.py
Coordinates fine-tuning of local models via MLX LoRA scripts.
"""
import os
import logging
import sys
from subprocess import SubprocessError
from typing import Dict, Any

from core.config import get_config
from core.runtime.errors import record_degradation
from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Learning.LoraTrainer")

_LORA_TRAINING_ERRORS = (OSError, RuntimeError, SubprocessError, TimeoutError, TypeError, ValueError)


class LoraTrainer:
    """Invokes fine-tuning adapters locally under memory-aware profiles."""

    def __init__(self):
        self.config = get_config()

    async def train_adapter(self, dataset_path: str, output_path: str) -> Dict[str, Any]:
        """Proposes and executes local mlx_lm.lora commands if resources permit."""
        model_path = self.config.llm.local_cortex_path
        if not model_path or not os.path.exists(model_path):
            return {"status": "skipped", "reason": "Active GGUF/MLX model path not configured"}

        # MLX LoRA script invocation command
        cmd = [
            sys.executable, "-m", "mlx_lm.lora",
            "--model", model_path,
            "--data", dataset_path,
            "--train",
            "--iters", "100",
            "--batch-size", "2",
            "--adapter-file", output_path
        ]

        logger.info("Initiating local model parameter adaptation: %s", " ".join(cmd))
        try:
            res = await get_subprocess_gateway().run_async(
                cmd,
                capture_output=True,
                timeout=600.0,
                offline_tooling=True,
                source="training_tooling:lora_trainer",
            )
            if res.returncode == 0:
                return {
                    "status": "success",
                    "adapter_path": output_path,
                    "stdout": res.stdout[:1000]
                }
            return {"status": "failed", "error": res.stderr}
        except _LORA_TRAINING_ERRORS as e:
            record_degradation("learning.lora_trainer", e)
            logger.error("LoRA fine-tuning invocation failed: %s", e)
            return {"status": "failed", "error": str(e)}
