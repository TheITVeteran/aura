"""core/learning/lora_trainer.py
Coordinates fine-tuning of local models via MLX LoRA scripts.
"""
import subprocess
import os
import logging
from typing import Dict, Any
from core.config import get_config

logger = logging.getLogger("Learning.LoraTrainer")


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
            "python", "-m", "mlx_lm.lora",
            "--model", model_path,
            "--data", dataset_path,
            "--train",
            "--iters", "100",
            "--batch-size", "2",
            "--adapter-file", output_path
        ]

        logger.info("Initiating local model parameter adaptation: %s", " ".join(cmd))
        try:
            # Run fine-tuning as background process using subprocess
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=600.0)
            if res.returncode == 0:
                return {
                    "status": "success",
                    "adapter_path": output_path,
                    "stdout": res.stdout[:1000]
                }
            return {"status": "failed", "error": res.stderr}
        except Exception as e:
            logger.error("LoRA fine-tuning invocation failed: %s", e)
            return {"status": "failed", "error": str(e)}
