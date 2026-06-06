"""core/learning/experience_collector.py
Collects event traces and compiles training datasets for LLM weight adaptation.
"""
import json
from pathlib import Path
from typing import Any, Dict, List

from core.config import get_config
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway

_EXPERIENCE_COLLECTOR_ERRORS = (OSError, RuntimeError, TypeError, ValueError)


class ExperienceCollector:
    """Converts autobiographical life episodes into instruction-fine-tuning format."""

    def __init__(self):
        cfg = get_config()
        self.data_path = Path(cfg.paths.data_dir) / "training_experiences.json"

    def collect_and_save(self, events: List[Dict[str, Any]]) -> int:
        samples = []
        for e in events:
            # Only collect samples where action occurred and outcome is known
            did = e.get("did", {}).get("action")
            if did:
                samples.append({
                    "instruction": f"Perform a {did} action with constraints.",
                    "input": json.dumps(e.get("chose", {}).get("params", {})),
                    "output": json.dumps(e.get("what_happened", {}))
                })

        if samples:
            try:
                get_file_write_gateway().write_text(
                    self.data_path,
                    json.dumps(samples, indent=4, sort_keys=True),
                    source="learning.experience_collector",
                )
            except _EXPERIENCE_COLLECTOR_ERRORS as exc:
                record_degradation("learning.experience_collector", exc)
                return 0
                
        return len(samples)
