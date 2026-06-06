"""core/learning/experience_collector.py
Collects event traces and compiles training datasets for LLM weight adaptation.
"""
from typing import Dict, List, Any
import json
import os
from core.config import get_config


class ExperienceCollector:
    """Converts autobiographical life episodes into instruction-fine-tuning format."""

    def __init__(self):
        cfg = get_config()
        self.data_path = os.path.join(cfg.paths.data_dir, "training_experiences.json")

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
            with open(self.data_path, "w", encoding="utf-8") as f:
                json.dump(samples, f, indent=4)
                
        return len(samples)
