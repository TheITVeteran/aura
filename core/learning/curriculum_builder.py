"""core/learning/curriculum_builder.py
Curriculum builder formatting training batches.
"""
from typing import List, Dict, Any


class CurriculumBuilder:
    """Structures curriculum batches for fine-tuning cycles."""

    def build_batches(self, samples: List[Dict[str, Any]], batch_size: int = 4) -> List[List[Dict[str, Any]]]:
        trainable = [s for s in samples if s.get("trainable", True)]
        # Split into batches
        batches = [trainable[i:i + batch_size] for i in range(0, len(trainable), batch_size)]
        return batches
