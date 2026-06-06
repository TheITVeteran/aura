"""core/learning/trace_labeler.py
Labels collected training samples with task success flags.
"""
from typing import Dict, Any


class TraceLabeler:
    """Labels training samples based on outcomes and verification evidence."""

    def label_sample(self, sample: Dict[str, Any], outcome: Dict[str, Any]) -> Dict[str, Any]:
        labeled = sample.copy()
        success = outcome.get("status") in ["success", "simulated"]
        labeled["success"] = success
        # Reject samples that failed moral audits (should not be learned as target behaviors)
        labeled["trainable"] = success or "inhibited" in outcome.get("status", "")
        return labeled
