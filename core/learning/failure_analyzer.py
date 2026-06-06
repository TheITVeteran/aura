"""core/learning/failure_analyzer.py
Analyzes execution logs to identify causal roots of failures.
"""
from typing import Dict, Any


class FailureAnalyzer:
    """Classifies errors to isolate infrastructure bugs from policy violations."""

    def analyze_failure(self, receipt: Dict[str, Any]) -> str:
        error_msg = receipt.get("error", "").lower()
        if "permission" in error_msg:
            return "causal_root:unauthorized_privilege_infringement"
        if "timeout" in error_msg:
            return "causal_root:resource_exhaustion_timeout"
        if "syntax" in error_msg:
            return "causal_root:code_syntax_violation"
            
        return "causal_root:unknown_environmental_deviation"
