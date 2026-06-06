"""core/security/audit_log.py
Durable append-only audit logger for tracking agent executions.
"""
import time
import os
import json
import logging
from core.config import get_config

logger = logging.getLogger("Security.AuditLogger")


class SecurityAuditLogger:
    """Writes security events to an append-only file in the logs directory."""

    def __init__(self):
        self.config = get_config()
        self.log_path = os.path.join(self.config.paths.log_dir, "security_audit.jsonl")

    def log_event(self, action: str, details: Dict[str, Any]) -> None:
        event = {
            "timestamp": time.time(),
            "action": action,
            "details": details
        }
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            logger.error("Failed to append security audit event: %s", e)
