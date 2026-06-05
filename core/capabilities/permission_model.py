"""core/capabilities/permission_model.py — Per-Action Risk Classification
=========================================================================
Every action Aura takes is classified by risk level BEFORE execution:

    LOW:     open app, create local file, read screen — auto-approved
    MEDIUM:  change wallpaper, download image, write to cloud — logged
    HIGH:    send email, delete files, post publicly — requires user confirmation
    BLOCKED: rm -rf, credential changes, money operations — always refused

The PermissionRiskModel integrates with UnifiedWill to gate actions.
It maintains per-modality toggles (camera, mic, screen, files, network)
and supports preset profiles (DemoSafe, FullAutonomy, etc.).
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Set, Tuple

from core.container import ServiceContainer
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.PermissionModel")


# ---------------------------------------------------------------------------
# Risk levels
# ---------------------------------------------------------------------------

class RiskLevel(IntEnum):
    """Risk classification for actions."""
    LOW = 0         # Auto-approved, logged
    MEDIUM = 1      # Auto-approved if trusted mode, logged prominently
    HIGH = 2        # Requires explicit user confirmation
    BLOCKED = 3     # Always refused, no override


# ---------------------------------------------------------------------------
# Permission decisions
# ---------------------------------------------------------------------------

@dataclass
class PermissionDecision:
    """The result of a permission check."""
    action: str
    target: str
    risk_level: RiskLevel
    approved: bool
    reason: str
    requires_confirmation: bool = False
    modality: str = ""              # which modality this touches
    receipt_id: str = ""
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.receipt_id:
            payload = f"{self.timestamp}|{self.action}|{self.target}|{self.approved}"
            self.receipt_id = hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Modality toggles
# ---------------------------------------------------------------------------

@dataclass
class ModalityPermissions:
    """Per-modality permission toggles."""
    camera: bool = False
    microphone: bool = True         # On by default for voice
    screen_recording: bool = True   # On by default for screen perception
    file_write: bool = True         # On by default (sandboxed)
    file_delete: bool = False       # Off by default
    network_read: bool = True       # On for web search
    network_write: bool = False     # Off for posting/sending
    clipboard: bool = True          # On for paste operations
    app_control: bool = True        # On for launching/focusing apps
    system_settings: bool = True    # On for wallpaper, volume, etc.
    email: bool = False             # Off by default
    cloud_write: bool = False       # Off by default (Google Docs, etc.)


# ---------------------------------------------------------------------------
# Risk classification rules
# ---------------------------------------------------------------------------

# Action patterns → risk levels
_RISK_RULES: List[Tuple[str, RiskLevel, str]] = [
    # BLOCKED — never allowed
    (r"rm\s+-rf\s+/", RiskLevel.BLOCKED, "Recursive root delete"),
    (r"sudo\s+rm", RiskLevel.BLOCKED, "Privileged delete"),
    (r"mkfs|format\s+disk", RiskLevel.BLOCKED, "Disk format"),
    (r"shutdown|reboot|halt", RiskLevel.BLOCKED, "System shutdown"),
    (r"credential.*change|password.*change", RiskLevel.BLOCKED, "Credential modification"),
    (r"send.*money|transfer.*funds|payment", RiskLevel.BLOCKED, "Financial operation"),
    (r"delete.*keychain|security\s+delete", RiskLevel.BLOCKED, "Keychain delete"),
    (r"launchctl\s+unload", RiskLevel.BLOCKED, "Service unload"),

    # HIGH — requires user confirmation
    (r"send.*email|compose.*email", RiskLevel.HIGH, "Email send"),
    (r"post.*public|tweet|publish", RiskLevel.HIGH, "Public post"),
    (r"delete\s+(?:file|folder|directory)", RiskLevel.HIGH, "File deletion"),
    (r"share.*document|share.*file", RiskLevel.HIGH, "Document sharing"),
    (r"upload.*file|upload.*data", RiskLevel.HIGH, "File upload"),
    (r"install.*package|pip\s+install|brew\s+install", RiskLevel.HIGH, "Package install"),
    (r"modify.*system|change.*permission", RiskLevel.HIGH, "System modification"),
    (r"git\s+push|git\s+force", RiskLevel.HIGH, "Remote push"),

    # MEDIUM — auto-approved but logged
    (r"change.*wallpaper|set.*wallpaper|set.*background", RiskLevel.MEDIUM, "Wallpaper change"),
    (r"download.*image|download.*file", RiskLevel.MEDIUM, "File download"),
    (r"write.*google\s*doc|create.*google\s*doc", RiskLevel.MEDIUM, "Cloud document"),
    (r"run.*command|execute.*command|shell.*command", RiskLevel.MEDIUM, "Command execution"),
    (r"create.*pdf|export.*pdf", RiskLevel.MEDIUM, "PDF creation"),
    (r"change.*volume|set.*volume", RiskLevel.MEDIUM, "Volume change"),
    (r"change.*appearance|dark\s*mode|light\s*mode", RiskLevel.MEDIUM, "Appearance change"),
    (r"open.*browser.*tab", RiskLevel.MEDIUM, "Browser tab open"),

    # LOW — auto-approved
    (r"open.*app|launch.*app|activate.*app", RiskLevel.LOW, "App launch"),
    (r"create.*file|create.*folder|create.*note", RiskLevel.LOW, "Local file creation"),
    (r"read.*screen|get.*screen|screenshot", RiskLevel.LOW, "Screen read"),
    (r"read.*file|list.*file|check.*file", RiskLevel.LOW, "File read"),
    (r"get.*window|get.*app|frontmost", RiskLevel.LOW, "App query"),
    (r"search.*web|web.*search", RiskLevel.LOW, "Web search"),
    (r"open.*url|browse.*url", RiskLevel.LOW, "URL open"),
    (r"type.*text|paste.*text|keystroke", RiskLevel.LOW, "Text input"),
    (r"click|scroll|hotkey", RiskLevel.LOW, "UI interaction"),
]


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

class PermissionRiskModel:
    """Per-action risk classification and approval gating.

    Every action must pass through this model before execution.
    The model classifies risk, checks modality permissions, and returns
    a PermissionDecision with receipt.

    In trusted/demo mode, MEDIUM actions are auto-approved.
    HIGH actions always require explicit user confirmation.
    BLOCKED actions are always refused.
    """

    def __init__(self) -> None:
        self.modality = ModalityPermissions()
        self._trusted_mode: bool = False        # Auto-approve MEDIUM
        self._demo_safe_mode: bool = False       # Block HIGH, restrict MEDIUM
        self._decision_history: list[PermissionDecision] = []
        self._max_history = 500
        self._escalation_window_s = 60.0
        self._escalation_threshold = 3          # 3+ MEDIUM in 60s → escalate
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        ServiceContainer.register_instance("permission_model", self, required=False)
        self._started = True
        logger.info(
            "PermissionRiskModel ONLINE (trusted=%s, demo_safe=%s)",
            self._trusted_mode, self._demo_safe_mode,
        )

    def set_trusted_mode(self, enabled: bool) -> None:
        """Enable/disable trusted mode (auto-approve MEDIUM)."""
        self._trusted_mode = enabled
        logger.info("Permission model: trusted_mode=%s", enabled)

    def set_demo_safe_mode(self, enabled: bool) -> None:
        """Enable/disable demo-safe mode (block HIGH, restrict MEDIUM)."""
        self._demo_safe_mode = enabled
        if enabled:
            self.modality.network_write = False
            self.modality.email = False
            self.modality.cloud_write = False
            self.modality.file_delete = False
        logger.info("Permission model: demo_safe_mode=%s", enabled)

    # ------------------------------------------------------------------
    # Risk classification
    # ------------------------------------------------------------------

    def classify_risk(self, action: str, target: str = "") -> Tuple[RiskLevel, str]:
        """Classify the risk level of an action.

        Returns (risk_level, reason).
        """
        combined = f"{action} {target}".lower()

        # Check against pattern rules (ordered by severity: blocked first)
        for pattern, level, reason in _RISK_RULES:
            if re.search(pattern, combined, re.IGNORECASE):
                return level, reason

        # Default: LOW for unknown actions
        return RiskLevel.LOW, "No matching risk pattern"

    # ------------------------------------------------------------------
    # Permission check
    # ------------------------------------------------------------------

    def check_permission(
        self,
        action: str,
        target: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> PermissionDecision:
        """Check if an action is permitted.

        This is the main entry point. All capability calls should check
        permission before executing.
        """
        context = context or {}
        risk_level, reason = self.classify_risk(action, target)
        modality = self._detect_modality(action, target)

        # Check modality permissions
        modality_allowed = self._check_modality(modality)
        if not modality_allowed:
            decision = PermissionDecision(
                action=action, target=target[:200],
                risk_level=risk_level, approved=False,
                reason=f"Modality '{modality}' is disabled",
                modality=modality,
            )
            self._record_decision(decision)
            return decision

        # Apply risk level decision
        approved = False
        requires_confirmation = False

        if risk_level == RiskLevel.BLOCKED:
            approved = False
            reason = f"BLOCKED: {reason}"

        elif risk_level == RiskLevel.HIGH:
            if self._demo_safe_mode:
                approved = False
                reason = f"BLOCKED in demo-safe mode: {reason}"
            else:
                # Check if user already pre-approved this class of action
                pre_approved = context.get("user_explicitly_authorized", False)
                if pre_approved:
                    approved = True
                    reason = f"HIGH risk, user pre-approved: {reason}"
                else:
                    approved = False
                    requires_confirmation = True
                    reason = f"Requires user confirmation: {reason}"

        elif risk_level == RiskLevel.MEDIUM:
            if self._demo_safe_mode:
                # In demo-safe mode, MEDIUM requires confirmation
                pre_approved = context.get("user_explicitly_authorized", False)
                if pre_approved:
                    approved = True
                    reason = f"MEDIUM risk, user pre-approved in demo mode: {reason}"
                else:
                    approved = False
                    requires_confirmation = True
                    reason = f"Requires confirmation in demo-safe mode: {reason}"
            elif self._trusted_mode:
                approved = True
                reason = f"Auto-approved in trusted mode: {reason}"
            else:
                # Check escalation: too many MEDIUM actions in short window
                if self._should_escalate():
                    approved = False
                    requires_confirmation = True
                    reason = f"Escalated due to rapid MEDIUM actions: {reason}"
                else:
                    approved = True
                    reason = f"MEDIUM risk, auto-approved: {reason}"

        elif risk_level == RiskLevel.LOW:
            approved = True
            reason = f"LOW risk: {reason}"

        decision = PermissionDecision(
            action=action, target=target[:200],
            risk_level=risk_level, approved=approved,
            reason=reason, requires_confirmation=requires_confirmation,
            modality=modality,
        )
        self._record_decision(decision)
        return decision

    def _detect_modality(self, action: str, target: str) -> str:
        """Detect which modality an action touches."""
        combined = f"{action} {target}".lower()
        if any(w in combined for w in ("camera", "photo", "vision")):
            return "camera"
        if any(w in combined for w in ("mic", "microphone", "listen", "voice", "speech")):
            return "microphone"
        if any(w in combined for w in ("screen", "screenshot", "ocr", "capture")):
            return "screen_recording"
        if any(w in combined for w in ("delete", "remove", "trash")):
            return "file_delete"
        if any(w in combined for w in ("write", "create", "save", "export", "move")):
            return "file_write"
        if any(w in combined for w in ("download", "fetch", "request", "search")):
            return "network_read"
        if any(w in combined for w in ("post", "send", "upload", "share", "publish")):
            return "network_write"
        if any(w in combined for w in ("email", "mail")):
            return "email"
        if any(w in combined for w in ("google doc", "cloud doc", "drive")):
            return "cloud_write"
        if any(w in combined for w in ("clipboard", "paste", "copy")):
            return "clipboard"
        if any(w in combined for w in ("wallpaper", "volume", "appearance", "dark mode", "setting")):
            return "system_settings"
        if any(w in combined for w in ("launch", "open", "activate", "focus", "app")):
            return "app_control"
        return "app_control"  # default

    def _check_modality(self, modality: str) -> bool:
        """Check if a modality is enabled."""
        return getattr(self.modality, modality, True)

    def _should_escalate(self) -> bool:
        """Check if too many MEDIUM actions happened recently."""
        now = time.time()
        recent_medium = sum(
            1 for d in self._decision_history[-20:]
            if d.risk_level == RiskLevel.MEDIUM
            and d.approved
            and (now - d.timestamp) < self._escalation_window_s
        )
        return recent_medium >= self._escalation_threshold

    def _record_decision(self, decision: PermissionDecision) -> None:
        """Record a decision for audit and escalation tracking."""
        self._decision_history.append(decision)
        if len(self._decision_history) > self._max_history:
            self._decision_history = self._decision_history[-self._max_history:]

        # Log to LifeTrace for significant decisions
        if decision.risk_level >= RiskLevel.MEDIUM:
            try:
                from core.runtime.life_trace import get_life_trace
                event_type = "action_executed" if decision.approved else "initiative_blocked"
                get_life_trace().record(
                    event_type=event_type,
                    origin="permission_model",
                    action_taken={
                        "action": decision.action,
                        "target": decision.target[:200],
                        "risk_level": decision.risk_level.name,
                    },
                    result={
                        "approved": decision.approved,
                        "reason": decision.reason[:200],
                        "modality": decision.modality,
                        "receipt_id": decision.receipt_id,
                    },
                )
            except (ImportError, AttributeError, RuntimeError) as e:
                record_degradation("permission_model.life_trace", e)

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    def get_recent_decisions(self, limit: int = 20) -> List[Dict[str, Any]]:
        return [
            {
                "action": d.action,
                "target": d.target[:60],
                "risk_level": d.risk_level.name,
                "approved": d.approved,
                "reason": d.reason[:100],
                "modality": d.modality,
                "receipt_id": d.receipt_id,
            }
            for d in self._decision_history[-limit:]
        ]

    def get_status(self) -> Dict[str, Any]:
        total = len(self._decision_history)
        approved = sum(1 for d in self._decision_history if d.approved)
        blocked = sum(1 for d in self._decision_history if not d.approved)
        return {
            "trusted_mode": self._trusted_mode,
            "demo_safe_mode": self._demo_safe_mode,
            "total_decisions": total,
            "approved": approved,
            "blocked": blocked,
            "modality_permissions": {
                k: v for k, v in vars(self.modality).items()
            },
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional[PermissionRiskModel] = None


def get_permission_model() -> PermissionRiskModel:
    global _instance
    if _instance is None:
        _instance = PermissionRiskModel()
    return _instance


__all__ = [
    "PermissionRiskModel",
    "PermissionDecision",
    "RiskLevel",
    "ModalityPermissions",
    "get_permission_model",
]
