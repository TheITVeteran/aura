"""core/capabilities/__init__.py — Capability Layer Boot Sequence
=================================================================
Wires up all Phase 2-6 modules into the ServiceContainer.

Call `await boot_capabilities()` during Aura's startup to bring all
capability providers online in dependency order.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Capabilities")


async def boot_capabilities() -> Dict[str, Any]:
    """Boot all capability providers in dependency order.

    Order matters:
    1. AppRegistry (discovers what's installed)
    2. CapabilityDiscovery (scans machine capabilities)
    3. PermissionModel (gates all actions)
    4. HostAutomation (executes actions)
    5. PostActionVerifier (verifies outcomes)
    6. BrowserController (web automation)
    7. DocumentService (file creation)
    8. FileBroker (sandboxed file ops)
    9. WebAssetHandler (image search/download)
    10. OSSettings (wallpaper, volume, etc.)
    11. ClipboardManager (clipboard ops)
    12. SourceSummarizer (multi-source summarization)
    13. ScreenPerception (visual perception)

    Planning layer:
    14. TaskDecomposer (NL → TaskGraph)
    15. RecoveryEngine (failure recovery)
    16. MissionState (durable mission progress)

    Voice layer:
    17. VoiceSession (narration + session management)
    18. WakeWord (always-listening detection)

    Philosophical layer:
    19. BehavioralProof (Path A functionalist evidence)
    20. MindStateExporter (mind state export/import)

    Returns a status dict.
    """
    start = time.time()
    booted: List[str] = []
    failed: List[str] = []

    async def _boot(name: str, factory):
        """Boot a single provider, recording success/failure."""
        try:
            instance = factory()
            if hasattr(instance, "start"):
                await instance.start()
            booted.append(name)
        except (ImportError, AttributeError, RuntimeError, TypeError, OSError) as e:
            failed.append(name)
            record_degradation(f"boot.{name}", e)
            logger.warning("Failed to boot %s: %s", name, e)

    # --- Tier 1: Foundation ---

    try:
        from core.capabilities.capabilities.app_registry import get_app_registry
        await _boot("app_registry", get_app_registry)
    except ImportError as e:
        failed.append("app_registry")
        record_degradation("boot.app_registry", e)

    try:
        from core.capabilities.capabilities.capability_discovery import get_capability_discovery
        await _boot("capability_discovery", get_capability_discovery)
    except ImportError as e:
        failed.append("capability_discovery")
        record_degradation("boot.capability_discovery", e)

    try:
        from core.capabilities.capabilities.permission_model import get_permission_model
        await _boot("permission_model", get_permission_model)
    except ImportError as e:
        failed.append("permission_model")
        record_degradation("boot.permission_model", e)

    try:
        from core.capabilities.capabilities.host_automation import get_host_automation
        await _boot("host_automation", get_host_automation)
    except ImportError as e:
        failed.append("host_automation")
        record_degradation("boot.host_automation", e)

    try:
        from core.capabilities.capabilities.post_action_verifier import get_post_action_verifier
        await _boot("post_action_verifier", get_post_action_verifier)
    except ImportError as e:
        failed.append("post_action_verifier")
        record_degradation("boot.post_action_verifier", e)

    # --- Tier 2: Adapters ---

    try:
        from core.capabilities.capabilities.browser_controller import get_browser_controller
        await _boot("browser_controller", get_browser_controller)
    except ImportError as e:
        failed.append("browser_controller")
        record_degradation("boot.browser_controller", e)

    try:
        from core.capabilities.capabilities.document_service import get_document_service
        await _boot("document_service", get_document_service)
    except ImportError as e:
        failed.append("document_service")
        record_degradation("boot.document_service", e)

    try:
        from core.capabilities.capabilities.file_broker import get_file_broker
        await _boot("file_broker", get_file_broker)
    except ImportError as e:
        failed.append("file_broker")
        record_degradation("boot.file_broker", e)

    try:
        from core.capabilities.capabilities.web_asset_handler import get_web_asset_handler
        await _boot("web_asset_handler", get_web_asset_handler)
    except ImportError as e:
        failed.append("web_asset_handler")
        record_degradation("boot.web_asset_handler", e)

    try:
        from core.capabilities.capabilities.os_settings import get_os_settings
        await _boot("os_settings", get_os_settings)
    except ImportError as e:
        failed.append("os_settings")
        record_degradation("boot.os_settings", e)

    try:
        from core.capabilities.capabilities.clipboard_manager import get_clipboard_manager
        await _boot("clipboard_manager", get_clipboard_manager)
    except ImportError as e:
        failed.append("clipboard_manager")
        record_degradation("boot.clipboard_manager", e)

    try:
        from core.capabilities.capabilities.source_summarizer import get_source_summarizer
        await _boot("source_summarizer", get_source_summarizer)
    except ImportError as e:
        failed.append("source_summarizer")
        record_degradation("boot.source_summarizer", e)

    # --- Tier 2: Perception ---

    try:
        from core.perception.screen_perception import get_screen_perception
        await _boot("screen_perception", get_screen_perception)
    except ImportError as e:
        failed.append("screen_perception")
        record_degradation("boot.screen_perception", e)

    try:
        from core.perception.perceptual_pump import get_perceptual_pump
        await _boot("perceptual_pump", get_perceptual_pump)
    except ImportError as e:
        failed.append("perceptual_pump")
        record_degradation("boot.perceptual_pump", e)

    # --- Tier 3: Planning ---

    try:
        from core.planning.task_decomposer import get_task_decomposer
        await _boot("task_decomposer", get_task_decomposer)
    except ImportError as e:
        failed.append("task_decomposer")
        record_degradation("boot.task_decomposer", e)

    try:
        from core.planning.recovery_engine import get_recovery_engine
        await _boot("recovery_engine", get_recovery_engine)
    except ImportError as e:
        failed.append("recovery_engine")
        record_degradation("boot.recovery_engine", e)

    try:
        from core.planning.mission_state import get_mission_state
        await _boot("mission_state", get_mission_state)
    except ImportError as e:
        failed.append("mission_state")
        record_degradation("boot.mission_state", e)

    # --- Tier 3: Voice ---

    try:
        from core.voice.voice_session import get_voice_session_manager
        await _boot("voice_session", get_voice_session_manager)
    except ImportError as e:
        failed.append("voice_session")
        record_degradation("boot.voice_session", e)

    try:
        from core.voice.wake_word import get_wake_word_detector
        await _boot("wake_word", get_wake_word_detector)
    except ImportError as e:
        failed.append("wake_word")
        record_degradation("boot.wake_word", e)

    # --- Tier 3: Philosophical ---

    try:
        from core.phenomenal_substrate.philosophical_stance import get_behavioral_proof
        await _boot("behavioral_proof", get_behavioral_proof)
    except ImportError as e:
        failed.append("behavioral_proof")
        record_degradation("boot.behavioral_proof", e)

    try:
        from core.self.mind_state_export import get_mind_state_exporter
        await _boot("mind_state_exporter", get_mind_state_exporter)
    except ImportError as e:
        failed.append("mind_state_exporter")
        record_degradation("boot.mind_state_exporter", e)

    duration = time.time() - start
    status = {
        "booted": booted,
        "failed": failed,
        "total": len(booted) + len(failed),
        "success_count": len(booted),
        "failure_count": len(failed),
        "duration_ms": round(duration * 1000, 1),
    }

    logger.info(
        "Capability layer ONLINE: %d/%d providers booted in %.0fms%s",
        len(booted), len(booted) + len(failed), duration * 1000,
        f" (FAILED: {', '.join(failed)})" if failed else "",
    )
    return status


async def get_capabilities_status() -> Dict[str, Any]:
    """Get status of all booted capability providers."""
    from core.container import ServiceContainer

    providers = [
        "app_registry", "capability_discovery", "permission_model",
        "host_automation", "post_action_verifier", "browser_controller",
        "document_service", "file_broker", "web_asset_handler",
        "os_settings", "clipboard_manager", "source_summarizer",
        "screen_perception", "perceptual_pump", "task_decomposer", "recovery_engine",
        "mission_state", "voice_session", "wake_word",
        "behavioral_proof", "mind_state_exporter",
    ]

    status = {}
    for name in providers:
        try:
            provider = ServiceContainer.get(name, default=None)
            if provider and hasattr(provider, "get_status"):
                status[name] = provider.get_status()
            elif provider:
                status[name] = {"started": True}
            else:
                status[name] = {"started": False}
        except (ImportError, AttributeError, RuntimeError):
            status[name] = {"started": False, "error": "unavailable"}

    return status


__all__ = ["boot_capabilities", "get_capabilities_status"]
