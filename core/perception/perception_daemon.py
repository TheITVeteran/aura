"""Always-on PerceptionDaemon for Aura.

Maintains continuous environmental context (screen, window focus, clipboard, audio, entity tracking)
and updates a rolling perceptual memory.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
import uuid
import dataclasses
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.container import ServiceContainer
from core.runtime.errors import record_degradation
from core.event_bus import get_event_bus, EventPriority

logger = logging.getLogger("Aura.PerceptionDaemon")




class PerceptionDaemon:
    """Always-on perception loop and rolling short/medium term memory."""

    _instance: Optional[PerceptionDaemon] = None
    _lock = asyncio.Lock()

    def __init__(self, *, check_interval_s: float = 2.0):
        self.check_interval = check_interval_s
        self.running = False
        self._tasks: List[asyncio.Task] = []

        # Rolling buffers (thread-safe deques)
        self._short_term_buffer: deque[dict[str, Any]] = deque(maxlen=200)   # last ~5-10 mins
        self._medium_term_buffer: deque[dict[str, Any]] = deque(maxlen=2000) # last ~24 hours

        # Entity tracking
        self._entities: Dict[str, Dict[str, Any]] = {}
        self._entity_aliases: Dict[str, str] = {}

        # Attention state
        self.user_focus = "unknown"
        self.aura_focus = "unknown"
        self.joint_attention_score = 0.5
        self.attention_lock = asyncio.Lock()

        # Telemetry & States
        self.user_active = True
        self.last_user_activity = time.time()
        self.last_clipboard_hash = ""
        self.last_active_window = ""

        # Privacy configs
        self.privacy_mode = False
        self.redacted_patterns = ["password", "token", "key", "secret", "private"]

        logger.info("📡 PerceptionDaemon initialized.")

    @classmethod
    async def get(cls) -> PerceptionDaemon:
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def get_sync(cls) -> PerceptionDaemon:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def start(self) -> None:
        if self.running:
            return
        self.running = True
        logger.info("📡 PerceptionDaemon starting background sensory loops...")

        # Spawn background loops
        self._tasks.append(asyncio.create_task(self._main_perceptual_loop(), name="perception_daemon.main"))
        self._tasks.append(asyncio.create_task(self._attention_alignment_loop(), name="perception_daemon.attention"))

        logger.info("📡 PerceptionDaemon is ONLINE.")

    async def stop(self) -> None:
        self.running = False
        for task in self._tasks:
            if not task.done():
                task.cancel()
        self._tasks.clear()
        logger.info("📡 PerceptionDaemon is OFFLINE.")

    def register_moment(self, source: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Insert a perceptual moment, apply privacy filters, and publish to the EventBus."""
        now = time.time()
        meta = dict(metadata or {})
        
        # Privacy redact
        filtered_content = content
        if self.privacy_mode or any(p in content.lower() for p in self.redacted_patterns):
            filtered_content = "<redacted: privacy policy>"
            meta["redacted"] = True

        moment = {
            "moment_id": f"pmom-{uuid.uuid4().hex[:8]}",
            "timestamp": now,
            "source": source,
            "content": filtered_content,
            "metadata": meta,
        }

        # Add to buffers
        self._short_term_buffer.append(moment)
        self._medium_term_buffer.append(moment)

        # Publish autonomic sensory event
        try:
            get_event_bus().publish_threadsafe(
                topic="aura/perception/moment",
                data=moment,
                priority=EventPriority.AUTONOMIC,
            )
        except Exception as e:
            logger.debug("Daemon failed to publish sensory moment to EventBus: %s", e)

        return moment

    async def _main_perceptual_loop(self) -> None:
        """Poll clipboard, active window, and file changes continuously."""
        while self.running:
            try:
                await asyncio.sleep(self.check_interval)

                # 1. Active Window Focus Check
                window = await self._check_active_window()
                if window and window != self.last_active_window:
                    self.last_active_window = window
                    self.register_moment(
                        source="window_focus",
                        content=f"User switched focus application to: {window}",
                        metadata={"app_name": window}
                    )
                    self.last_user_activity = time.time()
                    self.user_active = True

                # 2. Clipboard Change Check
                clipboard = await self._check_clipboard()
                if clipboard:
                    clip_hash = hashlib.sha256(clipboard.encode("utf-8")).hexdigest()
                    if clip_hash != self.last_clipboard_hash:
                        self.last_clipboard_hash = clip_hash
                        # Only store short snippet for safety
                        snippet = clipboard[:200]
                        if len(clipboard) > 200:
                            snippet += "..."
                        self.register_moment(
                            source="clipboard",
                            content=f"Clipboard changed: {snippet}",
                            metadata={"char_count": len(clipboard)}
                        )
                        self.last_user_activity = time.time()
                        self.user_active = True

                # 3. User Idle State Assessment
                idle_time = time.time() - self.last_user_activity
                if idle_time > 120.0 and self.user_active:
                    self.user_active = False
                    self.register_moment(
                        source="user_presence",
                        content="User has become idle (inactive for >2 mins)",
                        metadata={"idle_seconds": idle_time}
                    )
                elif idle_time <= 10.0 and not self.user_active:
                    self.user_active = True
                    self.register_moment(
                        source="user_presence",
                        content="User has resumed activity",
                        metadata={"idle_seconds": idle_time}
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                record_degradation("perception_daemon.main_loop", e)
                logger.debug("Error in PerceptionDaemon main loop: %s", e)
                await asyncio.sleep(self.check_interval * 2)

    async def _attention_alignment_loop(self) -> None:
        """Analyze rolling sensory moment topics to estimate shared attention."""
        while self.running:
            try:
                await asyncio.sleep(15.0)

                # Look at recent moments in last 30s to update joint attention
                recent = self.get_recent_moments(duration_seconds=30.0)
                async with self.attention_lock:
                    if recent:
                        sources = {m["source"] for m in recent}
                        if "window_focus" in sources or "clipboard" in sources:
                            self.user_focus = self.last_active_window or "desktop"
                            self.joint_attention_score = min(1.0, self.joint_attention_score + 0.1)
                        else:
                            self.joint_attention_score = max(0.2, self.joint_attention_score - 0.05)
                    else:
                        self.joint_attention_score = max(0.1, self.joint_attention_score - 0.1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                record_degradation("perception_daemon.attention_loop", e)
                logger.debug("Error in PerceptionDaemon attention loop: %s", e)
                await asyncio.sleep(15.0)

    async def _check_clipboard(self) -> Optional[str]:
        try:
            # Safe non-blocking execution of pbpaste (macOS standard clipboard grabber)
            proc = await asyncio.create_subprocess_exec(
                "pbpaste",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=1.0)
            if proc.returncode == 0:
                return stdout.decode("utf-8", errors="ignore").strip()
            return None
        except Exception:
            return None

    async def _check_active_window(self) -> Optional[str]:
        try:
            # Safe non-blocking check of active window process on macOS via osascript
            cmd = ["osascript", "-e", 'tell application "System Events" to get name of first application process whose frontmost is true']
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=1.5)
            if proc.returncode == 0:
                return stdout.decode("utf-8", errors="ignore").strip()
            return None
        except Exception:
            return None

    # --- Public API surface ------------------------------------------------

    def get_recent_moments(self, source: Optional[str] = None, duration_seconds: float = 300.0) -> List[Dict[str, Any]]:
        now = time.time()
        cutoff = now - duration_seconds
        # Search short term first
        res = [m for m in self._short_term_buffer if m["timestamp"] >= cutoff]
        if not res and duration_seconds > 300.0:
            # Fallback to medium term
            res = [m for m in self._medium_term_buffer if m["timestamp"] >= cutoff]
        
        if source:
            res = [m for m in res if m["source"] == source]
        return res

    def track_entity(self, entity_type: str, name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Track/retrieve stable ID for files, browser tabs, tasks, users, etc."""
        alias_key = f"{entity_type}::{name}".lower()
        if alias_key in self._entity_aliases:
            entity_id = self._entity_aliases[alias_key]
            self._entities[entity_id]["last_seen"] = time.time()
            if metadata:
                self._entities[entity_id]["metadata"].update(metadata)
            return entity_id

        entity_id = f"ent-{uuid.uuid4().hex[:8]}"
        self._entity_aliases[alias_key] = entity_id
        self._entities[entity_id] = {
            "entity_id": entity_id,
            "type": entity_type,
            "name": name,
            "created_at": time.time(),
            "last_seen": time.time(),
            "metadata": dict(metadata or {}),
        }
        logger.info("🆕 Tracking entity: %s (type=%s, ID=%s)", name, entity_type, entity_id)
        return entity_id

    async def active_perceive(self, probe_type: str, query: Optional[str] = None) -> Dict[str, Any]:
        """Force a perception probe like a manual screen capture or file verification."""
        logger.info("🔍 Active perception triggered: probe_type=%s, query=%s", probe_type, query)
        
        if probe_type == "screen_ocr":
            # Delegate to computer use or vision engine
            vision = ServiceContainer.get("vision_engine", default=None)
            if vision and hasattr(vision, "analyze_moment"):
                try:
                    desc = await vision.analyze_moment(prompt=query or "Describe current text contents.")
                    self.register_moment(source="active_ocr", content=desc)
                    return {"ok": True, "result": desc}
                except Exception as e:
                    return {"ok": False, "error": str(e)}
            return {"ok": False, "error": "vision_engine_unavailable"}
            
        elif probe_type == "file_status":
            if not query:
                return {"ok": False, "error": "missing file path query"}
            p = Path(query)
            if p.exists():
                stat = p.stat()
                desc = f"File {p.name} exists, size={stat.st_size} bytes, modified={stat.st_mtime}"
                self.register_moment(source="active_file_check", content=desc, metadata={"path": str(p)})
                return {"ok": True, "result": desc}
            return {"ok": False, "error": "file_not_found"}

        return {"ok": False, "error": f"unsupported probe type: {probe_type}"}


def get_perception_daemon() -> PerceptionDaemon:
    return PerceptionDaemon.get_sync()
