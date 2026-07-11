"""core/planning/mission_state.py — Durable Mission Progress with Crash-Resume
===============================================================================
Persists mission state to SQLite so tasks survive process restarts.
If Aura crashes halfway through, she can resume or explain exactly where
she stopped.

Each mission is a TaskGraph + metadata. Progress is tracked per-step
in the database with receipts, artifacts, and error logs.

Integrates with:
- TaskDecomposer: to create new missions from objectives
- TaskGraph: to track step-by-step progress
- PostActionVerifier: to verify each step
- RecoveryEngine: to handle failures
- LifeTraceLedger: to log audit events
- InitiativeSynthesizer: active missions submit impulses for their next steps
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from core.container import ServiceContainer
from core.planning.task_graph import TaskGraph, TaskNode
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway

logger = logging.getLogger("Aura.MissionState")


class MissionStatus(StrEnum):
    PLANNING = "planning"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Mission:
    """A single mission with its task graph and metadata."""
    mission_id: str
    objective: str
    status: MissionStatus = MissionStatus.PLANNING
    graph: TaskGraph | None = None
    source: str = ""                    # "voice", "text", "initiative"
    priority: float = 0.5              # 0-1
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    error_summary: str = ""
    narration_log: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "objective": self.objective,
            "status": self.status.value,
            "source": self.source,
            "priority": self.priority,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "error_summary": self.error_summary,
            "narration_log": self.narration_log[-10:],
            "progress": self.graph.get_progress() if self.graph else {},
        }


class MissionState:
    """Durable mission progress with crash-resume and proof bundling.

    Persisted to SQLite so missions survive process restarts.

    Usage:
        ms = get_mission_state()
        mission = await ms.create_mission("Find a squid image and set as wallpaper")
        # ... MissionExecutor runs the graph ...
        ms.update_mission_status(mission.mission_id, MissionStatus.COMPLETED)
    """

    DB_NAME = "missions.db"

    def __init__(self, data_dir: str | None = None) -> None:
        self._data_dir = Path(data_dir) if data_dir else Path.home() / ".aura" / "data" / "missions"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._data_dir / self.DB_NAME
        self._active_missions: dict[str, Mission] = {}
        self._conn: sqlite3.Connection | None = None
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        self._init_db()
        self._load_active_missions()
        ServiceContainer.register_instance("mission_state", self, required=False)
        self._started = True
        active_count = len(self._active_missions)
        logger.info(
            "MissionState ONLINE — %d active mission(s) loaded from %s",
            active_count, self._db_path,
        )

    def close(self) -> None:
        """Commit and close mission persistence idempotently."""

        connection = self._conn
        self._conn = None
        self._started = False
        if connection is None:
            return
        try:
            connection.commit()
        finally:
            connection.close()

    def _init_db(self) -> None:
        """Initialize the SQLite database."""
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS missions (
                mission_id TEXT PRIMARY KEY,
                objective TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'planning',
                source TEXT DEFAULT '',
                priority REAL DEFAULT 0.5,
                graph_json TEXT,
                narration_json TEXT DEFAULT '[]',
                error_summary TEXT DEFAULT '',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                completed_at REAL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS mission_steps (
                step_id TEXT NOT NULL,
                mission_id TEXT NOT NULL,
                action TEXT NOT NULL,
                params_json TEXT DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'pending',
                result_json TEXT DEFAULT '{}',
                receipt_id TEXT DEFAULT '',
                artifacts_json TEXT DEFAULT '[]',
                error TEXT DEFAULT '',
                started_at REAL DEFAULT 0,
                completed_at REAL DEFAULT 0,
                PRIMARY KEY (mission_id, step_id),
                FOREIGN KEY (mission_id) REFERENCES missions(mission_id)
            );
            CREATE INDEX IF NOT EXISTS idx_missions_status ON missions(status);
            CREATE INDEX IF NOT EXISTS idx_steps_mission ON mission_steps(mission_id);
        """)
        self._conn.commit()

    def _load_active_missions(self) -> None:
        """Load active missions from SQLite on startup."""
        if not self._conn:
            return
        try:
            cursor = self._conn.execute(
                "SELECT mission_id, objective, status, source, priority, graph_json, "
                "narration_json, error_summary, created_at, updated_at, completed_at "
                "FROM missions WHERE status IN ('planning', 'active', 'paused')"
            )
            for row in cursor.fetchall():
                mission_id, objective, status, source, priority, graph_json, \
                    narration_json, error_summary, created_at, updated_at, completed_at = row

                mission = Mission(
                    mission_id=mission_id,
                    objective=objective,
                    status=MissionStatus(status),
                    source=source or "",
                    priority=priority,
                    created_at=created_at,
                    updated_at=updated_at,
                    completed_at=completed_at,
                    error_summary=error_summary or "",
                )

                # Restore narration log
                try:
                    mission.narration_log = json.loads(narration_json or "[]")
                except (json.JSONDecodeError, TypeError):
                    mission.narration_log = []

                # Restore task graph
                if graph_json:
                    try:
                        mission.graph = TaskGraph.from_json(graph_json)
                    except (json.JSONDecodeError, KeyError, TypeError) as e:
                        logger.warning("Failed to restore graph for %s: %s", mission_id, e)

                self._active_missions[mission_id] = mission
        except (sqlite3.Error, ValueError) as e:
            record_degradation("mission_state.load", e)

    # ------------------------------------------------------------------
    # Mission lifecycle
    # ------------------------------------------------------------------

    async def create_mission(
        self,
        objective: str,
        source: str = "text",
        priority: float = 0.5,
        context: dict[str, Any] | None = None,
    ) -> Mission:
        """Create a new mission from an objective.

        Decomposes the objective into a TaskGraph and persists to SQLite.
        """
        # Decompose into task graph
        decomposer = ServiceContainer.get("task_decomposer", default=None)
        if decomposer is None:
            from core.planning.task_decomposer import get_task_decomposer
            decomposer = get_task_decomposer()

        graph = await decomposer.decompose(objective, context)

        mission = Mission(
            mission_id=graph.mission_id,
            objective=objective,
            status=MissionStatus.ACTIVE,
            graph=graph,
            source=source,
            priority=priority,
        )
        mission.narration_log.append(
            f"Mission created: {objective[:100]} ({graph.total_steps} steps)"
        )

        self._active_missions[mission.mission_id] = mission
        self._persist_mission(mission)

        # Log to LifeTrace
        self._log_to_life_trace("initiative_proposed", mission.mission_id, {
            "objective": objective[:200],
            "steps": graph.total_steps,
            "source": source,
        })

        logger.info(
            "Mission created: %s — '%s' (%d steps)",
            mission.mission_id, objective[:50], graph.total_steps,
        )
        return mission

    async def advance_mission(self, mission_id: str) -> TaskNode | None:
        """Execute the next ready node in a mission's graph.

        Returns the node that was executed, or None if no nodes are ready.
        This is called by the MissionExecutor or InitiativeSynthesizer.
        """
        mission = self._active_missions.get(mission_id)
        if not mission or not mission.graph:
            return None

        if mission.status != MissionStatus.ACTIVE:
            return None

        graph = mission.graph
        node = graph.get_next_node()
        if node is None:
            if graph.is_complete:
                await self._complete_mission(mission)
            return None

        # Mark running
        graph.mark_running(node.task_id)
        mission.narration_log.append(f"Starting: {node.description or node.action}")
        self._persist_mission(mission)

        # Execute the action
        try:
            result = await self._execute_node(node)

            if result.get("success", False):
                # Verify
                verification_ok = await self._verify_node(node)
                if verification_ok:
                    graph.mark_succeeded(
                        node.task_id,
                        result=result,
                        receipt_id=result.get("receipt_id", ""),
                        artifacts=result.get("artifacts", []),
                    )
                    mission.narration_log.append(f"✓ {node.description or node.action}")
                else:
                    # Verification failed — try recovery
                    recovered = await self._try_recovery(mission, node, "verification_failed")
                    if not recovered:
                        graph.mark_failed(node.task_id, "Verification failed after execution")
                        mission.narration_log.append(f"✗ {node.description}: verification failed")
            else:
                # Execution failed — try recovery
                error = result.get("error", "Unknown error")
                recovered = await self._try_recovery(mission, node, error)
                if not recovered:
                    graph.mark_failed(node.task_id, error)
                    mission.narration_log.append(f"✗ {node.description}: {error[:100]}")

        except (RuntimeError, OSError, TypeError, ValueError) as e:
            error_msg = str(e)
            recovered = await self._try_recovery(mission, node, error_msg)
            if not recovered:
                graph.mark_failed(node.task_id, error_msg)
                mission.narration_log.append(f"✗ {node.description}: {error_msg[:100]}")

        # Persist updated state
        self._persist_mission(mission)

        # Check if mission is complete
        if graph.is_complete:
            await self._complete_mission(mission)

        return node

    async def _execute_node(self, node: TaskNode) -> dict[str, Any]:
        """Execute a single task node using the appropriate capability."""
        # Check permission first
        try:
            perm_model = ServiceContainer.get("permission_model", default=None)
            if perm_model:
                decision = perm_model.check_permission(node.action, str(node.params))
                if not decision.approved:
                    if decision.requires_confirmation:
                        return {"success": False, "error": f"Needs user approval: {decision.reason}"}
                    return {"success": False, "error": f"Permission denied: {decision.reason}"}
        except (ImportError, AttributeError, RuntimeError):
            pass  # No permission model — proceed

        # Route to the appropriate handler
        action = node.action
        params = node.params

        try:
            host = ServiceContainer.get("host_automation", default=None)
            if host is None:
                from core.capabilities.host_automation import get_host_automation
                host = get_host_automation()

            if action == "launch_app":
                receipt = await host.launch_app(params.get("name", ""))
                return {"success": receipt.success, "result": receipt.result, "error": receipt.error, "receipt_id": receipt.receipt_id}

            elif action == "focus_app":
                receipt = await host.focus_app(params.get("name", ""))
                return {"success": receipt.success, "result": receipt.result, "error": receipt.error, "receipt_id": receipt.receipt_id}

            elif action == "close_app":
                receipt = await host.close_app(params.get("name", ""))
                return {"success": receipt.success, "error": receipt.error, "receipt_id": receipt.receipt_id}

            elif action == "type_text":
                receipt = await host.type_text(params.get("text", ""))
                return {"success": receipt.success, "error": receipt.error, "receipt_id": receipt.receipt_id}

            elif action == "hotkey":
                keys = params.get("keys", [])
                receipt = await host.hotkey(*keys)
                return {"success": receipt.success, "error": receipt.error, "receipt_id": receipt.receipt_id}

            elif action == "menu_select":
                receipt = await host.menu_select(params.get("app", ""), params.get("path", []))
                return {"success": receipt.success, "error": receipt.error, "receipt_id": receipt.receipt_id}

            elif action == "take_screenshot":
                receipt = await host.take_screenshot(params.get("save_path", ""))
                return {"success": receipt.success, "result": receipt.result, "artifacts": [receipt.result] if receipt.result else [], "receipt_id": receipt.receipt_id}

            elif action == "run_command":
                receipt = await host.run_command(params.get("command", ""))
                return {"success": receipt.success, "result": receipt.result, "error": receipt.error, "receipt_id": receipt.receipt_id}

            elif action == "create_folder":
                path = Path(params.get("path", ""))
                await asyncio.to_thread(path.mkdir, parents=True, exist_ok=True)
                exists = await asyncio.to_thread(path.exists)
                return {"success": exists, "result": str(path)}

            elif action == "create_text_file":
                path = Path(params.get("path", ""))
                content = params.get("content", "")
                path.parent.mkdir(parents=True, exist_ok=True)
                await get_file_write_gateway().write_text_async(
                    path,
                    content,
                    encoding="utf-8",
                    source="mission_state.create_text_file",
                )
                return {"success": path.exists(), "result": str(path), "artifacts": [str(path)]}

            elif action in ("create_pdf", "render_pdf"):
                try:
                    doc_service = ServiceContainer.get("document_service", default=None)
                    if doc_service:
                        success = await doc_service.create_pdf(
                            params.get("path", ""),
                            params.get("title", "Document"),
                            params.get("body", ""),
                        )
                        return {"success": success, "result": params.get("path", ""), "artifacts": [params.get("path", "")]}
                except (ImportError, AttributeError):
                    pass
                return {"success": False, "error": "PDF service not available"}

            elif action in ("search_web", "search_and_open"):
                try:
                    browser = ServiceContainer.get("browser_controller", default=None)
                    if browser:
                        receipt = await browser.search_and_open(params.get("query", ""), params.get("count", 3))
                        return {"success": receipt.success, "result": receipt.result, "receipt_id": receipt.receipt_id}
                except (ImportError, AttributeError):
                    pass
                return {"success": False, "error": "Browser controller not available"}

            elif action == "open_url":
                try:
                    browser = ServiceContainer.get("browser_controller", default=None)
                    if browser:
                        receipt = await browser.open_url(params.get("url", ""))
                        return {"success": receipt.success, "receipt_id": receipt.receipt_id}
                except (ImportError, AttributeError):
                    pass
                # Fallback: use system open
                receipt = await host.run_command(f"open {params.get('url', '')}")
                return {"success": receipt.success, "receipt_id": receipt.receipt_id}

            elif action == "search_images":
                try:
                    asset_handler = ServiceContainer.get("web_asset_handler", default=None)
                    if asset_handler:
                        results = await asset_handler.search_images(params.get("query", ""))
                        return {"success": bool(results), "result": results}
                except (ImportError, AttributeError):
                    pass
                return {"success": False, "error": "Web asset handler not available"}

            elif action == "download_image":
                try:
                    asset_handler = ServiceContainer.get("web_asset_handler", default=None)
                    if asset_handler:
                        path = await asset_handler.download_image(params.get("url", ""), params.get("save_dir", ""))
                        return {"success": bool(path), "result": path, "artifacts": [path] if path else []}
                except (ImportError, AttributeError):
                    pass
                return {"success": False, "error": "Web asset handler not available"}

            elif action == "set_wallpaper":
                try:
                    os_settings = ServiceContainer.get("os_settings", default=None)
                    if os_settings:
                        receipt = await os_settings.set_wallpaper(params.get("image_path", ""))
                        return {"success": receipt.success, "receipt_id": receipt.receipt_id}
                except (ImportError, AttributeError):
                    pass
                return {"success": False, "error": "OS settings adapter not available"}

            elif action == "get_wallpaper":
                try:
                    os_settings = ServiceContainer.get("os_settings", default=None)
                    if os_settings:
                        path = await os_settings.get_wallpaper()
                        return {"success": bool(path), "result": path}
                except (ImportError, AttributeError):
                    pass
                return {"success": False, "error": "OS settings adapter not available"}

            elif action == "get_screen_text":
                receipt = await host.get_screen_text()
                return {"success": receipt.success, "result": receipt.result, "receipt_id": receipt.receipt_id}

            elif action == "notify_user":
                logger.info("NARRATION: %s", params.get("message", ""))
                return {"success": True, "result": params.get("message", "")}

            elif action == "wait":
                await asyncio.sleep(float(params.get("seconds", 1.0)))
                return {"success": True}

            else:
                return {"success": False, "error": f"Unknown action: {action}"}

        except (RuntimeError, OSError, TypeError, ValueError) as e:
            return {"success": False, "error": str(e)}

    async def _verify_node(self, node: TaskNode) -> bool:
        """Run post-action verification for a node."""
        if node.verification == "true" or not node.verification:
            return True

        try:
            verifier = ServiceContainer.get("post_action_verifier", default=None)
            if verifier is None:
                from core.capabilities.post_action_verifier import get_post_action_verifier
                verifier = get_post_action_verifier()

            result = await verifier.verify(node.verification, node.verification_args)
            node.verification_result = result.to_dict()
            return result.success
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation("mission_state.verify", e)
            logger.debug("Verification failed for %s: %s", node.task_id, e)
            node.verification_result = {
                "predicate": node.verification,
                "success": False,
                "evidence": f"Verifier unavailable: {e}",
            }
            return False

    async def _try_recovery(self, mission: Mission, node: TaskNode, error: str) -> bool:
        """Attempt recovery for a failed node."""
        try:
            recovery = ServiceContainer.get("recovery_engine", default=None)
            if recovery is None:
                from core.planning.recovery_engine import get_recovery_engine
                recovery = get_recovery_engine()

            return await recovery.recover(mission, node, error)
        except (ImportError, AttributeError, RuntimeError) as e:
            logger.debug("Recovery unavailable: %s", e)
            return False

    async def _complete_mission(self, mission: Mission) -> None:
        """Mark a mission as complete and generate proof bundle."""
        if not mission.graph:
            return

        if mission.graph.is_successful:
            mission.status = MissionStatus.COMPLETED
            mission.narration_log.append("✓ Mission completed successfully")
        else:
            mission.status = MissionStatus.FAILED
            mission.error_summary = mission.graph.get_failure_summary()
            mission.narration_log.append(f"✗ Mission failed: {mission.error_summary[:100]}")

        mission.completed_at = time.time()
        self._persist_mission(mission)

        # Log to LifeTrace
        event_type = "mission_complete" if mission.status == MissionStatus.COMPLETED else "mission_failed"
        self._log_to_life_trace(event_type, mission.mission_id, {
            "objective": mission.objective[:200],
            "status": mission.status.value,
            "steps_completed": mission.graph.completed_steps,
            "steps_total": mission.graph.total_steps,
            "duration_s": round(mission.completed_at - mission.created_at, 1),
        })

        logger.info(
            "Mission %s: %s — %d/%d steps",
            mission.status.value, mission.mission_id,
            mission.graph.completed_steps, mission.graph.total_steps,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_mission(self, mission_id: str) -> Mission | None:
        return self._active_missions.get(mission_id)

    def list_active_missions(self) -> list[Mission]:
        return [m for m in self._active_missions.values()
                if m.status in (MissionStatus.PLANNING, MissionStatus.ACTIVE, MissionStatus.PAUSED)]

    def list_all_missions(self, limit: int = 50) -> list[dict[str, Any]]:
        """List all missions from SQLite."""
        if not self._conn:
            return []
        try:
            cursor = self._conn.execute(
                "SELECT mission_id, objective, status, created_at, updated_at, completed_at "
                "FROM missions ORDER BY created_at DESC LIMIT ?",
                (limit,)
            )
            return [
                {"mission_id": r[0], "objective": r[1], "status": r[2],
                 "created_at": r[3], "updated_at": r[4], "completed_at": r[5]}
                for r in cursor.fetchall()
            ]
        except sqlite3.Error:
            return []

    def get_mission_proof(self, mission_id: str) -> dict[str, Any] | None:
        """Get proof bundle for a mission."""
        mission = self._active_missions.get(mission_id)
        if mission and mission.graph:
            return mission.graph.get_proof_bundle()
        return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_mission(self, mission: Mission) -> None:
        """Save mission state to SQLite."""
        if not self._conn:
            return
        try:
            graph_json = mission.graph.to_json() if mission.graph else ""
            narration_json = json.dumps(mission.narration_log[-50:])
            self._conn.execute(
                "INSERT OR REPLACE INTO missions "
                "(mission_id, objective, status, source, priority, graph_json, "
                "narration_json, error_summary, created_at, updated_at, completed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    mission.mission_id, mission.objective, mission.status.value,
                    mission.source, mission.priority, graph_json,
                    narration_json, mission.error_summary,
                    mission.created_at, time.time(), mission.completed_at,
                ),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            record_degradation("mission_state.persist", e)

    def _log_to_life_trace(self, event_type: str, mission_id: str, data: dict[str, Any]) -> None:
        try:
            from core.runtime.life_trace import get_life_trace
            get_life_trace().record(
                event_type=event_type,
                origin="mission_state",
                action_taken={"mission_id": mission_id},
                result=data,
            )
        except (ImportError, AttributeError, RuntimeError):
            pass

    def update_mission_status(self, mission_id: str, status: MissionStatus) -> None:
        mission = self._active_missions.get(mission_id)
        if mission:
            mission.status = status
            mission.updated_at = time.time()
            self._persist_mission(mission)

    def get_status(self) -> dict[str, Any]:
        active = [m for m in self._active_missions.values() if m.status == MissionStatus.ACTIVE]
        return {
            "total_missions": len(self._active_missions),
            "active": len(active),
            "active_missions": [
                {"id": m.mission_id, "objective": m.objective[:60], "progress": m.graph.get_progress() if m.graph else {}}
                for m in active[:5]
            ],
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: MissionState | None = None


def get_mission_state() -> MissionState:
    global _instance
    if _instance is None:
        _instance = MissionState()
    return _instance


def close_mission_state() -> dict[str, object]:
    """Close the singleton without constructing it during root teardown."""

    if _instance is None:
        return {"clean": True, "closed": False, "reason": "not_initialized"}
    try:
        _instance.close()
    except (OSError, RuntimeError, sqlite3.Error) as exc:
        return {
            "clean": False,
            "closed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {"clean": True, "closed": True}


__all__ = [
    "MissionState",
    "Mission",
    "MissionStatus",
    "close_mission_state",
    "get_mission_state",
]
