"""interface/server.py
────────────────────
Aura Luna — FastAPI entry-point.

Decomposed: Routes live in interface/routes/*, auth in interface/auth.py,
WebSocket infrastructure in interface/websocket_manager.py, event bridge
in interface/event_bridge.py. This file retains only:
  - Imports and app creation
  - Lifespan context manager
  - Middleware stack
  - WebSocket endpoint and broadcaster
  - SPA catch-all
  - Entry-point
"""
# ruff: noqa: E402
# This module bootstraps logging, middleware, and route registration in phases;
# several imports intentionally stay next to the phase they wire.
from __future__ import annotations

# ── stdlib ────────────────────────────────────────────────────
import asyncio
import contextvars
import hmac
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

import psutil

# ── Third-party ───────────────────────────────────────────────
import uvicorn
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from core.runtime.errors import record_degradation
from core.runtime.shutdown_coordinator import is_shutdown_requested

try:
    from fastapi.responses import ORJSONResponse
except ImportError:
    ORJSONResponse = JSONResponse
from fastapi.staticfiles import StaticFiles

try:
    import sounddevice as sd
except ImportError:
    sd = None  # Audio features degrade gracefully

# ── Internal — logging first (no other internal imports before this) ──
from core.config import config
from core.container import ServiceContainer
from core.event_bus import get_event_bus

bus = get_event_bus()
from core.logging_config import setup_logging

logger = setup_logging("Aura.Server")

from core.health.boot_status import build_boot_health_snapshot
from core.runtime_tools import get_runtime_state
from core.utils.task_tracker import TaskTracker
from core.version import VERSION, version_string

PROJECT_ROOT = config.paths.project_root
_server_task_tracker = TaskTracker(name="AuraServer", max_concurrent=128)
_SERVER_BOUNDARY_ERRORS = (
    AttributeError,
    ConnectionError,
    ImportError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)

_HTTP_REQUESTS_TOTAL = Counter(
    "aura_http_requests_total",
    "HTTP requests served by the Aura interface.",
    ("method", "path", "status"),
)
_HTTP_REQUEST_LATENCY_SECONDS = Histogram(
    "aura_http_request_latency_seconds",
    "HTTP request latency for the Aura interface.",
    ("method", "path"),
)
_HTTP_REQUESTS_IN_PROGRESS = Gauge(
    "aura_http_requests_in_progress",
    "HTTP requests currently being processed by the Aura interface.",
)


def _route_template(request: Request) -> str:
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    if isinstance(path, str) and path:
        return path
    return request.url.path


def _spawn_server_task(coro, *, name: str) -> asyncio.Task:
    return _server_task_tracker.create_task(coro, name=name)


def _spawn_server_bounded_task(coro, *, name: str) -> asyncio.Task:
    return _server_task_tracker.bounded_track(coro, name=name)

logger.info("🚀 KERNEL LIFESPAN: Starting... EventBus ID: %s", bus._bus_id)

# Diagnostic: Identify process role
_is_proxy = os.environ.get("AURA_GUI_PROXY") == "1"
logger.info("📡 [PROCESS_BOOT] PID: %s | Role: %s", os.getpid(), "GUI_PROXY" if _is_proxy else "KERNEL")

# Lazy-loaded heavy subsystems (via lifespan)
_LocalBrain       = None
_LatentCore       = None
_PredictiveSelf   = None
_FastMouth        = None
_LocalVision      = None
_voice_engine_fn  = None


# ── WebSocket broadcast infrastructure (extracted to interface/websocket_manager.py) ──
from interface.websocket_manager import (
    MessageBroadcastBus as MessageBroadcastBus,
)
from interface.websocket_manager import (
    WebSocketManager as WebSocketManager,
)
from interface.websocket_manager import (
    broadcast_bus,
    log_queue,
    runtime_heartbeat_payload,
    ws_manager,
)

# Wire task spawner into ws_manager now that _spawn_server_task is defined
ws_manager.set_task_spawner(lambda coro, name: _spawn_server_task(coro, name=name))


main_loop: asyncio.AbstractEventLoop | None = None
_event_bridge_task: asyncio.Task | None = None


class _QueueHandler(logging.Handler):
    """Sends structured log records to the async broadcast queue.
    Implements a circular buffer for log_queue to prevent OOM/silencing.
    """

    _recursion_guard: contextvars.ContextVar[bool] = contextvars.ContextVar(
        "_qh_recursion_guard", default=False
    )
    _overflow_logged: bool = False
    _dropped_count: int = 0
    _last_overflow_warning_at: float = 0.0

    @staticmethod
    def _proof_logging_active() -> bool:
        return any(os.environ.get(name) for name in ("AURA_PROOF_RUN", "AURA_AGI_MAX_TASKS", "AURA_TESTING"))

    @classmethod
    def _should_buffer_record(cls, record: logging.LogRecord) -> bool:
        if cls._proof_logging_active() and record.levelno < logging.WARNING:
            return False
        return True

    @staticmethod
    def _entry_is_warning_or_worse(entry: Any) -> bool:
        if not isinstance(entry, dict):
            return False
        level = str(entry.get("level") or "").strip().lower()
        return level in {"warning", "error", "critical", "fatal"}

    @classmethod
    def _should_warn_for_overflow(cls, record: logging.LogRecord, dropped_entry: Any) -> bool:
        return record.levelno >= logging.WARNING or cls._entry_is_warning_or_worse(dropped_entry)

    def emit(self, record: logging.LogRecord) -> None:
        if self._recursion_guard.get():
            return
        if not self._should_buffer_record(record):
            return
        token = self._recursion_guard.set(True)
        try:
            msg = self.format(record)
            if "Error receiving data from connection" in msg or "Stream broken" in msg:
                return

            log_entry = {
                "type": "log",
                "message": msg,
                "level": record.levelname.lower(),
                "timestamp": record.created,
                "module": record.name
            }

            queue_was_full = len(log_queue) >= log_queue.maxlen
            dropped_entry = log_queue[0] if queue_was_full and log_queue else None
            log_queue.append(log_entry)

            if queue_was_full:
                self._dropped_count += 1
                if self._should_warn_for_overflow(record, dropped_entry):
                    last = self._last_overflow_warning_at
                    if record.created - last >= 60.0:
                        logger.warning(
                            "Log buffer dropped warning/error records while at capacity; "
                            "circular buffer preserved newest records (dropped=%d).",
                            self._dropped_count,
                        )
                        self._last_overflow_warning_at = record.created

            if main_loop is not None and not main_loop.is_closed() and main_loop.is_running():
                publish_coro = broadcast_bus.publish(log_entry)
                try:
                    asyncio.run_coroutine_threadsafe(publish_coro, main_loop)
                except _SERVER_BOUNDARY_ERRORS:
                    try:
                        publish_coro.close()
                    except _SERVER_BOUNDARY_ERRORS as close_exc:
                        print(f"CRITICAL LOG CLOSE FALLBACK: {close_exc}", file=sys.stderr)
                    raise

        except _SERVER_BOUNDARY_ERRORS:
            print(f"CRITICAL LOG FALLBACK: {record.levelname} - {record.getMessage()}", file=sys.stderr)
        finally:
            self._recursion_guard.reset(token)


# Attach queue handler to root logger
_qh = _QueueHandler()
_qh.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s", "%H:%M:%S"))
logging.getLogger().addHandler(_qh)


# ── Event bridge functions (extracted to interface/event_bridge.py) ──
from interface.auth import _restore_owner_session_from_request, validate_runtime_security_request
from interface.event_bridge import mycelial_ui_callback, run_event_bridge

# ── Shared helpers ──
from interface.helpers import _notify_user_spoke

# ── Lifespan ──────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start all subsystems on boot; shut them down cleanly on exit."""
    global main_loop
    global _LocalBrain, _LatentCore
    global _PredictiveSelf, _FastMouth, _LocalVision, _voice_engine_fn

    main_loop = asyncio.get_running_loop()
    logger.info("Aura Server %s starting… (Lifespan Enter)", version_string("short"))

    # Initialize EventBus loop for threadsafe publication from background tasks
    from core.event_bus import get_event_bus
    get_event_bus().set_loop(main_loop)

    # 0. Global Registration
    is_gui_proxy = os.environ.get("AURA_GUI_PROXY") == "1"
    from core.service_registration import register_all_services
    register_all_services(is_proxy=is_gui_proxy)

    if is_gui_proxy:
        bus = ServiceContainer.get("actor_bus", default=None)
        if bus:
            logger.info("📡 Igniting deferred ActorBus transports...")
            bus.start_transports()

    # 0.1 Mycelial Network
    from core.mycelium import MycelialNetwork

    mycelial = ServiceContainer.get("mycelial_network", default=None)
    if not mycelial:
        mycelial = MycelialNetwork()
        ServiceContainer.register_instance("mycelial_network", mycelial)
        _spawn_server_bounded_task(
            asyncio.to_thread(mycelial.map_infrastructure, base_dir=str(config.paths.project_root)),
            name="server.mycelium.map_infrastructure",
        )

    ServiceContainer.register_instance("mycelium", mycelial)

    mycelial.set_ui_callback(mycelial_ui_callback)
    if is_gui_proxy:
        logger.info("📡 GUI Proxy: Mycelial Network synchronized.")

    # Ensure data directories exist
    config.paths.create_directories()
    logger.info("📡 Lifespan: Directories verified.")

    # ── Boot heavy subsystems (each gracefully degraded) ──
    from core.utils.safe_import import async_safe_import, is_missing

    if not is_gui_proxy:
        try:
            mod = await async_safe_import("core.local_chat_brain", optional=True)
            if not is_missing(mod):
                _LocalBrain = mod.LocalChatBrain
            else:
                logger.warning("LocalBrain (legacy) unavailable — Fallback mode active")
        except _SERVER_BOUNDARY_ERRORS as _exc:
            record_degradation('server', _exc)
            logger.warning("Optional legacy local brain import failed; continuing degraded: %s", _exc)

        try:
            mod = await async_safe_import("core.latent.latent_core", optional=True)
            if not is_missing(mod):
                _LatentCore = mod.LatentCore
        except _SERVER_BOUNDARY_ERRORS as _exc:
            record_degradation('server', _exc)
            logger.warning("Optional latent core import failed; continuing degraded: %s", _exc)

        try:
            mod = await async_safe_import("core.predictive.predictive_self_model", optional=True)
            if not is_missing(mod):
                _PredictiveSelf = mod.PredictiveSelfModel
        except _SERVER_BOUNDARY_ERRORS as _exc:
            record_degradation('server', _exc)
            logger.warning("Optional predictive self model import failed; continuing degraded: %s", _exc)

        try:
            mod = await async_safe_import("core.senses.tts_stream", optional=True)
            if not is_missing(mod):
                _FastMouth = mod.FastMouth
        except _SERVER_BOUNDARY_ERRORS as _exc:
            record_degradation('server', _exc)
            logger.warning("Optional TTS stream import failed; continuing degraded: %s", _exc)

        try:
            mod = await async_safe_import("core.senses.screen_vision", optional=True)
            if not is_missing(mod):
                _LocalVision = mod.LocalVision
        except _SERVER_BOUNDARY_ERRORS as _exc:
            record_degradation('server', _exc)
            logger.warning("Optional local vision import failed; continuing degraded: %s", _exc)

        try:
            mod = await async_safe_import("core.senses.voice_engine", optional=True)
            if not is_missing(mod):
                _voice_engine_fn = mod.get_voice_engine
                try:
                    _ve_check = _voice_engine_fn()
                    if _ve_check is None:
                        logger.warning("⚠️ Voice engine factory returned None — voice features unavailable.")
                        _voice_engine_fn = None
                    else:
                        logger.info("✓ Voice engine health check passed.")
                except _SERVER_BOUNDARY_ERRORS as ve_err:
                    record_degradation('server', ve_err)
                    logger.warning("⚠️ Voice engine health check failed: %s — disabling voice.", ve_err)
                    _voice_engine_fn = None
        except _SERVER_BOUNDARY_ERRORS as _exc:
            record_degradation('server', _exc)
            logger.warning("Optional voice engine import failed; continuing degraded: %s", _exc)
    else:
        logger.info("📡 GUI Proxy Mode: Skipping heavy subsystem initialization (Brain, TTS, Vision).")

    # Share voice engine factory with privacy route module
    from interface.routes.privacy import set_voice_engine_fn
    set_voice_engine_fn(_voice_engine_fn)

    # ── Trigger cognitive substrate ──
    if not is_gui_proxy:
        logger.info("📡 Kernel Mode: Orchestrator startup deferred to aura_main (to prevent double-boot).")
    else:
        logger.info("📡 GUI Proxy Mode: Cognitive Orchestrator boot SKIPPED.")

    # ── Start WS broadcaster ──
    _spawn_server_task(_ws_broadcaster(), name="ws_broadcaster")

    # ── Bridge EventBus to WS broadcaster (Live HUD) ──
    is_gui_proxy = os.environ.get("AURA_GUI_PROXY") == "1"
    global _event_bridge_task
    if _event_bridge_task is None or _event_bridge_task.done():
        _event_bridge_task = _spawn_server_task(
            run_event_bridge(is_gui_proxy=is_gui_proxy), name="event_bus_bridge"
        )
    else:
        logger.debug("EventBridge task already running; skipping redundant spawn.")

    logger.info("Aura Server online — %s", version_string("full"))
    try:
        yield  # ← app is live here
    finally:
        # ── Shutdown ──
        logger.info("Aura Server shutting down…")
        await _server_task_tracker.shutdown(timeout=2.0)
        _event_bridge_task = None
        main_loop = None


# ── App ───────────────────────────────────────────────────────

app = FastAPI(
    title="Aura Luna Agent",
    description="Secure interface for the Aura Luna autonomous engine.",
    version=VERSION,
    lifespan=lifespan,
)

# 0.1 Prometheus instrumentation. Kept native to avoid Starlette-version
# coupling in third-party middleware.
@app.middleware("http")
async def prometheus_metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    status = 500
    _HTTP_REQUESTS_IN_PROGRESS.inc()
    try:
        response = await call_next(request)
        status = int(response.status_code)
        return response
    finally:
        path = _route_template(request)
        method = request.method
        elapsed = max(0.0, time.perf_counter() - start)
        _HTTP_REQUEST_LATENCY_SECONDS.labels(method=method, path=path).observe(elapsed)
        _HTTP_REQUESTS_TOTAL.labels(method=method, path=path, status=str(status)).inc()
        _HTTP_REQUESTS_IN_PROGRESS.dec()


@app.get("/metrics", include_in_schema=False)
@app.get("/metrics/prometheus", include_in_schema=False)
async def prometheus_metrics_endpoint():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# 0.2 Correlation ID Middleware & Context

correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar('correlation_id', default='')

@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
    correlation_id.set(req_id)
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = req_id
    return response

# SEC-02: Defense-in-depth token verification middleware
@app.middleware("http")
async def verify_token_middleware(request: Request, call_next):
    try:
        validate_runtime_security_request(request)
    except HTTPException as exc:
        return Response(status_code=exc.status_code, content=str(exc.detail))
    return await call_next(request)

# ── Storage & Resource Management ─────────────────────────────

DATA_DIR = Path(config.paths.data_dir)
UPLOAD_DIR = DATA_DIR / "uploads"
GEN_IMAGES_DIR = DATA_DIR / "generated_images"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
GEN_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
}


def _cache_policy_for_path(path: str) -> dict[str, str] | None:
    normalized = str(path or "")
    live_shell_paths = {
        "/",
        "/static/aura.css",
        "/static/aura.js",
        "/static/manifest.json",
        "/static/service-worker.js",
    }
    if normalized in live_shell_paths or normalized.endswith("/index.html"):
        return dict(NO_CACHE_HEADERS)
    if normalized.startswith(("/static", "/data")):
        return {"Cache-Control": "public, max-age=31536000, immutable"}
    return None


# Mount static files for uploads and generated media
app.mount("/data/uploads", StaticFiles(directory=UPLOAD_DIR, html=False), name="uploads")
app.mount("/data/generated_images", StaticFiles(directory=GEN_IMAGES_DIR, html=False), name="generated_images")


@app.middleware("http")
async def add_cache_headers(request: Request, call_next):
    response = await call_next(request)
    policy = _cache_policy_for_path(request.url.path)
    if policy and hasattr(response, "headers"):
        for key, value in policy.items():
            cast(Response, response).headers[key] = value
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if not config.security.internal_only_mode else [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Api-Token", "X-Idempotency-Key", "Authorization"],
)

STATIC_DIR = config.paths.project_root / "interface" / "static"
SHELL_DIST_DIR = STATIC_DIR / "shell" / "dist"
LEGACY_UI_INDEX = STATIC_DIR / "index.html"


def _react_shell_enabled() -> bool:
    """Keep the original Aura HUD as the canonical shell unless explicitly opted in."""
    return os.environ.get("AURA_ENABLE_REACT_SHELL", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Magnum Opus: Request ID Middleware ─────────────────────────

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Inject a unique request ID for distributed tracing and error correlation."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:12])
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ── Magnum Opus: Global Exception Handler ─────────────────────

from datetime import UTC, datetime


def _phenomenal_error_status(envelope) -> int:
    """Map graceful error envelopes to truthful HTTP status codes."""
    state = str(getattr(envelope, "phenomenal_state", "") or "")
    if state == "permission_denied":
        return 403
    if state == "disk_pressure":
        return 507
    if state in {"cognitive_fog", "metabolic_strain", "model_unavailable", "network_offline"}:
        return 503
    return 500


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Phenomenal error envelope for every unhandled exception.

    The user never sees a Python traceback. core/resilience/phenomenal_error_map
    classifies the exception, pushes a substrate signal (cognitive fog,
    sensory deprivation, etc.), and emits the four-button recovery envelope
    that the frontend's error_banner.js renders automatically.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(
        "Unhandled exception [req=%s] %s: %s",
        request_id, type(exc).__name__, exc,
        exc_info=True,
    )
    try:
        from core.resilience.phenomenal_error_map import PhenomenalRaise, build_envelope
        if isinstance(exc, PhenomenalRaise):
            envelope = exc.envelope
        else:
            envelope = build_envelope(exc, correlation_id=request_id)
        http_status = _phenomenal_error_status(envelope)
        return JSONResponse(
            status_code=http_status,
            content={
                "ok": False,
                "status": "phenomenal",
                "http_status": http_status,
                "envelope": envelope.to_dict(),
                "user_message": envelope.user_message,
                "request_id": request_id,
                "timestamp": datetime.now(tz=UTC).isoformat(),
            },
        )
    except _SERVER_BOUNDARY_ERRORS as inner:
        record_degradation('server', inner)
        # Fall back to a structured 500 only when the envelope builder
        # itself crashes — should never happen in practice, but we never
        # want this handler to compound the problem.
        logger.error("phenomenal envelope build failed: %s", inner)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "message": "An unexpected error occurred. Aura's cognitive systems are recovering.",
                "request_id": request_id,
                "timestamp": datetime.now(tz=UTC).isoformat(),
            },
        )


# ── Route Registration ────────────────────────────────────────
# Extracted route modules
from core.health.system_health import router as system_health_router
from core.session.checkpointing import CheckpointService
from interface import memory_ui
from interface.routes import chat as chat_routes
from interface.routes import dashboard as dashboard_routes
from interface.routes import inner_state as inner_state_routes
from interface.routes import interaction_signals as interaction_signal_routes
from interface.routes import memory as memory_routes
from interface.routes import multimodal as multimodal_routes
from interface.routes import performance as performance_routes
from interface.routes import privacy as privacy_routes
from interface.routes import rpc as rpc_routes
from interface.routes import settings as settings_routes
from interface.routes import subsystems as subsystem_routes
from interface.routes import system as system_routes
from interface.routes import mission_control as mission_control_routes

checkpoint_service = CheckpointService()

app.include_router(system_health_router, prefix="/api/health", tags=["health"])
app.include_router(memory_ui.router, prefix="/memory", tags=["memory"])
app.include_router(chat_routes.router, prefix="/api", tags=["chat"])
app.include_router(system_routes.router, prefix="/api", tags=["system"])
app.include_router(subsystem_routes.router, prefix="/api", tags=["subsystems"])
app.include_router(memory_routes.router, prefix="/api", tags=["memory-api"])
app.include_router(interaction_signal_routes.router, prefix="/api", tags=["interaction-signals"])
app.include_router(privacy_routes.router, prefix="/api", tags=["privacy"])
app.include_router(rpc_routes.router, prefix="/rpc", tags=["rpc"])
app.include_router(inner_state_routes.router, tags=["proof-surface"])
app.include_router(dashboard_routes.router, prefix="/api", tags=["dashboard"])
app.include_router(dashboard_routes.trace_router, prefix="/api", tags=["trace"])
app.include_router(settings_routes.router, prefix="/api", tags=["settings"])
app.include_router(multimodal_routes.router, prefix="/api", tags=["multimodal"])
app.include_router(performance_routes.router, prefix="/api", tags=["performance"])
app.include_router(mission_control_routes.router, prefix="/api", tags=["mission_control"])

_system_collect_liquid_state_payload = system_routes._collect_liquid_state_payload


def _collect_conversation_lane_status() -> dict[str, Any]:
    return chat_routes._collect_conversation_lane_status()


def _conversation_lane_is_standby(lane: dict[str, Any] | None) -> bool:
    return chat_routes._conversation_lane_is_standby(lane)


def _collect_liquid_state_payload(
    ls_data: dict[str, Any],
    *,
    runtime_state: dict[str, Any],
    homeostasis_data: dict[str, Any],
) -> dict[str, Any]:
    return _system_collect_liquid_state_payload(
        ls_data,
        runtime_state=runtime_state,
        homeostasis_data=homeostasis_data,
    )


def _sync_legacy_system_exports() -> None:
    system_routes._restore_owner_session_from_request = _restore_owner_session_from_request
    system_routes._collect_conversation_lane_status = _collect_conversation_lane_status
    system_routes._conversation_lane_is_standby = _conversation_lane_is_standby
    system_routes._collect_liquid_state_payload = _collect_liquid_state_payload
    system_routes._collect_legacy_shell_status = _collect_legacy_shell_status
    system_routes.build_boot_health_snapshot = build_boot_health_snapshot
    system_routes.get_runtime_state = get_runtime_state
    system_routes.psutil = psutil


def _collect_stability_details() -> dict[str, Any]:
    _sync_legacy_system_exports()
    return system_routes._collect_stability_details()


def _collect_runtime_capabilities(conversation_lane: dict[str, Any] | None = None) -> dict[str, Any]:
    _sync_legacy_system_exports()
    return system_routes._collect_runtime_capabilities(conversation_lane)


def _collect_legacy_shell_status() -> dict[str, Any]:
    react_shell_enabled = _react_shell_enabled()
    return {
        "shell": "legacy_shell" if LEGACY_UI_INDEX.exists() else "react_shell",
        "legacy_fallback_available": LEGACY_UI_INDEX.exists(),
        "experimental_shell_available": (SHELL_DIST_DIR / "index.html").exists(),
        "experimental_shell_enabled": react_shell_enabled,
        "canonical_shell": "legacy_shell" if LEGACY_UI_INDEX.exists() and not react_shell_enabled else "react_shell",
    }


# ── Compatibility re-exports ──────────────────────────────────────
# These functions were refactored into interface/routes/ but existing tests
# and internal callers still import them from interface.server.

ChatRequest = chat_routes.ChatRequest
api_chat = chat_routes.api_chat
_foreground_timeout_for_lane = chat_routes._foreground_timeout_for_lane
_conversation_lane_user_message = chat_routes._conversation_lane_user_message
_log_exchange = chat_routes._log_exchange
api_action_log = subsystem_routes.api_action_log


async def api_health(request: Request):
    _sync_legacy_system_exports()
    return await system_routes.api_health(request)


async def api_ui_bootstrap(request: Request = None):
    _sync_legacy_system_exports()
    return await system_routes.api_ui_bootstrap(request)


async def api_memory_episodic(limit: int = 20, offset: int = 0):
    return await memory_routes.api_memory_episodic(limit=limit, offset=offset)


# ── WebSocket broadcaster ─────────────────────────────────────

async def _ws_broadcaster() -> None:
    """Forward messages from broadcast_bus to all WebSocket clients."""
    q = await broadcast_bus.subscribe()
    try:
        while not is_shutdown_requested():
            try:
                ptr, ts, msg = await asyncio.wait_for(q.get(), timeout=10.0)

                if ws_manager.count() == 0:
                    q.task_done()
                    continue

                if isinstance(msg, str):
                    try:
                        msg = json.loads(msg)
                    except json.JSONDecodeError:
                        msg = {"type": "message", "content": msg}
                elif not isinstance(msg, dict):
                    msg = {"type": "message", "content": str(msg)}

                try:
                    await asyncio.wait_for(ws_manager.broadcast(msg), timeout=15.0)
                except TimeoutError:
                    logger.warning("WS Broadcaster timeout - serious delivery lag detected")

                q.task_done()
            except TimeoutError:
                continue  # Pulsing
            except asyncio.CancelledError:
                break
            except _SERVER_BOUNDARY_ERRORS as e:
                record_degradation('server', e)
                logger.error("WebSocket broadcaster error: %s", e)
                await asyncio.sleep(1.0)
    finally:
        await broadcast_bus.unsubscribe(q)


# ── Routes — UI ───────────────────────────────────────────────

from interface.auth import _require_internal


@app.get("/", include_in_schema=False)
async def serve_ui(request: Request):
    """Main entry point for the Sovereign HUD."""
    _require_internal(request)
    ui = LEGACY_UI_INDEX if LEGACY_UI_INDEX.exists() else (SHELL_DIST_DIR / "index.html")
    if not ui.exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="UI not built")
    return FileResponse(str(ui), headers=NO_CACHE_HEADERS)


@app.get("/telemetry", include_in_schema=False)
async def serve_telemetry(request: Request):
    _require_internal(request)
    p = STATIC_DIR / "telemetry.html"
    return FileResponse(str(p), headers=NO_CACHE_HEADERS) if p.exists() else ORJSONResponse({"error": "not found"}, status_code=404)

# ── Routes — Checkpoints (Phase 5A) ───────────────────────────

@app.post("/api/checkpoints/save", tags=["checkpoints"])
async def save_checkpoint(request: Request):
    """Manually trigger a conversation checkpoint save."""
    _require_internal(request)
    data = await request.json()
    
    label = data.get("label", "manual")
    # In a full integration, these states would be pulled from the active KernelInterface
    messages = data.get("messages", [])
    
    filepath = checkpoint_service.save(
        messages=messages,
        label=label
    )
    if filepath:
        return {"ok": True, "filepath": filepath}
    return JSONResponse(status_code=500, content={"ok": False, "error": "Save failed"})

@app.post("/api/checkpoints/restore", tags=["checkpoints"])
async def restore_checkpoint(request: Request):
    """Restore conversation from a checkpoint."""
    _require_internal(request)
    data = await request.json()
    
    label = data.get("label")
    if label:
        cp = checkpoint_service.restore_by_label(label)
    else:
        cp = checkpoint_service.restore_latest()
        
    if cp:
        # Here we would inject the state back into the KernelInterface
        return {"ok": True, "turn_count": cp.turn_count, "messages": len(cp.messages)}
    return JSONResponse(status_code=404, content={"ok": False, "error": "Checkpoint not found"})

# ── Routes — WebSocket ────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)

    expected = os.environ.get("AURA_API_TOKEN", "")
    host = ws.client.host if ws.client else "unknown"
    is_local = host in ("127.0.0.1", "::1", "localhost")

    authenticated = not bool(expected) or is_local
    auth_timeout = 5.0

    try:
        if not authenticated:
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=auth_timeout)
                data = json.loads(raw)
                if data.get("type") == "auth" and hmac.compare_digest(data.get("token", ""), expected):
                    authenticated = True
                    await ws.send_text(json.dumps({"type": "auth_success"}))
                else:
                    await ws.send_text(json.dumps({"type": "error", "message": "Unauthorized"}))
                    await ws.close(code=4001, reason="Unauthorized")
                    return
            except TimeoutError:
                await ws.close(code=4001, reason="Auth Timeout")
                return
            except json.JSONDecodeError:
                await ws.close(code=4001, reason="Invalid Auth Payload")
                return
        elif is_local and expected:
            await ws.send_text(json.dumps({"type": "auth_success", "note": "local_trust"}))

        while not is_shutdown_requested():
            msg = await ws.receive()

            if msg.get("type") == "websocket.disconnect":
                break

            if "text" in msg:
                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    await ws.send_text(json.dumps({"type": "error", "message": "Invalid JSON"}))
                    continue

                msg_type = data.get("type")
                if msg_type == "user_message":
                    content = data.get("content", "")
                    if content:
                        logger.debug("WS: Received user_message: %s", content[:100])
                        _notify_user_spoke(content)

                        async def _handle_ws_message(ws_ref, user_content: str):
                            """Process user message and send response back via WebSocket."""
                            try:
                                reply = await chat_routes._run_cognitive_engine_chat_turn(
                                    user_content,
                                    visible_user_message=user_content,
                                    origin="desktop-ui",
                                    timeout_s=300.0,
                                    lane=chat_routes._collect_conversation_lane_status(),
                                    source="desktop_websocket",
                                    require_engine=True,
                                )
                                
                                # [FIX] WebSocket path fallback if CognitiveEngine fails
                                if not reply:
                                    logger.warning(
                                        "🔧 [FALLBACK] WebSocket CognitiveEngine produced no reply. Attempting graceful fallback.",
                                    )
                                    from core.kernel.kernel_interface import KernelInterface
                                    ki = KernelInterface.get_instance()
                                    if ki.is_ready():
                                        logger.info("[FALLBACK] Trying KernelInterface as fallback for WebSocket...")
                                        try:
                                            reply = await asyncio.wait_for(
                                                ki.process(user_content, origin="desktop-ui", priority=True),
                                                timeout=120.0,
                                            )
                                            if reply:
                                                logger.info("[FALLBACK] KernelInterface recovered WebSocket reply (len=%d).", len(reply))
                                        except TimeoutError:
                                            logger.warning("[FALLBACK] KernelInterface fallback timed out for WebSocket.")
                                        except _SERVER_BOUNDARY_ERRORS as e:
                                            record_degradation('server', e)
                                            logger.warning("[FALLBACK] KernelInterface fallback failed for WebSocket: %s", e)
                                    
                                    # If still no reply, try orchestrator
                                    if not reply:
                                        logger.info("[FALLBACK] Trying orchestrator as final fallback for WebSocket...")
                                        try:
                                            from core.container import ServiceContainer
                                            orch = ServiceContainer.get("orchestrator", default=None)
                                            if orch:
                                                reply = await asyncio.wait_for(
                                                    orch.process_user_input_priority(user_content, origin="desktop-ui", timeout_sec=120.0),
                                                    timeout=120.0,
                                                )
                                                if reply:
                                                    logger.info("[FALLBACK] Orchestrator recovered WebSocket reply (len=%d).", len(reply))
                                        except TimeoutError:
                                            logger.warning("[FALLBACK] Orchestrator fallback timed out for WebSocket.")
                                        except _SERVER_BOUNDARY_ERRORS as e:
                                            record_degradation('server', e)
                                            logger.warning("[FALLBACK] Orchestrator fallback failed for WebSocket: %s", e)
                                
                                if not reply:
                                    await ws_ref.send_text(json.dumps({
                                        "type": "aura_message",
                                        "content": (
                                            "The desktop WebSocket chat path requires CognitiveEngine, and it did not "
                                            "return a clean reply. I refused the legacy fallback so this surface cannot "
                                            "display an incoherent answer."
                                        ),
                                        "status": "desktop_cognitive_engine_unavailable",
                                    }))
                                    return

                                reply = await chat_routes._stabilize_user_facing_reply(
                                    user_content,
                                    reply,
                                )
                                await ws_ref.send_text(json.dumps({
                                    "type": "aura_message",
                                    "content": reply,
                                }))
                            except TimeoutError:
                                logger.error("WS: live CognitiveEngine processing timed out")
                                await ws_ref.send_text(json.dumps({
                                    "type": "aura_message",
                                    "content": "The live reasoning lane exceeded its timeout. I logged the timeout and preserved this turn instead of fabricating a recovered answer.",
                                }))
                            except _SERVER_BOUNDARY_ERRORS as e:
                                record_degradation('server', e)
                                logger.error("WS: Message handling failed: %s (%s)", type(e).__name__, e, exc_info=True)
                                await ws_ref.send_text(json.dumps({
                                    "type": "aura_message",
                                    "content": "The live message handler failed before a coherent answer formed. I logged the failure with the current turn context.",
                                }))

                        _spawn_server_bounded_task(
                            _handle_ws_message(ws, content),
                            name="server.ws.handle_message",
                        )
                elif msg_type == "ping":
                    await ws.send_text(json.dumps(runtime_heartbeat_payload("pong")))

            elif "bytes" in msg:
                if _voice_engine_fn:
                    ve = _voice_engine_fn()
                    if ve:
                        _spawn_server_bounded_task(
                            ve.feed_chunk(msg["bytes"]),
                            name="server.ws.feed_chunk",
                        )

    except WebSocketDisconnect as _exc:
        logger.debug("Suppressed WebSocketDisconnect: %s", _exc)
    except _SERVER_BOUNDARY_ERRORS as exc:
        record_degradation('server', exc)
        logger.debug("WS error: %s", exc)
    finally:
        await ws_manager.disconnect(ws)


# ── SPA Catch-all — v6.0 Traverse Hardened ────────────────────

@app.get("/{path:path}", include_in_schema=False)
async def spa_catchall(path: str, request: Request):
    """Secure catch-all to support SPA routing and static resolution with traversal protection."""
    _require_internal(request)

    if ".." in path or path.startswith("/") or "./" in path:
         fallback = LEGACY_UI_INDEX if LEGACY_UI_INDEX.exists() else (SHELL_DIST_DIR / "index.html")
         return FileResponse(str(fallback), headers=NO_CACHE_HEADERS)

    if path == "memory" or path.startswith("memory/"):
        dist_dir = STATIC_DIR / "memory" / "dist"
        if path == "memory":
            return FileResponse(str(dist_dir / "index.html"), headers=NO_CACHE_HEADERS)
        sub_path = path[len("memory/"):]
        if not sub_path:
            return FileResponse(str(dist_dir / "index.html"), headers=NO_CACHE_HEADERS)
        requested_path = (dist_dir / sub_path).resolve()
        if requested_path.is_file():
            return FileResponse(str(requested_path), headers=NO_CACHE_HEADERS)
        raw_path = (STATIC_DIR / "memory" / sub_path).resolve()
        if raw_path.is_file():
             return FileResponse(str(raw_path), headers=NO_CACHE_HEADERS)
        return FileResponse(str(dist_dir / "index.html"), headers=NO_CACHE_HEADERS)

    if path == "shell" or path.startswith("shell/"):
        if LEGACY_UI_INDEX.exists() and not _react_shell_enabled():
            return FileResponse(str(LEGACY_UI_INDEX), headers=NO_CACHE_HEADERS)
        dist_dir = SHELL_DIST_DIR
        if path == "shell":
            return FileResponse(str(dist_dir / "index.html"), headers=NO_CACHE_HEADERS)
        sub_path = path[len("shell/"):]
        requested_shell_path = (dist_dir / sub_path).resolve()
        if requested_shell_path.is_file():
            return FileResponse(str(requested_shell_path), headers=NO_CACHE_HEADERS)
        return FileResponse(str(dist_dir / "index.html"), headers=NO_CACHE_HEADERS)

    requested_path = (STATIC_DIR / path).resolve()

    if not str(requested_path).startswith(str(STATIC_DIR)) or not requested_path.exists():
         fallback = LEGACY_UI_INDEX if LEGACY_UI_INDEX.exists() else (SHELL_DIST_DIR / "index.html")
         return FileResponse(str(fallback), headers=NO_CACHE_HEADERS)

    if requested_path.is_file():
        return FileResponse(str(requested_path), headers=NO_CACHE_HEADERS)

    fallback = LEGACY_UI_INDEX if LEGACY_UI_INDEX.exists() else (SHELL_DIST_DIR / "index.html")
    return FileResponse(str(fallback), headers=NO_CACHE_HEADERS)


# ── Entry-point ───────────────────────────────────────────────

def main() -> None:
    from core.logging_config import setup_logging as _sl
    _sl(log_dir=config.paths.log_dir)

    host = "127.0.0.1" if config.security.internal_only_mode else "0.0.0.0"
    logger.info("Binding to %s:8000", host)

    uvicorn.run(
        "interface.server:app",
        host=host,
        port=8000,
        reload=False,
        log_level="warning",
        ws_ping_interval=20,
        ws_ping_timeout=10,
    )


if __name__ == "__main__":
    main()
