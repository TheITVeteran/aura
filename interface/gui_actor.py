import logging
import json
import os
import site
import sys
import threading
import time
from pathlib import Path
from typing import Any

# Setup Path Resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime.errors import record_degradation
from core.runtime.health_contract import REQUIRED_HEALTH_PROBE_GROUPS
from core.runtime.network_gateway import get_network_gateway


def _inject_project_venv_site_packages() -> None:
    """Mirror aura_main.py so GUI subprocesses see venv-installed deps."""
    venv_path = PROJECT_ROOT / ".venv"
    if not venv_path.exists():
        venv_path = PROJECT_ROOT / ".venv_aura"
    if not venv_path.exists():
        return

    lib_dir = venv_path / "lib"
    if not lib_dir.exists():
        return

    current_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    for py_dir in lib_dir.glob("python3.*"):
        if py_dir.name != current_version:
            continue
        site_packages = py_dir / "site-packages"
        if site_packages.exists() and str(site_packages) not in sys.path:
            sys.path.insert(0, str(site_packages))
            site.addsitedir(str(site_packages))


_inject_project_venv_site_packages()

logger = logging.getLogger("Aura.GUI")

_GUI_RECOVERABLE_ERRORS = (
    ImportError,
    RuntimeError,
    OSError,
    ValueError,
)

def _flush_logs_before_forced_exit() -> None:
    try:
        logging.shutdown()
    except (RuntimeError, OSError):
        pass


def _heartbeat_response_healthy(resp: Any) -> bool:
    """Accept only the canonical runtime readiness heartbeat as healthy."""
    if getattr(resp, "status_code", None) != 200:
        return False
    try:
        payload = resp.json()
    except (AttributeError, TypeError, ValueError):
        return False
    if not bool(payload.get("healthy") is True and payload.get("status") == "healthy"):
        return False
    probes = payload.get("required_probes")
    if not isinstance(probes, dict) or not bool(probes.get("all_passed", False)):
        return False
    for group, expected_components in REQUIRED_HEALTH_PROBE_GROUPS.items():
        probe = probes.get(group)
        if not isinstance(probe, dict) or not bool(probe.get("ok", False)):
            return False
        components = probe.get("components")
        if not isinstance(components, dict):
            return False
        if any(components.get(component) is not True for component in expected_components):
            return False
    return True


def _gateway_heartbeat_healthy(response: dict[str, Any]) -> bool:
    content = response.get("content") or b""
    text = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else str(content)
    try:
        payload = json.loads(text or "{}")
    except json.JSONDecodeError:
        return False

    class _Response:
        status_code = int(response.get("status_code") or 0)

        @staticmethod
        def json() -> dict[str, Any]:
            return payload

    return _heartbeat_response_healthy(_Response())


def gui_actor_entry(port: int, token: str = None):
    """Entry point for the GUI process."""
    logger.info("🚀 Aura GUI Actor starting on port %d...", port)
    # The intentional RuntimeError has been removed to allow boot.
    
    # 1. Standardize Environment for macOS/WebKit
    if sys.platform == "darwin":
        os.environ["OPENCV_VIDEOIO_AVFOUNDATION_USE_FRAME_RECEIVER"] = "0"
        os.environ["PYAV_SKIP_AVF_FRAME_RECEIVER"] = "1"
        os.environ["WEBKIT_DISABLE_COMPOSITING_MODE"] = "1"
    
    # 1.5 Set Proxy Mode Flag (though mostly unused now)
    os.environ["AURA_GUI_PROXY"] = "1"

    # 2. Setup Logging for the process
    from core.config import config
    from core.logging_config import setup_logging
    setup_logging(log_dir=config.paths.log_dir)
    
    logger.info(f"🎨 GUI Actor initiating Pure WebView (Port: {port})")

    # 4. Launch webview
    try:
        import webview

        from core.utils.port_check import wait_for_port
        
        app_url = f"http://127.0.0.1:{port}"
        
        # Wait for the Kernel API to be ready
        # Increased to 60.0s for slow Silicon model loads
        if wait_for_port(port, timeout=60.0):
            logger.info(f"📡 API Server (Kernel) detected online on port {port}. Launching WebView...")
        else:
            logger.warning(f"⚠️ API Server (Kernel) NOT detected on port {port} after 60s. Attempting window creation anyway...")
        
        # No URL in create_window to prevent race conditions during startup
        window = webview.create_window(
            "Aura Zenith", 
            width=1280, height=820, min_size=(800, 600)
        )
        shutdown_event = threading.Event()

        # ISSUE #14 - window closure delay race condition
        def _on_closed():
            logger.info("🎨 Window closed. Forcing GUI process termination.")
            shutdown_event.set()
            _flush_logs_before_forced_exit()
            os._exit(0)
            
        window.events.closed += _on_closed

        def _on_shown():
            logger.info("🎨 WebView Window Shown. Initiating load...")
            time.sleep(1.0) # Grace period for WebKit
            try:
                window.load_url(app_url)
                logger.info(f"🔄 GUI Loaded: {app_url}")
            except _GUI_RECOVERABLE_ERRORS as e:
                record_degradation('gui_actor', e)
                logger.error(f"Failed to load URL in WebView: {e}")

        # Watchdog: Periodically check if the UI is responsive
        def _watchdog():
            logger.info("🐕 GUI Watchdog active.")
            consecutive_failures = 0
            while not shutdown_event.wait(20):
                try:
                    resp = get_network_gateway().request(
                        "GET",
                        f"{app_url}/api/health/heartbeat",
                        timeout=5,
                        source="gui_actor.watchdog",
                        read_only=True,
                    )
                    if _gateway_heartbeat_healthy(resp):
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                except (OSError, RuntimeError, TimeoutError, TypeError, ValueError):
                    consecutive_failures += 1
                
                if consecutive_failures >= 3:
                    logger.warning("🚨 [GUI WATCHDOG] Kernel API unreachable. Attempting reload.")
                    try:
                        window.load_url(app_url)
                    except _GUI_RECOVERABLE_ERRORS as _exc:
                        record_degradation('gui_actor', _exc)
                        logger.warning("GUI watchdog reload failed: %s", _exc)

                if consecutive_failures >= 6:
                    logger.critical("🛑 [GUI WATCHDOG] Kernel API unavailable for too long. Exiting stale WebView.")
                    _flush_logs_before_forced_exit()
                    os._exit(1)

        watchdog_thread = threading.Thread(target=_watchdog, daemon=True)
        watchdog_thread.start()

        # In Zenith, we use the functional start to load the URL after initialization
        webview.start(func=_on_shown, debug=False)
        
    except _GUI_RECOVERABLE_ERRORS as e:
        record_degradation('gui_actor', e)
        logger.error(f"❌ WebView Failure: {e}")
        time.sleep(5)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        gui_actor_entry(int(sys.argv[1]))
    else:
        gui_actor_entry(8000)
