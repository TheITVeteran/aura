import logging
import os
import queue

from core.runtime.errors import record_degradation

# Configure logging for the worker
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("core.senses.sensory_worker")
_SENSORY_INIT_ERRORS = (ImportError, AttributeError, RuntimeError, OSError)
_SENSORY_COMMAND_ERRORS = (AttributeError, RuntimeError, TypeError, ValueError, OSError)
_SENSORY_QUEUE_ERRORS = (BrokenPipeError, EOFError, OSError, RuntimeError)


def _screen_capture_preflight_allowed() -> bool:
    """Avoid triggering macOS prompts from the isolated worker."""
    try:
        import Quartz  # type: ignore

        preflight = getattr(Quartz, "CGPreflightScreenCaptureAccess", None)
        if callable(preflight):
            return bool(preflight())
    except (ImportError, AttributeError, RuntimeError) as exc:
        record_degradation('sensory_worker', exc)
        logger.debug("Sensory worker Quartz preflight unavailable: %s", exc)
    return os.getenv("AURA_ASSUME_SCREEN_PERMISSION", "0") == "1"

def sensory_worker_loop(request_queue, response_queue):
    """
    Isolated process for Vision and Audio capturing.
    Prevents cv2/sounddevice memory corruption from taking down the brain.
    """
    logger.info("[SENSORY] Isolated Worker started (PID: %d)", os.getpid())
    
    mss = None
    
    running = True
    while running:
        try:
            # Issue 30: Reduce polling latency from 1.0s to 0.1s
            req = request_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        except KeyboardInterrupt:
            break
        except _SENSORY_QUEUE_ERRORS as exc:
            record_degradation("sensory_worker", exc)
            logger.error("Sensory worker queue receive failed: %s", exc)
            break

        try:
            if not req:
                break

            cmd = req.get("command")

            if cmd == "init_vision":
                try:
                    if not _screen_capture_preflight_allowed():
                        _safe_put(response_queue, {"status": "error", "msg": "screen_permission_inactive"})
                        continue
                    import cv2 as _cv2
                    import mss as _mss
                    if not getattr(_cv2, "__version__", None):
                        raise RuntimeError("cv2 module missing version metadata")
                    mss = _mss
                    _safe_put(response_queue, {"status": "ok"})
                except _SENSORY_INIT_ERRORS as e:
                    record_degradation('sensory_worker', e)
                    _safe_put(response_queue, {"status": "error", "msg": str(e)})

            elif cmd == "capture_screen":
                if not mss:
                    _safe_put(response_queue, {"status": "error", "msg": "Vision not init"})
                    continue
                if not _screen_capture_preflight_allowed():
                    _safe_put(response_queue, {"status": "error", "msg": "screen_permission_inactive"})
                    continue
                with mss.mss() as sct:
                    monitor = sct.monitors[1]
                    sct_img = sct.grab(monitor)
                    # Convert to minimal bytes to send over queue
                    _safe_put(response_queue, {"status": "ok", "data": bytes(sct_img.raw)})

            elif cmd == "init_audio":
                try:
                    import sounddevice as _sd
                    if not callable(getattr(_sd, "query_devices", None)):
                        raise RuntimeError("sounddevice query_devices unavailable")
                    _safe_put(response_queue, {"status": "ok"})
                except _SENSORY_INIT_ERRORS as e:
                    record_degradation('sensory_worker', e)
                    _safe_put(response_queue, {"status": "error", "msg": str(e)})

            elif cmd == "ping":
                _safe_put(response_queue, {"status": "ok", "msg": "pong"})

            elif cmd == "exit":
                running = False
            else:
                _safe_put(response_queue, {"status": "error", "msg": f"Unknown command: {cmd}"})

        except _SENSORY_COMMAND_ERRORS as e:
            record_degradation('sensory_worker', e)
            logger.error("Sensory worker command failed: %s", e)
            _safe_put(response_queue, {"status": "error", "msg": str(e)})


def _safe_put(response_queue, payload: dict[str, object]) -> bool:
    try:
        response_queue.put(payload)
        return True
    except _SENSORY_QUEUE_ERRORS as exc:
        record_degradation("sensory_worker", exc)
        logger.error("Sensory worker response send failed: %s", exc)
        return False

if __name__ == "__main__":
    # This worker is intended to be started via mp.Process from the client
    ...
