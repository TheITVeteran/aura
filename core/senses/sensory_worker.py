import logging
import os
import queue

from core.runtime.errors import record_degradation
from core.security.screen_capture_policy import evaluate_screen_capture_admission

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
        return True
    return os.getenv("AURA_ASSUME_SCREEN_PERMISSION", "0") == "1"

def sensory_worker_loop(request_queue, response_queue):
    """
    Isolated process for Vision and Audio capturing.
    Prevents cv2/sounddevice memory corruption from taking down the brain.
    """
    os.environ["AURA_MEDIA_SIDECAR_PROCESS"] = "1"
    logger.info("[SENSORY] Isolated Worker started (PID: %d)", os.getpid())
    
    mss = None
    # The camera handle lives for the life of the worker, not per frame.
    # Reopening the device for every grab is what makes the macOS camera
    # indicator strobe and adds ~200ms to each read.
    camera = None

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
                    admission = evaluate_screen_capture_admission()
                    if not admission.allowed:
                        _safe_put(
                            response_queue,
                            {
                                "status": "error",
                                "msg": admission.reason.value,
                                "capture_admission": admission.to_receipt(),
                            },
                        )
                        continue
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
                admission = evaluate_screen_capture_admission()
                if not admission.allowed:
                    _safe_put(
                        response_queue,
                        {
                            "status": "error",
                            "msg": admission.reason.value,
                            "capture_admission": admission.to_receipt(),
                        },
                    )
                    continue
                with mss.mss() as sct:
                    monitor = sct.monitors[1]
                    sct_img = sct.grab(monitor)
                    # Convert to minimal bytes to send over queue
                    _safe_put(response_queue, {"status": "ok", "data": bytes(sct_img.raw)})

            # ── camera ────────────────────────────────────────────────
            #
            # `core/media/safe_imports.py` has always said "the sensory
            # worker is the production process boundary for camera/screen
            # media libraries" and "the sidecar sensory worker remains the
            # production camera path". It handled init_vision,
            # capture_screen, init_audio, ping and exit — the SCREEN, never
            # the camera. Meanwhile the main process is forbidden from
            # importing cv2 on macOS by an import guard that raises.
            #
            # So the camera could not be opened anywhere: not in the main
            # process, which is banned, and not in the sidecar, which had no
            # command for it. The documented production path did not exist,
            # and the claim outlived nothing — it was never true.
            elif cmd == "camera_open":
                try:
                    import cv2 as _cv2

                    index = int((req.get("data") or {}).get("index", 0))
                    if camera is not None:
                        camera.release()
                        camera = None
                    candidate = _cv2.VideoCapture(index)
                    if candidate is None or not candidate.isOpened():
                        if candidate is not None:
                            candidate.release()
                        _safe_put(
                            response_queue,
                            {"status": "error", "msg": "camera_would_not_open"},
                        )
                        continue
                    camera = candidate
                    _safe_put(response_queue, {"status": "ok"})
                except _SENSORY_INIT_ERRORS as e:
                    record_degradation('sensory_worker', e)
                    _safe_put(response_queue, {"status": "error", "msg": str(e)})

            elif cmd == "camera_frame":
                if camera is None:
                    _safe_put(
                        response_queue,
                        {"status": "error", "msg": "camera_not_open"},
                    )
                    continue
                try:
                    import cv2 as _cv2

                    ok, frame = camera.read()
                    if not ok or frame is None:
                        _safe_put(
                            response_queue, {"status": "error", "msg": "no_frame"}
                        )
                        continue
                    # JPEG rather than the raw array: a 1080p BGR frame is
                    # ~6MB and this crosses a multiprocessing queue every
                    # couple of seconds. Encoding here also keeps numpy out
                    # of the pickle, which is what the process boundary
                    # exists to avoid.
                    encoded, buffer = _cv2.imencode(".jpg", frame)
                    if not encoded:
                        _safe_put(
                            response_queue,
                            {"status": "error", "msg": "encode_failed"},
                        )
                        continue
                    height, width = frame.shape[:2]
                    _safe_put(
                        response_queue,
                        {
                            "status": "ok",
                            "data": buffer.tobytes(),
                            "width": int(width),
                            "height": int(height),
                        },
                    )
                except _SENSORY_COMMAND_ERRORS as e:
                    record_degradation('sensory_worker', e)
                    _safe_put(response_queue, {"status": "error", "msg": str(e)})

            elif cmd == "camera_close":
                if camera is not None:
                    try:
                        camera.release()
                    except _SENSORY_COMMAND_ERRORS as e:
                        record_degradation('sensory_worker', e)
                    camera = None
                _safe_put(response_queue, {"status": "ok"})

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
                # Release the device before the process goes away, so the
                # camera light goes out on shutdown rather than at whatever
                # moment the OS reclaims the handle.
                if camera is not None:
                    try:
                        camera.release()
                    except _SENSORY_COMMAND_ERRORS as e:
                        record_degradation('sensory_worker', e)
                    camera = None
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
