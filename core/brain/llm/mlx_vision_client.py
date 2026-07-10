from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import threading
import time
import uuid
from typing import Any

from core.runtime.shutdown_coordinator import is_shutdown_requested

from .mlx_vision_worker import _mlx_vision_worker_loop

logger = logging.getLogger("MLXVisionClient")

class MLXVisionClient:
    """
    Manages an isolated MLX vision model worker for multimodal inference.
    """
    
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._process: Any | None = None
        self._req_q: Any | None = None
        self._res_q: Any | None = None
        self._lock = threading.Lock()
        self._pending_requests: dict[str, dict[str, Any] | None] = {}
        self._listener_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._init_done = False

    @staticmethod
    def _safe_close_queue(queue_obj: Any) -> None:
        if queue_obj is None:
            return
        try:
            close = getattr(queue_obj, "close", None)
            if callable(close):
                close()
            join_thread = getattr(queue_obj, "join_thread", None)
            if callable(join_thread):
                join_thread()
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            logger.debug("Suppressed vision queue close error: %s", exc)

    def _close_queues(self) -> None:
        self._safe_close_queue(self._req_q)
        self._safe_close_queue(self._res_q)
        self._req_q = None
        self._res_q = None

    def _replace_queues(self, ctx: Any) -> None:
        self._close_queues()
        factory = ctx.Queue if hasattr(ctx, "Queue") else mp.Queue
        self._req_q = factory(maxsize=10)
        self._res_q = factory(maxsize=10)
        
    def start(self) -> bool:
        if is_shutdown_requested():
            logger.info("Vision worker start refused: runtime shutdown requested")
            return False
        with self._lock:
            if is_shutdown_requested():
                return False
            if self._process is not None and self._process.is_alive():
                return True
                
            logger.info("Starting MLX Vision Worker for %s", self.model_path)
            # Ensure spawn method for MLX Metal compatibility
            ctx: Any = mp.get_context("spawn") if hasattr(mp, "get_context") else mp
            self._stop_event.clear()
            self._pending_requests.clear()
            self._init_done = False
            self._replace_queues(ctx)
            
            process = ctx.Process(
                target=_mlx_vision_worker_loop,
                args=(self.model_path, self._req_q, self._res_q),
                daemon=True,
                name="MLX-Vision-Worker"
            )
            self._process = process
            if is_shutdown_requested():
                process, self._process = self._process, None
                self._close_queues()
                close = getattr(process, "close", None)
                if callable(close):
                    close()
                return False
            process.start()
            if is_shutdown_requested():
                logger.info("Vision worker crossed shutdown during spawn; terminating it")
                process.terminate()
                process.join(timeout=1.0)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1.0)
                self._process = None
                self._close_queues()
                return False
            try:
                from core.runtime.runtime_hygiene import get_runtime_hygiene

                get_runtime_hygiene().register_process_handle(
                    process,
                    kind="multiprocessing",
                    name=process.name,
                    source="mlx_vision_client.worker_owner",
                    command=f"MLX vision worker for {self.model_path}",
                )
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
                logger.debug("Vision worker runtime hygiene registration failed: %s", exc)
            
            self._listener_thread = threading.Thread(target=self._listener_loop, daemon=True)
            self._listener_thread.start()
            
            # Wait for init
            start_time = time.time()
            while time.time() - start_time < 30.0:
                if is_shutdown_requested():
                    self.stop()
                    return False
                if self._init_done:
                    return True
                time.sleep(0.1)
                
            logger.error("Vision worker failed to initialize within 30s")
            self.stop()
            return False

    def _listener_loop(self) -> None:
        while not self._stop_event.is_set() and not is_shutdown_requested():
            try:
                if self._res_q is None:
                    break
                msg: dict[str, Any] = self._res_q.get(timeout=1.0)
                status = msg.get("status")
                action = msg.get("action")
                
                if status == "heartbeat":
                    continue
                    
                if action == "init":
                    self._init_done = True
                    continue
                    
                req_id = msg.get("id")
                if req_id and req_id in self._pending_requests:
                    self._pending_requests[req_id] = msg
            except queue.Empty:
                continue
            except (OSError, ConnectionError, TimeoutError) as e:
                logger.error("Vision listener error: %s", e)

    def see(self, prompt: str, image_base64: str, max_tokens: int = 512, temp: float = 0.0) -> str:
        """
        Send a base64 image (with optional data:image prefix) and prompt to the vision model.
        Blocks until completion.
        """
        if not self.start():
            raise RuntimeError("Vision worker unavailable")
        if self._req_q is None:
            raise RuntimeError("Vision worker queue unavailable after start")
        
        req_id = str(uuid.uuid4())
        self._pending_requests[req_id] = None
        
        self._req_q.put({
            "id": req_id,
            "action": "see",
            "prompt": prompt,
            "image_base64": image_base64,
            "max_tokens": max_tokens,
            "temp": temp
        })
        
        # Wait for response
        while self._pending_requests[req_id] is None:
            if is_shutdown_requested():
                self.stop()
                raise RuntimeError("runtime_shutdown")
            time.sleep(0.1)
            if self._process and not self._process.is_alive():
                raise RuntimeError("Vision worker crashed during inference")
                
        resp = self._pending_requests.pop(req_id)
        if resp is None:  # pragma: no cover - loop exits only after response assignment
            raise RuntimeError("Vision worker returned no response payload")
        if resp.get("status") == "error":
            raise RuntimeError(f"Vision model error: {resp.get('message')}")
            
        return str(resp.get("response", ""))

    def stop(self) -> None:
        self._stop_event.set()
        if self._req_q:
            try:
                self._req_q.put(None)
            except (RuntimeError, AttributeError, TypeError, ValueError) as _exc:
                logger.debug("Suppressed %s in core.brain.llm.mlx_vision_client: %s", type(_exc).__name__, _exc)
        process, self._process = self._process, None
        if process:
            process.join(timeout=3.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)
        if self._listener_thread and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=2.0)
        self._listener_thread = None
        self._pending_requests.clear()
        self._init_done = False
        self._close_queues()
        logger.info("Vision worker stopped.")

    close = stop
    cleanup = stop
    on_stop = stop

    def __del__(self) -> None:
        try:
            self.stop()
        except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as _exc:
            logger.debug("Suppressed %s in core.brain.llm.mlx_vision_client: %s", type(_exc).__name__, _exc)
