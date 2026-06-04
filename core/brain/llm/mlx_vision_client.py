import logging
import multiprocessing as mp
import queue
import threading
import time
import uuid
from typing import Optional

from .mlx_vision_worker import _mlx_vision_worker_loop

logger = logging.getLogger("MLXVisionClient")

class MLXVisionClient:
    """
    Manages an isolated MLX vision model worker for multimodal inference.
    """
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._process: Optional[mp.Process] = None
        self._req_q = None
        self._res_q = None
        self._lock = threading.Lock()
        self._pending_requests = {}
        self._listener_thread = None
        self._stop_event = threading.Event()
        self._init_done = False

    @staticmethod
    def _safe_close_queue(queue_obj) -> None:
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

    def _replace_queues(self, ctx) -> None:
        self._close_queues()
        factory = ctx.Queue if hasattr(ctx, "Queue") else mp.Queue
        self._req_q = factory(maxsize=10)
        self._res_q = factory(maxsize=10)
        
    def start(self) -> None:
        with self._lock:
            if self._process is not None and self._process.is_alive():
                return
                
            logger.info("Starting MLX Vision Worker for %s", self.model_path)
            # Ensure spawn method for MLX Metal compatibility
            ctx = mp.get_context("spawn") if hasattr(mp, "get_context") else mp
            self._stop_event.clear()
            self._pending_requests.clear()
            self._init_done = False
            self._replace_queues(ctx)
            
            self._process = ctx.Process(
                target=_mlx_vision_worker_loop,
                args=(self.model_path, self._req_q, self._res_q),
                daemon=True,
                name="MLX-Vision-Worker"
            )
            self._process.start()
            
            self._listener_thread = threading.Thread(target=self._listener_loop, daemon=True)
            self._listener_thread.start()
            
            # Wait for init
            start_time = time.time()
            while time.time() - start_time < 30.0:
                if self._init_done:
                    break
                time.sleep(0.1)
                
            if not self._init_done:
                logger.error("Vision worker failed to initialize within 30s")

    def _listener_loop(self):
        while not self._stop_event.is_set():
            try:
                if self._res_q is None:
                    break
                msg = self._res_q.get(timeout=1.0)
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
        self.start()
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
            time.sleep(0.1)
            if self._process and not self._process.is_alive():
                raise RuntimeError("Vision worker crashed during inference")
                
        resp = self._pending_requests.pop(req_id)
        if resp.get("status") == "error":
            raise RuntimeError(f"Vision model error: {resp.get('message')}")
            
        return resp.get("response", "")

    def stop(self):
        self._stop_event.set()
        if self._req_q:
            try:
                self._req_q.put(None)
            except (RuntimeError, AttributeError, TypeError, ValueError) as _exc:
                logger.debug("Suppressed %s in core.brain.llm.mlx_vision_client: %s", type(_exc).__name__, _exc)
        if self._process:
            self._process.join(timeout=3.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
        if self._listener_thread and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=2.0)
        self._process = None
        self._listener_thread = None
        self._pending_requests.clear()
        self._init_done = False
        self._close_queues()
        logger.info("Vision worker stopped.")

    close = stop
    cleanup = stop
    on_stop = stop

    def __del__(self):
        try:
            self.stop()
        except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as _exc:
            logger.debug("Suppressed %s in core.brain.llm.mlx_vision_client: %s", type(_exc).__name__, _exc)
