import logging

# Issue 30: Unused dead imports removed
from core.runtime.service_registry import get_runtime_service
from core.runtime.shutdown_coordinator import is_shutdown_requested
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Senses.Mouth")

class FastMouth:
    """Shim for backward compatibility."""
    def __init__(self):
        self.engine = get_runtime_service("voice_engine", default=None)
        self._speak_task = None
        self._stream_task = None
    
    def speak(self, text: str):
        if self.engine:
            if is_shutdown_requested():
                return
            try:
                # Cancel previous task to prevent pile-up
                if self._speak_task and not self._speak_task.done():
                    self._speak_task.cancel()
                self._speak_task = get_task_tracker().create_task(
                    self.engine.speak(text),
                    name="tts_stream.speak",
                )
            except RuntimeError as _e:
                logger.debug('Ignored RuntimeError in tts_stream.py: %s', _e)

    def speak_stream(self, text_generator):
        if self.engine:
            if is_shutdown_requested():
                return
            try:
                if self._stream_task and not self._stream_task.done():
                    self._stream_task.cancel()
                self._stream_task = get_task_tracker().create_task(
                    self.engine.speak_stream(text_generator),
                    name="tts_stream.speak_stream",
                )
            except RuntimeError as _e:
                logger.debug('Ignored RuntimeError in tts_stream.py: %s', _e)

    def stop(self):
        for task in (self._speak_task, self._stream_task):
            if task and not task.done():
                task.cancel()
