import json
import logging
import logging.handlers
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Pattern, Union, Optional
import structlog
from structlog.dev import ConsoleRenderer

# ── Redaction Patterns ─────────────────────────────────────────

_REDACT_PATTERNS: list[tuple[Pattern[str], str]] = [
    (re.compile(r'(sk-[A-Za-z0-9\-_]{20,})', re.IGNORECASE), "[REDACTED_API_KEY]"),
    (re.compile(r'(Bearer\s+)[A-Za-z0-9\-_\.=]{10,}', re.IGNORECASE), r"\1[REDACTED_BEARER]"),
    (re.compile(r'(password["\s:=]+)[^\s"\']+', re.IGNORECASE), r"\1[REDACTED_PASS]"),
    (re.compile(r'(token["\s:=]+)[^\s"\']+', re.IGNORECASE), r"\1[REDACTED_TOKEN]"),
]

def redact_text(text: str) -> str:
    """Apply every redaction pattern to a rendered log line."""
    for pattern, replacement in _REDACT_PATTERNS:
        text = pattern.sub(replacement, text)
    return text

def _redact_processor(_: Any, __: Any, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Structlog processor to redact sensitive patterns in the event dict."""
    for key, value in event_dict.items():
        if isinstance(value, str):
            for pattern, replacement in _REDACT_PATTERNS:
                event_dict[key] = pattern.sub(replacement, event_dict[key])
    return event_dict


class JsonLineFormatter(logging.Formatter):
    """Render every record — structlog or plain stdlib — as one redacted JSON object per line.

    Structlog events arrive pre-rendered as JSON strings and pass through with
    logger/level/timestamp back-filled; anything else (third-party libraries,
    bare ``logging`` calls) is wrapped in the same envelope so the file sink
    stays machine-parseable end to end.
    """

    def format(self, record: logging.LogRecord) -> str:
        try:
            message = record.getMessage()
        except (TypeError, ValueError):
            message = str(record.msg)
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)

        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        payload: dict[str, Any] | None = None
        if message.startswith("{"):
            try:
                parsed = json.loads(message)
                if isinstance(parsed, dict):
                    payload = parsed
            except ValueError:
                payload = None
        if payload is None:
            payload = {"event": message}
        payload.setdefault("logger", record.name)
        payload.setdefault("level", record.levelname.lower())
        payload.setdefault("timestamp", timestamp)
        if record.exc_text:
            payload.setdefault("exc_info", record.exc_text)

        try:
            line = json.dumps(payload, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            line = json.dumps({
                "event": message, "logger": record.name,
                "level": record.levelname.lower(), "timestamp": timestamp,
            }, ensure_ascii=False, default=str)
        return redact_text(line)

# ── Main Entry-Point ─────────────────────────────────────────

_initialised: bool = False


def _resolve_log_dir(log_dir: Optional[Path]) -> Path:
    """Explicit argument wins, then AURA_LOG_DIR (test/CI hermeticity), then ~/.aura/logs."""
    if log_dir is not None:
        return Path(log_dir)
    env_log_dir = os.environ.get("AURA_LOG_DIR")
    if env_log_dir:
        return Path(env_log_dir)
    return Path.home() / ".aura" / "logs"

def setup_logging(
    name: str = "Aura",
    level: Union[str, int] = logging.INFO,
    log_dir: Optional[Path] = None,
    max_bytes: int = 100 * 1024 * 1024, # 100MB
    backup_count: int = 10,
) -> Any:
    """Configure structured logging and return a bound logger."""
    global _initialised
    
    if _initialised:
        return structlog.get_logger(name)

    # 1. Stdlib handlers for local file backup (structured JSON)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    log_dir = _resolve_log_dir(log_dir)

    file_handler = None
    for candidate in (Path(log_dir), Path(tempfile.gettempdir()) / "aura-logs"):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                candidate / "aura_json.log",
                maxBytes=max_bytes,
                backupCount=backup_count,
            )
            break
        except OSError:
            continue

    if file_handler is not None:
        file_handler.setFormatter(JsonLineFormatter())
        handlers.append(file_handler)

    # 2. Configure stdlib logging bridge
    root_logger = logging.getLogger()
    
    # If handlers already exist, we might be in a re-init or partial init.
    # Clear existing handlers to ensure our configuration is the single source of truth.
    if root_logger.handlers:
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)

    # Use basicConfig or manual handler addition to root
    for h in handlers:
        root_logger.addHandler(h)
    root_logger.setLevel(level)

    # 3. Structlog configuration
    from core.config import Environment, config
    
    # Zenith HUD consumes JSON, but developers prefer human-readable console output
    is_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    
    # Force JSON if explicitly requested or if we are in a production/silent environment
    if os.environ.get("AURA_LOG_JSON") == "1":
        renderer = structlog.processors.JSONRenderer()
    elif config.env == Environment.DEV and is_tty:
        renderer = ConsoleRenderer(colors=True)
    elif is_tty:
        renderer = ConsoleRenderer(colors=False) # Human-readable but no escape codes
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            _redact_processor,
            renderer
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Silence noisy libs
    for noisy in ("httpx", "httpcore", "urllib3", "asyncio", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if file_handler is None:
        logging.getLogger("Aura.Logging").warning(
            "File logging unavailable; continuing with stdout-only logging."
        )

    _initialised = True
    return structlog.get_logger(name)

def get_logger(name: str) -> Any:
    """Return a module-level bound logger."""
    return structlog.get_logger(name)
