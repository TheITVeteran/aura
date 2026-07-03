"""Low-level service metadata bridge for runtime modules.

This module deliberately does not import ``core.container``. The container may
install a read-only resolver here, allowing low-level runtime sinks to inspect
service metadata without creating upward dependency cycles.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

_Resolver = Callable[[str], str | None]
_ServiceResolver = Callable[[str, object | None], object | None]
_ServicePresenceResolver = Callable[[str], bool]
_RuntimeFlagResolver = Callable[[], bool]
_MetricCounterSink = Callable[[str, int], None]
_FileWriteBytesSink = Callable[[object, bytes, str], None]
_FileWriteTextSink = Callable[[object, str, str, str], None]
_resolver_lock = threading.Lock()
_failure_policy_resolver: _Resolver | None = None
_service_resolver: _ServiceResolver | None = None
_service_presence_resolver: _ServicePresenceResolver | None = None
_registration_locked_resolver: _RuntimeFlagResolver | None = None
_metric_counter_sink: _MetricCounterSink | None = None
_file_write_bytes_sink: _FileWriteBytesSink | None = None
_file_write_text_sink: _FileWriteTextSink | None = None


def install_failure_policy_resolver(resolver: _Resolver | None) -> None:
    """Install or clear the process-local service failure-policy resolver."""
    global _failure_policy_resolver
    with _resolver_lock:
        _failure_policy_resolver = resolver


def install_service_resolver(resolver: _ServiceResolver | None) -> None:
    """Install or clear the process-local service instance resolver."""
    global _service_resolver
    with _resolver_lock:
        _service_resolver = resolver


def install_service_presence_resolver(resolver: _ServicePresenceResolver | None) -> None:
    """Install or clear the process-local service presence resolver."""
    global _service_presence_resolver
    with _resolver_lock:
        _service_presence_resolver = resolver


def install_registration_locked_resolver(resolver: _RuntimeFlagResolver | None) -> None:
    """Install or clear the process-local registration-lock resolver."""
    global _registration_locked_resolver
    with _resolver_lock:
        _registration_locked_resolver = resolver


def install_metric_counter_sink(sink: _MetricCounterSink | None) -> None:
    """Install or clear a low-level metric counter sink."""
    global _metric_counter_sink
    with _resolver_lock:
        _metric_counter_sink = sink


def install_file_write_sinks(
    *,
    write_bytes: _FileWriteBytesSink | None = None,
    write_text: _FileWriteTextSink | None = None,
) -> None:
    """Install or clear low-level file-write sinks."""
    global _file_write_bytes_sink, _file_write_text_sink
    with _resolver_lock:
        _file_write_bytes_sink = write_bytes
        _file_write_text_sink = write_text


def get_service_failure_policy(service_name: str) -> str | None:
    """Return the configured failure policy for a service, if registered."""
    with _resolver_lock:
        resolver = _failure_policy_resolver
    if resolver is None:
        return None
    return resolver(service_name)


def get_runtime_service(service_name: str, default: object | None = None) -> object | None:
    """Return a registered runtime service without importing the container."""
    with _resolver_lock:
        resolver = _service_resolver
    if resolver is None:
        return default
    return resolver(service_name, default)


def has_runtime_service(service_name: str) -> bool:
    """Return True when a runtime service is registered, without instantiating it."""
    with _resolver_lock:
        resolver = _service_presence_resolver
    if resolver is not None:
        return bool(resolver(service_name))
    return get_runtime_service(service_name, default=None) is not None


def is_runtime_registration_locked() -> bool:
    """Return True when the live runtime service registry is locked."""
    with _resolver_lock:
        resolver = _registration_locked_resolver
    if resolver is None:
        return False
    return bool(resolver())


def increment_runtime_counter(name: str, amount: int = 1) -> bool:
    """Increment a runtime metric counter without importing observability."""
    with _resolver_lock:
        sink = _metric_counter_sink
    if sink is not None:
        sink(name, int(amount))
        return True

    for service_name in ("metrics", "metrics_collector"):
        metrics = get_runtime_service(service_name, default=None)
        increment = getattr(metrics, "increment_counter", None)
        if callable(increment):
            increment(name, int(amount))
            return True
    return False


def runtime_write_bytes(path: object, payload: bytes, *, source: str = "unknown") -> None:
    """Write bytes through the live file gateway when available, else atomically."""
    with _resolver_lock:
        sink = _file_write_bytes_sink
    if sink is not None:
        sink(path, bytes(payload), source)
        return

    from core.runtime.atomic_writer import atomic_write_bytes

    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(target, bytes(payload))


def runtime_write_text(
    path: object,
    text: str,
    *,
    encoding: str = "utf-8",
    source: str = "unknown",
) -> None:
    """Write text through the live file gateway when available, else atomically."""
    with _resolver_lock:
        sink = _file_write_text_sink
    if sink is not None:
        sink(path, str(text), encoding, source)
        return

    from core.runtime.atomic_writer import atomic_write_text

    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(target, str(text), encoding=encoding)
