"""core/runtime/network_gateway.py — Canonical Network Gateway.

All outbound network requests should flow through this module to ensure correct governance, logging, and audit.
"""
from __future__ import annotations

import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

from core.governance_context import (
    governance_runtime_active,
    require_governance,
)
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.NetworkGateway")
_NETWORK_DOMAINS = (
    "network_call",
    "cloud_call",
    "cloud_fallback",
    "tool_execution",
    "exploration",
)
_NETWORK_RECOVERABLE_ERRORS = (
    OSError,
    TimeoutError,
    TypeError,
    urllib.error.URLError,
    ValueError,
)
_HTTP_METHODS = {"DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"}


class NetworkGateway:
    """Single canonical owner for HTTP/Network requests."""

    def __init__(self) -> None:
        self._allowed_domains = _NETWORK_DOMAINS

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[dict[str, str]] = None,
        data: Optional[bytes | str] = None,
        timeout: float = 30.0,
        source: str = "unknown",
        read_only: bool = False,
    ) -> dict[str, Any]:
        """Perform a synchronous HTTP request."""
        method_text = _coerce_method(method)
        url_text = _coerce_url(url)
        timeout_s = _coerce_timeout(timeout)
        request_headers = _coerce_headers(headers)
        request_data = _coerce_data(data)

        if not read_only and governance_runtime_active():
            require_governance(
                f"network_gateway.request:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )

        req = urllib.request.Request(
            url_text,
            data=request_data,
            headers=request_headers,
            method=method_text,
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as response:
                return {
                    "status_code": response.status,
                    "headers": dict(response.info()),
                    "content": response.read(),
                    "url": response.url,
                    "ok": True,
                }
        except urllib.error.HTTPError as exc:
            return {
                "status_code": exc.code,
                "headers": dict(exc.headers or {}),
                "content": exc.read(),
                "ok": False,
                "error": str(exc),
            }
        except _NETWORK_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "network_gateway",
                exc,
                action="returned failed network action receipt",
            )
            logger.warning("Network gateway request failed: %s", exc)
            return {
                "status_code": 0,
                "headers": {},
                "content": b"",
                "ok": False,
                "error": str(exc),
            }


def _coerce_method(method: str) -> str:
    if not isinstance(method, str):
        raise TypeError("HTTP method must be a string")
    method_text = method.strip().upper()
    if method_text not in _HTTP_METHODS:
        raise ValueError(f"unsupported HTTP method: {method_text}")
    return method_text


def _coerce_url(url: str) -> str:
    if not isinstance(url, str):
        raise TypeError("URL must be a string")
    url_text = url.strip()
    parsed = urllib.parse.urlparse(url_text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("network gateway URLs must be absolute http(s) URLs")
    return url_text


def _coerce_timeout(timeout: float) -> float:
    try:
        timeout_s = float(timeout)
    except (TypeError, ValueError) as exc:
        raise ValueError("network timeout must be numeric") from exc
    if timeout_s <= 0:
        raise ValueError("network timeout must be positive")
    return min(timeout_s, 120.0)


def _coerce_headers(headers: Optional[dict[str, str]]) -> dict[str, str]:
    if headers is None:
        return {}
    if not isinstance(headers, dict):
        raise TypeError("headers must be a mapping")
    return {str(key): str(value) for key, value in headers.items()}


def _coerce_data(data: Optional[bytes | str]) -> bytes | None:
    if data is None:
        return None
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return data.encode("utf-8")
    raise TypeError("request data must be bytes, string, or None")


_gateway: Optional[NetworkGateway] = None


def get_network_gateway() -> NetworkGateway:
    global _gateway
    if _gateway is None:
        _gateway = NetworkGateway()
    return _gateway


__all__ = ["NetworkGateway", "get_network_gateway"]
