"""core/runtime/network_gateway.py — Canonical Network Gateway.

All outbound network requests should flow through this module to ensure correct governance, logging, and audit.
"""
from __future__ import annotations

import asyncio
import ipaddress
import logging
import re
import socket
import ssl
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from core.governance_context import (
    governance_runtime_active,
    require_governance,
)
from core.runtime.authorization_receipt import read_verdict
from core.runtime.errors import NetworkEffectDenied, record_degradation

logger = logging.getLogger("Aura.NetworkGateway")
_NETWORK_DOMAINS = (
    "environment_action",
    "external_action",
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
_WEBSOCKET_SCHEMES = {"ws", "wss"}
_STREAM_SCHEMES = {"tcp", "tls"}
_HOSTNAME = re.compile(
    r"^(?=.{1,253}$)(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)(?:\.(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?))*\.?$"
)
_CLOUD_METADATA_ADDRESSES = frozenset(
    {"169.254.169.254", "100.100.100.200", "fd00:ec2::254"}
)
_OPERATIONAL_TELEMETRY_SOURCE_PREFIXES = (
    "observability:",
    "telemetry:",
)


@dataclass(frozen=True)
class WebSocketAdmission:
    """One admitted, address-pinned WebSocket connection.

    The receipt deliberately excludes headers because they may contain bearer
    credentials.  ``peer_address`` is the exact address selected during the
    governed resolution and verified after the TCP connection completed.
    """

    connection: Any
    destination_host: str
    destination_port: int
    peer_address: str
    secure: bool
    source: str
    read_only: bool


@dataclass(frozen=True)
class StreamAdmission:
    """One admitted, address-pinned byte stream with no credential material."""

    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    destination_host: str
    destination_port: int
    peer_address: str
    secure: bool
    peer_certificate_sha256: str
    source: str
    read_only: bool


class NetworkGateway:
    """Single canonical owner for HTTP/Network requests."""

    def __init__(self) -> None:
        self._allowed_domains = _NETWORK_DOMAINS

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        data: bytes | str | None = None,
        proxies: dict[str, str] | None = None,
        timeout: float = 30.0,
        source: str = "unknown",
        read_only: bool = False,
        operational_telemetry: bool = False,
        suppress_degradation: bool = False,
    ) -> dict[str, Any]:
        """Perform a synchronous HTTP request."""
        method_text = _coerce_method(method)
        url_text = _coerce_url(url)
        timeout_s = _coerce_timeout(timeout)
        request_headers = _coerce_headers(headers)
        request_data = _coerce_data(data)
        request_proxies = _coerce_proxies(proxies)

        try:
            from core.security.defensive_runtime import validate_outbound_network

            defensive_receipt = validate_outbound_network(
                method=method_text,
                url=url_text,
                data_length=len(request_data or b""),
                source=source,
            )
            # CP126 (fail-open class). This was
            # `defensive_receipt.get("allowed", True)`, so a receipt that
            # stated no verdict — a partial dict, an early return, a
            # validator that did not understand the question — was read as
            # permission to make the request. Absence of a check reported as
            # a passed check, on the outbound network boundary.
            verdict = read_verdict(defensive_receipt)
            if not verdict.allows:
                if not verdict.is_stated:
                    # Refusing is right; saying WHY matters, because an
                    # unstated verdict is a broken validator and looks
                    # nothing like a policy decision from the outside.
                    record_degradation(
                        "network_gateway.defensive_runtime",
                        RuntimeError(
                            f"outbound preflight stated no verdict: {verdict.reason}"
                        ),
                        severity="warning",
                        action="refused the request rather than reading an absent verdict as permission",
                        enforce_failure_policy=False,
                    )
                return {
                    "status_code": 0,
                    "headers": {},
                    "content": b"",
                    "ok": False,
                    "error": str(
                        defensive_receipt.get("reason")
                        or verdict.reason
                        or "blocked_by_defensive_runtime"
                    ),
                    "defensive_runtime": defensive_receipt,
                    "defensive_verdict": verdict.to_dict(),
                }
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "network_gateway.defensive_runtime",
                exc,
                action="continued network request after defensive preflight failed",
            )

        telemetry_bypass = _validate_operational_telemetry_bypass(
            operational_telemetry=operational_telemetry,
            source=source,
        )
        if not read_only and not telemetry_bypass and governance_runtime_active():
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
            opener = (
                urllib.request.build_opener(urllib.request.ProxyHandler(request_proxies))
                if request_proxies
                else None
            )
            open_request = opener.open if opener is not None else urllib.request.urlopen
            with open_request(req, timeout=timeout_s) as response:
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
            if suppress_degradation:
                logger.debug(
                    "Network gateway request failed without degradation emission "
                    "(source=%s): %s",
                    source,
                    exc,
                )
            else:
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

    async def request_async(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        data: bytes | str | None = None,
        proxies: dict[str, str] | None = None,
        timeout: float = 30.0,  # noqa: ASYNC109 - forwarded to urllib.
        source: str = "unknown",
        read_only: bool = False,
        operational_telemetry: bool = False,
        suppress_degradation: bool = False,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self.request,
            method,
            url,
            headers=headers,
            data=data,
            proxies=proxies,
            timeout=timeout,
            source=source,
            read_only=read_only,
            operational_telemetry=operational_telemetry,
            suppress_degradation=suppress_degradation,
        )

    async def connect_websocket(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        open_timeout: float = 10.0,
        close_timeout: float = 5.0,
        ping_interval: float = 20.0,
        ping_timeout: float = 10.0,
        max_size: int = 1_048_576,
        max_queue: int = 64,
        source: str = "unknown",
        read_only: bool = False,
        allow_private_target: bool = False,
    ) -> WebSocketAdmission:
        """Open a governed WebSocket pinned to one authorized DNS result.

        Resolution occurs once at this boundary.  The selected numeric address
        is passed to the TCP connector while the original host remains the
        HTTP Host and TLS SNI identity.  This closes the resolve-then-connect
        DNS-rebinding window present in ordinary URL preflight checks.

        Private, loopback, and link-local destinations are denied unless the
        owning adapter explicitly declares that it connects to a configured
        local device.  Cloud metadata, multicast, unspecified, and reserved
        addresses are never eligible.
        """

        url_text, host, port, secure = _coerce_websocket_url(url)
        request_headers = _coerce_headers(headers)
        open_timeout_s = _coerce_timeout(open_timeout)
        close_timeout_s = _coerce_timeout(close_timeout)
        ping_interval_s = _coerce_timeout(ping_interval)
        ping_timeout_s = _coerce_timeout(ping_timeout)
        max_size_value = _coerce_positive_int(max_size, name="max_size", maximum=16_777_216)
        max_queue_value = _coerce_positive_int(max_queue, name="max_queue", maximum=1_024)

        try:
            from core.security.defensive_runtime import validate_outbound_network

            defensive_receipt = validate_outbound_network(
                method="CONNECT",
                url=url_text,
                data_length=0,
                source=source,
            )
            verdict = read_verdict(defensive_receipt)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "network_gateway.websocket_preflight",
                exc,
                severity="warning",
                action="refused WebSocket admission after defensive preflight failed",
                enforce_failure_policy=False,
            )
            raise NetworkEffectDenied("websocket_defensive_preflight_failed") from exc
        if not verdict.allows:
            reason = str(
                defensive_receipt.get("reason")
                or verdict.reason
                or "blocked_by_defensive_runtime"
            )
            raise NetworkEffectDenied(f"websocket_admission_denied:{reason}")

        if not read_only and governance_runtime_active():
            require_governance(
                f"network_gateway.connect_websocket:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )

        addresses = await _resolve_websocket_addresses(
            host,
            port,
            timeout_s=min(open_timeout_s, 10.0),
        )
        approved = _approve_websocket_addresses(
            addresses,
            allow_private_target=allow_private_target,
        )

        try:
            import websockets
        except ImportError as exc:
            raise NetworkEffectDenied("websocket_dependency_unavailable") from exc

        failures: list[BaseException] = []
        for address in approved:
            connection: Any | None = None
            try:
                connection = await websockets.connect(
                    url_text,
                    additional_headers=request_headers,
                    open_timeout=open_timeout_s,
                    close_timeout=close_timeout_s,
                    ping_interval=ping_interval_s,
                    ping_timeout=ping_timeout_s,
                    max_size=max_size_value,
                    max_queue=max_queue_value,
                    compression=None,
                    proxy=None,
                    host=address,
                    port=port,
                )
                try:
                    peer_address = _websocket_peer_address(connection)
                except NetworkEffectDenied:
                    await connection.close(code=1008, reason="peer address unavailable")
                    raise
                if _normalize_ip(peer_address) != _normalize_ip(address):
                    await connection.close(code=1008, reason="network address changed")
                    raise NetworkEffectDenied("websocket_peer_address_mismatch")
                logger.info(
                    "WebSocket admitted source=%s host=%s port=%s peer=%s secure=%s read_only=%s",
                    source,
                    host,
                    port,
                    peer_address,
                    secure,
                    read_only,
                )
                return WebSocketAdmission(
                    connection=connection,
                    destination_host=host,
                    destination_port=port,
                    peer_address=peer_address,
                    secure=secure,
                    source=source,
                    read_only=read_only,
                )
            except NetworkEffectDenied:
                raise
            except (
                websockets.exceptions.WebSocketException,
                OSError,
                RuntimeError,
                TimeoutError,
                ValueError,
            ) as exc:
                failures.append(exc)
                if connection is not None:
                    await connection.close(code=1011, reason="connection admission failed")

        cause = failures[-1] if failures else RuntimeError("no approved address was attempted")
        raise NetworkEffectDenied("websocket_connection_failed") from cause

    async def connect_stream(
        self,
        endpoint: str,
        *,
        open_timeout: float = 10.0,
        read_limit: int = 65_536,
        source: str = "unknown",
        read_only: bool = False,
        allow_private_target: bool = False,
        expected_certificate_sha256: str = "",
    ) -> StreamAdmission:
        """Open a governed TCP/TLS byte stream pinned to an admitted address."""

        endpoint_text, host, port, secure = _coerce_stream_endpoint(endpoint)
        timeout_s = _coerce_timeout(open_timeout)
        read_limit_value = _coerce_positive_int(
            read_limit,
            name="read_limit",
            maximum=16_777_216,
        )
        certificate_pin = str(expected_certificate_sha256 or "").strip().lower()
        if certificate_pin and not _is_sha256_digest(certificate_pin):
            raise ValueError("expected_certificate_sha256 must be a sha256 digest")
        if certificate_pin and not secure:
            raise ValueError("a certificate pin requires a TLS stream")

        try:
            from core.security.defensive_runtime import validate_outbound_network

            defensive_receipt = validate_outbound_network(
                method="CONNECT",
                url=endpoint_text,
                data_length=0,
                source=source,
            )
            verdict = read_verdict(defensive_receipt)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "network_gateway.stream_preflight",
                exc,
                severity="warning",
                action="refused stream admission after defensive preflight failed",
                enforce_failure_policy=False,
            )
            raise NetworkEffectDenied("stream_defensive_preflight_failed") from exc
        if not verdict.allows:
            reason = str(
                defensive_receipt.get("reason")
                or verdict.reason
                or "blocked_by_defensive_runtime"
            )
            raise NetworkEffectDenied(f"stream_admission_denied:{reason}")
        if not read_only and governance_runtime_active():
            require_governance(
                f"network_gateway.connect_stream:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )

        addresses = await _resolve_network_addresses(
            host,
            port,
            timeout_s=min(timeout_s, 10.0),
            error_prefix="stream",
        )
        approved = _approve_network_addresses(
            addresses,
            allow_private_target=allow_private_target,
            error_prefix="stream",
        )
        context = ssl.create_default_context() if secure else None
        failures: list[BaseException] = []
        for address in approved:
            writer: asyncio.StreamWriter | None = None
            try:
                async with asyncio.timeout(timeout_s):
                    reader, writer = await asyncio.open_connection(
                        host=address,
                        port=port,
                        ssl=context,
                        server_hostname=host if secure else None,
                        limit=read_limit_value,
                    )
                peer_address = _stream_peer_address(writer)
                if _normalize_ip(peer_address, error_prefix="stream") != _normalize_ip(
                    address,
                    error_prefix="stream",
                ):
                    await _close_stream_writer(writer)
                    raise NetworkEffectDenied("stream_peer_address_mismatch")
                certificate_sha256 = _stream_certificate_sha256(writer)
                if secure and not certificate_sha256:
                    await _close_stream_writer(writer)
                    raise NetworkEffectDenied("stream_peer_certificate_unavailable")
                if certificate_pin and certificate_sha256 != certificate_pin:
                    await _close_stream_writer(writer)
                    raise NetworkEffectDenied("stream_peer_certificate_mismatch")
                logger.info(
                    "Stream admitted source=%s host=%s port=%s peer=%s secure=%s read_only=%s",
                    source,
                    host,
                    port,
                    peer_address,
                    secure,
                    read_only,
                )
                return StreamAdmission(
                    reader=reader,
                    writer=writer,
                    destination_host=host,
                    destination_port=port,
                    peer_address=peer_address,
                    secure=secure,
                    peer_certificate_sha256=certificate_sha256,
                    source=source,
                    read_only=read_only,
                )
            except NetworkEffectDenied:
                if writer is not None:
                    await _close_stream_writer(writer)
                raise
            except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
                failures.append(exc)
                if writer is not None:
                    await _close_stream_writer(writer)
        cause = failures[-1] if failures else RuntimeError("no approved address was attempted")
        raise NetworkEffectDenied("stream_connection_failed") from cause


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


def _coerce_websocket_url(url: str) -> tuple[str, str, int, bool]:
    if not isinstance(url, str):
        raise TypeError("WebSocket URL must be a string")
    url_text = url.strip()
    if any(character in url_text for character in ("\n", "\r", "\x00", " ")):
        raise ValueError("WebSocket URL contains whitespace or control characters")
    parsed = urllib.parse.urlparse(url_text)
    scheme = parsed.scheme.lower()
    if scheme not in _WEBSOCKET_SCHEMES or not parsed.hostname:
        raise ValueError("network gateway WebSocket URLs must be absolute ws(s) URLs")
    if parsed.username or parsed.password:
        raise ValueError("WebSocket URL must not contain credentials")
    try:
        port = parsed.port or (443 if scheme == "wss" else 80)
    except ValueError as exc:
        raise ValueError("WebSocket URL has an invalid port") from exc
    return url_text, parsed.hostname.lower(), port, scheme == "wss"


def _coerce_stream_endpoint(endpoint: str) -> tuple[str, str, int, bool]:
    if not isinstance(endpoint, str):
        raise TypeError("stream endpoint must be a string")
    endpoint_text = endpoint.strip()
    if any(character in endpoint_text for character in ("\n", "\r", "\x00", " ")):
        raise ValueError("stream endpoint contains whitespace or control characters")
    parsed = urllib.parse.urlparse(endpoint_text)
    scheme = parsed.scheme.lower()
    if (
        scheme not in _STREAM_SCHEMES
        or not parsed.hostname
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("stream endpoints must be absolute tcp:// or tls:// authorities")
    if parsed.username or parsed.password:
        raise ValueError("stream endpoint must not contain credentials")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("stream endpoint has an invalid port") from exc
    if port is None:
        raise ValueError("stream endpoint requires an explicit port")
    return endpoint_text, parsed.hostname.lower(), port, scheme == "tls"


def build_stream_endpoint(host: str, port: int, *, secure: bool = False) -> str:
    """Build one unambiguous stream endpoint from a host and numeric port."""

    if not isinstance(host, str):
        raise TypeError("stream host must be a string")
    normalized = host.strip()
    if any(character in normalized for character in ("\n", "\r", "\x00", " ")):
        raise ValueError("stream host contains whitespace or control characters")
    try:
        ip = ipaddress.ip_address(normalized.split("%", 1)[0])
    except ValueError:
        if not _HOSTNAME.fullmatch(normalized):
            raise ValueError("stream host must be an IP address or DNS name") from None
        authority_host = normalized.rstrip(".").lower()
    else:
        authority_host = f"[{normalized}]" if ip.version == 6 else normalized
    if isinstance(port, bool):
        raise TypeError("stream port must be an integer")
    try:
        normalized_port = int(port)
    except (TypeError, ValueError) as exc:
        raise TypeError("stream port must be an integer") from exc
    if not 1 <= normalized_port <= 65_535:
        raise ValueError("stream port must lie inside [1, 65535]")
    return f"{'tls' if secure else 'tcp'}://{authority_host}:{normalized_port}"


def _coerce_timeout(timeout: float) -> float:
    try:
        timeout_s = float(timeout)
    except (TypeError, ValueError) as exc:
        raise ValueError("network timeout must be numeric") from exc
    if timeout_s <= 0:
        raise ValueError("network timeout must be positive")
    return min(timeout_s, 120.0)


def _coerce_positive_int(value: int, *, name: str, maximum: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if normalized <= 0 or normalized > maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")
    return normalized


async def _resolve_websocket_addresses(host: str, port: int, *, timeout_s: float) -> tuple[str, ...]:
    return await _resolve_network_addresses(
        host,
        port,
        timeout_s=timeout_s,
        error_prefix="websocket",
    )


async def _resolve_network_addresses(
    host: str,
    port: int,
    *,
    timeout_s: float,
    error_prefix: str,
) -> tuple[str, ...]:
    loop = asyncio.get_running_loop()
    try:
        async with asyncio.timeout(timeout_s):
            infos = await loop.getaddrinfo(
                host,
                port,
                type=socket.SOCK_STREAM,
                proto=socket.IPPROTO_TCP,
            )
    except (OSError, TimeoutError, ValueError) as exc:
        raise NetworkEffectDenied(f"{error_prefix}_destination_resolution_failed") from exc
    addresses = tuple(dict.fromkeys(str(info[4][0]) for info in infos if info[4]))
    if not addresses:
        raise NetworkEffectDenied(f"{error_prefix}_destination_resolved_to_no_addresses")
    return addresses


def _normalize_ip(value: str, *, error_prefix: str = "websocket") -> str:
    candidate = str(value or "").split("%", 1)[0]
    try:
        return str(ipaddress.ip_address(candidate))
    except ValueError as exc:
        raise NetworkEffectDenied(f"{error_prefix}_peer_address_invalid") from exc


def _approve_websocket_addresses(
    addresses: tuple[str, ...],
    *,
    allow_private_target: bool,
) -> tuple[str, ...]:
    return _approve_network_addresses(
        addresses,
        allow_private_target=allow_private_target,
        error_prefix="websocket",
    )


def _approve_network_addresses(
    addresses: tuple[str, ...],
    *,
    allow_private_target: bool,
    error_prefix: str,
) -> tuple[str, ...]:
    approved: list[str] = []
    for address in addresses:
        normalized = _normalize_ip(address, error_prefix=error_prefix)
        ip = ipaddress.ip_address(normalized)
        if normalized in _CLOUD_METADATA_ADDRESSES:
            raise NetworkEffectDenied(f"{error_prefix}_cloud_metadata_target_denied")
        if ip.is_multicast or ip.is_unspecified or ip.is_reserved:
            raise NetworkEffectDenied(f"{error_prefix}_non_unicast_target_denied")
        non_public = ip.is_private or ip.is_loopback or ip.is_link_local
        if non_public and not allow_private_target:
            raise NetworkEffectDenied(
                f"{error_prefix}_private_target_requires_explicit_scope"
            )
        approved.append(normalized)
    return tuple(approved)


def _websocket_peer_address(connection: Any) -> str:
    transport = getattr(connection, "transport", None)
    peer = transport.get_extra_info("peername") if transport is not None else None
    if not isinstance(peer, tuple) or not peer:
        raise NetworkEffectDenied("websocket_peer_address_unavailable")
    return str(peer[0])


def _stream_peer_address(writer: asyncio.StreamWriter) -> str:
    peer = writer.get_extra_info("peername")
    if not isinstance(peer, tuple) or not peer:
        raise NetworkEffectDenied("stream_peer_address_unavailable")
    return str(peer[0])


def _stream_certificate_sha256(writer: asyncio.StreamWriter) -> str:
    ssl_object = writer.get_extra_info("ssl_object")
    certificate = ssl_object.getpeercert(binary_form=True) if ssl_object is not None else None
    if not isinstance(certificate, bytes) or not certificate:
        return ""
    import hashlib

    return "sha256:" + hashlib.sha256(certificate).hexdigest()


async def _close_stream_writer(writer: asyncio.StreamWriter) -> None:
    writer.close()
    try:
        await asyncio.wait_for(writer.wait_closed(), timeout=2.0)
    except (ConnectionError, OSError, RuntimeError):
        logger.debug("Stream writer raised while closing", exc_info=True)
    except TimeoutError:
        logger.warning("Stream writer close exceeded the 2s shutdown budget")


def _is_sha256_digest(value: str) -> bool:
    if not value.startswith("sha256:") or len(value) != 71:
        return False
    return all(character in "0123456789abcdef" for character in value[7:])


def _coerce_headers(headers: dict[str, str] | None) -> dict[str, str]:
    if headers is None:
        return {}
    if not isinstance(headers, dict):
        raise TypeError("headers must be a mapping")
    return {str(key): str(value) for key, value in headers.items()}


def _coerce_data(data: bytes | str | None) -> bytes | None:
    if data is None:
        return None
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return data.encode("utf-8")
    raise TypeError("request data must be bytes, string, or None")


def _coerce_proxies(proxies: dict[str, str] | None) -> dict[str, str] | None:
    if proxies is None:
        return None
    if not isinstance(proxies, dict):
        raise TypeError("proxies must be a mapping")
    normalized: dict[str, str] = {}
    for scheme, proxy_url in proxies.items():
        scheme_text = str(scheme).lower()
        if scheme_text not in {"http", "https"}:
            raise ValueError(f"unsupported proxy scheme: {scheme_text}")
        proxy_text = str(proxy_url).strip()
        parsed = urllib.parse.urlparse(proxy_text)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("proxy URLs must be absolute http(s) URLs")
        normalized[scheme_text] = proxy_text
    return normalized or None


def _validate_operational_telemetry_bypass(*, operational_telemetry: bool, source: str) -> bool:
    if not operational_telemetry:
        return False
    if not any(source.startswith(prefix) for prefix in _OPERATIONAL_TELEMETRY_SOURCE_PREFIXES):
        raise ValueError(
            "operational telemetry network bypass requires a source prefix of "
            f"{', '.join(_OPERATIONAL_TELEMETRY_SOURCE_PREFIXES)}"
        )
    logger.info("operational telemetry network request source=%s", source)
    return True


_gateway: NetworkGateway | None = None


def get_network_gateway() -> NetworkGateway:
    global _gateway
    if _gateway is None:
        _gateway = NetworkGateway()
    return _gateway


__all__ = [
    "NetworkGateway",
    "StreamAdmission",
    "WebSocketAdmission",
    "build_stream_endpoint",
    "get_network_gateway",
]
