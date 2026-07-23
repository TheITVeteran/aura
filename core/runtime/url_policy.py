"""Canonical outbound-URL policy for observation and fetch paths.

This is the single place that decides whether Aura may point a fetcher or a
browser at a URL. It was previously private to ``core.actuators.web_actuators``,
which meant the sensory actor — the component that actually drives a real
browser — had no policy at all and passed caller-controlled URL text straight
through (CP126 3bba0f36).

The policy is deliberately two-part:

* :func:`validate_fetch_url_static` is network-free, so it is cheap enough to
  use as a validation predicate.
* :func:`validate_fetch_url` adds the resolved-address check, which is what
  actually stops an allowlisted hostname from being rebound to a loopback,
  private, link-local or cloud-metadata address.

Honest bound: without connection-level address pinning, resolve-then-connect
remains a TOCTOU window. Callers that can re-check the landing URL after
redirects should do so — see :func:`describe_decision`.
"""
from __future__ import annotations

import ipaddress
import os
import socket
import urllib.parse
from typing import Any

#: Hosts that are always acceptable targets for observation.
DEFAULT_FETCH_ALLOWLIST = {
    "wikipedia.org",
    "python.org",
    "github.com",
    "pypi.org",
    "stackoverflow.com",
    "w3schools.com",
}

#: Ports an observation request may use. A browser pointed at an arbitrary port
#: is a port scanner.
ALLOWED_PORTS = {80, 443}

CLOUD_METADATA_ADDRESSES = {"169.254.169.254", "fd00:ec2::254", "100.100.100.200"}


def allowed_fetch_domains() -> set[str]:
    extra = os.environ.get("AURA_WEB_FETCH_ALLOWLIST", "")
    names = {part.strip().lower() for part in extra.split(",") if part.strip()}
    return DEFAULT_FETCH_ALLOWLIST | names


def ip_is_public(ip_text: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_text)
    except ValueError:
        return False
    if ip_text in CLOUD_METADATA_ADDRESSES:
        return False
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def validate_fetch_url_static(url: Any) -> tuple[str | None, str]:
    """Static URL policy (no network): scheme, userinfo, port, host allowlist."""
    if not isinstance(url, str) or not url.strip():
        return None, "url is missing"
    text = url.strip()
    if any(char in text for char in ("\n", "\r", "\x00", " ")):
        return None, "url contains whitespace or control characters"
    parsed = urllib.parse.urlparse(text)
    scheme = parsed.scheme.lower()
    allow_http = str(os.environ.get("AURA_WEB_FETCH_ALLOW_HTTP", "")).strip().lower() in {
        "1", "true", "yes", "on",
    }
    if scheme != "https" and not (allow_http and scheme == "http"):
        return None, f"scheme '{scheme}' not allowed (https required)"
    if parsed.username or parsed.password or "@" in (parsed.netloc or ""):
        return None, "url must not contain credentials (userinfo)"
    host = (parsed.hostname or "").lower()
    if not host:
        return None, "url has no host"
    try:
        port = parsed.port
    except ValueError:
        return None, "url has an invalid port"
    if port is not None and port not in ALLOWED_PORTS:
        return None, f"port {port} is not permitted for observation"
    allowed = allowed_fetch_domains()
    if not any(host == domain or host.endswith("." + domain) for domain in allowed):
        return None, f"host '{host}' is not in the fetch allowlist"
    return text, ""


def validate_fetch_url(url: Any) -> tuple[str | None, str]:
    """Full fetch policy: static checks + resolved-IP SSRF guard (does DNS)."""
    validated, error = validate_fetch_url_static(url)
    if validated is None:
        return None, error
    parsed = urllib.parse.urlparse(validated)
    host = (parsed.hostname or "").lower()
    scheme = parsed.scheme.lower()
    # Bind the allowlisted name to its resolved addresses — reject if ANY
    # resolves to a non-public target (defeats DNS rebinding to internal hosts).
    try:
        infos = socket.getaddrinfo(
            host,
            parsed.port or (443 if scheme == "https" else 80),
            proto=socket.IPPROTO_TCP,
        )
    except (socket.gaierror, OSError, ValueError) as exc:
        return None, f"host '{host}' did not resolve: {exc}"
    addrs = {info[4][0] for info in infos}
    if not addrs:
        return None, f"host '{host}' resolved to no addresses"
    for addr in addrs:
        if not ip_is_public(addr):
            return None, f"host '{host}' resolves to a non-public address ({addr})"
    return validated, ""


def describe_decision(url: Any, *, principal: str = "", stage: str = "request") -> dict[str, Any]:
    """A receipt for one policy decision, safe to log and return to a caller."""
    validated, error = validate_fetch_url(url)
    parsed = urllib.parse.urlparse(str(url or "").strip())
    return {
        "stage": stage,
        "principal": principal,
        "url": str(url or "")[:2048],
        "host": (parsed.hostname or "").lower(),
        "scheme": parsed.scheme.lower(),
        "allowed": validated is not None,
        "reason": error,
        "policy": "core.runtime.url_policy",
    }
