"""One-request Python sandbox worker.

This module is intentionally self-contained so it can be copied into the
native macOS sandbox and launched with ``python -I``. The trusted parent owns
governance, validation, deadlines, framing, and process lifecycle; this worker
only executes one already-admitted payload and then exits.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import resource
import struct
import sys
import traceback
from typing import Any, BinaryIO

PROTOCOL_VERSION = 1
MAX_REQUEST_BYTES = 256 * 1024
MAX_RESPONSE_BYTES = 1024 * 1024
MAX_CAPTURE_BYTES = 128 * 1024
_HEADER = struct.Struct("!I")
_WORKER_EXECUTION_ERRORS = (Exception, KeyboardInterrupt, SystemExit)


class ProtocolError(ValueError):
    pass


class _BoundedTextCapture(io.TextIOBase):
    def __init__(self, limit: int = MAX_CAPTURE_BYTES) -> None:
        self._limit = max(1, int(limit))
        self._parts: list[bytes] = []
        self._size = 0
        self.truncated = False

    @property
    def encoding(self) -> str:
        return "utf-8"

    def writable(self) -> bool:
        return True

    def write(self, value: str) -> int:
        text = str(value)
        encoded = text.encode("utf-8", errors="replace")
        remaining = self._limit - self._size
        if remaining > 0:
            piece = encoded[:remaining]
            self._parts.append(piece)
            self._size += len(piece)
        if len(encoded) > max(0, remaining):
            self.truncated = True
        return len(text)

    def getvalue(self) -> str:
        value = b"".join(self._parts).decode("utf-8", errors="replace")
        if self.truncated:
            value += "\n[output truncated by sandbox limit]\n"
        return value


def encode_frame(payload: dict[str, Any], *, max_bytes: int) -> bytes:
    body = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if len(body) > max_bytes:
        raise ProtocolError(f"frame exceeds {max_bytes} bytes")
    return _HEADER.pack(len(body)) + body


def read_frame(stream: BinaryIO, *, max_bytes: int) -> dict[str, Any]:
    header = stream.read(_HEADER.size)
    if not header:
        raise EOFError("frame stream closed")
    if len(header) != _HEADER.size:
        raise ProtocolError("incomplete frame header")
    (size,) = _HEADER.unpack(header)
    if size <= 0 or size > max_bytes:
        raise ProtocolError(f"invalid frame size {size}")
    body = stream.read(size)
    if len(body) != size:
        raise ProtocolError("incomplete frame body")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("frame body is not UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("frame payload must be an object")
    return payload


def write_frame(stream: BinaryIO, payload: dict[str, Any], *, max_bytes: int) -> None:
    stream.write(encode_frame(payload, max_bytes=max_bytes))
    stream.flush()


def _apply_resource_limits() -> list[str]:
    """Install sandbox rlimits, reporting which ones did NOT take.

    CP126 6c13255a. Every getrlimit/setrlimit failure was swallowed with
    `continue`, and the worker then announced ready and executed code
    regardless. The parent had no way to know whether CPU, file-size,
    descriptor or address-space limits were actually in force — so an
    unsandboxed worker was indistinguishable from a sandboxed one, which
    defeats the point of having limits at all.

    Returns the names of the limits that could not be applied.
    """
    unapplied: list[str] = []
    limits = (
        (resource.RLIMIT_CPU, 12, 12),
        (resource.RLIMIT_FSIZE, 2 * 1024 * 1024, 2 * 1024 * 1024),
        (resource.RLIMIT_NOFILE, 64, 64),
    )
    if hasattr(resource, "RLIMIT_AS"):
        limits += ((resource.RLIMIT_AS, 2 * 1024**3, 2 * 1024**3),)
    names = {
        resource.RLIMIT_CPU: "cpu",
        resource.RLIMIT_FSIZE: "file_size",
        resource.RLIMIT_NOFILE: "open_files",
    }
    if hasattr(resource, "RLIMIT_AS"):
        names[resource.RLIMIT_AS] = "address_space"
    for kind, soft, hard in limits:
        try:
            current_soft, current_hard = resource.getrlimit(kind)
            bounded_hard = min(hard, current_hard) if current_hard >= 0 else hard
            bounded_soft = min(soft, bounded_hard)
            resource.setrlimit(kind, (bounded_soft, bounded_hard))
        except (OSError, ValueError):
            unapplied.append(names.get(kind, str(kind)))
    return unapplied


def _validate_request(payload: dict[str, Any]) -> tuple[str, str, str]:
    if payload.get("version") != PROTOCOL_VERSION or payload.get("kind") != "execute":
        raise ProtocolError("unsupported sandbox request envelope")
    request_id = str(payload.get("request_id") or "")
    authority_id = str(payload.get("authority_id") or "")
    code = payload.get("code")
    if not request_id or len(request_id) > 96:
        raise ProtocolError("request_id is required and bounded")
    if not authority_id or len(authority_id) > 128:
        raise ProtocolError("authority_id is required and bounded")
    if not isinstance(code, str):
        raise ProtocolError("code must be a string")
    if len(code.encode("utf-8")) > MAX_REQUEST_BYTES:
        raise ProtocolError("code exceeds sandbox request limit")
    return request_id, authority_id, code


def _execute_request(payload: dict[str, Any]) -> dict[str, Any]:
    request_id, authority_id, code = _validate_request(payload)
    capture = _BoundedTextCapture()
    success = False
    namespace: dict[str, Any] = {
        "__name__": "__aura_sandbox__",
        "__file__": "<aura_sandbox>",
    }
    with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
        try:
            code_object = compile(code, "<aura_sandbox>", "exec")
            exec(code_object, namespace, namespace)
            success = True
        except _WORKER_EXECUTION_ERRORS:
            traceback.print_exc(file=capture)
    return {
        "version": PROTOCOL_VERSION,
        "kind": "result",
        "request_id": request_id,
        "authority_id": authority_id,
        "success": success,
        "output": capture.getvalue(),
        "output_truncated": capture.truncated,
    }


def main() -> None:
    input_stream = getattr(sys.stdin, "buffer", sys.stdin)
    output_stream = getattr(sys.stdout, "buffer", sys.stdout)
    site_packages = os.environ.pop("AURA_SANDBOX_SITE_PACKAGES", "").strip()
    if site_packages:
        sys.path.insert(0, site_packages)
    unapplied_limits = _apply_resource_limits()
    write_frame(
        output_stream,
        {
            "version": PROTOCOL_VERSION,
            "kind": "ready",
            "worker_pid": os.getpid(),
            # The parent decides what to do about a partially-sandboxed
            # worker; it cannot decide anything if we do not tell it.
            "sandbox_limits_applied": not unapplied_limits,
            "unapplied_limits": unapplied_limits,
        },
        max_bytes=MAX_RESPONSE_BYTES,
    )
    try:
        request = read_frame(input_stream, max_bytes=MAX_REQUEST_BYTES)
        response = _execute_request(request)
    except (EOFError, ProtocolError, TypeError, ValueError) as exc:
        response = {
            "version": PROTOCOL_VERSION,
            "kind": "protocol_error",
            "request_id": "",
            "authority_id": "",
            "success": False,
            "output": f"Sandbox protocol error: {exc}",
            "output_truncated": False,
        }
    write_frame(output_stream, response, max_bytes=MAX_RESPONSE_BYTES)


if __name__ == "__main__":
    main()
