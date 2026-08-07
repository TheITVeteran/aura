"""interface/routes/media.py — serving the bytes so playback happens in the chat.

A media card without a byte source is a link with better styling. For this to
be playback rather than a hand-off, the page needs somewhere to point a
``<video>`` or ``<audio>`` element, and that means an endpoint that behaves
the way media elements expect.

Two things are non-negotiable and both are easy to get subtly wrong:

**Range requests.** A media element issues ``Range: bytes=0-`` immediately and
then seeks by asking for arbitrary windows. An endpoint that only ever returns
200 with the whole file will *play* — which is why this defect ships — but the
scrubber will not work, and on a large file the element buffers the entire
thing before starting. So 206 with a correct ``Content-Range`` is the normal
path, not an optimisation.

**The path is never the caller's.** Requests name an opaque id; the id is
resolved through the media index, which only ever contains files under roots
the user configured. There is no filename in the URL to sanitise, because
there is no filename in the URL. A traversal attempt cannot even be spelled.

Reads are streamed in chunks from a thread, never slurped: a 4 GB video read
into memory on the event loop would take the whole runtime down with it, and
this is the one route whose files are routinely larger than RAM.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from core.media.library import MediaItem, get_media_library
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Routes.Media")

router = APIRouter()

# Read size per chunk. Large enough that a long file is not a syscall storm,
# small enough that a seek is answered promptly and one request cannot pin a
# large buffer per connection.
CHUNK_BYTES = 256 * 1024

# A single range request may not ask for more than this in one response. The
# element will simply ask for the next window; the point is that one request
# cannot be made to stream a whole file in a single unbounded response.
MAX_RANGE_BYTES = 16 * 1024 * 1024

_RANGE_RE = re.compile(r"^bytes=(?P<start>\d*)-(?P<end>\d*)$")

_MEDIA_ERRORS = (AttributeError, OSError, RuntimeError, TypeError, ValueError)


def _require_local(request: Request) -> None:
    """Media is owner-surface only.

    The index is built from the user's home directory. Serving it to a paired
    phone would quietly turn "let Aura read my texts" into "let Aura read my
    Downloads folder out loud", which is not a grant anybody made.
    """
    from interface.auth import request_has_allowed_local_browser_origin

    client = getattr(request, "client", None)
    host = str(getattr(client, "host", "") or "")
    if host not in ("127.0.0.1", "::1", "localhost"):
        raise HTTPException(status_code=403, detail="Media is available on this machine only")
    if not request_has_allowed_local_browser_origin(request):
        raise HTTPException(status_code=403, detail="Media requires a local browser origin")


def _parse_range(header: str, size: int) -> tuple[int, int] | None:
    """Resolve a Range header to inclusive byte offsets, or None for the whole file.

    Returns None for an absent or unsatisfiable-but-ignorable header. Raises
    416 only for a range that is genuinely outside the file, which is what a
    media element uses to discover it has stale metadata.
    """
    raw = str(header or "").strip()
    if not raw:
        return None
    match = _RANGE_RE.match(raw)
    if not match:
        # Multi-range and unit-less forms. Answering with the whole entity is
        # explicitly allowed and is what every real client copes with.
        return None

    start_text = match.group("start")
    end_text = match.group("end")

    if not start_text and not end_text:
        return None

    if not start_text:
        # "bytes=-500" — the final 500 bytes. Used by container probes.
        length = int(end_text)
        if length <= 0:
            raise HTTPException(status_code=416, detail="Unsatisfiable range")
        start = max(0, size - length)
        end = size - 1
    else:
        start = int(start_text)
        end = int(end_text) if end_text else size - 1

    if start >= size or start < 0:
        raise HTTPException(
            status_code=416,
            detail="Unsatisfiable range",
            headers={"Content-Range": f"bytes */{size}"},
        )
    end = min(end, size - 1)
    if end < start:
        raise HTTPException(
            status_code=416,
            detail="Unsatisfiable range",
            headers={"Content-Range": f"bytes */{size}"},
        )
    end = min(end, start + MAX_RANGE_BYTES - 1)
    return start, end


async def _stream_file(item: MediaItem, start: int, end: int) -> Any:
    """Yield the requested window in bounded chunks, off the event loop."""
    import anyio

    remaining = end - start + 1
    try:
        handle = await anyio.to_thread.run_sync(lambda: open(item.path, "rb"))  # noqa: SIM115
    except OSError as exc:
        record_degradation(
            "media.route",
            exc,
            action="could not open a media file that the index listed",
            severity="warning",
        )
        return

    try:
        await anyio.to_thread.run_sync(handle.seek, start)
        while remaining > 0:
            size = min(CHUNK_BYTES, remaining)
            chunk = await anyio.to_thread.run_sync(handle.read, size)
            if not chunk:
                return
            remaining -= len(chunk)
            yield chunk
    finally:
        await anyio.to_thread.run_sync(handle.close)


@router.get("/media/search")
async def media_search(request: Request, q: str = "", kind: str = "") -> JSONResponse:
    """What is playable on this machine for a given query."""
    _require_local(request)
    try:
        library = get_media_library()
        scan = library.index()
        items = library.search(q, kind=kind) if q else list(scan.items[:12])
        return JSONResponse(
            {
                "query": q,
                "items": [item.to_dict() for item in items],
                "library": scan.narrative(),
                "truncated": scan.truncated,
            }
        )
    except _MEDIA_ERRORS as exc:
        record_degradation("media.route", exc, action="media search failed")
        return JSONResponse({"error": "media_search_failed", "detail": str(exc)}, status_code=500)


@router.get("/media/stream/{item_id}")
async def media_stream(item_id: str, request: Request) -> Any:
    """Serve one indexed file, honouring Range so seeking works."""
    _require_local(request)

    item = get_media_library().get(item_id)
    if item is None:
        # Covers both "never indexed" and "indexed but since deleted". The
        # id is opaque, so this is also the answer to a guessed id.
        raise HTTPException(status_code=404, detail="No such media item")

    try:
        size = item.path.stat().st_size
    except OSError as exc:
        record_degradation(
            "media.route", exc, action="media file disappeared between index and read"
        )
        raise HTTPException(status_code=404, detail="Media item is no longer readable") from exc

    window = _parse_range(request.headers.get("range", ""), size)

    common = {
        "Content-Type": item.mime_type,
        # Without this the element renders no scrubber and cannot seek, which
        # looks exactly like a broken file to the person watching.
        "Accept-Ranges": "bytes",
        "Cache-Control": "private, max-age=3600",
        # The bytes are the user's own files; nothing else may embed them.
        "X-Content-Type-Options": "nosniff",
    }

    if window is None:
        return StreamingResponse(
            _stream_file(item, 0, size - 1),
            status_code=200,
            headers={**common, "Content-Length": str(size)},
        )

    start, end = window
    return StreamingResponse(
        _stream_file(item, start, end),
        status_code=206,
        headers={
            **common,
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Content-Length": str(end - start + 1),
        },
    )


__all__ = ["router"]
