"""Serving media the way a media element actually asks for it.

A `<video>` or `<audio>` element issues `Range: bytes=0-` immediately and then
seeks by requesting arbitrary windows. An endpoint that only ever answers 200
with the whole file *plays* — which is exactly why that defect ships and
survives a manual check — but the scrubber does not work, and on a large file
the element buffers the entire thing before it starts. So 206 with a correct
`Content-Range` is the normal path here, not an optimisation, and these tests
exercise the forms real clients send rather than the one form that is easy.

The other property is the security one, and it is structural rather than
defensive: requests name an opaque id, the id resolves through the media
index, and the index only ever contains files under configured roots. There
is no filename in the URL to sanitise because there is no filename in the URL
— a traversal attempt cannot be spelled.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.media.library import MediaLibrary


@pytest.fixture()
def media_file(tmp_path: Path) -> tuple[MediaLibrary, str, bytes]:
    root = tmp_path / "Music"
    root.mkdir(parents=True)
    payload = bytes(range(256)) * 40  # 10,240 bytes with distinguishable content
    (root / "Test Track.mp3").write_bytes(payload)
    library = MediaLibrary(roots=(root,))
    item = library.search("Test Track")[0]
    return library, item.item_id, payload


def _parse(header: str, size: int):
    from interface.routes.media import _parse_range

    return _parse_range(header, size)


# ── the ranges real clients send ─────────────────────────────────────────


def test_no_range_header_means_the_whole_file() -> None:
    assert _parse("", 10_240) is None


def test_open_ended_range_is_the_opening_request() -> None:
    """`bytes=0-` is the first thing every media element sends."""
    assert _parse("bytes=0-", 10_240) == (0, 10_239)


def test_a_seek_asks_for_a_window() -> None:
    assert _parse("bytes=4096-8191", 10_240) == (4096, 8191)


def test_a_suffix_range_reads_the_tail() -> None:
    """`bytes=-500` — used by container probes looking for trailing metadata."""
    assert _parse("bytes=-500", 10_240) == (9_740, 10_239)


def test_an_end_past_the_file_is_clamped_rather_than_refused() -> None:
    """Clients routinely over-ask; refusing would break ordinary playback."""
    assert _parse("bytes=10000-99999", 10_240) == (10_000, 10_239)


def test_a_start_past_the_file_is_a_416() -> None:
    """This is how a client discovers its metadata is stale."""
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as excinfo:
        _parse("bytes=20000-", 10_240)
    assert excinfo.value.status_code == 416
    assert excinfo.value.headers["Content-Range"] == "bytes */10240"


def test_an_inverted_range_is_a_416() -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as excinfo:
        _parse("bytes=900-100", 10_240)
    assert excinfo.value.status_code == 416


def test_a_multi_range_falls_back_to_the_whole_entity() -> None:
    """Answering with the whole entity is allowed and is what clients cope with."""
    assert _parse("bytes=0-99,200-299", 10_240) is None


def test_a_malformed_range_does_not_raise() -> None:
    """A bad header must not turn playback into a 500."""
    for header in ("bytes=abc-def", "items=0-10", "bytes=", "garbage"):
        assert _parse(header, 10_240) is None


def test_one_request_cannot_stream_an_unbounded_amount() -> None:
    """The element will simply ask for the next window."""
    from interface.routes.media import MAX_RANGE_BYTES

    huge = 4 * 1024 * 1024 * 1024
    start, end = _parse("bytes=0-", huge)
    assert start == 0
    assert end - start + 1 == MAX_RANGE_BYTES


# ── the bytes that come back ─────────────────────────────────────────────


def test_the_streamed_window_is_exactly_the_bytes_requested(media_file) -> None:
    """An off-by-one here is audible as a click and visible as a glitch."""
    import asyncio

    from interface.routes.media import _stream_file

    library, item_id, payload = media_file
    item = library.get(item_id)
    assert item is not None

    async def collect(start: int, end: int) -> bytes:
        return b"".join([chunk async for chunk in _stream_file(item, start, end)])

    assert asyncio.run(collect(0, 9)) == payload[:10]
    assert asyncio.run(collect(4096, 8191)) == payload[4096:8192]
    assert asyncio.run(collect(len(payload) - 1, len(payload) - 1)) == payload[-1:]


def test_a_deleted_file_yields_nothing_rather_than_raising(media_file) -> None:
    """The index can outlive the file; a 500 mid-playback is worse than silence."""
    import asyncio

    from interface.routes.media import _stream_file

    library, item_id, _payload = media_file
    item = library.get(item_id)
    assert item is not None
    item.path.unlink()

    async def collect() -> bytes:
        return b"".join([chunk async for chunk in _stream_file(item, 0, 100)])

    assert asyncio.run(collect()) == b""


# ── the surface it is served to ──────────────────────────────────────────


def test_media_is_owner_surface_only() -> None:
    """The index is built from the user's home directory.

    Serving it to a paired phone would quietly turn "let Aura read my texts"
    into "let Aura read my Downloads folder", which is not a grant anyone made.
    """
    from types import SimpleNamespace

    from fastapi import HTTPException

    from interface.routes.media import _require_local

    remote = SimpleNamespace(
        client=SimpleNamespace(host="192.168.1.40"), headers={}, scope={}
    )
    with pytest.raises(HTTPException) as excinfo:
        _require_local(remote)
    assert excinfo.value.status_code == 403
