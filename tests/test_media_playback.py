"""Playing what is actually on this machine, and saying so when it is not.

Ask any shipped assistant to play a song and the best case is a hand-off: a
card that opens a streaming app, a link, a new tab. Nothing plays where you
asked for it, and nothing plays at all without a network. Aura runs on the
machine that has the files, which makes local playback the fast path, the
private path, and the only path that works on a plane.

Two properties matter more than the feature itself.

**Requests to play are told apart from talking about music.** People discuss
songs constantly, and an assistant that starts playback because a band was
mentioned is worse than one that cannot play anything.

**A miss produces facts, not an apology.** When nothing matches and there is
no network, what gets recorded is what was searched, what the connectivity
reading actually said, and what is still possible — so the sentence she says
is hers rather than a string written months ago in whichever module failed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.conversation.failure_context import bind_failure_ledger, pending_failure_context
from core.media.library import MediaLibrary, reset_media_library_for_test
from core.media.playback import parse_play_request, resolve_play_request


@pytest.fixture(autouse=True)
def _clean_library():
    reset_media_library_for_test()
    yield
    reset_media_library_for_test()


@pytest.fixture()
def library_root(tmp_path: Path) -> Path:
    music = tmp_path / "Music" / "Miles Davis" / "Kind of Blue"
    music.mkdir(parents=True)
    for name in ("So What.mp3", "Blue in Green.mp3", "Flamenco Sketches.m4a"):
        (music / name).write_bytes(b"\x00" * 2048)
    movies = tmp_path / "Movies"
    movies.mkdir(parents=True)
    (movies / "Wedding Toast.mp4").write_bytes(b"\x00" * 4096)
    (movies / "notes.txt").write_text("not media")
    return tmp_path


# ── telling a request from a remark ──────────────────────────────────────


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("play Kind of Blue", "kind of blue"),
        ("put on some Miles Davis", "miles davis"),
        ("can you play the wedding toast video", "wedding toast"),
        ("play me that song by Radiohead", "radiohead"),
        ("put on some music by Miles Davis", "miles davis"),
    ],
)
def test_play_requests_are_recognised(message: str, expected: str) -> None:
    what, _kind = parse_play_request(message)
    assert what.lower() == expected


@pytest.mark.parametrize(
    "message",
    [
        "I was listening to Kind of Blue earlier",
        "that Miles Davis album is incredible",
        "what did you think of the wedding video",
        "playing guitar is harder than it looks",
    ],
)
def test_talking_about_music_is_not_a_request_to_play_it(message: str) -> None:
    """The failure that would make the feature intolerable."""
    what, _kind = parse_play_request(message)
    assert what == ""


def test_a_trailing_clause_is_not_part_of_the_title() -> None:
    what, _kind = parse_play_request("play Kind of Blue and turn the lights down")
    assert what.lower() == "kind of blue"


def test_the_medium_is_inferred_when_it_is_stated() -> None:
    assert parse_play_request("play the wedding toast video")[1] == "video"
    assert parse_play_request("play that song again")[1] == "audio"


# ── finding it ───────────────────────────────────────────────────────────


def test_local_media_is_found_and_playable(library_root: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "core.media.playback.get_media_library",
        lambda: MediaLibrary(roots=(library_root,)),
    )
    resolution = resolve_play_request("play Blue in Green")
    assert resolution.playable, resolution.to_dict()
    assert resolution.item is not None
    assert resolution.item.title == "Blue in Green"
    assert resolution.item.kind == "audio"


def test_a_video_request_does_not_return_audio(library_root: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "core.media.playback.get_media_library",
        lambda: MediaLibrary(roots=(library_root,)),
    )
    resolution = resolve_play_request("play the wedding toast video")
    assert resolution.playable
    assert resolution.item is not None
    assert resolution.item.kind == "video"


def test_the_exact_match_outranks_the_longer_one(tmp_path: Path, monkeypatch) -> None:
    """"Blue in Green" beats "Blue in Green (Remastered Live 1997)"."""
    root = tmp_path / "Music"
    root.mkdir(parents=True)
    (root / "Blue in Green (Remastered Live 1997).mp3").write_bytes(b"\x00" * 16)
    (root / "Blue in Green.mp3").write_bytes(b"\x00" * 16)
    monkeypatch.setattr(
        "core.media.playback.get_media_library", lambda: MediaLibrary(roots=(root,))
    )
    resolution = resolve_play_request("play Blue in Green")
    assert resolution.item is not None
    assert resolution.item.title == "Blue in Green"


def test_non_media_files_are_never_offered(library_root: Path) -> None:
    library = MediaLibrary(roots=(library_root,))
    titles = {item.title for item in library.index().items}
    assert "notes" not in titles
    assert "So What" in titles


def test_paths_are_never_exposed_only_a_folder_name(library_root: Path) -> None:
    """This payload ends up in a chat log; a home directory layout must not."""
    library = MediaLibrary(roots=(library_root,))
    item = library.search("So What")[0]
    payload = item.to_dict()
    assert "path" not in payload
    assert str(library_root) not in repr(payload)
    assert payload["folder"] == "Kind of Blue"


# ── the index is the allowlist ───────────────────────────────────────────


def test_only_indexed_ids_resolve_to_a_file(library_root: Path) -> None:
    """There is no filename in the URL, so traversal cannot be spelled."""
    library = MediaLibrary(roots=(library_root,))
    library.index()
    assert library.get("../../etc/passwd") is None
    assert library.get("") is None
    assert library.get("0" * 24) is None


def test_an_id_stops_resolving_once_the_file_is_gone(library_root: Path) -> None:
    library = MediaLibrary(roots=(library_root,))
    item = library.search("So What")[0]
    assert library.get(item.item_id) is not None
    item.path.unlink()
    assert library.get(item.item_id) is None


def test_replacing_a_file_invalidates_its_id(library_root: Path) -> None:
    """A stale id resolving to different bytes plays something nobody asked for."""
    library = MediaLibrary(roots=(library_root,))
    before = library.search("So What")[0].item_id
    target = library_root / "Music" / "Miles Davis" / "Kind of Blue" / "So What.mp3"
    target.write_bytes(b"\x01" * 9999)
    after = MediaLibrary(roots=(library_root,)).search("So What")[0].item_id
    assert before != after


# ── the honest miss ──────────────────────────────────────────────────────


def test_offline_and_absent_produces_facts_she_can_say(
    library_root: Path, monkeypatch
) -> None:
    """The whole point. No canned sentence anywhere in this path.

    What she is handed is: what was searched, what the connectivity probe
    actually reported, and what still works. The sentence is hers.
    """
    monkeypatch.setattr(
        "core.media.playback.get_media_library",
        lambda: MediaLibrary(roots=(library_root,)),
    )
    monkeypatch.setenv("AURA_FORCE_OFFLINE", "1")
    from core.runtime import connectivity

    connectivity._PROBE = None

    with bind_failure_ledger():
        resolution = resolve_play_request("play something by Coltrane")
        block = pending_failure_context()

    assert resolution.status == "offline"
    assert "Coltrane" in block or "coltrane" in block
    # The real reading, not an assumption about being offline.
    assert "connectivity probe" in block
    assert "AURA_FORCE_OFFLINE" in block
    # What remains, so a bounded failure is not over-generalised.
    assert "still works" in block
    assert "playable locally" in block
    # And nothing she is expected to recite.
    for canned in ("I'm sorry", "I am unable", "I can't play", "Unfortunately"):
        assert canned.lower() not in block.lower()


def test_a_remark_about_music_records_no_failure_at_all() -> None:
    """Nothing was attempted, so there is nothing to explain."""
    with bind_failure_ledger():
        resolution = resolve_play_request("that album is incredible")
        assert resolution.status == "not_a_request"
        assert pending_failure_context() == ""
