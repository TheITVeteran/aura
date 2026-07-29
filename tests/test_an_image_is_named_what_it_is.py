"""An image file should be named what it actually is.

Live 2026-07-28: "download it to my Desktop, and set it as my wallpaper."
A real grizzly image arrived and landed as grizzly_bear_wallpaper.png —
containing JPEG data. `file` reported "JPEG image data, JFIF standard 1.01"
for a .png.

Nothing downstream had lied. The planner asks for a .png because that is what
it wrote into the plan, and the web returns whatever the web returns. The
extension was the only dishonest part, and anything that trusts a suffix —
another tool, a preview, a person looking at their Desktop — inherits that.

Bryan asked for exactly this check: make sure the image is what it says it
is. The bytes are the authority, not the filename.
"""

import pytest

from core.skills.computer_use import _image_suffix_from_bytes

pytestmark = pytest.mark.unit

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 16
JPEG = b"\xff\xd8\xff\xe0" + b"0" * 16
GIF = b"GIF89a" + b"0" * 16
WEBP = b"RIFF" + b"0000" + b"WEBP" + b"0" * 8


@pytest.mark.parametrize("raw,expected", [
    (PNG, ".png"),
    (JPEG, ".jpg"),
    (GIF, ".gif"),
    (WEBP, ".webp"),
])
def test_the_type_is_read_from_the_bytes(raw: bytes, expected: str):
    assert _image_suffix_from_bytes(raw) == expected


@pytest.mark.parametrize("raw", [b"", b"short", b"not an image at all", None, "text"])
def test_unrecognised_content_claims_nothing(raw):
    """Unknown must not guess a suffix — a wrong rename is worse than none."""
    assert _image_suffix_from_bytes(raw) == ""


def test_the_fetcher_renames_to_the_sniffed_type():
    import inspect

    from core.skills import computer_use

    source = inspect.getsource(computer_use.ComputerUseSkill._fetch_topic_image)
    assert "_image_suffix_from_bytes(raw)" in source
    assert "path.with_suffix(sniffed)" in source
