"""A downloaded file is not an image because its first bytes say so.

CP126 (critical), core/capabilities/web_asset_handler.py: "MIME rejection is
fail-open and magic-byte checks do not validate an image."

Both halves were true. A non-image Content-Type only logged a warning and
processing continued — the code said so out loud:

    if not content_type.startswith("image/"):
        logger.warning("Not an image: %s (type=%s)", url[:60], content_type)
        # Try anyway — some servers don't set correct type

And the validator accepted two to eight leading bytes with no structural
decode, no truncation check, no decompression-ratio check, no frame count,
no pixel count, and no test for anything appended after the image.

A third defect surfaced while fixing those. The WEBP rule appeared twice:
once correctly as ``RIFF`` + ``WEBP`` at offset 8, and once — FIRST — as a
bare ``data[8:12] == b"WEBP"``. Any file whose ninth through twelfth bytes
spelled WEBP was admitted as an image regardless of the rest of it.

The polyglot is the case worth naming. Pillow decodes a well-formed PNG
with a shell script stapled to the end without complaint, because "are the
pixels valid" is a different question from "is this file ONLY an image".
Only the container's own declared extent answers the second one.
"""
from __future__ import annotations

import io

import pytest

from core.capabilities.web_asset_handler import (
    _MAX_TRAILING_BYTES,
    WebAssetHandler,
    _structural_image_check,
)

pytest.importorskip("PIL", reason="structural admission needs a decoder")


def _image(fmt: str, width: int = 8, height: int = 8) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (1, 2, 3)).save(buffer, fmt)
    return buffer.getvalue()


def _admit(data: bytes):
    ok, fmt = WebAssetHandler._validate_image_header(data)
    if not ok:
        return None
    return _structural_image_check(data, fmt)


class TestGenuineImagesStillWork:
    """Over-refusal is the opposite failure and just as real."""

    @pytest.mark.parametrize("fmt", ["PNG", "JPEG", "GIF", "BMP", "WEBP"])
    def test_a_real_image_is_admitted(self, fmt):
        admission = _admit(_image(fmt))
        assert admission is not None, f"{fmt} failed the magic-byte prefilter"
        assert admission.ok, admission.reason
        assert admission.structurally_verified is True
        assert admission.width == 8 and admission.height == 8

    def test_a_small_metadata_trailer_is_tolerated(self):
        """Legitimate files carry trailing metadata; the slack is deliberate."""
        assert _admit(_image("PNG") + b"X" * 100).ok is True


class TestPolyglotsAreRefused:
    """The named case: decodable as an image AND as something else."""

    def test_a_png_with_an_appended_script_is_refused(self):
        hostile = _image("PNG") + b"\n<script>alert(1)</script>" + b"A" * 8000
        admission = _admit(hostile)
        assert admission.ok is False
        assert "trailing_data_after_image" in admission.reason

    def test_a_jpeg_with_an_appended_binary_is_refused(self):
        hostile = _image("JPEG") + b"\x7fELF" + b"\x00" * 9000
        admission = _admit(hostile)
        assert admission.ok is False
        assert "trailing_data_after_image" in admission.reason

    def test_pillow_alone_would_have_accepted_it(self):
        """Pins WHY the container-extent check exists, by showing the
        decoder disagreeing with it."""
        from PIL import Image

        hostile = _image("PNG") + b"A" * (_MAX_TRAILING_BYTES + 1000)
        # The decoder is perfectly happy.
        with Image.open(io.BytesIO(hostile)) as decoded:
            decoded.load()
            assert decoded.format == "PNG"
        # The admission check is not.
        assert _admit(hostile).ok is False


class TestMalformedBodiesAreRefused:
    def test_a_truncated_image_is_refused(self):
        real = _image("PNG")
        admission = _admit(real[: len(real) // 2])
        assert admission.ok is False
        assert "structural_decode_failed" in admission.reason

    def test_magic_bytes_with_no_body_are_refused(self):
        admission = _admit(b"\xff\xd8\xff" + b"\x00" * 300)
        assert admission.ok is False


class TestTheMagicBytePrefilter:
    def test_the_loose_webp_rule_is_gone(self):
        """The actual bug: any file with "WEBP" at offset 8 was an image."""
        javascript = b"var x=1;WEBP" + b"alert(1);" * 40
        ok, _fmt = WebAssetHandler._validate_image_header(javascript)
        assert ok is False

    def test_html_is_not_an_image(self):
        ok, _fmt = WebAssetHandler._validate_image_header(
            b"<!DOCTYPE html><html><body>hi</body></html>" + b" " * 200,
        )
        assert ok is False

    def test_a_short_buffer_is_not_an_image(self):
        assert WebAssetHandler._validate_image_header(b"\xff\xd8")[0] is False


class TestTheMimeGateFailsClosed:
    @pytest.mark.asyncio
    async def test_a_non_image_content_type_is_never_written(self, tmp_path, monkeypatch):
        """The defect, driven end to end rather than read from the source.

        A server returning text/html used to have its body persisted under
        an image filename. Nothing may reach the write gateway now.
        """
        from core.capabilities import web_asset_handler as mod

        writes: list = []

        class _Net:
            async def request_async(self, *_a, **_kw):
                return {
                    "ok": True,
                    "headers": {"Content-Type": "text/html; charset=utf-8"},
                    "content": b"<!DOCTYPE html><html>" + b"x" * 500,
                }

        class _Writer:
            async def write_bytes_async(self, *a, **kw):
                writes.append(a)

        monkeypatch.setattr(mod, "get_network_gateway", lambda: _Net())
        monkeypatch.setattr(mod, "get_file_write_gateway", lambda: _Writer())

        result = await mod.WebAssetHandler().download_image(
            "https://example.invalid/x", save_dir=str(tmp_path),
        )
        assert result == ""
        assert writes == [], "a non-image body reached the write gateway"

    @pytest.mark.asyncio
    async def test_a_real_image_still_downloads(self, tmp_path, monkeypatch):
        """The gate must not refuse everything to look safe."""
        from core.capabilities import web_asset_handler as mod

        writes: list = []

        class _Net:
            async def request_async(self, *_a, **_kw):
                return {
                    "ok": True,
                    "headers": {"Content-Type": "image/png"},
                    "content": _image("PNG", 32, 32),
                }

        class _Writer:
            async def ensure_directory_async(self, path, **_kw):
                return str(path)

            async def write_bytes_async(self, path, data, **kw):
                writes.append((path, len(data)))

        monkeypatch.setattr(mod, "get_network_gateway", lambda: _Net())
        monkeypatch.setattr(mod, "get_file_write_gateway", lambda: _Writer())

        result = await mod.WebAssetHandler().download_image(
            "https://example.invalid/pic.png", save_dir=str(tmp_path),
        )
        assert result, "a genuine PNG was refused"
        assert len(writes) == 1

    def test_a_non_image_content_type_returns_before_writing(self):
        import inspect

        from core.capabilities import web_asset_handler

        source = inspect.getsource(web_asset_handler.WebAssetHandler.download_image)
        mime_block = source[source.index("declared_mime"):]
        refusal = mime_block[: mime_block.index("Read data with size limit")]
        assert 'return ""' in refusal


class TestHonestyAboutWhatWasChecked:
    def test_a_missing_decoder_is_reported_not_promoted(self, monkeypatch):
        """Absence of a check must not be reported as a passed check.

        Pillow is optional. Without it there is no structural verification,
        and the admission says so instead of claiming the image is clean.
        """
        import builtins

        # Build the sample BEFORE hiding the decoder — the helper needs it.
        sample = _image("PNG")
        real_import = builtins.__import__

        def _no_pil(name, *args, **kwargs):
            if name == "PIL" or name.startswith("PIL."):
                raise ImportError("no pillow")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_pil)
        admission = _structural_image_check(sample, "png")
        assert admission.structurally_verified is False
        assert "pillow_unavailable" in admission.reason

    def test_the_verdict_is_serializable_with_its_provenance(self):
        payload = _admit(_image("PNG")).to_dict()
        assert payload["valid"] is True
        assert payload["structurally_verified"] is True
        assert payload["format"] == "png"
