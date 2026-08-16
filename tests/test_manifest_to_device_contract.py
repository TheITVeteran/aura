from __future__ import annotations

from pathlib import Path

import pytest

from core.governance_context import local_internal_governed_scope
from core.skills import manifest_to_device as manifest_module
from core.skills.manifest_to_device import ManifestToDeviceSkill


class _NetworkGateway:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    async def request_async(self, *args: object, **kwargs: object) -> dict[str, object]:
        self.calls.append((args, kwargs))
        return self.response


class _FileGateway:
    def __init__(self) -> None:
        self.events: list[tuple[str, Path, bytes | None]] = []

    async def ensure_directory_async(self, path: object, **_kwargs: object) -> str:
        target = Path(path)
        self.events.append(("ensure", target, None))
        return str(target)

    async def write_bytes_async(
        self, path: object, payload: bytes, **_kwargs: object
    ) -> None:
        self.events.append(("write", Path(path), bytes(payload)))


def test_constructor_has_no_filesystem_side_effect(monkeypatch: pytest.MonkeyPatch) -> None:
    def _unexpected_gateway() -> object:
        raise AssertionError("skill construction must not reach the write gateway")

    monkeypatch.setattr(manifest_module, "get_file_write_gateway", _unexpected_gateway)
    skill = ManifestToDeviceSkill()
    assert skill.desktop_path.name == "Aura_Manifests"


@pytest.mark.asyncio
async def test_manifest_download_is_governed_bounded_and_receipted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    network = _NetworkGateway(
        {
            "status_code": 200,
            "ok": True,
            "headers": {"Content-Type": "image/png"},
            "content": b"real-png-payload",
            "url": "https://cdn.example.test/orca.png",
        }
    )
    files = _FileGateway()
    monkeypatch.setattr(manifest_module, "get_network_gateway", lambda: network)
    monkeypatch.setattr(manifest_module, "get_file_write_gateway", lambda: files)

    skill = ManifestToDeviceSkill()
    skill.desktop_path = tmp_path / "Aura_Manifests"
    with local_internal_governed_scope("test.manifest", domain="tool_execution"):
        result = await skill.execute(
            {"url": "https://example.test/orca.png", "filename": "orca.png"}, {}
        )

    assert result["ok"] is True
    assert result["bytes_written"] == len(b"real-png-payload")
    assert result["source_url"] == "https://cdn.example.test/orca.png"
    assert result["path"] == str(skill.desktop_path / "orca.png")
    assert [event[0] for event in files.events] == ["ensure", "write"]
    assert files.events[1][2] == b"real-png-payload"
    assert network.calls[0][1]["read_only"] is True
    assert network.calls[0][1]["max_response_bytes"] == 64 * 1024 * 1024
    assert network.calls[0][1]["public_network_only"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("filename", ["../escape.png", "nested/escape.png", "..", "bad\\name.png"])
async def test_manifest_rejects_path_bearing_filename_before_network(
    monkeypatch: pytest.MonkeyPatch, filename: str
) -> None:
    network = _NetworkGateway({"status_code": 200, "ok": True, "content": b"x"})
    monkeypatch.setattr(manifest_module, "get_network_gateway", lambda: network)
    skill = ManifestToDeviceSkill()

    with local_internal_governed_scope("test.manifest", domain="tool_execution"):
        result = await skill.execute(
            {"url": "https://example.test/orca.png", "filename": filename}, {}
        )

    assert result["ok"] is False
    assert "filename" in result["error"]
    assert network.calls == []


@pytest.mark.asyncio
async def test_manifest_refuses_empty_download_without_writing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    network = _NetworkGateway(
        {"status_code": 200, "ok": True, "headers": {}, "content": b""}
    )
    files = _FileGateway()
    monkeypatch.setattr(manifest_module, "get_network_gateway", lambda: network)
    monkeypatch.setattr(manifest_module, "get_file_write_gateway", lambda: files)
    skill = ManifestToDeviceSkill()

    with local_internal_governed_scope("test.manifest", domain="tool_execution"):
        result = await skill.execute({"url": "https://example.test/empty"}, {})

    assert result["ok"] is False
    assert "empty body" in result["error"]
    assert files.events == []
