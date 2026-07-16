from __future__ import annotations

from pathlib import Path

import pytest
from starlette.requests import Request

from core.runtime.launch_provenance import (
    RUNTIME_SHELL_ASSETS,
    capture_runtime_shell_assets,
)
from core.runtime.runtime_shell_snapshot import (
    clear_runtime_shell_snapshots,
    publish_runtime_shell_snapshot,
    runtime_shell_snapshot_asset,
    runtime_shell_snapshot_known,
)


def _write_shell(root: Path) -> None:
    for index, relative in enumerate(RUNTIME_SHELL_ASSETS):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"asset-{index}:{relative}\n".encode())


def _request(path: str, *, query: str = "", referer: str = "") -> Request:
    headers = [(b"host", b"127.0.0.1:8000")]
    if referer:
        headers.append((b"referer", referer.encode()))
    return Request(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": query.encode(),
            "headers": headers,
            "client": ("127.0.0.1", 50000),
            "server": ("127.0.0.1", 8000),
        }
    )


@pytest.fixture(autouse=True)
def _isolated_snapshots():
    clear_runtime_shell_snapshots()
    yield
    clear_runtime_shell_snapshots()


def test_published_revision_keeps_exact_bytes_after_source_mutation(tmp_path: Path):
    _write_shell(tmp_path)
    digest, assets = capture_runtime_shell_assets(tmp_path)
    revision = "a" * 64
    expected = assets["interface/static/aura.js"]

    publish_runtime_shell_snapshot(
        revision_token=revision,
        shell_assets_sha256=digest,
        assets=assets,
    )
    (tmp_path / "interface/static/aura.js").write_bytes(b"mutated after attestation")

    assert runtime_shell_snapshot_known(revision) is True
    assert runtime_shell_snapshot_asset(revision, "/static/aura.js") == expected
    assert runtime_shell_snapshot_asset("b" * 64, "/static/aura.js") is None


def test_snapshot_publication_rejects_tampered_bytes(tmp_path: Path):
    _write_shell(tmp_path)
    digest, assets = capture_runtime_shell_assets(tmp_path)
    assets["interface/static/aura.js"] = b"tampered"

    with pytest.raises(RuntimeError, match="do not match"):
        publish_runtime_shell_snapshot(
            revision_token="a" * 64,
            shell_assets_sha256=digest,
            assets=assets,
        )


def test_capture_rejects_symlinked_parent_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from core.runtime import launch_provenance

    target = tmp_path / "target"
    target.mkdir()
    (target / "aura.js").write_bytes(b"payload")
    interface = tmp_path / "interface"
    interface.mkdir()
    (interface / "static").symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(
        launch_provenance,
        "RUNTIME_SHELL_ASSETS",
        ("interface/static/aura.js",),
    )

    with pytest.raises(RuntimeError, match="symlink"):
        launch_provenance.capture_runtime_shell_assets(tmp_path)


@pytest.mark.asyncio
async def test_server_serves_only_known_frozen_revision_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from interface import server

    _write_shell(tmp_path)
    digest, assets = capture_runtime_shell_assets(tmp_path)
    revision = "a" * 64
    publish_runtime_shell_snapshot(
        revision_token=revision,
        shell_assets_sha256=digest,
        assets=assets,
    )
    expected = assets["interface/static/aura.js"]
    monkeypatch.setattr(server, "validate_runtime_security_request", lambda _request: None)

    async def forbidden(_request):
        raise AssertionError("revision-addressed shell request fell through to mutable disk")

    response = await server.serve_immutable_runtime_shell(
        _request("/static/aura.js", query=f"_aura_runtime={revision}"),
        forbidden,
    )
    assert response.status_code == 200
    assert response.body == expected
    assert response.headers["x-aura-runtime-revision"] == revision
    assert response.headers["cache-control"].endswith("immutable")

    unknown = await server.serve_immutable_runtime_shell(
        _request("/static/aura.js", query=f"_aura_runtime={'b' * 64}"),
        forbidden,
    )
    assert unknown.status_code == 409


@pytest.mark.asyncio
async def test_revision_marked_document_gets_frozen_unaddressed_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from interface import server

    _write_shell(tmp_path)
    digest, assets = capture_runtime_shell_assets(tmp_path)
    revision = "c" * 64
    publish_runtime_shell_snapshot(
        revision_token=revision,
        shell_assets_sha256=digest,
        assets=assets,
    )
    monkeypatch.setattr(server, "validate_runtime_security_request", lambda _request: None)

    async def forbidden(_request):
        raise AssertionError("revision referrer fell through to mutable disk")

    response = await server.serve_immutable_runtime_shell(
        _request(
            "/static/aura.js",
            referer=f"http://127.0.0.1:8000/?_aura_runtime={revision}",
        ),
        forbidden,
    )
    assert response.status_code == 200
    assert response.body == assets["interface/static/aura.js"]
    assert response.headers["cache-control"].startswith("no-store")
