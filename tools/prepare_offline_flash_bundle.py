#!/usr/bin/env python3
"""Copy a catalog's original programs into a hashed offline browser bundle.

Catalog rows are tab-separated title, uploaded file path and optional archive
identifier. Unavailable or invalid programs remain rows with explicit errors.
The bundle preserves original bytes; it does not implement game strategies.
Ruffle is pinned and checked against the package registry's integrity digest.

A program may load other files from where it was kept (its levels, its pictures), by paths relative to it. Each case's
other files are fetched from the same archive item its program came from (``--dependencies``), kept under
``games/<case>/`` with their hashes, and the case is played with that folder as its base: offline 2026-10-11 a trap
game asked for its levels, which its archive item holds and the bundle did not, and stood at LOADING LEVEL.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import lzma
import sys
import tarfile
import urllib.request
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MAX_PROGRAM_BYTES = 64 * 1024 * 1024
RUFFLE_VERSION = "0.7.1"


def swf_metadata(data: bytes) -> dict:
    """Validate the whole decompressed byte extent before accepting a program."""
    if len(data) < 8 or data[:3] not in (b"FWS", b"CWS", b"ZWS"):
        raise ValueError("not a Flash program")
    declared = int.from_bytes(data[4:8], "little")
    if not 8 <= declared <= MAX_PROGRAM_BYTES:
        raise ValueError("program extent exceeds the bounded reader")
    if data[:3] == b"CWS":
        decoder = zlib.decompressobj()
        body = decoder.decompress(data[8:], declared - 7)
        if not decoder.eof or decoder.unconsumed_tail or decoder.unused_data:
            raise ValueError("compressed extent is incomplete or contains trailing data")
    elif data[:3] == b"ZWS":
        if len(data) < 17:
            raise ValueError("missing compressed properties")
        compressed = data[12:17] + (declared - 8).to_bytes(8, "little") + data[17:]
        decoder = lzma.LZMADecompressor(format=lzma.FORMAT_ALONE, memlimit=MAX_PROGRAM_BYTES)
        body = decoder.decompress(compressed, max_length=declared - 7)
        if not decoder.eof or decoder.unused_data:
            raise ValueError("compressed extent is incomplete or contains trailing data")
    else:
        body = data[8:]
    if len(body) + 8 != declared:
        raise ValueError(f"declared extent {declared} differs from measured extent {len(body) + 8}")
    if not body:
        raise ValueError("empty program body")
    # RECT is bit-packed, followed by the fixed8 frame rate and frame count.
    rect_bytes = (5 + 4 * (body[0] >> 3) + 7) // 8
    if len(body) < rect_bytes + 4:
        raise ValueError("missing timeline header")
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
            "declared_bytes": declared, "compression": data[:3].decode("ascii"),
            "version": data[3], "frame_rate": int.from_bytes(body[rect_bytes:rect_bytes + 2], "little") / 256,
            "frames": int.from_bytes(body[rect_bytes + 2:rect_bytes + 4], "little")}


def write_bytes(target: Path, data: bytes) -> None:
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    with local_internal_governed_scope("offline_environment_bundle"):
        get_file_write_gateway().write_bytes(target, data, source="offline_environment_bundle")


def install_ruffle(destination: Path) -> dict:
    address = f"https://registry.npmjs.org/@ruffle-rs/ruffle/{RUFFLE_VERSION}"
    with urllib.request.urlopen(address, timeout=30) as answer:
        metadata = json.load(answer)
    dist = metadata["dist"]
    with urllib.request.urlopen(dist["tarball"], timeout=60) as answer:
        data = answer.read(MAX_PROGRAM_BYTES + 1)
    algorithm, expected = dist["integrity"].split("-", 1)
    if algorithm != "sha512" or len(data) > MAX_PROGRAM_BYTES:
        raise ValueError("unsupported package integrity or oversized package")
    actual = base64.b64encode(hashlib.sha512(data).digest()).decode("ascii")
    if actual != expected:
        raise ValueError("emulator package failed its integrity check")
    files = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as package:
        for member in package.getmembers():
            relative = Path(member.name)
            if (not member.isfile() or relative.parts[:1] != ("package",)
                    or ".." in relative.parts or member.size > MAX_PROGRAM_BYTES):
                raise ValueError("unexpected package member")
            stream = package.extractfile(member)
            if stream is None:
                raise ValueError("package member cannot be read")
            content = stream.read()
            relative = Path(*relative.parts[1:])
            write_bytes(destination / relative, content)
            files[str(relative)] = hashlib.sha256(content).hexdigest()
    return {"name": metadata["name"], "version": metadata["version"],
            "integrity": dist["integrity"], "files": files}


#: The page each game is played on. It names the game as a page that hosts one does, in its title and over the game:
#: a game played from a page that called itself "Offline program" was looked up by that name (offline 2026-10-11).
PLAYER = """<!doctype html><html><head><meta charset="utf-8"><title>Offline program</title>
<style>body{margin:0;background:#111;color:#eee;font:15px sans-serif}h1{margin:0;padding:6px 12px;font-size:16px;height:20px}
#stage{width:100vw;height:calc(100vh - 32px)}ruffle-player{width:100%;height:100%}</style></head><body>
<h1 id="title"></h1><div id="stage"></div>
<script>window.RufflePlayer={config:{autoplay:'auto',splashScreen:true,allowScriptAccess:false,
openUrlMode:'deny',logLevel:'info'}};</script><script src="ruffle/ruffle.js"></script>
<script>const number=new URLSearchParams(location.search).get('case');
if(!/^\\d+$/.test(number)) throw Error('a numeric case is required');
fetch('manifest.json').then(r=>r.json()).then(m=>{const c=m.cases.find(c=>String(c.id)===number);
if(c){document.title=c.title;document.getElementById('title').textContent=c.title;}});
const player=window.RufflePlayer.newest().createPlayer();
document.getElementById('stage').appendChild(player);
player.ruffle().load({url:'programs/'+number+'.swf',base:'games/'+number+'/'}).catch(error=>{
document.getElementById('stage').textContent='Program failed to load: '+error;throw error;});
</script></body></html>"""


def prepare(catalog: Path, assets: Path, output: Path, expected_count: int | None = None) -> dict:
    rows = [line.split("\t") for line in catalog.read_text().splitlines() if line.strip()]
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"catalog has {len(rows)} rows; expected {expected_count}")
    cases = []
    for number, row in enumerate(rows):
        if len(row) < 2 or not row[0].strip():
            raise ValueError(f"invalid catalog row {number}")
        source = assets / Path(row[1]).name
        case = {"id": number, "title": row[0], "catalog_source": row[1],
                "archive_identifier": row[2] if len(row) > 2 else "", "status": "unresolved"}
        try:
            data = source.read_bytes()
            case["program"] = swf_metadata(data)
            write_bytes(output / "programs" / f"{number}.swf", data)
            case.update(status="asset_verified", url=f"index.html?case={number}")
        except (OSError, ValueError, zlib.error, lzma.LZMAError) as error:
            case["error"] = f"{type(error).__name__}: {error}"
        cases.append(case)
    emulator = install_ruffle(output / "ruffle")
    write_bytes(output / "index.html", PLAYER.encode())
    manifest = {"schema_version": 1, "catalog_sha256": hashlib.sha256(catalog.read_bytes()).hexdigest(),
                "emulator": emulator, "cases": cases, "passed": 0,
                "qualification": "unmeasured", "randomness": "original program RNG is not controlled",
                "dependency_completeness": "not established; capture failed requests during execution"}
    write_bytes(output / "manifest.json", json.dumps(manifest, indent=2).encode())
    return manifest


#: Files of an archive item that are its record, not the program's: never fetched as dependencies.
_RECORD = (".xml", ".sqlite", ".torrent", "_thumb.jpg", "_screenshot.jpg")
#: The largest dependency fetched, and how long one fetch may take.
MAX_DEPENDENCY_BYTES = 32 * 1024 * 1024
FETCH_TIMEOUT_S = 60


def _download_dependency(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=FETCH_TIMEOUT_S) as answer:
        data = answer.read(MAX_DEPENDENCY_BYTES + 1)
    if len(data) > MAX_DEPENDENCY_BYTES:
        raise ValueError("dependency exceeds the bounded reader")
    return data


def _item_files(identifier: str) -> list[dict]:
    with urllib.request.urlopen(f"https://archive.org/metadata/{urllib.request.quote(identifier, safe='')}", timeout=FETCH_TIMEOUT_S) as answer:
        data = answer.read(4 * 1024 * 1024 + 1)
    if len(data) > 4 * 1024 * 1024:
        raise ValueError("archive metadata exceeds the bounded reader")
    return list(json.loads(data).get("files") or [])


def fetch_dependencies(output: Path, list_files=_item_files, fetch=None) -> dict:
    """Each verified case's other files, from the archive item its program came from: kept under ``games/<case>/`` by
    their own paths, with their hashes, in the manifest. A file that is the program itself, the item's record, too
    large, or that leaves the case's folder is not kept."""
    fetch = fetch or _download_dependency
    manifest = json.loads((output / "manifest.json").read_text())
    for case in manifest["cases"]:
        identifier = case.get("archive_identifier") or ""
        if case.get("status") != "asset_verified" or not identifier:
            continue
        program = Path(case.get("catalog_source") or "").name
        kept: dict[str, str] = {}
        try:
            files = list_files(identifier)
        except (OSError, ValueError) as error:
            case["dependencies_error"] = f"{type(error).__name__}: {error}"
            continue
        folder = (output / "games" / str(case["id"])).resolve()
        for item in files:
            name = str(item.get("name") or "")
            try:
                size = int(item.get("size") or 0)
            except (TypeError, ValueError):
                case.setdefault("dependency_errors", {})[name] = "invalid declared extent"
                continue
            if (not name or name.endswith(_RECORD) or name == program or name.lower().endswith(".swf") and "/" not in name
                    and size == int(case["program"]["bytes"])):
                continue
            if not 0 <= size <= MAX_DEPENDENCY_BYTES:
                case.setdefault("dependency_errors", {})[name] = "declared extent exceeds the bounded reader"
                continue
            target = (folder / name).resolve()
            if not target.is_relative_to(folder):
                continue
            try:
                data = fetch(f"https://archive.org/download/{urllib.request.quote(identifier, safe='')}/{urllib.request.quote(name)}")
                if len(data) > MAX_DEPENDENCY_BYTES or size and len(data) != size:
                    raise ValueError("downloaded extent differs from the bounded archive record")
            except (OSError, ValueError) as error:
                case.setdefault("dependency_errors", {})[name] = f"{type(error).__name__}: {error}"
                continue
            write_bytes(target, data)
            kept[name] = hashlib.sha256(data).hexdigest()
        case["dependencies"] = kept
    manifest["dependency_completeness"] = "unestablished; listed archive dependencies retained with hashes and explicit fetch errors"
    write_bytes(output / "manifest.json", json.dumps(manifest, indent=2).encode())
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expect-count", type=int)
    parser.add_argument("--dependencies", action="store_true", help="fetch each case's other files into the bundle")
    args = parser.parse_args()
    if args.dependencies:
        manifest = fetch_dependencies(args.out)
        print(json.dumps({case["id"]: sorted(case.get("dependencies") or {}) for case in manifest["cases"]
                          if case.get("dependencies")}), flush=True)
        return 0
    if args.catalog is None or args.assets is None:
        parser.error("--catalog and --assets are needed to prepare a bundle")
    manifest = prepare(args.catalog, args.assets, args.out, args.expect_count)
    cases = manifest["cases"]
    print(json.dumps({"cases": len(cases), "assets_verified": sum(case["status"] == "asset_verified" for case in cases),
                      "passed": 0, "unresolved_assets": [{"id": case["id"], "error": case["error"]}
                                                       for case in cases if "error" in case]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
