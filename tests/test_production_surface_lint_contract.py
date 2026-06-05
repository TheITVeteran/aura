from __future__ import annotations

import ast

from tools.production_surface_lint import AstLinter


def _findings(source: str) -> list[str]:
    visitor = AstLinter("core/example.py")
    visitor.visit(ast.parse(source))
    return [finding.kind for finding in visitor.findings]


def test_lint_blocks_asyncio_subprocess_exec() -> None:
    kinds = _findings(
        """
import asyncio

async def main():
    await asyncio.create_subprocess_exec("python", "-V")
"""
    )

    assert "unapproved_direct_subprocess" in kinds


def test_lint_blocks_asyncio_subprocess_shell() -> None:
    kinds = _findings(
        """
import asyncio

async def main():
    await asyncio.create_subprocess_shell("python -V")
"""
    )

    assert "unapproved_direct_subprocess" in kinds


def test_lint_blocks_subprocess_callable_wrapped_in_to_thread() -> None:
    kinds = _findings(
        """
import asyncio
import subprocess

async def main():
    await asyncio.to_thread(subprocess.run, ["python", "-V"])
"""
    )

    assert "unapproved_direct_subprocess" in kinds


def test_lint_blocks_network_callable_wrapped_in_to_thread() -> None:
    kinds = _findings(
        """
import asyncio
import requests

async def main():
    await asyncio.to_thread(requests.get, "https://example.com")
"""
    )

    assert "unapproved_direct_network" in kinds


def test_lint_blocks_network_callable_import_alias_wrapped_in_to_thread() -> None:
    kinds = _findings(
        """
import asyncio
from requests import get as http_get

async def main():
    await asyncio.to_thread(http_get, "https://example.com")
"""
    )

    assert "unapproved_direct_network" in kinds


def test_lint_blocks_urllib_request_module_alias() -> None:
    kinds = _findings(
        """
import urllib.request as ureq

def main():
    return ureq.urlopen("https://example.com")
"""
    )

    assert "unapproved_direct_network" in kinds


def test_lint_blocks_urllib_urlretrieve_alias() -> None:
    kinds = _findings(
        """
from urllib.request import urlretrieve

def main():
    urlretrieve("https://example.com/model.onnx", "model.onnx")
"""
    )

    assert "unapproved_direct_network" in kinds


def test_lint_blocks_subprocess_import_alias_wrapped_in_to_thread() -> None:
    kinds = _findings(
        """
import asyncio
from subprocess import run as proc_run

async def main():
    await asyncio.to_thread(proc_run, ["python", "-V"])
"""
    )

    assert "unapproved_direct_subprocess" in kinds


def test_lint_blocks_os_spawn_family() -> None:
    kinds = _findings(
        """
import os

def main():
    return os.posix_spawnp("python", ["python", "-V"], os.environ.copy())
"""
    )

    assert "unapproved_direct_subprocess" in kinds


def test_lint_blocks_os_spawn_alias() -> None:
    kinds = _findings(
        """
from os import spawnvp

def main():
    return spawnvp(0, "python", ["python", "-V"])
"""
    )

    assert "unapproved_direct_subprocess" in kinds


def test_lint_blocks_path_constructor_write_text() -> None:
    kinds = _findings(
        """
from pathlib import Path

def save():
    Path("runtime.txt").write_text("unsafe")
"""
    )

    assert "unapproved_direct_file_write" in kinds


def test_lint_allows_wave_open_on_bytesio_buffer() -> None:
    kinds = _findings(
        """
import io
import wave

def synthesize():
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
"""
    )

    assert "unapproved_direct_file_write" not in kinds


def test_lint_allows_file_write_gateway_factory_write_text() -> None:
    kinds = _findings(
        """
from core.runtime.file_write_gateway import get_file_write_gateway

def save(path):
    get_file_write_gateway().write_text(path, "safe", source="test")
"""
    )

    assert "unapproved_direct_file_write" not in kinds
