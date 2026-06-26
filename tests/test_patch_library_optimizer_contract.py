from __future__ import annotations

import asyncio
import json
from pathlib import Path

from core.optimizer import Optimizer
from core.patch_library import AVAILABLE_PATCHES, GitInitPatch, PatchStrategy, PipInstallPatch


class RecordingPatch(PatchStrategy):
    name = "recording"

    def __init__(self) -> None:
        self.received: list[str] = []

    def match(self, failure_reason: str) -> bool:
        return "needs-repair" in failure_reason

    async def apply(self, failure_reason: str) -> bool:
        self.received.append(failure_reason)
        return True


def test_patch_library_exports_available_patches() -> None:
    names = {patch.name for patch in AVAILABLE_PATCHES}

    assert "git_init_fix" in names
    assert "pip_install_fix" in names


def test_git_repair_defaults_to_governed_proposal(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AURA_ROOT", str(tmp_path / ".aura"))
    monkeypatch.delenv("AURA_ALLOW_AUTONOMIC_GIT_REPAIR", raising=False)

    result = asyncio.run(GitInitPatch().apply("fatal: not a git repository"))

    proposals = list((tmp_path / ".aura" / "data" / "repair_proposals").glob("git_init_fix-*.md"))
    assert result is False
    assert len(proposals) == 1
    text = proposals[0].read_text(encoding="utf-8")
    assert "not executed automatically" in text
    assert "git init" in text


def test_pip_install_defaults_to_governed_proposal(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AURA_ROOT", str(tmp_path / ".aura"))
    monkeypatch.delenv("AURA_ALLOW_AUTONOMIC_PIP_INSTALL", raising=False)

    result = asyncio.run(PipInstallPatch().apply("ModuleNotFoundError: No module named 'aiohttp'"))

    proposals = list((tmp_path / ".aura" / "data" / "repair_proposals").glob("pip_install_fix.aiohttp-*.md"))
    assert result is False
    assert len(proposals) == 1
    text = proposals[0].read_text(encoding="utf-8")
    assert "not executed automatically" in text
    assert "pip install aiohttp" in text


def test_optimizer_passes_failure_signature_to_patch_and_archives(tmp_path: Path) -> None:
    hard_examples = tmp_path / "hard_examples.json"
    hard_examples.write_text(
        json.dumps([{"reason": "needs-repair", "outcome": {"detail": "broken"}}]),
        encoding="utf-8",
    )
    patch = RecordingPatch()

    asyncio.run(Optimizer(str(hard_examples), patches=[patch]).run())

    assert len(patch.received) == 1
    assert "needs-repair" in patch.received[0]
    assert "broken" in patch.received[0]
    assert json.loads(hard_examples.read_text(encoding="utf-8")) == []
    assert hard_examples.with_suffix(".json.processed").exists()
