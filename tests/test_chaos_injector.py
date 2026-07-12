"""Contract tests for tools/chaos/injector.py.

A chaos tool that has rotted is worse than none — it manufactures false
confidence. These tests pin: (1) the documented catalogue exactly matches
the registered faults (no claimed-but-unbuilt entries); (2) every fault
that mutates state restores it (full apply→restore round-trips under a
shortened restore window); (3) the random entry point never raises.
"""

from __future__ import annotations

import asyncio
import os
import re

import pytest

from tools.chaos import injector


def _documented_catalogue() -> set[str]:
    """Fault names from the module docstring's Catalogue section."""
    doc = injector.__doc__ or ""
    section = doc.split("Catalogue", 1)[1].split("Roadmap", 1)[0]
    return set(re.findall(r"^\s{2}([a-z_]+)\s+—", section, flags=re.MULTILINE))


class TestCatalogueHonesty:
    def test_docstring_matches_registry_exactly(self):
        assert _documented_catalogue() == set(injector._FAULTS), (
            "docstring catalogue and fault registry drifted — a documented "
            "fault that is not registered is a false capability claim"
        )

    def test_roadmap_faults_are_not_registered(self):
        for name in ("kill_subprocess", "corrupt_sqlite_row",
                     "break_memory_facade", "break_agency_pathway"):
            assert name not in injector._FAULTS


@pytest.fixture()
def fast_restore(monkeypatch):
    monkeypatch.setenv("AURA_CHAOS_RESTORE_SECONDS", "0.05")


async def _drain_restores():
    """Give scheduled restore tasks time to run under the shortened window."""
    await asyncio.sleep(0.3)


class TestRoundTrips:
    @pytest.mark.asyncio
    async def test_model_load_failure_applies_and_restores(
        self, fast_restore, monkeypatch
    ):
        monkeypatch.setenv("AURA_MODEL", "/real/model/path")
        out = await injector._FAULTS["force_model_load_failure"]()
        assert out["applied"] is True
        assert os.environ["AURA_MODEL"] != "/real/model/path"
        await _drain_restores()
        assert os.environ["AURA_MODEL"] == "/real/model/path"

    @pytest.mark.asyncio
    async def test_expire_api_keys_round_trip(self, fast_restore, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-original")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        out = await injector._FAULTS["expire_api_keys"]()
        assert out["applied"] is True
        assert os.environ["OPENAI_API_KEY"] == "invalid-injection"
        await _drain_restores()
        assert os.environ["OPENAI_API_KEY"] == "sk-original"
        assert "ANTHROPIC_API_KEY" not in os.environ, (
            "restore must unset keys that were absent before injection"
        )

    @pytest.mark.asyncio
    async def test_sever_network_round_trip(self, fast_restore, monkeypatch):
        monkeypatch.delenv("HTTPS_PROXY", raising=False)
        out = await injector._FAULTS["sever_network"]()
        assert out["applied"] is True
        assert os.environ["HTTPS_PROXY"] == "http://127.0.0.1:1"
        await _drain_restores()
        assert "HTTPS_PROXY" not in os.environ

    @pytest.mark.asyncio
    async def test_fill_disk_bounded_in_safe_target_and_cleaned(
        self, fast_restore, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("AURA_CHAOS_DISK_TARGET_DIR", str(tmp_path))
        monkeypatch.setenv("AURA_CHAOS_DISK_MAX_MB", "2")
        monkeypatch.setenv("AURA_CHAOS_DISK_RESTORE_SECONDS", "0.05")
        out = await injector._FAULTS["fill_disk"]()
        assert out["applied"] is True
        assert out["bytes_written"] == 2 * 1024 * 1024
        await _drain_restores()
        assert not list(tmp_path.glob("aura-chaos-disk-pressure-*")), (
            "pressure files must be cleaned after the restore window"
        )

    @pytest.mark.asyncio
    async def test_delete_vector_index_honest_when_no_target(self):
        out = await injector._FAULTS["delete_vector_index"]()
        # On hosts without a vector_index the fault must say so, not lie.
        if not out["applied"]:
            assert out["reason"] == "no_target"

    @pytest.mark.asyncio
    async def test_loop_lag_reports_measured_lag(self):
        out = await injector._FAULTS["induce_event_loop_lag"]()
        assert out["applied"] is True
        assert out["lagged_ms"] >= 1000


class TestRandomEntryPoint:
    @pytest.mark.asyncio
    async def test_inject_random_fault_never_raises(
        self, fast_restore, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("AURA_CHAOS_DISK_TARGET_DIR", str(tmp_path))
        monkeypatch.setenv("AURA_CHAOS_DISK_MAX_MB", "1")
        monkeypatch.setenv("AURA_MODEL", "/x")
        for _ in range(12):
            out = await injector.inject_random_fault()
            assert "kind" in out and "applied" in out
        await _drain_restores()
