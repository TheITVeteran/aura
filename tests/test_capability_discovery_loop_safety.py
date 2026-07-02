"""Capability discovery must never block the event loop with disk I/O.

All 12 loop-wedge crash dumps recorded on the live desktop (July 2026) showed
the identical main-thread stack: _discover_writable_dirs running a governed
fsync write ON the event loop while the disk thrashed, freezing the loop for
~1200s until the external liveness sentinel SIGKILLed the process tree — a
20-minute crash-restart cycle. These tests pin the fix.
"""
import asyncio
import threading

import pytest

from core.capabilities.capability_discovery import CapabilityDiscovery, CapabilityReport


class TestWritableDirProbeOffLoop:
    def test_probe_runs_off_the_event_loop_thread(self, monkeypatch, tmp_path):
        probe_threads: list[threading.Thread] = []
        real_probe = CapabilityDiscovery._probe_writable_dir

        def recording_probe(d):
            probe_threads.append(threading.current_thread())
            return real_probe(d)

        monkeypatch.setattr(
            CapabilityDiscovery, "_probe_writable_dir", staticmethod(recording_probe)
        )
        monkeypatch.setattr(
            "core.capabilities.capability_discovery.Path.home", lambda: tmp_path
        )

        async def scenario():
            loop_thread = threading.current_thread()
            report = CapabilityReport()
            await CapabilityDiscovery()._discover_writable_dirs(report)
            return loop_thread, report

        loop_thread, report = asyncio.run(scenario())
        assert probe_threads, "probe never ran"
        assert all(t is not loop_thread for t in probe_threads), (
            "writable-dir probe executed on the event-loop thread — this is "
            "the exact stack that froze the live runtime 12 times"
        )
        assert report.writable_directories, "probe should find writable dirs"

    def test_probe_timeout_records_degradation_and_continues(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "core.capabilities.capability_discovery.Path.home", lambda: tmp_path
        )

        def hanging_probe(d):
            import time
            time.sleep(0.3)
            raise OSError("disk gone")

        monkeypatch.setattr(
            CapabilityDiscovery, "_probe_writable_dir", staticmethod(hanging_probe)
        )

        async def scenario():
            report = CapabilityReport()
            await CapabilityDiscovery()._discover_writable_dirs(report)
            return report

        report = asyncio.run(scenario())
        assert report.writable_directories == []

    def test_package_probe_does_not_import_modules(self, monkeypatch):
        """find_spec must be used — importing numpy/PIL at boot stalls the loop."""
        import builtins

        real_import = builtins.__import__
        imported: list[str] = []
        heavyweights = {"numpy", "PIL", "reportlab", "pyautogui", "pytesseract", "fpdf"}

        def watching_import(name, *args, **kwargs):
            if name in heavyweights:
                imported.append(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", watching_import)

        async def scenario():
            report = CapabilityReport()
            await CapabilityDiscovery()._discover_python_packages(report)
            return report

        report = asyncio.run(scenario())
        assert imported == [], f"package probe imported heavyweight modules: {imported}"
        assert report.has_python_packages.get("psutil") is True


class TestNonDurableAtomicWrites:
    def test_durable_false_skips_fsync(self, tmp_path, monkeypatch):
        import core.runtime.atomic_writer as aw

        fsync_calls: list[int] = []
        monkeypatch.setattr(aw, "_fsync_file", lambda fd: fsync_calls.append(fd))
        monkeypatch.setattr(aw, "_fsync_dir", lambda d: fsync_calls.append(-1))

        target = tmp_path / "probe.txt"
        aw.atomic_write_text(target, "probe", durable=False)
        assert target.read_text() == "probe"
        assert fsync_calls == [], "durable=False must not fsync"

    def test_durable_default_still_fsyncs(self, tmp_path, monkeypatch):
        import core.runtime.atomic_writer as aw

        fsync_calls: list[int] = []
        monkeypatch.setattr(aw, "_fsync_file", lambda fd: fsync_calls.append(fd))
        monkeypatch.setattr(aw, "_fsync_dir", lambda d: fsync_calls.append(-1))

        aw.atomic_write_text(tmp_path / "state.json", "{}")
        assert fsync_calls, "default writes must stay durable"

    def test_non_durable_write_is_still_atomic(self, tmp_path):
        from core.runtime.atomic_writer import DEFAULT_TEMP_PREFIX, atomic_write_text

        target = tmp_path / "swap.txt"
        atomic_write_text(target, "one", durable=False)
        atomic_write_text(target, "two", durable=False)
        assert target.read_text() == "two"
        leftovers = list(tmp_path.glob(f"{DEFAULT_TEMP_PREFIX}*"))
        assert leftovers == []

    def test_gateway_passes_durability_through(self, tmp_path, monkeypatch):
        import core.runtime.file_write_gateway as gw

        captured: dict = {}

        def fake_atomic_write_text(path, text, *, encoding="utf-8", durable=True):
            captured["durable"] = durable
            captured["path"] = path

        monkeypatch.setattr(gw, "atomic_write_text", fake_atomic_write_text)
        gw.FileWriteGateway().write_text(
            tmp_path / "f.txt", "x", source="test", durable=False
        )
        assert captured["durable"] is False


@pytest.mark.parametrize("method", ["_discover_tools", "_discover_python_packages"])
def test_sibling_probes_complete_quickly(method):
    """Sibling probes should finish without touching the network or hanging."""

    async def scenario():
        report = CapabilityReport()
        await getattr(CapabilityDiscovery(), method)(report)
        return report

    asyncio.run(asyncio.wait_for(scenario(), timeout=30))
