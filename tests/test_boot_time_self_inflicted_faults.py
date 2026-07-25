"""Two faults a healthy runtime raised against itself, every single run.

Both showed up in the 2026-07-25 boot as a degradation, an INCIDENT and a
MARGINAL fault on a runtime with nothing actually wrong. Together with the
welfare storm they are what drove cortisol into crisis and produced 15
``SubstrateAuthority BLOCKED … neurochemical_cortisol_crisis`` entries — real
work refused because of noise the system generated about itself.

  1. TerminalMonitor saved its blacklist through ``atomic_write_json``, which
     wraps the list in a ``{schema, schema_name, schema_version, payload}``
     envelope, and then read it back expecting a bare list. It also corrupted
     itself: an earlier read iterated the dict, so the envelope's own KEYS were
     saved back as blacklist entries.

  2. The nightly LoRA run collected its training examples and then threw them
     away at the write, because a maintenance write without
     ``local_internal_governed_scope`` is refused by the live runtime.
"""
from __future__ import annotations

import inspect
import json

import pytest

pytestmark = pytest.mark.unit


class TestBlacklistRoundTrips:
    @pytest.fixture()
    def monitor(self, tmp_path, monkeypatch):
        from core import terminal_monitor as tm

        monkeypatch.setattr(tm, "BLACKLIST_PATH", tmp_path / "terminal_blacklist.json")
        return tm

    def test_what_the_writer_writes_the_reader_can_read(self, monitor):
        """The exact live failure: envelope written, bare list expected."""
        from core.runtime.atomic_writer import atomic_write_json

        atomic_write_json(
            monitor.BLACKLIST_PATH,
            ["error one", "error two"],
            schema_version=1,
            schema_name="terminal_error_blacklist",
        )
        loaded = monitor.TerminalMonitor.__new__(
            monitor.TerminalMonitor
        )._load_blacklist()

        assert loaded == {"error one", "error two"}

    def test_a_bare_legacy_list_still_loads(self, monitor):
        monitor.BLACKLIST_PATH.write_text(json.dumps(["legacy"]), encoding="utf-8")
        assert monitor.TerminalMonitor.__new__(
            monitor.TerminalMonitor
        )._load_blacklist() == {"legacy"}

    def test_self_inflicted_envelope_keys_are_scrubbed(self, monitor):
        """The live file had its own envelope keys saved as blacklist entries."""
        monitor.BLACKLIST_PATH.write_text(
            json.dumps(
                {
                    "payload": [
                        "a real error fingerprint",
                        "payload", "schema", "schema_name", "schema_version",
                    ],
                    "schema": "terminal_error_blacklist",
                    "schema_name": "terminal_error_blacklist",
                    "schema_version": 1,
                }
            ),
            encoding="utf-8",
        )
        assert monitor.TerminalMonitor.__new__(
            monitor.TerminalMonitor
        )._load_blacklist() == {"a real error fingerprint"}

    def test_genuine_corruption_still_starts_clean(self, monitor):
        monitor.BLACKLIST_PATH.write_text("{not json", encoding="utf-8")
        assert monitor.TerminalMonitor.__new__(
            monitor.TerminalMonitor
        )._load_blacklist() == set()


class TestNightlyLoraWriteIsGoverned:
    def test_the_training_write_runs_inside_a_governed_scope(self):
        from core.adaptation.nightly_lora import NightlyLoRATrainer

        src = inspect.getsource(NightlyLoRATrainer.run)
        assert "local_internal_governed_scope" in src, (
            "an ungoverned maintenance write is refused by the live runtime, "
            "so the collected training data is discarded at the last step"
        )
        scope_at = src.index("local_internal_governed_scope")
        write_at = src.index("write_text_async")
        assert scope_at < write_at, "the scope must wrap the write, not follow it"
