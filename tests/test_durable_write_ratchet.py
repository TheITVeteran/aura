"""Ratchet: no NEW file writes may bypass the atomic write gateway.

CLAUDE.md states the rule — "All consequential file writes go through
core/runtime/file_write_gateway.py" — and nothing enforced it.

A direct write is not atomic. The old contents are truncated the moment the
file is opened and the new contents land in pieces; a crash, an OOM kill or
a liveness-sentinel SIGKILL between those two moments leaves a truncated
file where durable state used to be. This runtime is killed that way often
enough that the window is not theoretical, and the failure is silent in the
worst way: the next boot reads a valid-looking short file and carries on
with half a state.

The measured answer was good news — core/ and interface/ carry exactly two
direct writes, both crash-time forensic dumps that must NOT use the gateway
(it allocates, locks and fsyncs, and those are precisely what hang when the
loop is already wedged or the allocator is already failing). So this is not
a cleanup; it is the gate that keeps it clean.

If this fails on code you just wrote: use get_file_write_gateway() (or the
*_async methods from async code), or write to a temp path and os.replace.
If you legitimately need a raw write while the process is dying, add it to
EXEMPT_CRASH_DUMPS with the reason.
"""
from __future__ import annotations

from core.runtime.durable_write_audit import (
    EXEMPT_CRASH_DUMPS,
    DirectWrite,
    scan_direct_writes,
)


class TestNoUnexemptedBypasses:
    def test_no_write_bypasses_the_gateway(self):
        report = scan_direct_writes()
        offenders = [f"{w.path}:{w.line} ({w.call} in {w.function})" for w in report.enforceable]
        assert not offenders, (
            "these writes bypass the atomic gateway:\n  " + "\n  ".join(offenders)
        )

    def test_the_exemptions_still_exist(self):
        """An exemption for code that has moved is a stale allowance, and the
        next real bypass would hide behind it."""
        found = scan_direct_writes().keys
        stale = sorted(EXEMPT_CRASH_DUMPS - found)
        assert not stale, f"exempted writes no longer present: {stale}"

    def test_the_exemption_list_stays_small(self):
        """Exemptions are for dying-process forensics, not convenience."""
        assert len(EXEMPT_CRASH_DUMPS) <= 4


class TestTheScannerRecognisesCompliantWrites:
    """A scanner that cries wolf gets muted, which is worse than not having
    one. An earlier version reported 153 then 13 bypasses; every one of
    those was its own false positive."""

    def test_gateway_writes_are_not_flagged(self, tmp_path):
        source = tmp_path / "core" / "x.py"
        source.parent.mkdir(parents=True)
        source.write_text(
            "def save(d):\n"
            "    file_gateway = get_file_write_gateway()\n"
            "    file_gateway.write_text(d / 'a.json', '{}', source='x')\n",
            encoding="utf-8",
        )
        assert scan_direct_writes(roots=("core",), repo_root=tmp_path).writes == []

    def test_the_temp_then_replace_idiom_is_not_flagged(self, tmp_path):
        """Writing a temp file and os.replace-ing it IS the atomic write."""
        source = tmp_path / "core" / "y.py"
        source.parent.mkdir(parents=True)
        source.write_text(
            "def save(run_dir, payload):\n"
            "    manifest_tmp = run_dir / '.m.tmp'\n"
            "    manifest_tmp.write_text(payload)\n"
            "    os.replace(manifest_tmp, run_dir / 'm.json')\n",
            encoding="utf-8",
        )
        assert scan_direct_writes(roots=("core",), repo_root=tmp_path).writes == []

    def test_a_path_join_is_judged_by_its_base(self, tmp_path):
        source = tmp_path / "core" / "z.py"
        source.parent.mkdir(parents=True)
        source.write_text(
            "def save(gateway, d):\n"
            "    gateway.write_text(d / 'nested' / 'a.json', '{}')\n",
            encoding="utf-8",
        )
        assert scan_direct_writes(roots=("core",), repo_root=tmp_path).writes == []

    def test_reads_are_not_flagged(self, tmp_path):
        source = tmp_path / "core" / "r.py"
        source.parent.mkdir(parents=True)
        source.write_text(
            "def load(p):\n"
            "    with open(p) as fh:\n"
            "        return fh.read()\n"
            "def load2(p):\n"
            "    return p.read_text()\n",
            encoding="utf-8",
        )
        assert scan_direct_writes(roots=("core",), repo_root=tmp_path).writes == []


class TestTheScannerActuallyCatchesBypasses:
    """A gate that cannot fail is not a gate."""

    def test_a_raw_write_text_is_caught(self, tmp_path):
        source = tmp_path / "core" / "bad.py"
        source.parent.mkdir(parents=True)
        source.write_text(
            "def save(state_path, payload):\n"
            "    state_path.write_text(payload)\n",
            encoding="utf-8",
        )
        found = scan_direct_writes(roots=("core",), repo_root=tmp_path).writes
        assert [w.call for w in found] == ["write_text"]
        assert found[0].function == "save"

    def test_a_raw_open_for_writing_is_caught(self, tmp_path):
        source = tmp_path / "core" / "bad2.py"
        source.parent.mkdir(parents=True)
        source.write_text(
            "def save(p, payload):\n"
            "    with open(p, 'w') as fh:\n"
            "        fh.write(payload)\n",
            encoding="utf-8",
        )
        assert len(scan_direct_writes(roots=("core",), repo_root=tmp_path).writes) == 1

    def test_append_mode_is_caught(self, tmp_path):
        source = tmp_path / "core" / "bad3.py"
        source.parent.mkdir(parents=True)
        source.write_text(
            "def log(p, line):\n"
            "    with open(p, mode='a') as fh:\n"
            "        fh.write(line)\n",
            encoding="utf-8",
        )
        assert len(scan_direct_writes(roots=("core",), repo_root=tmp_path).writes) == 1

    def test_write_bytes_is_caught(self, tmp_path):
        source = tmp_path / "core" / "bad4.py"
        source.parent.mkdir(parents=True)
        source.write_text(
            "def save(state_path, blob):\n"
            "    state_path.write_bytes(blob)\n",
            encoding="utf-8",
        )
        assert len(scan_direct_writes(roots=("core",), repo_root=tmp_path).writes) == 1


class TestBaselineKeysAreStable:
    def test_a_key_ignores_the_line_number(self):
        """A write does not become a new defect because something above it
        grew by three lines."""
        first = DirectWrite("core/a.py", 10, "write_text", "Cls.save")
        second = DirectWrite("core/a.py", 99, "write_text", "Cls.save")
        assert first.key() == second.key()

    def test_a_key_separates_different_functions(self):
        first = DirectWrite("core/a.py", 10, "write_text", "Cls.save")
        second = DirectWrite("core/a.py", 10, "write_text", "Cls.other")
        assert first.key() != second.key()
