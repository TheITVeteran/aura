"""Contract tests for the operator CLI (`aura doctor|conformance|chaos|plugin|...`).

The CLI surface existed and was wired into aura_main but had no dedicated test coverage —
this locks its contract: every subcommand parses, dispatches to a registered handler, and
returns a JSON-serializable dict with the documented shape and an `ok` boolean.
"""
from __future__ import annotations

import json

import pytest

from core.runtime import operator_cli


# ── parser + registry ──────────────────────────────────────────────────────

def test_all_documented_commands_parse():
    parser = operator_cli.build_parser()
    for argv in (
        ["doctor"], ["doctor", "--bundle"], ["conformance"],
        ["backup"], ["restore", "--snapshot", "/tmp/x"], ["migrate", "--dry-run"],
        ["verify-state"], ["verify-memory"], ["rebuild-index"], ["chaos"],
        ["plugin", "list"], ["plugin", "approve", "/tmp/p.py"], ["plugin", "scan"],
    ):
        ns = parser.parse_args(argv)
        assert ns.command == argv[0]


def test_every_subcommand_has_a_registered_handler():
    parser = operator_cli.build_parser()
    # the top-level choices the parser exposes must each have a handler
    choices = parser._subparsers._group_actions[0].choices  # type: ignore[attr-defined]
    for name in choices:
        assert name in operator_cli.COMMAND_HANDLERS, f"no handler registered for '{name}'"


def test_result_is_json_serializable():
    result = operator_cli.run_command(["chaos"])
    json.dumps(result, default=str)   # must not raise
    assert "command" in result and "ok" in result


# ── doctor ──────────────────────────────────────────────────────────────────

def test_doctor_runs_and_reports_checks(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    result = operator_cli.run_command(["doctor"])
    assert result["command"] == "doctor"
    assert isinstance(result["ok"], bool)
    checks = result["checks"]
    # environment-independent checks must be present and pass
    assert checks["python_version"]["ok"] is True
    assert checks["sqlite_available"]["ok"] is True
    assert checks["atomic_writer_round_trip"]["ok"] is True
    assert checks["data_dir_writable"]["ok"] is True


# ── conformance ──────────────────────────────────────────────────────────────

def test_conformance_runs_static_proofs(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    result = operator_cli.run_command(["conformance"])
    assert result["command"] == "conformance"
    assert isinstance(result["ok"], bool)
    # it must be honest that this is a static fixture proof, not live runtime
    assert result["evidence_scope"] == "static_contract_fixture"
    assert result["live_runtime_verified"] is False
    assert "report" in result and "results" in result["report"]
    assert result["ok"], f"static conformance proofs should pass: {result['report']}"


# ── chaos ────────────────────────────────────────────────────────────────────

def test_chaos_smoke_passes():
    result = operator_cli.run_command(["chaos"])
    assert result["command"] == "chaos"
    assert result["ok"] is True
    assert isinstance(result["fired"], list)


# ── plugin ───────────────────────────────────────────────────────────────────

def test_plugin_list_and_scan(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    listed = operator_cli.run_command(["plugin", "list"])
    assert listed["command"] == "plugin list" and listed["ok"] is True
    assert isinstance(listed["entries"], list)

    scanned = operator_cli.run_command(["plugin", "scan"])
    assert scanned["command"] == "plugin scan"
    assert "scanned" in scanned and isinstance(scanned["report"], list)


def test_plugin_approve_missing_file(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    result = operator_cli.run_command(["plugin", "approve", str(tmp_path / "nope.py")])
    assert result["ok"] is False and "not found" in result["error"].lower()


# ── main() exit code contract ────────────────────────────────────────────────

def test_main_returns_zero_on_ok(capsys):
    rc = operator_cli.main(["chaos"])
    assert rc == 0
    out = capsys.readouterr().out
    assert json.loads(out)["command"] == "chaos"   # prints JSON for runbooks/CI


# ── drift guard: aura_main must dispatch exactly the CLI's commands ──────────

def test_aura_main_dispatch_set_matches_cli_handlers():
    """Every operator command the CLI registers must be reachable via `aura <cmd>`.

    Catches the gap where a command is added to the CLI but not to aura_main's dispatch
    set (or vice versa), which would make it silently unreachable from the `aura` binary.
    """
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "aura_main.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    dispatch: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "operator_commands" for t in node.targets
        ):
            if isinstance(node.value, ast.Set):
                dispatch = {e.value for e in node.value.elts if isinstance(e, ast.Constant)}
            break
    assert dispatch, "could not find operator_commands dispatch set in aura_main.py"
    assert dispatch == set(operator_cli.COMMAND_HANDLERS), (
        f"aura_main dispatch and CLI handlers drifted: "
        f"only in aura_main={dispatch - set(operator_cli.COMMAND_HANDLERS)}, "
        f"only in CLI={set(operator_cli.COMMAND_HANDLERS) - dispatch}"
    )
