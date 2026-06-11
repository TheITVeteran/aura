"""Tests for the typed mutation evaluator + quarantine."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

import core.self_modification.mutation_safety as mutation_safety_mod
from core.self_modification.mutation_safety import (
    MutationDiagnostics,
    MutationOutcome,
    QuarantineStore,
    SafeMutationEvaluator,
)


@pytest.fixture
def evaluator(tmp_path: Path) -> SafeMutationEvaluator:
    quarantine = QuarantineStore(tmp_path / "quarantine")
    return SafeMutationEvaluator(
        timeout_seconds=10.0,
        memory_mb=256,
        quarantine=quarantine,
    )


# ---------------------------------------------------------------------------
# typed outcomes
# ---------------------------------------------------------------------------
def test_passed_for_clean_module(evaluator):
    diag = evaluator.evaluate("def add(a, b):\n    return a + b\n")
    assert diag.outcome is MutationOutcome.PASSED
    assert diag.exit_code == 0
    assert diag.quarantine_path is None


def test_compile_fail_on_syntax_error(evaluator):
    diag = evaluator.evaluate("def broken(:\n    pass\n")
    assert diag.outcome is MutationOutcome.COMPILE_FAIL
    assert diag.quarantine_path is not None
    assert "SyntaxError" in diag.traceback_text


def test_import_fail(evaluator):
    diag = evaluator.evaluate("import definitely_not_a_real_pkg_xyz_42\n")
    assert diag.outcome is MutationOutcome.IMPORT_FAIL
    assert diag.quarantine_path is not None


def test_runtime_exception(evaluator):
    diag = evaluator.evaluate(
        textwrap.dedent(
            """
            def boom():
                return 1 / 0
            boom()
            """
        )
    )
    assert diag.outcome is MutationOutcome.RUNTIME_EXCEPTION
    assert "ZeroDivisionError" in diag.traceback_text


def test_custom_runtime_exception_still_gets_typed(evaluator):
    diag = evaluator.evaluate(
        textwrap.dedent(
            """
            class CandidateSpecificFailure(Exception):
                pass

            raise CandidateSpecificFailure("candidate escaped explicit classes")
            """
        )
    )
    assert diag.outcome is MutationOutcome.RUNTIME_EXCEPTION
    assert diag.quarantine_path is not None
    assert "CandidateSpecificFailure" in diag.traceback_text
    assert diag.extra.get("uncaught") is True


def test_assertion_fail_in_module_body(evaluator):
    diag = evaluator.evaluate("assert 1 == 2, 'nope'\n")
    assert diag.outcome is MutationOutcome.ASSERTION_FAIL
    assert "AssertionError" in diag.traceback_text


def test_assertion_fail_in_test(evaluator):
    diag = evaluator.evaluate(
        "def add(a, b): return a + b\n",
        test_source="assert add(1, 1) == 3\n",
    )
    assert diag.outcome is MutationOutcome.ASSERTION_FAIL
    assert diag.quarantine_path is not None


def test_test_passes(evaluator):
    diag = evaluator.evaluate(
        "def add(a, b): return a + b\n",
        test_source="assert add(2, 2) == 4\n",
    )
    assert diag.outcome is MutationOutcome.PASSED


def test_timeout(tmp_path):
    evaluator = SafeMutationEvaluator(
        timeout_seconds=1.0,
        memory_mb=256,
        quarantine=QuarantineStore(tmp_path / "q"),
    )
    diag = evaluator.evaluate(
        textwrap.dedent(
            """
            import time
            time.sleep(60)
            """
        )
    )
    assert diag.outcome is MutationOutcome.TIMEOUT
    assert diag.quarantine_path is not None
    assert diag.runtime_seconds < 5.0  # sanity: parent killed the child quickly


def test_parent_does_not_crash_on_malformed_mutation(evaluator):
    """The whole point of the typed evaluator: a malformed mutation
    must degrade to a diagnostic, never crash the parent process."""
    catastrophic_sources = [
        "def \xff broken_token():\n    pass\n",  # not valid utf-8 token
        "raise SystemExit('bye')\n",
        "import sys; sys.exit(99)\n",
        "1/0\n",
        "assert False\n",
        "this is not python at all !!!\n",
        "while True: pass\n",  # would hang without a small timeout
    ]
    fast_evaluator = SafeMutationEvaluator(
        timeout_seconds=2.0,
        memory_mb=256,
        quarantine=evaluator.quarantine,
    )
    outcomes = []
    for src in catastrophic_sources:
        diag = fast_evaluator.evaluate(src)
        outcomes.append(diag.outcome)
        # Either way, the parent kept running — that is the assertion.
    # And every catastrophic input ended with a typed (non-PASSED) outcome.
    assert MutationOutcome.PASSED not in outcomes


# ---------------------------------------------------------------------------
# quarantine
# ---------------------------------------------------------------------------
def test_quarantine_layout(evaluator, tmp_path):
    diag = evaluator.evaluate("def f():\n    return 1/0\nf()\n")
    assert diag.quarantine_path is not None
    entry = Path(diag.quarantine_path)
    assert entry.exists()
    assert (entry / "source.py").exists()
    assert (entry / "result.json").exists()
    assert (entry / "stdout.log").exists()
    assert (entry / "stderr.log").exists()
    payload = json.loads((entry / "result.json").read_text(encoding="utf-8"))
    assert payload["outcome"] == "runtime_exception"


def test_quarantine_writes_artifacts_through_file_gateway(tmp_path, monkeypatch):
    calls = []

    class FakeFileWriteGateway:
        def write_text(self, path, text, *, encoding="utf-8", source="unknown"):
            target = Path(path)
            calls.append((target.name, source))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding=encoding)

    monkeypatch.setattr(
        mutation_safety_mod,
        "get_file_write_gateway",
        lambda: FakeFileWriteGateway(),
    )

    diag = MutationDiagnostics(
        outcome=MutationOutcome.RUNTIME_EXCEPTION,
        runtime_seconds=0.1,
        exit_code=14,
        stdout="out",
        stderr="err",
    )

    entry = QuarantineStore(tmp_path / "quarantine").quarantine(
        source="raise RuntimeError('bad')\n",
        test_source="assert False\n",
        diagnostics=diag,
    )

    assert entry.exists()
    assert {
        ("source.py", "core.self_modification.mutation_safety.quarantine_source"),
        ("test.py", "core.self_modification.mutation_safety.quarantine_test"),
        ("stdout.log", "core.self_modification.mutation_safety.quarantine_stdout"),
        ("stderr.log", "core.self_modification.mutation_safety.quarantine_stderr"),
        ("result.json", "core.self_modification.mutation_safety.quarantine_result"),
    }.issubset(set(calls))


def test_evaluator_uses_gateways_for_temp_files_and_child_process(tmp_path, monkeypatch):
    file_write_sources = []
    subprocess_calls = []

    class FakeFileWriteGateway:
        def write_text(self, path, text, *, encoding="utf-8", source="unknown"):
            file_write_sources.append(source)
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding=encoding)

    class FakeProcess:
        returncode = 0

        def communicate(self, timeout=None):
            return (
                b'__MUTATION_RESULT__:{"outcome":"passed","traceback":"","extra":{}}\n',
                b"",
            )

    class FakeSubprocessGateway:
        def spawn(self, argv, **kwargs):
            subprocess_calls.append((tuple(argv), kwargs))
            return FakeProcess()

    monkeypatch.setattr(
        mutation_safety_mod,
        "get_file_write_gateway",
        lambda: FakeFileWriteGateway(),
    )
    monkeypatch.setattr(
        mutation_safety_mod,
        "get_subprocess_gateway",
        lambda: FakeSubprocessGateway(),
    )

    evaluator = SafeMutationEvaluator(
        timeout_seconds=1.0,
        memory_mb=128,
        quarantine=QuarantineStore(tmp_path / "q"),
    )
    diag = evaluator.evaluate("x = 1\n", test_source="assert x == 1\n")

    assert diag.outcome is MutationOutcome.PASSED
    assert {
        "core.self_modification.mutation_safety.candidate_source",
        "core.self_modification.mutation_safety.test_source",
        "core.self_modification.mutation_safety.bootstrap_source",
    }.issubset(set(file_write_sources))
    assert subprocess_calls
    _argv, kwargs = subprocess_calls[0]
    assert kwargs["source"] == "core.self_modification.mutation_safety.evaluator_subprocess"
    assert kwargs["text"] is False


def test_passed_does_not_quarantine(evaluator):
    diag = evaluator.evaluate("x = 1\n")
    assert diag.outcome is MutationOutcome.PASSED
    assert diag.quarantine_path is None
    # Quarantine root may have been created but contains nothing.
    entries = evaluator.quarantine.list_entries()
    assert entries == []


def test_quarantine_failure_is_visible_in_diagnostics():
    class FailingQuarantine:
        def quarantine(self, **_kwargs):
            self.invoked = True
            raise RuntimeError("quarantine disk unavailable")

    quarantine = FailingQuarantine()
    evaluator = SafeMutationEvaluator(
        timeout_seconds=2.0,
        memory_mb=128,
        quarantine=quarantine,
    )

    diag = evaluator.evaluate("raise RuntimeError('candidate failed')\n")

    assert quarantine.invoked is True
    assert diag.outcome is MutationOutcome.RUNTIME_EXCEPTION
    assert diag.quarantine_path is None
    assert diag.extra["quarantine_error"] == {
        "type": "RuntimeError",
        "message": "quarantine disk unavailable",
    }


def test_import_free_mutations_skip_site_bootstrap():
    assert SafeMutationEvaluator._python_startup_flags("x = 1\n", None) == ["-S"]


def test_importing_mutations_keep_site_bootstrap_available():
    assert SafeMutationEvaluator._python_startup_flags("import requests\n", None) == []
    assert SafeMutationEvaluator._python_startup_flags("x = 1\n", "import pytest\n") == []


def test_quarantine_is_per_invocation(evaluator):
    # Two different broken mutations get two different quarantine dirs.
    a = evaluator.evaluate("syntax error here:::")
    b = evaluator.evaluate("import not_a_real_module_zzz")
    assert a.quarantine_path != b.quarantine_path
    entries = evaluator.quarantine.list_entries()
    assert len(entries) == 2


# ---------------------------------------------------------------------------
# diagnostics shape
# ---------------------------------------------------------------------------
def test_diagnostics_has_required_fields(evaluator):
    diag = evaluator.evaluate("syntax_error_here !!!:")
    d = diag.to_dict()
    assert d["outcome"] == "compile_fail"
    assert "runtime_seconds" in d
    assert "exit_code" in d
    assert "stdout" in d
    assert "stderr" in d
    assert d["quarantine_path"] is not None


def test_outcome_enum_completeness():
    expected = {
        "passed",
        "compile_fail",
        "import_fail",
        "runtime_exception",
        "assertion_fail",
        "timeout",
        "oom",
    }
    assert {o.value for o in MutationOutcome} == expected
