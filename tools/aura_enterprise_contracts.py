"""Shared data contracts and policy inventory for the enterprise gate."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

EXCLUDED_DIRS = {
    ".git",
    ".agents",
    ".claude",
    ".aura_architect",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".venv_aura",
    "__pycache__",
    "archive",
    "artifacts",
    "build",
    "data",
    "dist",
    "htmlcov",
    "logs",
    "node_modules",
    "scratch",
    "test_vdb",
    "venv",
}

DEFAULT_PRODUCTION_DIRS = {
    "core",
    "executors",
    "infrastructure",
    "interface",
    "llm",
    "security",
    "senses",
    "skills",
}
DEFAULT_PRODUCTION_FILES = {"aura_main.py"}

ALLOW_DYNAMIC_CODE = {
    # TOCTOU-hardened frozen-source loading: executes EXACTLY the curriculum
    # bytes it hashed into the training receipt — the exec IS the security
    # feature (importing the module path again could race a source edit).
    "tools/recurrence_native_train_v2.py",
    "core/agency/repl_daemon.py",
    "core/runtime/dynamic_execution_gateway.py",
    "core/sandbox/bash_daemon.py",
    "core/sandbox/runner.py",
    "core/self_modification/mutation_safety.py",
    "core/self_modification/shadow_runtime.py",
    "security/code_sandbox.py",
    "security/sandbox.py",
}

#: The organs. Everything the running Aura does with a child process goes
#: through core/runtime/subprocess_gateway.py, which is what carries the
#: source label, the accelerator claim, the shutdown interlock and the
#: read-only assertion. A direct spawn in here has none of that.
#:
#: Outside these roots — tests, tools, scripts, the launcher — spawning child
#: processes IS the job: a containment proof needs a real child to kill, an
#: operator driver orchestrates the gates it runs, and the out-of-process
#: sentinels must outlive the runtime they watch. This replaced a
#: ninety-entry file-name allowlist, most of whose entries no longer had a
#: subprocess call in them at all; the ones that did were tools and tests.
GATEWAY_OWNED_ROOTS = ("core/", "interface/")
SUBPROCESS_GATEWAY_MODULE = "core/runtime/subprocess_gateway.py"


def subprocess_must_use_gateway(rel: str) -> bool:
    return rel.startswith(GATEWAY_OWNED_ROOTS) and rel != SUBPROCESS_GATEWAY_MODULE

ALLOW_BLOCKING_SLEEP_IN_ASYNC = {
    # This chaos fault deliberately stalls the loop to verify lag detection
    # and recovery alarms. It is not production request handling.
    "tools/chaos/injector.py",
}


_TMP_PATH_PREFIX = "/" + "tmp" + "/"
_USERS_PATH_PREFIX = "/" + "Users" + "/"
_HOME_PATH_PREFIX = "/" + "home" + "/"
_WINDOWS_USERS_PREFIX = "C:" + "\\\\" + "Users" + "\\\\"

TEXT_PATTERNS = {
    # The match extends over the WHOLE path, not just the prefix. Two rules
    # below compare the matched text against other evidence in the file, and
    # a match of "/Users/" alone carries none of the information they need.
    "hardcoded_local_path": re.compile(
        rf"({re.escape(_USERS_PATH_PREFIX)}|"
        rf"{re.escape(_HOME_PATH_PREFIX)}[^/\s]+/|"
        rf"{re.escape(_WINDOWS_USERS_PREFIX)}|"
        rf"{re.escape(_TMP_PATH_PREFIX)})[^\s\"'`,)\]}}]*"
    ),
    # "notimplemented" is deliberately absent. The word boundary means it
    # never matched NotImplementedError; the only thing it could match was the
    # bare ``NotImplemented`` singleton, which is Python's binary-operator
    # protocol — the CORRECT return from __eq__ for an unrelated type — and
    # every one of its five occurrences in this repo was exactly that.
    "placeholder_stub_mock": re.compile(
        r"\b(placeholder|stub|mock|dummy|not implemented)\b",
        re.IGNORECASE,
    ),
    "potential_secret": re.compile(
        r"(sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9_]{20,}|xox[baprs]-[A-Za-z0-9-]{10,})"
    ),
    # ``pytest.mark.skip`` (no "if") is unconditional. ``skipif`` is excluded
    # by the word boundary, and a bare ``pytest.skip()`` is filtered below by
    # whether anything guards it — a precondition is not parked debt.
    "pytest_skip_xfail": re.compile(
        r"pytest\.mark\.skip\b|pytest\.skip\b|\bxfail\b", re.IGNORECASE
    ),
}

#: Credential-shaped strings that cannot be credentials.
#:
#: All ten "potential_secret" findings in the repo are test fixtures: fake
#: keys written so the redaction code can be tested against them. A scanner
#: that flags its own fixtures teaches people to ignore it, and an ignored
#: secret scanner is worse than none — the day it finds a real key, that
#: finding arrives in a list nobody reads.
#:
#: The exclusions are properties of the VALUE, not of the file it sits in, so
#: a real key pasted into a test is still caught:
#:   * AKIAIOSFODNN7EXAMPLE is AWS's own published example key.
#:   * A body that is the alphabet in sequence is not entropy.
#:   * EXAMPLE/PLACEHOLDER/REDACTED/XXXX bodies announce themselves.
_NON_SECRET_LITERALS = re.compile(
    r"""(?x)
    AKIAIOSFODNN7EXAMPLE
    | (?:sk-|ghp_|xox[baprs]-)?
      (?:abcdefghijklmnopqrstuvwxyz|abcdefghijklmnopqrstuvwx)
    | (?:EXAMPLE|PLACEHOLDER|REDACTED|FAKE|DUMMY|SAMPLE|TESTKEY)
    | X{8,}
    # One character, repeated. "sk-aaaaaaaaaaaaaaaaaaaaaa" is the same
    # argument as the alphabet-in-sequence case above and the same argument
    # as X{8,}: a body with no entropy cannot be a key, whoever wrote it and
    # wherever it sits. Generalised rather than adding 'a' to a list,
    # because the next fixture will use a different letter.
    | (?:sk-|ghp_|xox[baprs]-)(?P<repeated>[A-Za-z0-9])(?P=repeated){9,}
    """,
    re.IGNORECASE,
)


def _is_non_secret_literal(text: str) -> bool:
    """Whether this credential-shaped match is a known non-secret."""
    return bool(_NON_SECRET_LITERALS.search(str(text or "")))
TODO_MARKER_PATTERN = re.compile(
    r"^(TODO|FIXME|XXX|HACK)\b(?:\([^)]*\))?\s*(?::|-|\s|$)",
    re.IGNORECASE,
)

FAILURE_KINDS = {
    "baseline_regression",
    "compile_failure",
    "pytest_collect_failure",
    "pytest_collect_timeout",
    "syntax_error",
}


@dataclass(frozen=True)
class Finding:
    severity: str
    kind: str
    file: str
    line: int = 0
    detail: str = ""


@dataclass
class GateReport:
    root: str
    generated_at_unix: float
    python_files: int = 0
    compile_ok: bool | None = None
    pytest_collect_ok: bool | None = None
    pytest_collect_seconds: float | None = None
    pytest_collect_output_tail: str = ""
    findings: list[Finding] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for finding in self.findings:
            out[finding.kind] = out.get(finding.kind, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))

    def severity_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for finding in self.findings:
            out[finding.severity] = out.get(finding.severity, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: kv[0]))

    def high_or_critical_count(self) -> int:
        return sum(1 for finding in self.findings if finding.severity in {"high", "critical"})

    def to_json_dict(self) -> dict:
        payload = asdict(self)
        payload["findings"] = [asdict(finding) for finding in self.findings]
        payload["counts"] = self.counts()
        payload["severity_counts"] = self.severity_counts()
        payload["high_or_critical_count"] = self.high_or_critical_count()
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_json_dict(), indent=2, sort_keys=True)


def rel_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def iter_py(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        rel_parts = path.relative_to(root).parts
        if any(part in EXCLUDED_DIRS for part in rel_parts):
            continue
        yield path


def is_production(rel: str) -> bool:
    first = rel.split("/", 1)[0]
    return first in DEFAULT_PRODUCTION_DIRS or rel in DEFAULT_PRODUCTION_FILES

__all__ = [
    "ALLOW_BLOCKING_SLEEP_IN_ASYNC",
    "ALLOW_DYNAMIC_CODE",
    "DEFAULT_PRODUCTION_DIRS",
    "DEFAULT_PRODUCTION_FILES",
    "EXCLUDED_DIRS",
    "FAILURE_KINDS",
    "Finding",
    "GATEWAY_OWNED_ROOTS",
    "GateReport",
    "SUBPROCESS_GATEWAY_MODULE",
    "TEXT_PATTERNS",
    "TODO_MARKER_PATTERN",
    "_is_non_secret_literal",
    "is_production",
    "iter_py",
    "rel_path",
    "subprocess_must_use_gateway",
]
