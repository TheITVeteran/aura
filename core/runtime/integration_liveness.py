"""Liveness probe for optional integrations — catches silent decay.

The failure this exists to catch has a signature: a subsystem imports, its
tests stay green, and it does nothing. Aura's core stays honest because the
core is exercised constantly; optional dependencies and external
integrations decay quietly because nothing runs them. The XTTS/Coqui voice
path was dead from an import error and nothing noticed.

The mechanism that hides it is specific and common — nineteen call sites in
this codebase use ``importlib.util.find_spec`` as an availability check:

    def _tts_dependency_available() -> bool:
        return importlib.util.find_spec("TTS") is not None

``find_spec`` answers "is this module present on disk", which is NOT the
question. A package whose ``__init__`` raises on a moved API, a missing
transitive dependency, or an incompatible version is *present* and
*unimportable* — so the check reports available, the readiness surface
reports healthy, and the feature is dead. Absence of a check reported as a
passed check, one more time.

This probe asks the real question by performing the real import, and
distinguishes three states that the boolean conflated:

``live``    the module imports; the integration can actually run
``broken``  the module is installed but importing it RAISES — the
            silent-decay defect, and the only state that is a failure
``absent``  the module is not installed — legitimate for an optional
            dependency, and honestly reported rather than hidden

Each import runs in a subprocess under a timeout. Heavy scientific
dependencies pull hundreds of megabytes and can hang on a bad driver; an
in-process probe would leak that memory into whatever ran it and could wedge
the gate. A subprocess reclaims everything on exit and can be killed.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Per-import ceiling. A dependency that cannot import within this is broken
# for practical purposes even if it would eventually succeed: the runtime
# imports these on a user-facing path.
IMPORT_TIMEOUT_S = 90.0

LIVE = "live"
BROKEN = "broken"
ABSENT = "absent"


@dataclass(frozen=True)
class Integration:
    """One optional integration and what it actually powers.

    ``powers`` is written for a human reading a failure report: it must say
    what the user loses, because that is what decides whether a broken
    integration is urgent.
    """

    name: str
    module: str
    powers: str
    user_facing: bool = False
    # Attributes that must exist after import. An import that succeeds
    # against a package whose API moved is still a dead integration, and
    # this is where that is caught.
    requires_attrs: tuple[str, ...] = ()
    # "module:function" that the runtime calls BEFORE importing this
    # dependency — a compatibility shim, a path fix, an env var.
    #
    # This matters more than it looks. Probing a raw import in a vacuum
    # answers a question nobody asked: Coqui TTS fails to import against
    # current transformers, and Aura installs a shim that repairs it. A
    # probe without the preflight reports a defect that does not exist in
    # the running system; a probe WITH it answers the real question — does
    # this work the way Aura actually invokes it. The preflight is part of
    # the integration, so it is part of the test.
    preflight: str = ""


# Declared optional integrations. Membership here is a claim that the
# codebase has a real code path behind the dependency — not that it is
# installed.
INTEGRATIONS: tuple[Integration, ...] = (
    Integration(
        name="coqui_tts",
        module="TTS.api",
        powers="XTTS voice cloning and high-quality speech output",
        user_facing=True,
        requires_attrs=("TTS",),
        # Coqui does not import against current transformers without this;
        # voice_engine._load_tts_api installs it before importing, so the
        # probe must too or it reports a defect the runtime does not have.
        preflight=(
            "core.utils.transformers_tts_compat:install_transformers_tts_compat"
        ),
    ),
    Integration(
        name="faster_whisper",
        module="faster_whisper",
        powers="speech-to-text; without it Aura cannot hear",
        user_facing=True,
        requires_attrs=("WhisperModel",),
    ),
    Integration(
        name="piper",
        module="piper",
        powers="fast local neural speech synthesis (fallback voice)",
        user_facing=True,
    ),
    Integration(
        name="pyttsx3",
        module="pyttsx3",
        powers="native OS speech synthesis (last-resort voice)",
        user_facing=True,
    ),
    Integration(
        name="sounddevice",
        module="sounddevice",
        powers="microphone capture and audio playback",
        user_facing=True,
    ),
    Integration(
        name="mlx",
        module="mlx.core",
        powers="the local model substrate; Aura's actual mind",
        user_facing=True,
    ),
    Integration(
        name="mlx_lm",
        module="mlx_lm",
        powers="model loading and generation on the MLX substrate",
        user_facing=True,
    ),
)


@dataclass
class ProbeResult:
    integration: Integration
    state: str
    detail: str = ""
    duration_s: float = 0.0

    @property
    def is_defect(self) -> bool:
        """Only 'broken' is a defect. Absence of an optional dep is not."""
        return self.state == BROKEN

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.integration.name,
            "module": self.integration.module,
            "powers": self.integration.powers,
            "user_facing": self.integration.user_facing,
            "preflight": self.integration.preflight,
            "state": self.state,
            "detail": self.detail,
            "duration_s": round(self.duration_s, 3),
        }


# Runs in the child. Distinguishes "not installed" from "installed and
# raising", which is the whole point of the probe.
_PROBE_SOURCE = """
import importlib, importlib.util, json, sys

module = sys.argv[1]
preflight = sys.argv[2]
required = [a for a in sys.argv[3:] if a]

if preflight:
    # A preflight that fails is itself a broken integration: the runtime
    # cannot import the dependency without it either.
    try:
        pf_module, _, pf_attr = preflight.partition(":")
        getattr(importlib.import_module(pf_module), pf_attr)()
    except BaseException as exc:
        print(json.dumps({
            "state": "broken",
            "detail": f"preflight {preflight} failed: {type(exc).__name__}: {exc}"[:400],
        }))
        sys.exit(0)
# Import FIRST, and use the failure to classify. Probing find_spec ahead of
# the import is fragile in exactly the case that matters: a preflight shim
# may legitimately install the module, and find_spec raises on a module in
# sys.modules with no __spec__. The import is the ground truth; find_spec is
# only needed afterwards to tell "never installed" from "installed and
# raising".
top_level = module.split(".")[0]
try:
    imported = importlib.import_module(module)
except BaseException as exc:
    detail = f"{type(exc).__name__}: {exc}"[:400]
    # A ModuleNotFoundError naming the TOP-LEVEL module means the optional
    # dependency simply is not installed, which is legitimate. Anything else
    # — including a missing transitive dependency — means the integration is
    # present and broken, because the code path cannot run either way.
    absent = (
        isinstance(exc, ModuleNotFoundError)
        and getattr(exc, "name", None) == top_level
    )
    print(json.dumps({
        "state": "absent" if absent else "broken",
        "detail": "not installed" if absent else detail,
    }))
    sys.exit(0)
missing = [a for a in required if not hasattr(imported, a)]
if missing:
    print(json.dumps({
        "state": "broken",
        "detail": "imported but missing attributes: " + ", ".join(missing),
    }))
    sys.exit(0)
print(json.dumps({"state": "live", "detail": ""}))
"""


def probe(
    integration: Integration,
    *,
    timeout_s: float = IMPORT_TIMEOUT_S,
    python: str | None = None,
) -> ProbeResult:
    """Import one integration for real, in a subprocess, under a timeout."""
    started = time.monotonic()
    argv = [
        python or sys.executable,
        "-c",
        _PROBE_SOURCE,
        integration.module,
        integration.preflight,
        *integration.requires_attrs,
    ]
    # The child needs the repo root importable: preflights live in `core.*`,
    # and a subprocess does not inherit the parent's sys.path edits.
    env = dict(os.environ)
    repo_root = str(Path(__file__).resolve().parent.parent.parent)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{existing}" if existing else repo_root
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return ProbeResult(
            integration,
            BROKEN,
            f"import did not complete within {timeout_s:.0f}s",
            time.monotonic() - started,
        )
    except OSError as exc:
        return ProbeResult(
            integration, BROKEN, f"probe could not run: {exc}",
            time.monotonic() - started,
        )

    duration = time.monotonic() - started
    payload: dict[str, Any] | None = None
    for line in reversed((completed.stdout or "").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            candidate = json.loads(line)
        except ValueError:
            continue
        if isinstance(candidate, dict) and "state" in candidate:
            payload = candidate
            break

    if payload is None:
        # A child that produced no verdict tells us nothing good: the import
        # crashed the interpreter, or wrote nothing. Either way it is not
        # evidence of a working integration.
        stderr = (completed.stderr or "").strip().splitlines()
        tail = stderr[-1] if stderr else f"exit code {completed.returncode}"
        return ProbeResult(integration, BROKEN, f"probe produced no verdict: {tail}"[:400], duration)

    state = str(payload.get("state") or "")
    if state not in {LIVE, BROKEN, ABSENT}:
        return ProbeResult(integration, BROKEN, f"unrecognized probe verdict: {state}", duration)
    return ProbeResult(integration, state, str(payload.get("detail") or ""), duration)


@dataclass
class LivenessReport:
    results: list[ProbeResult] = field(default_factory=list)

    @property
    def broken(self) -> list[ProbeResult]:
        return [r for r in self.results if r.state == BROKEN]

    @property
    def absent(self) -> list[ProbeResult]:
        return [r for r in self.results if r.state == ABSENT]

    @property
    def live(self) -> list[ProbeResult]:
        return [r for r in self.results if r.state == LIVE]

    @property
    def ok(self) -> bool:
        """Broken integrations are failures; absent optional ones are not."""
        return not self.broken

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "aura.integration_liveness.v1",
            "checked_at_unix": round(time.time(), 3),
            "total": len(self.results),
            "live": len(self.live),
            "broken": len(self.broken),
            "absent": len(self.absent),
            "ok": self.ok,
            "results": [r.to_dict() for r in self.results],
        }


def probe_all(
    integrations: tuple[Integration, ...] = INTEGRATIONS,
    *,
    timeout_s: float = IMPORT_TIMEOUT_S,
    python: str | None = None,
) -> LivenessReport:
    return LivenessReport(
        [probe(i, timeout_s=timeout_s, python=python) for i in integrations]
    )


__all__ = [
    "ABSENT",
    "BROKEN",
    "INTEGRATIONS",
    "IMPORT_TIMEOUT_S",
    "LIVE",
    "Integration",
    "LivenessReport",
    "ProbeResult",
    "probe",
    "probe_all",
]
