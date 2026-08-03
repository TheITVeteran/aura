"""Drive a resume verifier through the real detached runner.

Every resume verifier is a separate process: ``tools/run_detached_step.py``
builds the environment, the verifier reads it, and the verdict travels back as
JSON on stdout.  A test that sets that environment by hand proves the
verifier's half only -- it cannot notice the runner changing what it supplies.
That is exactly how bd2961d3d stranded five verifiers at once while every one
of their unit tests stayed green.

``accepted_verdict`` closes the loop: the runner builds and supplies the
environment, the verdict is serialised over stdout the way a real child would,
and only what the runner accepts comes back.  Any ``AURA_DETACHED_*`` variable
the test set itself is cleared first, so a variable the runner stops supplying
fails the test instead of leaking in from the fixture.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from tools import run_detached_step

PLAN_SHA256 = "1" * 64
COMMAND_SHA256 = "2" * 64
PRIOR_JOURNAL_HEAD_SHA256 = "3" * 64


def accepted_verdict(
    monkeypatch: pytest.MonkeyPatch,
    build: Callable[[Mapping[str, str]], Any],
    *,
    cwd: Path,
    prior_attempt: int = 1,
) -> dict[str, Any]:
    """Return the verdict ``build`` produced, once the runner has accepted it.

    ``build`` receives the environment the runner supplies; verifiers that read
    ``os.environ`` directly see exactly that environment and nothing else.
    """
    monkeypatch.setattr(
        run_detached_step,
        "_verify_execution_manifest_current",
        lambda _manifest: None,
    )
    plan = {
        "plan_sha256": PLAN_SHA256,
        "command_sha256": COMMAND_SHA256,
        "cwd": str(cwd),
        "execution_environment": {},
        "resume_verifier_command": ["/usr/bin/true"],
        "resume_verifier_execution_manifest": {},
    }

    def _run(*_args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        environment = dict(kwargs["env"])
        with monkeypatch.context() as context:
            for key in list(os.environ):
                if key.startswith("AURA_DETACHED_"):
                    context.delenv(key, raising=False)
            for key, value in environment.items():
                context.setenv(key, value)
            verdict = build(environment)
        return subprocess.CompletedProcess(
            args=["/usr/bin/true"],
            returncode=0,
            stdout=json.dumps(verdict),
            stderr="",
        )

    monkeypatch.setattr(run_detached_step.subprocess, "run", _run)
    return run_detached_step._run_resume_verifier(
        plan,
        cwd,
        prior_attempt=prior_attempt,
        prior_journal_head_sha256=PRIOR_JOURNAL_HEAD_SHA256,
    )
