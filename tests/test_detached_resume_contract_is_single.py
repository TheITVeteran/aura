"""There is exactly one detached-resume contract in the tree.

``tools/run_detached_step.py`` is the only consumer of a resume verdict, and it
accepts exactly one schema version and one evidence transport.  Every resume
verifier is a separate process, so a version bump in the runner cannot break
its implementations at import time -- it breaks them at 3am, mid-campaign, as
an opaque non-zero exit.  That is what happened at bd2961d3d, which moved the
runner to ``resume_verdict.v3``/``resume_evidence.v2`` over ``stdout-v3`` and
left four of the five verifiers emitting the old file-transport contract.

These checks are deliberately source-level: they hold for verifiers that no
current test can afford to run end to end.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_RUNNER = _ROOT / "tools" / "run_detached_step.py"

_VERDICT_VERSIONS = re.compile(r"resume_verdict\.v(\d+)")
_EVIDENCE_VERSIONS = re.compile(r"resume_evidence\.v(\d+)")

# Every module that builds a resume verdict for the runner to accept.
_RESUME_VERIFIERS = (
    "tools/resume_durable_external_verifier_job.py",
    "tools/verify_structured_sft_research_resume.py",
    "tools/verify_latent_cortex_campaign_resume.py",
    "tools/replay_verified_recurrent_policy_states.py",
    "tools/prepare_resident_recurrent_grpo_campaign.py",
    "tools/run_resident_recurrent_sft_bootstrap_campaign.py",
)


def _runner_versions() -> tuple[str, str]:
    source = _RUNNER.read_text(encoding="utf-8")
    verdicts = set(_VERDICT_VERSIONS.findall(source))
    evidence = set(_EVIDENCE_VERSIONS.findall(source))
    assert len(verdicts) == 1, f"runner accepts several verdict versions: {sorted(verdicts)}"
    assert len(evidence) == 1, f"runner accepts several evidence versions: {sorted(evidence)}"
    return verdicts.pop(), evidence.pop()


@pytest.mark.parametrize("relative", _RESUME_VERIFIERS)
def test_resume_verifier_emits_the_runner_contract(relative: str) -> None:
    verdict_version, evidence_version = _runner_versions()
    source = (_ROOT / relative).read_text(encoding="utf-8")

    emitted_verdicts = set(_VERDICT_VERSIONS.findall(source))
    emitted_evidence = set(_EVIDENCE_VERSIONS.findall(source))
    assert emitted_verdicts == {verdict_version}, (
        f"{relative} emits resume_verdict {sorted(emitted_verdicts)}; "
        f"the runner accepts only v{verdict_version}"
    )
    assert emitted_evidence == {evidence_version}, (
        f"{relative} emits resume_evidence {sorted(emitted_evidence)}; "
        f"the runner accepts only v{evidence_version}"
    )


@pytest.mark.parametrize("relative", _RESUME_VERIFIERS)
def test_resume_verifier_uses_the_stdout_transport(relative: str) -> None:
    source = (_ROOT / relative).read_text(encoding="utf-8")
    assert "AURA_DETACHED_RESUME_EVIDENCE_PATH" not in source, (
        f"{relative} still reads the retired evidence-file environment variable; "
        "the runner sets AURA_DETACHED_RESUME_EVIDENCE_TRANSPORT=stdout-v3 instead"
    )
    assert "AURA_DETACHED_RESUME_EVIDENCE_TRANSPORT" in source, (
        f"{relative} does not bind the resume evidence transport, so it would "
        "accept being driven by a runner speaking a contract it does not implement"
    )


def test_runner_sets_only_the_stdout_transport() -> None:
    source = _RUNNER.read_text(encoding="utf-8")
    assert "AURA_DETACHED_RESUME_EVIDENCE_PATH" not in source
    assert '"AURA_DETACHED_RESUME_EVIDENCE_TRANSPORT": "stdout-v3"' in source
