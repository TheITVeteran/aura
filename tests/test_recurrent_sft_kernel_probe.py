from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from core.learning import recurrent_sft_kernel_probe as probe


def _spec(tmp_path: Path) -> dict:
    profile = tmp_path / "profile.sb"
    profile.write_text("(version 1)\n(deny default)\n", encoding="ascii")
    targets = {}
    for role in (
        "evaluator_read",
        "production_write",
        "resident_read",
        "training_write",
    ):
        target = tmp_path / role
        target.write_bytes(role.encode("ascii"))
        targets[role] = target
    return probe.build_kernel_probe_spec(
        sandbox_executable=Path("/usr/bin/sandbox-exec"),
        profile=profile,
        python=Path("/usr/bin/python3"),
        targets=targets,
    )


def _observations() -> dict:
    denied = {"denied": True, "errno": 1, "result": None}
    return {
        "evaluator_read": {"denied": False, "errno": None, "result": 1},
        "network": dict(denied),
        "process_fork": dict(denied),
        "production_write": dict(denied),
        "resident_read": dict(denied),
        "training_write": dict(denied),
    }


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def test_kernel_probe_executes_and_receipt_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(tmp_path)
    stdout = (
        json.dumps(
            _observations(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )
    monkeypatch.setattr(
        probe.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            0,
            stdout=stdout,
            stderr=b"",
        ),
    )

    receipt = probe.execute_kernel_probe(
        spec,
        contract_sha256="a" * 64,
        environment={"PATH": "/usr/bin"},
        cwd=tmp_path,
    )

    assert receipt["all_expectations_met"] is True
    assert (
        probe.validate_kernel_probe_receipt(
            receipt,
            spec=spec,
            contract_sha256="a" * 64,
        )
        == receipt
    )


def test_kernel_probe_rejects_observation_or_target_rebinding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(tmp_path)
    observations = _observations()
    observations["network"] = {"denied": False, "errno": None, "result": None}
    stdout = (
        json.dumps(observations, sort_keys=True, separators=(",", ":")).encode(
            "ascii"
        )
        + b"\n"
    )
    monkeypatch.setattr(
        probe.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            0,
            stdout=stdout,
            stderr=b"",
        ),
    )
    with pytest.raises(
        probe.RecurrentSFTKernelProbeError,
        match="network_expectation_failed",
    ):
        probe.execute_kernel_probe(
            spec,
            contract_sha256="a" * 64,
            environment={"PATH": "/usr/bin"},
            cwd=tmp_path,
        )

    rebound = copy.deepcopy(spec)
    target = Path(rebound["targets"]["training_write"]["path"])
    target.write_bytes(b"rebound")
    with pytest.raises(
        probe.RecurrentSFTKernelProbeError,
        match="spec_binding_mismatch",
    ):
        probe.validate_kernel_probe_spec(rebound)


def test_contained_validation_does_not_reread_denied_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(tmp_path)
    stdout = (
        json.dumps(
            _observations(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )
    monkeypatch.setattr(
        probe.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            0,
            stdout=stdout,
            stderr=b"",
        ),
    )
    receipt = probe.execute_kernel_probe(
        spec,
        contract_sha256="a" * 64,
        environment={"PATH": "/usr/bin"},
        cwd=tmp_path,
    )
    Path(spec["targets"]["resident_read"]["path"]).write_bytes(b"changed")

    assert (
        probe.validate_kernel_probe_receipt(
            receipt,
            spec=spec,
            contract_sha256="a" * 64,
            rebind_files=False,
        )
        == receipt
    )
    with pytest.raises(
        probe.RecurrentSFTKernelProbeError,
        match="spec_binding_mismatch",
    ):
        probe.validate_kernel_probe_receipt(
            receipt,
            spec=spec,
            contract_sha256="a" * 64,
        )


def test_contained_validation_still_rejects_command_rebinding(
    tmp_path: Path,
) -> None:
    rebound = copy.deepcopy(_spec(tmp_path))
    rebound["command"][0] = "/usr/bin/false"
    rebound["command_sha256"] = _canonical_sha256(rebound["command"])
    body = dict(rebound)
    body.pop("spec_sha256")
    rebound["spec_sha256"] = _canonical_sha256(body)

    with pytest.raises(
        probe.RecurrentSFTKernelProbeError,
        match="spec_command_invalid",
    ):
        probe.validate_kernel_probe_spec(rebound, rebind_files=False)
