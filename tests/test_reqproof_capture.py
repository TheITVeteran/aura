"""Command-capture contracts for the requirement proof control plane."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from reqproof_testkit import make_registry_dict, make_requirement

from tools.reqproof.capture import (
    ProofCaptureError,
    ProofSpecRegistry,
    assert_pushed_clean_source,
    capture_proof,
    load_proof_specs,
    validate_spec_targets,
)
from tools.reqproof.evidence import (
    EvidenceLedger,
    load_evidence_ledger,
    write_evidence_ledger_atomic,
)
from tools.reqproof.schema import Registry


def _hashed_specs(specs: list[dict]) -> dict:
    body = {"schema_version": 1, "specs": specs}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return {
        **body,
        "content_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def _spec(**overrides) -> dict:
    base = {
        "id": "bounded-proof",
        "command": ["{python}", "-m", "pytest", "-q", "tests/test_one.py"],
        "cwd": ".",
        "timeout_seconds": 30,
        "max_output_bytes": 4096,
        "source_paths": ["core/a.py", "tests/test_one.py"],
        "evidence_targets": [
            {
                "requirement_id": "TEST-001",
                "evidence_class": "test",
                "acceptance_ids": ["A1"],
            }
        ],
    }
    base.update(overrides)
    return base


class FakeGateway:
    def __init__(
        self,
        *,
        head: str = "a" * 40,
        remote: str | None = None,
        status: str = "",
    ):
        self.head = head
        self.remote = remote or head
        self.status = status

    def run(self, argv, **kwargs):
        command = tuple(argv)
        if command[-2:] == ("rev-parse", "HEAD"):
            stdout = self.head + "\n"
        elif command[-2:] == ("rev-parse", "origin/main"):
            stdout = self.remote + "\n"
        elif "status" in command:
            stdout = self.status
        else:
            raise AssertionError(command)
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")


class CaptureGateway(FakeGateway):
    def __init__(
        self,
        *,
        command_returncode: int = 0,
        stdout: str | None = None,
        timeout: bool = False,
    ):
        super().__init__()
        self.command_returncode = command_returncode
        self.stdout = stdout
        self.timeout = timeout
        self.command_kwargs = None

    def run(self, argv, **kwargs):
        if tuple(argv[:2]) == ("git", "rev-parse") or "status" in argv:
            return super().run(argv, **kwargs)
        self.command_kwargs = kwargs
        if self.timeout:
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        return subprocess.CompletedProcess(
            tuple(argv),
            self.command_returncode,
            stdout=(
                self.stdout
                if self.stdout is not None
                else "2 passed in 0.01s\n" if self.command_returncode == 0 else ""
            ),
            stderr="proof failed\n" if self.command_returncode else "",
        )


def _capture_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    (tmp_path / "core").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "config").mkdir()
    (tmp_path / "core" / "a.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "tests" / "test_one.py").write_text(
        "def test_one(): assert True\n", encoding="utf-8"
    )
    spec_path = tmp_path / "config" / "specs.json"
    spec_path.write_text(json.dumps(_hashed_specs([_spec()])), encoding="utf-8")
    registry = Registry.from_dict(
        make_registry_dict(
            [make_requirement(evidence_required=["implementation", "test"])]
        )
    )
    registry_path = tmp_path / "config" / "registry.json"
    registry_path.write_text(registry.to_canonical_json(), encoding="utf-8")
    ledger_path = tmp_path / "config" / "ledger.json"
    write_evidence_ledger_atomic(EvidenceLedger.empty_for(registry), ledger_path)
    return spec_path, registry_path, ledger_path


def test_checked_registry_round_trips_and_targets_real_acceptance(
    tmp_path: Path,
) -> None:
    path = tmp_path / "specs.json"
    path.write_text(json.dumps(_hashed_specs([_spec()])), encoding="utf-8")
    specs = load_proof_specs(path)
    registry = Registry.from_dict(
        make_registry_dict(
            [make_requirement(evidence_required=["implementation", "test"])]
        )
    )

    validate_spec_targets(specs, registry)
    assert specs.by_id()["bounded-proof"].timeout_seconds == 30


def test_spec_hash_tampering_and_shell_placeholders_fail_closed() -> None:
    data = _hashed_specs([_spec()])
    data["specs"][0]["timeout_seconds"] = 31
    with pytest.raises(ProofCaptureError, match="content hash mismatch"):
        ProofSpecRegistry.from_dict(data)

    with pytest.raises(ProofCaptureError, match="unsupported placeholder"):
        ProofSpecRegistry.from_dict(
            _hashed_specs([_spec(command=["sh", "-c", "{payload}"])])
        )
    with pytest.raises(ProofCaptureError, match="may not invoke a shell"):
        ProofSpecRegistry.from_dict(
            _hashed_specs([_spec(command=["sh", "-c", "pytest -q"])])
        )


def test_specs_reject_unsorted_sources_targets_and_unknown_cells() -> None:
    with pytest.raises(ProofCaptureError, match="source_paths must be sorted"):
        ProofSpecRegistry.from_dict(
            _hashed_specs([_spec(source_paths=["tests/z.py", "core/a.py"])])
        )

    specs = ProofSpecRegistry.from_dict(_hashed_specs([_spec()]))
    registry = Registry.from_dict(
        make_registry_dict([make_requirement(evidence_required=["implementation"])])
    )
    with pytest.raises(ProofCaptureError, match="unrequired class"):
        validate_spec_targets(specs, registry)


def test_source_must_equal_pushed_main_and_be_clean(tmp_path: Path) -> None:
    assert assert_pushed_clean_source(FakeGateway(), tmp_path) == "a" * 40
    with pytest.raises(ProofCaptureError, match="not exact pushed main"):
        assert_pushed_clean_source(FakeGateway(remote="b" * 40), tmp_path)
    with pytest.raises(ProofCaptureError, match="dirty"):
        assert_pushed_clean_source(FakeGateway(status=" M core/a.py\n"), tmp_path)


def test_capture_writes_hash_bound_receipt_and_ledger_cell(tmp_path: Path) -> None:
    spec_path, registry_path, ledger_path = _capture_fixture(tmp_path)
    gateway = CaptureGateway()

    receipt_path = capture_proof(
        root=tmp_path,
        spec_registry_path=spec_path,
        registry_path=registry_path,
        ledger_path=ledger_path,
        artifact_root=tmp_path / "artifacts" / "reqproof" / "evidence",
        proof_id="bounded-proof",
        record=True,
        gateway=gateway,
    )

    receipt = json.loads(receipt_path.read_text())
    ledger = load_evidence_ledger(ledger_path)
    assert receipt["verdict"] == "pass"
    assert receipt["source_commit"] == "a" * 40
    assert receipt["stdout"] == "2 passed in 0.01s\n"
    assert [item["path"] for item in receipt["source_manifest"]] == [
        "core/a.py",
        "tests/test_one.py",
    ]
    assert ledger.entries[0].evidence.ref == receipt_path.relative_to(tmp_path).as_posix()
    assert ledger.entries[0].acceptance_ids == ("A1",)
    assert gateway.command_kwargs["offline_tooling"] is True
    assert gateway.command_kwargs["accelerator_capability"] == "none"
    assert gateway.command_kwargs["stdin_devnull"] is True


def test_failed_command_leaves_no_receipt_or_ledger_entry(tmp_path: Path) -> None:
    spec_path, registry_path, ledger_path = _capture_fixture(tmp_path)
    before = ledger_path.read_bytes()

    with pytest.raises(ProofCaptureError, match="failed with exit 1"):
        capture_proof(
            root=tmp_path,
            spec_registry_path=spec_path,
            registry_path=registry_path,
            ledger_path=ledger_path,
            artifact_root=tmp_path / "artifacts" / "reqproof" / "evidence",
            proof_id="bounded-proof",
            record=True,
            gateway=CaptureGateway(command_returncode=1),
        )

    assert ledger_path.read_bytes() == before
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize(
    ("gateway", "message"),
    [
        (CaptureGateway(timeout=True), "exceeded 30s"),
        (CaptureGateway(stdout="x" * 5000), "exceeds 4096-byte contract"),
    ],
)
def test_timeout_and_output_overflow_leave_no_evidence(
    tmp_path: Path,
    gateway: CaptureGateway,
    message: str,
) -> None:
    spec_path, registry_path, ledger_path = _capture_fixture(tmp_path)
    before = ledger_path.read_bytes()

    with pytest.raises(ProofCaptureError, match=message):
        capture_proof(
            root=tmp_path,
            spec_registry_path=spec_path,
            registry_path=registry_path,
            ledger_path=ledger_path,
            artifact_root=tmp_path / "artifacts" / "reqproof" / "evidence",
            proof_id="bounded-proof",
            record=True,
            gateway=gateway,
        )

    assert ledger_path.read_bytes() == before
    assert not (tmp_path / "artifacts").exists()


def test_checked_repo_spec_is_valid() -> None:
    root = Path(__file__).resolve().parents[1]
    specs = load_proof_specs(root / "config" / "requirement_proof_specs.json")
    registry = Registry.from_dict(
        json.loads((root / "config" / "requirement_registry.json").read_text())
    )
    validate_spec_targets(specs, registry)
    assert specs.by_id()["model-lane-contract-tests"].command[0] == "{python}"
