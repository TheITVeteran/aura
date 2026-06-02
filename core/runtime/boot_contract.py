"""Canonical boot/runtime contract for Aura proof and launch surfaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BootServiceRequirement:
    name: str
    owner_file: str
    required_for: str
    failure_policy: str
    evidence_tokens: tuple[str, ...]
    aliases: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BootContractIssue:
    code: str
    service: str
    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


BOOT_SERVICE_REQUIREMENTS: tuple[BootServiceRequirement, ...] = (
    BootServiceRequirement(
        name="unified_will",
        owner_file="core/governance/will.py",
        required_for="governed decisions and consequential action",
        failure_policy="fail-closed",
        evidence_tokens=('ServiceContainer.register_instance("unified_will"', "def verify_receipt"),
        aliases=("will", "core.will"),
    ),
    BootServiceRequirement(
        name="being_runtime",
        owner_file="core/service_registration.py",
        required_for="state-grounded AuraNow self-report and LAMP runtime",
        failure_policy="degrade_with_receipt",
        evidence_tokens=("container.register(\n        'being_runtime'", "get_being_runtime"),
        aliases=("aura_now_runtime",),
    ),
    BootServiceRequirement(
        name="aura_now",
        owner_file="core/being/runtime.py",
        required_for="Cortex-facing live state packet",
        failure_policy="degrade_with_receipt",
        evidence_tokens=('ServiceContainer.register_instance("aura_now"', "def prompt_block"),
    ),
    BootServiceRequirement(
        name="memory_write_gateway",
        owner_file="core/memory/memory_write_gateway.py",
        required_for="governed durable memory writes",
        failure_policy="fail-closed",
        evidence_tokens=("class ConcreteMemoryWriteGateway", "_default_memory_governance_decide"),
    ),
    BootServiceRequirement(
        name="state_gateway",
        owner_file="core/state/state_gateway.py",
        required_for="governed runtime state mutation",
        failure_policy="fail-closed",
        evidence_tokens=("class ConcreteStateGateway", "_default_state_governance_decide"),
    ),
    BootServiceRequirement(
        name="inference_gate",
        owner_file="core/brain/inference_gate.py",
        required_for="bounded live model response generation",
        failure_policy="fail-closed",
        evidence_tokens=("class InferenceGate", "inference_gate_generation_timeout"),
    ),
    BootServiceRequirement(
        name="llm_router",
        owner_file="core/providers/cognitive_provider.py",
        required_for="model routing and launch response path",
        failure_policy="fail-closed",
        evidence_tokens=("create_llm_router", "container.register('llm_router'"),
    ),
    BootServiceRequirement(
        name="capability_engine",
        owner_file="core/providers/cognitive_provider.py",
        required_for="governed tool and skill execution",
        failure_policy="fail-closed",
        evidence_tokens=("create_capability_engine", "container.register('capability_engine'"),
    ),
)

CANONICAL_PROOF_ARTIFACT_DIRS: tuple[str, ...] = (
    "agi_live",
    "agency_emergence_boxed_entity",
    "external_live_validation",
    "unified_system_scenario",
    "continual_learning",
    "novel_environment_adaptation",
    "longevity_soak",
    "person_box_proof",
)


def validate_boot_contract(
    root: str | Path,
    *,
    requirements: tuple[BootServiceRequirement, ...] = BOOT_SERVICE_REQUIREMENTS,
) -> list[BootContractIssue]:
    repo_root = Path(root).resolve()
    issues: list[BootContractIssue] = []
    for requirement in requirements:
        path = repo_root / requirement.owner_file
        if not path.exists():
            issues.append(
                BootContractIssue(
                    code="BOOT_CONTRACT_OWNER_MISSING",
                    service=requirement.name,
                    path=requirement.owner_file,
                    message="Boot contract owner file is missing.",
                )
            )
            continue
        source = path.read_text(encoding="utf-8", errors="ignore")
        for token in requirement.evidence_tokens:
            if token not in source:
                issues.append(
                    BootContractIssue(
                        code="BOOT_CONTRACT_TOKEN_MISSING",
                        service=requirement.name,
                        path=requirement.owner_file,
                        message=f"Required boot evidence token not found: {token}",
                    )
                )
    return issues


def boot_contract_report(root: str | Path) -> dict[str, Any]:
    issues = validate_boot_contract(root)
    return {
        "schema": "aura.boot_contract.v1",
        "ok": not issues,
        "services": [requirement.to_dict() for requirement in BOOT_SERVICE_REQUIREMENTS],
        "canonical_proof_artifact_dirs": list(CANONICAL_PROOF_ARTIFACT_DIRS),
        "issues": [issue.to_dict() for issue in issues],
    }
