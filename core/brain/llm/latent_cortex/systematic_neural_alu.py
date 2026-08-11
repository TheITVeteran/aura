"""A learned periodic neural ALU that generalizes beyond trained moduli."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import mlx.core as mx
import mlx.nn as nn

from core.brain.llm.latent_cortex.typed_action_compiler import TypedActionProgram

SYSTEMATIC_NEURAL_ALU_SCHEMA: Final = "aura.systematic_neural_alu.v1"
SYSTEMATIC_NEURAL_ALU_ARTIFACT_SCHEMA: Final = "aura.systematic_neural_alu_artifact.v1"
MAX_MODULUS: Final = 63
HARMONIC_COUNT: Final = 8
SYSTEMATIC_NEURAL_ALU_SOURCE_FILES: Final = (
    "core/brain/llm/latent_cortex/persistence.py",
    "core/brain/llm/latent_cortex/systematic_neural_alu.py",
    "core/learning/systematic_neural_alu_training.py",
)
DEFAULT_SYSTEMATIC_NEURAL_ALU_ARTIFACT: Final = (
    Path(__file__).resolve().parent / "assets" / "systematic_neural_alu_v1"
)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SystematicNeuralALUConfig:
    schema: str = SYSTEMATIC_NEURAL_ALU_SCHEMA
    max_modulus: int = MAX_MODULUS
    harmonic_count: int = HARMONIC_COUNT

    def __post_init__(self) -> None:
        if (
            self.schema != SYSTEMATIC_NEURAL_ALU_SCHEMA
            or self.max_modulus != MAX_MODULUS
            or self.harmonic_count != HARMONIC_COUNT
        ):
            raise ValueError("systematic neural ALU configuration is invalid")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SystematicNeuralALUResult:
    next_state: tuple[int, int, int]
    raw_value: float
    confidence_margin: float
    tissue_sha256: str
    input_sha256: str

    def receipt(self) -> dict[str, Any]:
        body = {
            "schema": SYSTEMATIC_NEURAL_ALU_SCHEMA,
            "next_state_sha256": _canonical_sha256(list(self.next_state)),
            "raw_value_sha256": _canonical_sha256(self.raw_value),
            "confidence_margin": self.confidence_margin,
            "tissue_sha256": self.tissue_sha256,
            "input_sha256": self.input_sha256,
            "teacher_available": False,
            "exact_operator_available": False,
        }
        return {**body, "receipt_sha256": _canonical_sha256(body)}


@dataclass(frozen=True, slots=True)
class SystematicNeuralALUExecution:
    states: tuple[tuple[int, int, int], ...]
    transition_receipts: tuple[dict[str, Any], ...]
    chain_sha256: str

    @property
    def terminal_state(self) -> tuple[int, int, int]:
        return self.states[-1]

    def receipt(self) -> dict[str, Any]:
        body = {
            "schema": SYSTEMATIC_NEURAL_ALU_SCHEMA,
            "state_count": len(self.states),
            "transition_count": len(self.transition_receipts),
            "initial_state_sha256": _canonical_sha256(list(self.states[0])),
            "terminal_state_sha256": _canonical_sha256(list(self.states[-1])),
            "transition_receipt_sha256s": [
                row["receipt_sha256"] for row in self.transition_receipts
            ],
            "chain_sha256": self.chain_sha256,
            "student_rollin": True,
            "teacher_available": False,
            "exact_operator_available": False,
        }
        return {**body, "receipt_sha256": _canonical_sha256(body)}


class SystematicNeuralALU(nn.Module):
    """Learn a raw operation and periodic residue decoding from examples.

    The architecture exposes scalar and bilinear interactions and a cyclic
    candidate basis. Which interaction implements each opcode, and how the
    cyclic basis is weighted, are learned parameters. Inference contains no
    modulo operation or transition lookup table.
    """

    def __init__(self, *, tissue_sha256: str = "unsealed") -> None:
        super().__init__()
        if not tissue_sha256:
            raise ValueError("systematic neural ALU identity is empty")
        self.config = SystematicNeuralALUConfig()
        self.tissue_sha256 = tissue_sha256
        self.raw_coefficients = mx.random.normal(
            (3, 4),
            key=mx.random.key(20260810197),
        ).astype(mx.float32) * 0.1
        self.harmonic_weights = mx.zeros((HARMONIC_COUNT,), dtype=mx.float32)

    def raw_batch(self, opcodes: Any, residues: Any, operands: Any) -> Any:
        coefficients = self.raw_coefficients[opcodes]
        return (
            coefficients[:, 0] * residues
            + coefficients[:, 1] * operands
            + coefficients[:, 2] * (residues * operands)
            + coefficients[:, 3]
        )

    def logits_batch(
        self,
        opcodes: Any,
        residues: Any,
        operands: Any,
        moduli: Any,
    ) -> Any:
        raw = self.raw_batch(opcodes, residues, operands)
        candidates = mx.arange(MAX_MODULUS, dtype=mx.float32)[None, :]
        phase = (raw[:, None] - candidates) / moduli[:, None]
        logits = mx.zeros_like(phase)
        for harmonic in range(1, HARMONIC_COUNT + 1):
            logits = logits + self.harmonic_weights[harmonic - 1] * mx.cos(
                2.0 * math.pi * harmonic * phase
            )
        return mx.where(candidates < moduli[:, None], logits, mx.array(-1e9))

    def transition(
        self,
        *,
        depth: int,
        state: tuple[int, int, int],
        action: tuple[int, int, int],
    ) -> SystematicNeuralALUResult:
        if (
            type(depth) is not int
            or not 1 <= depth <= 1_024
            or len(state) != 3
            or len(action) != 3
            or any(type(value) is not int for value in (*state, *action))
        ):
            raise ValueError("systematic neural ALU request is invalid")
        pc, residue, done = state
        opcode, operand, modulus = action
        if (
            not 2 <= modulus <= MAX_MODULUS
            or opcode not in (0, 1, 2)
            or not 0 <= residue < modulus
            or not 0 <= operand < modulus
            or pc < 0
            or pc >= depth
            or done != 0
        ):
            raise ValueError("systematic neural ALU request is outside support")
        opcodes = mx.array([opcode], dtype=mx.int32)
        residues = mx.array([residue], dtype=mx.float32)
        operands = mx.array([operand], dtype=mx.float32)
        moduli = mx.array([modulus], dtype=mx.float32)
        raw = self.raw_batch(opcodes, residues, operands)
        logits = self.logits_batch(opcodes, residues, operands, moduli)[0]
        ranking = mx.argsort(logits)
        predicted = ranking[-1]
        margin = logits[ranking[-1]] - logits[ranking[-2]]
        mx.eval(raw, predicted, margin)
        next_pc = pc + 1
        next_state = (next_pc, int(predicted.item()), int(next_pc == depth))
        payload = {
            "schema": SYSTEMATIC_NEURAL_ALU_SCHEMA,
            "depth": depth,
            "state": list(state),
            "action": list(action),
        }
        return SystematicNeuralALUResult(
            next_state=next_state,
            raw_value=float(raw.item()),
            confidence_margin=float(margin.item()),
            tissue_sha256=self.tissue_sha256,
            input_sha256=_canonical_sha256(payload),
        )


def execute_systematic_neural_program(
    program: TypedActionProgram,
    tissue: SystematicNeuralALU,
    *,
    actions: tuple[tuple[int, ...], ...] | None = None,
    max_steps: int | None = None,
) -> SystematicNeuralALUExecution:
    if (
        not isinstance(program, TypedActionProgram)
        or program.family != "modular"
        or not isinstance(tissue, SystematicNeuralALU)
    ):
        raise TypeError("systematic neural ALU requires a modular typed program")
    selected = program.actions if actions is None else actions
    if len(selected) != program.depth:
        raise ValueError("systematic neural ALU action count differs from depth")
    steps = program.depth if max_steps is None else max_steps
    if type(steps) is not int or not 1 <= steps <= program.depth:
        raise ValueError("systematic neural ALU step limit is invalid")
    states = [program.initial_state]
    receipts: list[dict[str, Any]] = []
    chain = _canonical_sha256(
        {
            "schema": SYSTEMATIC_NEURAL_ALU_SCHEMA,
            "program_sha256": program.program_sha256,
            "tissue_sha256": tissue.tissue_sha256,
        }
    )
    for action in selected[:steps]:
        result = tissue.transition(
            depth=program.depth,
            state=states[-1],
            action=action,
        )
        receipt = result.receipt()
        states.append(result.next_state)
        receipts.append(receipt)
        chain = _canonical_sha256(
            {
                "prior_chain_sha256": chain,
                "transition_receipt_sha256": receipt["receipt_sha256"],
            }
        )
    return SystematicNeuralALUExecution(
        states=tuple(states),
        transition_receipts=tuple(receipts),
        chain_sha256=chain,
    )


def load_systematic_neural_alu(
    artifact_dir: Path = DEFAULT_SYSTEMATIC_NEURAL_ALU_ARTIFACT,
) -> SystematicNeuralALU:
    directory = artifact_dir.expanduser().resolve(strict=True)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError("systematic neural ALU manifest is not an object")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    receipt = manifest.get("training_receipt")
    if (
        manifest.get("schema") != SYSTEMATIC_NEURAL_ALU_ARTIFACT_SCHEMA
        or manifest.get("manifest_sha256") != _canonical_sha256(body)
        or manifest.get("config") != SystematicNeuralALUConfig().to_dict()
        or manifest.get("teacher_removed_runtime") is not True
        or not isinstance(receipt, dict)
    ):
        raise RuntimeError("systematic neural ALU manifest commitment differs")
    receipt_body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if (
        receipt.get("schema") != "aura.systematic_neural_alu_training.v1"
        or receipt.get("receipt_sha256") != _canonical_sha256(receipt_body)
        or receipt.get("teacher_removed_before_evaluation") is not True
        or receipt.get("train_exact_accuracy") != 1.0
        or receipt.get("development_exact_accuracy") != 1.0
    ):
        raise RuntimeError("systematic neural ALU training receipt differs")
    repo_root = Path(__file__).resolve().parents[4]
    source_sha256s = receipt.get("source_sha256s")
    if not isinstance(source_sha256s, dict) or set(source_sha256s) != set(
        SYSTEMATIC_NEURAL_ALU_SOURCE_FILES
    ) or any(
        _file_sha256(repo_root / relative) != source_sha256s[relative]
        for relative in SYSTEMATIC_NEURAL_ALU_SOURCE_FILES
    ):
        raise RuntimeError("systematic neural ALU training source differs")
    weights_path = (directory / str(manifest.get("weights_file"))).resolve(strict=True)
    if weights_path.parent != directory or _file_sha256(weights_path) != manifest.get(
        "weights_sha256"
    ):
        raise RuntimeError("systematic neural ALU weights commitment differs")
    tensors = mx.load(str(weights_path))
    if set(tensors) != {"raw_coefficients", "harmonic_weights"} or tuple(
        tensors["raw_coefficients"].shape
    ) != (3, 4) or tuple(tensors["harmonic_weights"].shape) != (HARMONIC_COUNT,):
        raise RuntimeError("systematic neural ALU tensor inventory differs")
    tissue = SystematicNeuralALU(tissue_sha256=manifest["weights_sha256"])
    tissue.raw_coefficients = tensors["raw_coefficients"].astype(mx.float32)
    tissue.harmonic_weights = tensors["harmonic_weights"].astype(mx.float32)
    mx.eval(tissue.parameters())
    return tissue


def build_systematic_neural_alu_manifest(
    *,
    weights_sha256: str,
    training_receipt: dict[str, Any],
) -> dict[str, Any]:
    if len(weights_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in weights_sha256
    ):
        raise ValueError("systematic neural ALU weights commitment is invalid")
    body = {
        "schema": SYSTEMATIC_NEURAL_ALU_ARTIFACT_SCHEMA,
        "config": SystematicNeuralALUConfig().to_dict(),
        "weights_file": "weights.safetensors",
        "weights_sha256": weights_sha256,
        "teacher_removed_runtime": True,
        "training_receipt": training_receipt,
    }
    return {**body, "manifest_sha256": _canonical_sha256(body)}


__all__ = [
    "DEFAULT_SYSTEMATIC_NEURAL_ALU_ARTIFACT",
    "HARMONIC_COUNT",
    "MAX_MODULUS",
    "SYSTEMATIC_NEURAL_ALU_ARTIFACT_SCHEMA",
    "SYSTEMATIC_NEURAL_ALU_SCHEMA",
    "SYSTEMATIC_NEURAL_ALU_SOURCE_FILES",
    "SystematicNeuralALU",
    "build_systematic_neural_alu_manifest",
    "SystematicNeuralALUConfig",
    "SystematicNeuralALUExecution",
    "SystematicNeuralALUResult",
    "execute_systematic_neural_program",
    "load_systematic_neural_alu",
]
