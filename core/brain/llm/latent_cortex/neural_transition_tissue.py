"""Teacher-removed neural state transitions for bounded recurrent programs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import mlx.core as mx
import mlx.nn as nn

from core.brain.llm.latent_cortex.typed_action_compiler import TypedActionProgram

NEURAL_TRANSITION_TISSUE_SCHEMA: Final = "aura.neural_transition_tissue.v1"
NEURAL_TRANSITION_ARTIFACT_SCHEMA: Final = "aura.neural_transition_artifact.v1"
SUPPORTED_MODULI: Final = (13, 17, 19, 23)
MAX_RESIDUE: Final = max(SUPPORTED_MODULI)
DEFAULT_NEURAL_TRANSITION_ARTIFACT: Final = (
    Path(__file__).resolve().parent / "assets" / "neural_transition_tissue_v1"
)
NEURAL_TRANSITION_SOURCE_FILES: Final = (
    "core/brain/llm/latent_cortex/neural_transition_tissue.py",
    "core/brain/llm/latent_cortex/persistence.py",
    "core/brain/llm/latent_cortex/typed_transition_executor.py",
    "core/learning/neural_transition_training.py",
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
class NeuralTransitionTissueConfig:
    schema: str = NEURAL_TRANSITION_TISSUE_SCHEMA
    supported_moduli: tuple[int, ...] = SUPPORTED_MODULI
    max_residue: int = MAX_RESIDUE

    def __post_init__(self) -> None:
        if (
            self.schema != NEURAL_TRANSITION_TISSUE_SCHEMA
            or self.supported_moduli != SUPPORTED_MODULI
            or self.max_residue != MAX_RESIDUE
        ):
            raise ValueError("neural transition tissue configuration is invalid")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["supported_moduli"] = list(self.supported_moduli)
        return value


@dataclass(frozen=True, slots=True)
class NeuralTransitionResult:
    family: str
    next_state: tuple[int, ...]
    input_sha256: str
    tissue_sha256: str

    def receipt(self) -> dict[str, Any]:
        body = {
            "schema": NEURAL_TRANSITION_TISSUE_SCHEMA,
            "family": self.family,
            "input_sha256": self.input_sha256,
            "next_state_sha256": _canonical_sha256(list(self.next_state)),
            "tissue_sha256": self.tissue_sha256,
            "teacher_available": False,
            "student_rollin": True,
        }
        return {**body, "receipt_sha256": _canonical_sha256(body)}


@dataclass(frozen=True, slots=True)
class NeuralProgramExecution:
    family: str
    depth: int
    states: tuple[tuple[int, ...], ...]
    transition_receipts: tuple[dict[str, Any], ...]
    chain_sha256: str

    @property
    def terminal_state(self) -> tuple[int, ...]:
        return self.states[-1]

    def receipt(self) -> dict[str, Any]:
        body = {
            "schema": NEURAL_TRANSITION_TISSUE_SCHEMA,
            "family": self.family,
            "depth": self.depth,
            "state_count": len(self.states),
            "transition_count": len(self.transition_receipts),
            "initial_state_sha256": _canonical_sha256(list(self.states[0])),
            "terminal_state_sha256": _canonical_sha256(list(self.states[-1])),
            "transition_receipt_sha256s": [
                row["receipt_sha256"] for row in self.transition_receipts
            ],
            "chain_sha256": self.chain_sha256,
            "teacher_available": False,
            "student_rollin": True,
        }
        return {**body, "receipt_sha256": _canonical_sha256(body)}


class NeuralTransitionTissue(nn.Module):
    """Differentiable transition tables with a recurrent, teacher-free runtime.

    The tissue learns only the semantic value transition. Program-counter and
    terminal-bit updates remain structural invariants, just as sequence position
    and end-of-sequence handling do in a transformer runtime.
    """

    def __init__(
        self,
        config: NeuralTransitionTissueConfig | None = None,
        *,
        tissue_sha256: str = "unsealed",
    ) -> None:
        super().__init__()
        self.config = config or NeuralTransitionTissueConfig()
        if not tissue_sha256:
            raise ValueError("neural transition tissue identity is empty")
        self.tissue_sha256 = tissue_sha256
        self.boolean_logits = mx.zeros((32, 2), dtype=mx.float32)
        self.modular_logits = mx.zeros(
            (len(SUPPORTED_MODULI) * 3 * MAX_RESIDUE * MAX_RESIDUE, MAX_RESIDUE),
            dtype=mx.float32,
        )

    @staticmethod
    def boolean_key(*, opcode: int, value: int, operand: int, has_operand: int) -> int:
        return opcode * 8 + value * 4 + operand * 2 + has_operand

    @staticmethod
    def modular_key(*, modulus_index: int, opcode: int, residue: int, operand: int) -> int:
        return (
            modulus_index * 3 * MAX_RESIDUE * MAX_RESIDUE
            + opcode * MAX_RESIDUE * MAX_RESIDUE
            + residue * MAX_RESIDUE
            + operand
        )

    def boolean_batch(self, keys: Any) -> Any:
        return self.boolean_logits[keys]

    def modular_batch(self, keys: Any) -> Any:
        return self.modular_logits[keys]

    def _predict_value(
        self,
        *,
        family: str,
        state: tuple[int, ...],
        action: tuple[int, ...],
    ) -> int:
        if family == "boolean":
            value = state[1]
            opcode, operand, has_operand = action
            if value not in (0, 1) or operand not in (0, 1) or has_operand not in (0, 1):
                raise ValueError("Boolean neural transition value is outside {0,1}")
            if (opcode == 0 and (operand, has_operand) != (0, 0)) or (
                opcode in (1, 2, 3) and has_operand != 1
            ):
                raise ValueError("Boolean neural transition action is invalid")
            if opcode not in (0, 1, 2, 3):
                raise ValueError("Boolean neural transition opcode is unsupported")
            key = self.boolean_key(
                opcode=opcode,
                value=value,
                operand=operand,
                has_operand=has_operand,
            )
            predicted = mx.argmax(self.boolean_logits[key])
        elif family == "modular":
            residue = state[1]
            opcode, operand, modulus = action
            try:
                modulus_index = SUPPORTED_MODULI.index(modulus)
            except ValueError:
                raise ValueError("modulus is outside neural tissue support") from None
            if opcode not in (0, 1, 2):
                raise ValueError("modular neural transition opcode is unsupported")
            if not 0 <= residue < modulus or not 0 <= operand < modulus:
                raise ValueError("modular neural transition operand is invalid")
            key = self.modular_key(
                modulus_index=modulus_index,
                opcode=opcode,
                residue=residue,
                operand=operand,
            )
            logits = self.modular_logits[key]
            predicted = mx.argmax(logits[:modulus])
        else:
            raise ValueError(f"unsupported neural transition family: {family}")
        mx.eval(predicted)
        return int(predicted.item())

    def transition(
        self,
        *,
        family: str,
        depth: int,
        field_names: tuple[str, ...],
        state: tuple[int, ...],
        action_field_names: tuple[str, ...],
        action: tuple[int, ...],
    ) -> NeuralTransitionResult:
        expected_fields = ("pc", "value", "done") if family == "boolean" else (
            "pc",
            "residue",
            "done",
        )
        expected_action_fields = (
            ("opcode", "operand", "has_operand")
            if family == "boolean"
            else ("opcode", "operand", "modulus")
        )
        if (
            type(depth) is not int
            or not 1 <= depth <= 1_024
            or field_names != expected_fields
            or action_field_names != expected_action_fields
            or len(state) != 3
            or len(action) != 3
            or any(type(value) is not int or value < 0 for value in (*state, *action))
            or state[0] >= depth
            or state[-1] != 0
        ):
            raise ValueError("neural transition request is invalid")
        next_value = self._predict_value(family=family, state=state, action=action)
        next_pc = state[0] + 1
        next_state = (next_pc, next_value, int(next_pc == depth))
        input_payload = {
            "schema": NEURAL_TRANSITION_TISSUE_SCHEMA,
            "family": family,
            "depth": depth,
            "field_names": list(field_names),
            "state": list(state),
            "action_field_names": list(action_field_names),
            "action": list(action),
        }
        return NeuralTransitionResult(
            family=family,
            next_state=next_state,
            input_sha256=_canonical_sha256(input_payload),
            tissue_sha256=self.tissue_sha256,
        )


def execute_neural_action_program(
    program: TypedActionProgram,
    tissue: NeuralTransitionTissue,
    *,
    actions: tuple[tuple[int, ...], ...] | None = None,
    max_steps: int | None = None,
) -> NeuralProgramExecution:
    if not isinstance(program, TypedActionProgram) or not isinstance(
        tissue, NeuralTransitionTissue
    ):
        raise TypeError("neural program execution requires typed inputs")
    selected_actions = program.actions if actions is None else actions
    if len(selected_actions) != program.depth:
        raise ValueError("neural program action count differs from depth")
    step_count = program.depth if max_steps is None else max_steps
    if type(step_count) is not int or not 1 <= step_count <= program.depth:
        raise ValueError("neural program step limit is invalid")
    states = [program.initial_state]
    receipts: list[dict[str, Any]] = []
    chain = _canonical_sha256(
        {
            "schema": NEURAL_TRANSITION_TISSUE_SCHEMA,
            "program_sha256": program.program_sha256,
            "tissue_sha256": tissue.tissue_sha256,
        }
    )
    for action in selected_actions[:step_count]:
        result = tissue.transition(
            family=program.family,
            depth=program.depth,
            field_names=program.field_names,
            state=states[-1],
            action_field_names=program.action_field_names,
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
    return NeuralProgramExecution(
        family=program.family,
        depth=program.depth,
        states=tuple(states),
        transition_receipts=tuple(receipts),
        chain_sha256=chain,
    )


def load_neural_transition_tissue(
    artifact_dir: Path = DEFAULT_NEURAL_TRANSITION_ARTIFACT,
) -> NeuralTransitionTissue:
    directory = artifact_dir.expanduser().resolve(strict=True)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError("neural transition manifest is not an object")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    training_receipt = manifest.get("training_receipt")
    if (
        manifest.get("schema") != NEURAL_TRANSITION_ARTIFACT_SCHEMA
        or manifest.get("manifest_sha256") != _canonical_sha256(body)
        or manifest.get("config") != NeuralTransitionTissueConfig().to_dict()
        or manifest.get("teacher_removed_runtime") is not True
        or not isinstance(training_receipt, dict)
    ):
        raise RuntimeError("neural transition manifest commitment differs")
    receipt_body = {
        key: value for key, value in training_receipt.items() if key != "receipt_sha256"
    }
    final_metrics = training_receipt.get("final_metrics")
    if (
        training_receipt.get("schema") != "aura.neural_transition_training.v1"
        or training_receipt.get("receipt_sha256") != _canonical_sha256(receipt_body)
        or training_receipt.get("example_count") != 4_058
        or training_receipt.get("boolean_example_count") != 14
        or training_receipt.get("modular_example_count") != 4_044
        or training_receipt.get("teacher_removed_before_evaluation") is not True
        or not isinstance(training_receipt.get("source_sha256s"), dict)
        or not isinstance(final_metrics, dict)
        or final_metrics.get("exact_accuracy") != 1.0
    ):
        raise RuntimeError("neural transition training receipt differs")
    repo_root = Path(__file__).resolve().parents[4]
    source_sha256s = training_receipt["source_sha256s"]
    if set(source_sha256s) != set(NEURAL_TRANSITION_SOURCE_FILES) or any(
        _file_sha256(repo_root / relative) != source_sha256s[relative]
        for relative in NEURAL_TRANSITION_SOURCE_FILES
    ):
        raise RuntimeError("neural transition training source differs")
    weights_name = manifest.get("weights_file")
    if weights_name != "weights.safetensors":
        raise RuntimeError("neural transition weights path is invalid")
    weights_path = (directory / weights_name).resolve(strict=True)
    if weights_path.parent != directory or _file_sha256(weights_path) != manifest.get(
        "weights_sha256"
    ):
        raise RuntimeError("neural transition weights commitment differs")
    tensors = mx.load(str(weights_path))
    expected_shapes = {
        "boolean_logits": (32, 2),
        "modular_logits": (
            len(SUPPORTED_MODULI) * 3 * MAX_RESIDUE * MAX_RESIDUE,
            MAX_RESIDUE,
        ),
    }
    if set(tensors) != set(expected_shapes) or any(
        tuple(tensors[name].shape) != shape for name, shape in expected_shapes.items()
    ):
        raise RuntimeError("neural transition tensor inventory differs")
    tissue = NeuralTransitionTissue(tissue_sha256=manifest["weights_sha256"])
    tissue.boolean_logits = tensors["boolean_logits"].astype(mx.float32)
    tissue.modular_logits = tensors["modular_logits"].astype(mx.float32)
    mx.eval(tissue.parameters())
    return tissue


def build_neural_transition_manifest(
    *,
    weights_sha256: str,
    training_receipt: dict[str, Any],
) -> dict[str, Any]:
    if len(weights_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in weights_sha256
    ):
        raise ValueError("neural transition weights commitment is invalid")
    body = {
        "schema": NEURAL_TRANSITION_ARTIFACT_SCHEMA,
        "config": NeuralTransitionTissueConfig().to_dict(),
        "weights_file": "weights.safetensors",
        "weights_sha256": weights_sha256,
        "teacher_removed_runtime": True,
        "training_receipt": training_receipt,
    }
    return {**body, "manifest_sha256": _canonical_sha256(body)}


__all__ = [
    "DEFAULT_NEURAL_TRANSITION_ARTIFACT",
    "MAX_RESIDUE",
    "NEURAL_TRANSITION_ARTIFACT_SCHEMA",
    "NEURAL_TRANSITION_TISSUE_SCHEMA",
    "NEURAL_TRANSITION_SOURCE_FILES",
    "SUPPORTED_MODULI",
    "build_neural_transition_manifest",
    "NeuralProgramExecution",
    "NeuralTransitionResult",
    "NeuralTransitionTissue",
    "NeuralTransitionTissueConfig",
    "execute_neural_action_program",
    "load_neural_transition_tissue",
]
