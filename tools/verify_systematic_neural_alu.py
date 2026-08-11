#!/usr/bin/env python3
"""Independently verify the sealed systematic neural ALU on frozen moduli."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.systematic_neural_alu import (  # noqa: E402
    DEFAULT_SYSTEMATIC_NEURAL_ALU_ARTIFACT,
    SystematicNeuralALU,
    execute_systematic_neural_program,
    load_systematic_neural_alu,
)
from core.brain.llm.latent_cortex.typed_action_compiler import (  # noqa: E402
    compile_modular_operations,
)
from core.learning.systematic_neural_alu_training import (  # noqa: E402
    DEVELOPMENT_MODULI,
    FROZEN_TEST_MODULI,
    TRAIN_MODULI,
)

PREREGISTRATION = (
    REPO_ROOT
    / "artifacts"
    / "closeout"
    / "latent_cortex"
    / "cp197_systematic_neural_alu"
    / "preregistration.json"
)
FROZEN_DEPTHS = (1, 2, 4, 8, 16, 32, 64)
FROZEN_SEEDS = tuple(range(197_000, 197_016))


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _expected(opcode: int, residue: int, operand: int, modulus: int) -> int:
    raw = (
        residue + operand
        if opcode == 0
        else residue * operand
        if opcode == 1
        else residue - operand
    )
    return raw % modulus


def _program(modulus: int, depth: int, seed: int):
    rng = random.Random(f"cp197:{modulus}:{depth}:{seed}")
    initial = rng.randrange(modulus)
    symbols = ("+", "*", "-")
    operations = tuple(
        f"{rng.choice(symbols)}{rng.randrange(modulus)}" for _ in range(depth)
    )
    source = {
        "modulus": modulus,
        "depth": depth,
        "seed": seed,
        "initial": initial,
        "operations": operations,
    }
    return compile_modular_operations(
        initial=initial,
        modulus=modulus,
        operations=operations,
        public_source_sha256=_sha(source),
    )


def _reference_terminal(program) -> int:
    value = program.initial_state[1]
    for opcode, operand, modulus in program.actions:
        value = _expected(opcode, value, operand, modulus)
    return value


def verify_systematic_neural_alu_artifact(artifact_dir: Path) -> dict[str, Any]:
    preregistration_bytes = PREREGISTRATION.read_bytes()
    preregistration = json.loads(preregistration_bytes)
    if (
        preregistration.get("schema")
        != "aura.systematic_neural_alu_preregistration.v1"
        or tuple(preregistration.get("train_moduli", ())) != TRAIN_MODULI
        or tuple(preregistration.get("development_moduli", ()))
        != DEVELOPMENT_MODULI
        or tuple(preregistration.get("frozen_test_moduli", ()))
        != FROZEN_TEST_MODULI
        or tuple(preregistration.get("frozen_program_depths", ())) != FROZEN_DEPTHS
    ):
        raise RuntimeError("systematic neural ALU preregistration differs")
    tissue = load_systematic_neural_alu(artifact_dir)

    primitive_count = 0
    for modulus in FROZEN_TEST_MODULI:
        for residue in range(modulus):
            for operand in range(modulus):
                for opcode in (0, 1, 2):
                    observed = tissue.transition(
                        depth=2,
                        state=(0, residue, 0),
                        action=(opcode, operand, modulus),
                    ).next_state[1]
                    if observed != _expected(opcode, residue, operand, modulus):
                        raise RuntimeError("frozen primitive transfer failed")
                    primitive_count += 1

    program_count = 0
    transition_count = 0
    t1_lesion_changes = 0
    reversed_action_changes = 0
    reset_state_changes = 0
    sham_matches = 0
    sham = SystematicNeuralALU()
    for modulus in FROZEN_TEST_MODULI:
        for depth in FROZEN_DEPTHS:
            for seed in FROZEN_SEEDS:
                program = _program(modulus, depth, seed)
                complete = execute_systematic_neural_program(program, tissue)
                expected = _reference_terminal(program)
                if complete.terminal_state != (depth, expected, 1):
                    raise RuntimeError("frozen deep student roll-in failed")
                if depth > 1:
                    t1 = execute_systematic_neural_program(program, tissue, max_steps=1)
                    t1_lesion_changes += int(t1.terminal_state[1] != expected)
                reversed_result = execute_systematic_neural_program(
                    program,
                    tissue,
                    actions=tuple(reversed(program.actions)),
                )
                reversed_action_changes += int(
                    reversed_result.terminal_state[1] != expected
                )
                reset_program = type(program)(
                    schema=program.schema,
                    family=program.family,
                    depth=program.depth,
                    field_names=program.field_names,
                    initial_state=(0, (program.initial_state[1] + 1) % modulus, 0),
                    action_field_names=program.action_field_names,
                    actions=program.actions,
                    compiler_id=program.compiler_id,
                    public_source_sha256=program.public_source_sha256,
                )
                reset_result = execute_systematic_neural_program(reset_program, tissue)
                reset_state_changes += int(
                    reset_result.terminal_state[1] != expected
                )
                sham_result = execute_systematic_neural_program(program, sham)
                sham_matches += int(sham_result.terminal_state[1] == expected)
                program_count += 1
                transition_count += depth
    if (
        t1_lesion_changes < 250
        or reversed_action_changes < 100
        or reset_state_changes < 100
        or sham_matches > 200
    ):
        raise RuntimeError("systematic neural ALU causal controls are insufficient")
    body = {
        "schema": "aura.systematic_neural_alu_verification.v1",
        "verified": True,
        "teacher_available": False,
        "exact_operator_available": False,
        "lookup_table_available": False,
        "inference_modulo_operator_available": False,
        "tissue_sha256": tissue.tissue_sha256,
        "preregistration_sha256": hashlib.sha256(preregistration_bytes).hexdigest(),
        "train_moduli": list(TRAIN_MODULI),
        "development_moduli": list(DEVELOPMENT_MODULI),
        "frozen_test_moduli": list(FROZEN_TEST_MODULI),
        "frozen_primitive_count": primitive_count,
        "frozen_primitive_exact_accuracy": 1.0,
        "fresh_program_count": program_count,
        "fresh_transition_count": transition_count,
        "fresh_program_exact_accuracy": 1.0,
        "t1_lesion_changes": t1_lesion_changes,
        "reversed_action_changes": reversed_action_changes,
        "reset_state_changes": reset_state_changes,
        "sham_matches": sham_matches,
        "claim_boundary": (
            "systematic neural modular transition transfer and deep recurrent "
            "composition; symbolic action compilation remains and this is not "
            "resident-transformer Level 3, frontier reasoning, or a WOW Signal"
        ),
    }
    return {**body, "report_sha256": _sha(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifact",
        type=Path,
        nargs="?",
        default=DEFAULT_SYSTEMATIC_NEURAL_ALU_ARTIFACT,
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = verify_systematic_neural_alu_artifact(args.artifact)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        destination = args.report.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        scratch = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        scratch.write_text(encoded, encoding="utf-8")
        with scratch.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(scratch, destination)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
