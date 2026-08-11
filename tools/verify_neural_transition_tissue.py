#!/usr/bin/env python3
"""Independently verify Aura's sealed teacher-removed transition tissue."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.neural_transition_tissue import (  # noqa: E402
    DEFAULT_NEURAL_TRANSITION_ARTIFACT,
    SUPPORTED_MODULI,
    execute_neural_action_program,
    load_neural_transition_tissue,
)
from core.brain.llm.latent_cortex.typed_action_compiler import (  # noqa: E402
    compile_public_transition_program,
)
from core.learning.recurrence_curriculum import modular_chain, nested_boolean  # noqa: E402


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


def verify_neural_transition_artifact(artifact_dir: Path) -> dict[str, Any]:
    tissue = load_neural_transition_tissue(artifact_dir)
    primitive_count = 0
    boolean_actions = ((0, 0, 0),) + tuple(
        (opcode, operand, 1) for opcode in (1, 2, 3) for operand in (0, 1)
    )
    for value in (0, 1):
        for opcode, operand, has_operand in boolean_actions:
            observed = tissue.transition(
                family="boolean",
                depth=2,
                field_names=("pc", "value", "done"),
                state=(0, value, 0),
                action_field_names=("opcode", "operand", "has_operand"),
                action=(opcode, operand, has_operand),
            ).next_state[1]
            expected = (
                1 - value
                if opcode == 0
                else value & operand
                if opcode == 1
                else value | operand
                if opcode == 2
                else value ^ operand
            )
            if observed != expected:
                raise RuntimeError("Boolean neural primitive verification failed")
            primitive_count += 1
    for modulus in SUPPORTED_MODULI:
        for residue in range(modulus):
            for operand in range(modulus):
                for opcode in (0, 1, 2):
                    observed = tissue.transition(
                        family="modular",
                        depth=2,
                        field_names=("pc", "residue", "done"),
                        state=(0, residue, 0),
                        action_field_names=("opcode", "operand", "modulus"),
                        action=(opcode, operand, modulus),
                    ).next_state[1]
                    expected = (
                        (residue + operand) % modulus
                        if opcode == 0
                        else (residue * operand) % modulus
                        if opcode == 1
                        else (residue - operand) % modulus
                    )
                    if observed != expected:
                        raise RuntimeError("modular neural primitive verification failed")
                    primitive_count += 1

    program_count = 0
    transition_count = 0
    lesion_changes = 0
    action_shuffle_changes = 0
    sham_matches = 0
    from core.brain.llm.latent_cortex.neural_transition_tissue import NeuralTransitionTissue

    sham = NeuralTransitionTissue()
    for generator in (nested_boolean, modular_chain):
        for depth in (1, 2, 4, 8, 16, 32):
            for seed in range(196_200, 196_216):
                task = generator(depth, seed)
                program = compile_public_transition_program(task.prompt)
                complete = execute_neural_action_program(program, tissue)
                if complete.terminal_state != task.transition_trace.states[-1]:
                    raise RuntimeError("fresh neural student roll-in verification failed")
                lesion = execute_neural_action_program(program, tissue, max_steps=1)
                shuffled = execute_neural_action_program(
                    program,
                    tissue,
                    actions=tuple(reversed(program.actions)),
                )
                sham_result = execute_neural_action_program(program, sham)
                lesion_changes += int(lesion.states[-1][1] != complete.terminal_state[1])
                action_shuffle_changes += int(
                    shuffled.terminal_state[1] != complete.terminal_state[1]
                )
                sham_matches += int(sham_result.terminal_state == complete.terminal_state)
                program_count += 1
                transition_count += depth
    if lesion_changes < 100 or action_shuffle_changes < 50 or sham_matches > 50:
        raise RuntimeError("neural transition causal controls are insufficient")
    body = {
        "schema": "aura.neural_transition_independent_verification.v1",
        "verified": True,
        "teacher_available": False,
        "tissue_sha256": tissue.tissue_sha256,
        "primitive_count": primitive_count,
        "fresh_program_count": program_count,
        "fresh_transition_count": transition_count,
        "lesion_changes": lesion_changes,
        "action_shuffle_changes": action_shuffle_changes,
        "sham_matches": sham_matches,
        "claim_boundary": (
            "bounded finite-state neural algorithm acquisition and recurrent composition; "
            "not open-domain, resident-32B, or frontier reasoning evidence"
        ),
    }
    return {**body, "report_sha256": _sha(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, nargs="?", default=DEFAULT_NEURAL_TRANSITION_ARTIFACT)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = verify_neural_transition_artifact(args.artifact)
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
