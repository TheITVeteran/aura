"""Shared immutable result contracts for deterministic proof solvers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProofAnswer:
    answer: str
    solver: str
    confidence: float = 1.0


@dataclass(frozen=True)
class ProofAnswerValidation:
    valid: bool | None
    solver: str | None
    candidate_answer: str
    derived_answer: str | None = None
    reason: str = "unknown_prompt_shape"
