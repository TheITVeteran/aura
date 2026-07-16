"""Recursive Latent Cortex — the frozen checkpoint as a programmable reasoning machine.

Spec and honest-claims ladder: docs/RECURSIVE_LATENT_CORTEX.md

Worker-side package: everything here runs inside the MLX worker process on the
RESIDENT model (or, for tests/experiments, on any in-process mlx_lm model).
All mlx imports are lazy so the package imports cleanly on hosts without MLX.

Public surface:
    CortexConfig / LatentReasoningResult   — types.py
    LatentCortexEngine                     — engine.py (the integrated machine)
    run_experiment_*                       — experiments.py (falsification harness)
"""
from core.brain.llm.latent_cortex.types import (
    BranchConfig,
    ComputeBudget,
    CortexConfig,
    EpisodeReceipt,
    FastWeightsConfig,
    LatentOptConfig,
    LatentReasoningResult,
    RecurrenceConfig,
    WorkspaceConfig,
)

__all__ = [
    "BranchConfig",
    "ComputeBudget",
    "CortexConfig",
    "EpisodeReceipt",
    "FastWeightsConfig",
    "LatentOptConfig",
    "LatentReasoningResult",
    "RecurrenceConfig",
    "WorkspaceConfig",
]
