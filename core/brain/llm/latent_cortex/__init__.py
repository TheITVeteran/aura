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

from core.brain.llm.latent_cortex.latent_tree_search import (
    LatentTreeSearchConfig,
    build_empty_latent_tree_receipt,
    run_latent_tree_search,
    validate_latent_tree_receipt,
    validate_latent_tree_transaction,
)
from core.brain.llm.latent_cortex.latent_tree_search import (
    append_transaction as append_latent_tree_transaction,
)
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
from core.brain.llm.latent_cortex.virtual_quanta import (
    VirtualQuantaConfig,
    build_empty_virtual_quanta_receipt,
    run_virtual_quanta,
    validate_virtual_quanta_receipt,
)

__all__ = [
    "BranchConfig",
    "ComputeBudget",
    "CortexConfig",
    "EpisodeReceipt",
    "FastWeightsConfig",
    "LatentOptConfig",
    "LatentReasoningResult",
    "LatentTreeSearchConfig",
    "RecurrenceConfig",
    "VirtualQuantaConfig",
    "WorkspaceConfig",
    "build_empty_virtual_quanta_receipt",
    "build_empty_latent_tree_receipt",
    "append_latent_tree_transaction",
    "run_latent_tree_search",
    "validate_latent_tree_receipt",
    "validate_latent_tree_transaction",
    "run_virtual_quanta",
    "validate_virtual_quanta_receipt",
]
