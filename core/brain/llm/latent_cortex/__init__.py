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

from core.brain.llm.latent_cortex.answer_replacement import (
    ANSWER_REPLACEMENT_PRIVATE_SCHEMA,
    ANSWER_REPLACEMENT_SCHEMA,
    DEFAULT_REPLACEMENT_MARGIN,
    MAX_REPLACEMENT_OUTPUT_TOKENS,
    build_answer_replacement_receipt,
    validate_answer_replacement_receipt,
)
from core.brain.llm.latent_cortex.atomic_decomposition import (
    ATOMIC_DECOMPOSITION_SCHEMA,
    AtomKind,
    TransitionKind,
    atom_ids,
    build_atomic_decomposition,
    decomposition_check,
    validate_atomic_decomposition,
    validate_atomic_decomposition_envelope,
)
from core.brain.llm.latent_cortex.context_focus import (
    CONTEXT_FOCUS_ACTIONS,
    CONTEXT_FOCUS_SCHEMA,
    DEFAULT_CONTEXT_FOCUS_STRENGTH,
    apply_context_focus,
    context_sources_for_action,
    validate_context_focus_receipt,
)
from core.brain.llm.latent_cortex.deterministic_verifier_router import (
    DETERMINISTIC_ROUTER_SCHEMA,
    RouteOutcome,
    build_deterministic_router_receipt,
    router_check,
    validate_deterministic_router_envelope,
)
from core.brain.llm.latent_cortex.diagnostic_action_selector import (
    DIAGNOSTIC_ACTION_SELECTOR_SCHEMA,
    build_candidate_routes,
    build_diagnostic_action_selector_receipt,
    validate_diagnostic_action_selector_receipt,
)
from core.brain.llm.latent_cortex.disagreement_graph import (
    DISAGREEMENT_GRAPH_SCHEMA,
    build_disagreement_graph_receipt,
    decompose_branch_candidates,
    validate_disagreement_graph_receipt,
)
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
from core.brain.llm.latent_cortex.local_repair import (
    LOCAL_REPAIR_SCHEMA,
    build_local_repair_receipt,
    parse_local_repair_generation,
    prepare_local_repair_requests,
    validate_local_repair_receipt,
)
from core.brain.llm.latent_cortex.output_arbitration import (
    DEFAULT_ARBITRATION_MARGIN,
    MAX_ARBITRATION_OUTPUT_TOKENS,
    OUTPUT_ARBITRATION_SCHEMA,
    ArbitrationDecision,
    CandidateSource,
    OutputCandidate,
    assert_output_arbitration_no_regression,
    build_output_arbitration_receipt,
    validate_output_arbitration_receipt,
)
from core.brain.llm.latent_cortex.test_time_training import (
    CRITIC_RECALIBRATION_SCHEMA,
    MATCHED_COMPUTE_SCHEMA,
    PSEUDO_LABEL_ADMISSION_SCHEMA,
    TEST_TIME_TRAINING_SCHEMA,
    build_critic_recalibration_receipt,
    build_matched_compute_receipt,
    build_pseudo_label_admission,
    deterministic_sham_target,
    validate_critic_recalibration_receipt,
    validate_matched_compute_receipt,
    validate_pseudo_label_admission,
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
    "ANSWER_REPLACEMENT_PRIVATE_SCHEMA",
    "ANSWER_REPLACEMENT_SCHEMA",
    "ArbitrationDecision",
    "ATOMIC_DECOMPOSITION_SCHEMA",
    "AtomKind",
    "BranchConfig",
    "ComputeBudget",
    "CONTEXT_FOCUS_ACTIONS",
    "CONTEXT_FOCUS_SCHEMA",
    "CRITIC_RECALIBRATION_SCHEMA",
    "CortexConfig",
    "DEFAULT_CONTEXT_FOCUS_STRENGTH",
    "DEFAULT_ARBITRATION_MARGIN",
    "DEFAULT_REPLACEMENT_MARGIN",
    "DETERMINISTIC_ROUTER_SCHEMA",
    "DISAGREEMENT_GRAPH_SCHEMA",
    "DIAGNOSTIC_ACTION_SELECTOR_SCHEMA",
    "EpisodeReceipt",
    "FastWeightsConfig",
    "LatentOptConfig",
    "LatentReasoningResult",
    "LatentTreeSearchConfig",
    "LOCAL_REPAIR_SCHEMA",
    "MAX_ARBITRATION_OUTPUT_TOKENS",
    "MAX_REPLACEMENT_OUTPUT_TOKENS",
    "MATCHED_COMPUTE_SCHEMA",
    "PSEUDO_LABEL_ADMISSION_SCHEMA",
    "OUTPUT_ARBITRATION_SCHEMA",
    "OutputCandidate",
    "RecurrenceConfig",
    "RouteOutcome",
    "CandidateSource",
    "TransitionKind",
    "TEST_TIME_TRAINING_SCHEMA",
    "VirtualQuantaConfig",
    "WorkspaceConfig",
    "atom_ids",
    "apply_context_focus",
    "assert_output_arbitration_no_regression",
    "build_atomic_decomposition",
    "build_answer_replacement_receipt",
    "build_candidate_routes",
    "build_deterministic_router_receipt",
    "build_diagnostic_action_selector_receipt",
    "build_disagreement_graph_receipt",
    "build_empty_virtual_quanta_receipt",
    "build_empty_latent_tree_receipt",
    "build_local_repair_receipt",
    "build_output_arbitration_receipt",
    "build_critic_recalibration_receipt",
    "build_matched_compute_receipt",
    "build_pseudo_label_admission",
    "context_sources_for_action",
    "decomposition_check",
    "decompose_branch_candidates",
    "append_latent_tree_transaction",
    "run_latent_tree_search",
    "router_check",
    "parse_local_repair_generation",
    "prepare_local_repair_requests",
    "deterministic_sham_target",
    "validate_answer_replacement_receipt",
    "validate_context_focus_receipt",
    "validate_latent_tree_receipt",
    "validate_latent_tree_transaction",
    "validate_atomic_decomposition",
    "validate_atomic_decomposition_envelope",
    "validate_deterministic_router_envelope",
    "validate_diagnostic_action_selector_receipt",
    "validate_disagreement_graph_receipt",
    "validate_local_repair_receipt",
    "validate_output_arbitration_receipt",
    "validate_critic_recalibration_receipt",
    "validate_matched_compute_receipt",
    "validate_pseudo_label_admission",
    "run_virtual_quanta",
    "validate_virtual_quanta_receipt",
]
