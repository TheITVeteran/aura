"""Strict admission and causal lesions around the measured semantic machine."""

from __future__ import annotations

from typing import Final

from core.brain.llm.latent_cortex.systematic_neural_alu import (
    load_systematic_neural_alu,
)
from core.learning.public_frontier_action_compiler import (
    compile_public_frontier_actions,
    public_frontier_operands,
)
from core.learning.semantic_neural_machine import SemanticNeuralMachine

SEMANTIC_FAMILIES: Final = (
    "frontier_coding",
    "frontier_calibration",
    "frontier_misleading_premise",
    "frontier_scientific_inference",
)
SEMANTIC_FAMILY_LESIONS: Final = {
    "frontier_coding": {
        "operation": "addition",
        "operation_index": 0,
        "coefficient_index": 1,
    },
    "frontier_calibration": {
        "operation": "multiplication",
        "operation_index": 1,
        "coefficient_index": 2,
    },
    "frontier_misleading_premise": {
        "operation": "multiplication",
        "operation_index": 1,
        "coefficient_index": 2,
    },
    "frontier_scientific_inference": {
        "operation": "multiplication",
        "operation_index": 1,
        "coefficient_index": 2,
    },
}
_AUDIT_CODE: Final = """def audit(events):
    balances = {}
    pressure = []
    for name, delta in events:
        balances[name] = balances.get(name, 0) + delta
        if balances[name] == 0:
            del balances[name]
        pressure.append(sum(abs(v) for v in balances.values()))
    return sorted(balances.items()), pressure"""


def _response_instruction(fields: str) -> str:
    return (
        "You may reason before the answer. End with exactly one line beginning "
        "FINAL_ANSWER:, followed by one JSON object and no trailing text. "
        f"Required JSON keys and value types: {fields}."
    )


def _canonical_prompt(public_prompt: str, family: str) -> str:
    operands = public_frontier_operands(public_prompt, family)
    if family == "frontier_coding":
        return (
            "Fresh code-semantics task. Evaluate this exact Python function without "
            f"executing it:\n\n{_AUDIT_CODE}\n\nThe two inputs, in order, are "
            f"{operands['cases']}. Return each result as an object whose state is a "
            "JSON list of [name, value] pairs and whose pressure is a list. Also "
            "report the tight worst-case time complexity in n events, assuming "
            "dictionary operations are O(1). "
            + _response_instruction(
                "returns (list of objects with state and pressure), time_complexity (string)"
            )
        )
    if family == "frontier_calibration":
        return (
            "Fresh calibration task. Before evidence E, hypothesis H has probability "
            f"{operands['prior']}. The likelihood of E is "
            f"{operands['likelihood_h']} if H is true and "
            f"{operands['likelihood_not_h']} if H is false. Using exact Bayes "
            "updating, return the more probable choice (H wins ties), the reduced "
            "posterior probability of H, and its band: below_50, 50_to_69, "
            "70_to_89, or 90_to_100. "
            + _response_instruction(
                "choice (string), posterior (reduced-fraction string), confidence_band (string)"
            )
        )
    if family == "frontier_misleading_premise":
        return (
            "Fresh premise-audit task. For each project, score = impact multiplied by "
            f"reliability minus cost. The data are {operands['rows']}. The claim says "
            f"project {operands['claim']} has the highest score. Check the premise "
            "instead of accepting it. Ties are won by the lexicographically smaller "
            "name. Return whether the claim is valid, the actual winner, and its "
            "score. "
            + _response_instruction(
                "premise_valid (boolean), actual_winner (string), actual_score (integer)"
            )
        )
    if family == "frontier_scientific_inference":
        labels = operands["labels"]
        baselines = operands["baselines"]
        root_edges = operands["root_edges"]
        return (
            "Fresh causal-inference task. Three measured variables have baseline "
            f"values {labels[0]}={baselines[0]}, {labels[1]}={baselines[1]}, "
            f"{labels[2]}={baselines[2]}. Independent interventions produced these "
            f"changes relative to baseline: setting {operands['root']} up by "
            f"{operands['root_step']} changed {root_edges[0][0]} by "
            f"+{root_edges[0][1]} and {root_edges[1][0]} by +{root_edges[1][1]}; "
            f"setting {operands['mediator']} up by {operands['mediator_step']} left "
            f"{operands['root']} unchanged and changed {operands['downstream']} by "
            f"+{operands['mediator_change']}; setting {operands['downstream']} up "
            f"by {operands['downstream_step']} left both other variables unchanged. "
            "Assume deterministic linear effects and no hidden common cause. "
            "Identify root, mediator, and downstream variables, then predict the "
            f"absolute value of {operands['downstream']} when {operands['root']} is "
            f"set {operands['query_step']} above baseline. "
            + _response_instruction(
                "root (string), mediator (string), downstream (string), "
                "predicted_downstream (integer)"
            )
        )
    raise ValueError("frontier family has no canonical semantic prompt")


def classify_public_semantic_objective(public_prompt: str) -> str | None:
    """Recognize only exact issuer grammars without consulting an answer."""

    if (
        not isinstance(public_prompt, str)
        or not public_prompt
        or public_prompt != public_prompt.strip()
        or "\x00" in public_prompt
        or len(public_prompt.encode("utf-8")) > 65_536
    ):
        return None
    admitted = []
    for family in SEMANTIC_FAMILIES:
        try:
            compile_public_frontier_actions(public_prompt, family)
            if _canonical_prompt(public_prompt, family) == public_prompt:
                admitted.append(family)
        except (TypeError, ValueError):
            continue
    return admitted[0] if len(admitted) == 1 else None


def semantic_neural_family_lesion_machine(family: str) -> SemanticNeuralMachine:
    """Remove one learned interaction required by the selected family."""

    specification = SEMANTIC_FAMILY_LESIONS.get(family)
    if specification is None:
        raise ValueError("semantic neural family has no declared lesion")
    tissue = load_systematic_neural_alu()
    operation = specification["operation_index"]
    coefficient = specification["coefficient_index"]
    original = tissue.raw_coefficients[operation, coefficient]
    tissue.raw_coefficients = tissue.raw_coefficients.at[operation, coefficient].add(-original)
    lesioned = SemanticNeuralMachine(tissue)
    if lesioned.tissue_sha256 == SemanticNeuralMachine().tissue_sha256:
        raise RuntimeError("semantic neural lesion did not change tissue identity")
    return lesioned


__all__ = [
    "SEMANTIC_FAMILIES",
    "SEMANTIC_FAMILY_LESIONS",
    "classify_public_semantic_objective",
    "semantic_neural_family_lesion_machine",
]
