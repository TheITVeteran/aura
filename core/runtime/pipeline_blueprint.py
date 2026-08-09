"""core/runtime/pipeline_blueprint.py — the ordered phases, and which run when.

There is one blueprint of 29 phases, and it is regularly described as though
a user turn passes through all of them. It does not, and the difference is
architectural rather than incidental.

**Aura runs at two rates.** ``AuraKernel.tick(priority=True)`` serves the
person at the keyboard and suppresses the expensive phases outright — Φ,
full cognitive integration, inference, bonding, repair, memory
consolidation, identity reflection, initiative generation, the consciousness
phase, self-review, learning, evolution and several more. A healthy
foreground turn is closer to eleven phases plus conditional tool execution:

    proprioception → social context → sensory ingestion → memory retrieval
    → affect → motivation → executive closure → conversational dynamics
    → routing → unity → (tools) → response

The suppressed phases are not dormant. ``MindTick`` calls
``kernel.tick(objective, priority=False)`` on the same kernel, that
background tick traverses the complete pipeline, and its result is committed
back to the shared state repository. So the honest description is not "every
thought passes through 29 cognitive organs in sequence". It is: one
persistent cognitive runtime whose slower organs update shared state in the
background, while the foreground reads and updates a latency-bounded subset
of that same state.

That is a better architecture than the sequential reading — protecting
foreground latency is why the response arrives at all — but it must not be
described as the sequential one.

:func:`foreground_phase_attributes` and :func:`background_only_phase_names`
report the split from the code that enforces it, so prose about "the 29
phases" can be checked rather than believed. The kernel imports
:data:`BACKGROUND_ONLY_PHASES` rather than keeping its own copy, because two
copies of a list like this diverge and the doc always loses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.phases.affect_update import AffectUpdatePhase
from core.phases.cognitive_integration_phase import CognitiveIntegrationPhase
from core.phases.cognitive_routing import CognitiveRoutingPhase
from core.phases.consciousness_phase import ConsciousnessPhase
from core.phases.executive_closure import ExecutiveClosurePhase
from core.phases.identity_reflection import IdentityReflectionPhase
from core.phases.initiative_generation import InitiativeGenerationPhase
from core.phases.memory_consolidation import MemoryConsolidationPhase
from core.phases.memory_retrieval import MemoryRetrievalPhase
from core.phases.proprioceptive_loop import ProprioceptiveLoop
from core.phases.response_generation import ResponseGenerationPhase
from core.phases.sensory_ingestion import SensoryIngestionPhase
from core.phases.social_context_phase import SocialContextPhase
from core.phases.unity_binding import UnityBindingPhase


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    attribute_name: str
    phase_cls: type[Any]


_LEGACY_PIPELINE_PREFIX: tuple[PhaseSpec, ...] = (
    PhaseSpec("proprioceptive_loop", "proprioceptive_phase", ProprioceptiveLoop),
    PhaseSpec("social_context", "social_context_phase", SocialContextPhase),
    PhaseSpec("sensory_ingestion", "sensory_ingestion_phase", SensoryIngestionPhase),
    PhaseSpec("memory_retrieval", "memory_retrieval_phase", MemoryRetrievalPhase),
    PhaseSpec("affect_update", "affect_phase", AffectUpdatePhase),
    PhaseSpec(
        "cognitive_integration",
        "cognitive_integration_phase",
        CognitiveIntegrationPhase,
    ),
)

_LEGACY_PIPELINE_SUFFIX: tuple[PhaseSpec, ...] = (
    PhaseSpec("cognitive_routing", "routing_phase", CognitiveRoutingPhase),
    PhaseSpec("unity_binding", "unity_phase", UnityBindingPhase),
    PhaseSpec("response_generation", "response_phase", ResponseGenerationPhase),
    PhaseSpec("memory_consolidation", "memory_consolidation_phase", MemoryConsolidationPhase),
    PhaseSpec("identity_reflection", "identity_reflection_phase", IdentityReflectionPhase),
    PhaseSpec("initiative_generation", "initiative_generation_phase", InitiativeGenerationPhase),
    PhaseSpec("consciousness", "consciousness_phase", ConsciousnessPhase),
)

_EXECUTIVE_CLOSURE_SPEC = PhaseSpec(
    "executive_closure",
    "executive_closure_phase",
    ExecutiveClosurePhase,
)

_KERNEL_PIPELINE_ATTRIBUTE_ORDER: tuple[str, ...] = (
    "proprioceptive_phase",
    "social_context_phase",
    "sensory_ingestion_phase",
    "multimodal",
    "eternal",
    "memory_retrieval_phase",
    "perfect_emotion",
    "affect_phase",
    "phi_phase",
    "motivation_phase",
    "cognitive_integration",  # Learned cognitive systems (sentiment, anomaly, self-model, RL, plasticity)
    "executive_closure_phase",
    "evolution_guard",
    "growth",
    "evolution",
    "inference_phase",
    "conversational_dynamics_phase",
    "bonding_phase",
    "routing_phase",
    "unity_phase",
    "godmode_tools",
    "response_phase",
    "repair_phase",
    "memory_consolidation_phase",
    "identity_reflection_phase",
    "initiative_generation_phase",
    "consciousness_phase",
    "self_review_phase",
    "learning_phase",
)


def legacy_runtime_phase_specs(*, include_executive_closure: bool) -> tuple[PhaseSpec, ...]:
    specs = list(_LEGACY_PIPELINE_PREFIX)
    if include_executive_closure:
        specs.append(_EXECUTIVE_CLOSURE_SPEC)
    specs.extend(_LEGACY_PIPELINE_SUFFIX)
    return tuple(specs)


def instantiate_legacy_runtime_phases(
    owner: Any,
    *,
    include_executive_closure: bool,
) -> list[tuple[str, Any]]:
    return [
        (spec.name, spec.phase_cls(owner))
        for spec in legacy_runtime_phase_specs(
            include_executive_closure=include_executive_closure,
        )
    ]


def bind_legacy_runtime_phase_attributes(
    target: Any,
    owner: Any,
    *,
    include_executive_closure: bool,
) -> None:
    for spec in legacy_runtime_phase_specs(
        include_executive_closure=include_executive_closure,
    ):
        setattr(target, spec.attribute_name, spec.phase_cls(owner))


#: Phase CLASS names suppressed on a priority (user-facing) tick.
#:
#: The kernel held this set inline in ``_should_skip_priority_phase``, where
#: nothing outside the kernel could read it — so every description of the
#: pipeline had to be written from memory, and "29 phases per turn" is what
#: that produced. It lives here now, next to the order it modifies, and the
#: kernel imports it.
#:
#: ``GodModeToolPhase`` is in the set but conditional: it runs on a priority
#: tick when the turn's intent is SKILL or TASK. The kernel owns that rule
#: because it needs the live state to evaluate it.
BACKGROUND_ONLY_PHASES: frozenset[str] = frozenset(
    {
        "EternalMemoryPhase",
        "EternalGrowthEngine",
        "TrueEvolutionPhase",
        "NativeMultimodalBridge",
        "ShadowExecutionPhase",
        "PerfectEmotionPhase",
        "PhiConsciousnessPhase",
        "CognitiveIntegrationPhase",
        "InferencePhase",
        "BondingPhase",
        "RepairPhase",
        "MemoryConsolidationPhase",
        "IdentityReflectionPhase",
        "InitiativeGenerationPhase",
        "ConsciousnessPhase",
        "SelfReviewPhase",
        "LearningPhase",
        "LegacyPhase",
        "GodModeToolPhase",
    }
)

#: Attribute name → phase class name, for the kernel pipeline. Needed
#: because the order is expressed as attributes and the skip set as classes.
_ATTRIBUTE_TO_PHASE_CLASS: dict[str, str] = {
    "proprioceptive_phase": "ProprioceptiveLoop",
    "social_context_phase": "SocialContextPhase",
    "sensory_ingestion_phase": "SensoryIngestionPhase",
    "multimodal": "NativeMultimodalBridge",
    "eternal": "EternalMemoryPhase",
    "memory_retrieval_phase": "MemoryRetrievalPhase",
    "perfect_emotion": "PerfectEmotionPhase",
    "affect_phase": "AffectUpdatePhase",
    "phi_phase": "PhiConsciousnessPhase",
    "motivation_phase": "MotivationPhase",
    "cognitive_integration": "CognitiveIntegrationPhase",
    "executive_closure_phase": "ExecutiveClosurePhase",
    "evolution_guard": "ShadowExecutionPhase",
    "growth": "EternalGrowthEngine",
    "evolution": "TrueEvolutionPhase",
    "inference_phase": "InferencePhase",
    "conversational_dynamics_phase": "ConversationalDynamicsPhase",
    "bonding_phase": "BondingPhase",
    "routing_phase": "CognitiveRoutingPhase",
    "unity_phase": "UnityBindingPhase",
    "godmode_tools": "GodModeToolPhase",
    "response_phase": "ResponseGenerationPhase",
    "repair_phase": "RepairPhase",
    "memory_consolidation_phase": "MemoryConsolidationPhase",
    "identity_reflection_phase": "IdentityReflectionPhase",
    "initiative_generation_phase": "InitiativeGenerationPhase",
    "consciousness_phase": "ConsciousnessPhase",
    "self_review_phase": "SelfReviewPhase",
    "learning_phase": "LearningPhase",
}


def kernel_phase_attribute_order() -> tuple[str, ...]:
    return _KERNEL_PIPELINE_ATTRIBUTE_ORDER


def phase_class_for_attribute(attribute_name: str) -> str:
    return _ATTRIBUTE_TO_PHASE_CLASS.get(attribute_name, attribute_name)


def background_only_phase_names() -> tuple[str, ...]:
    """Phase classes a priority tick suppresses, in pipeline order."""
    return tuple(
        _ATTRIBUTE_TO_PHASE_CLASS[attribute]
        for attribute in _KERNEL_PIPELINE_ATTRIBUTE_ORDER
        if _ATTRIBUTE_TO_PHASE_CLASS.get(attribute) in BACKGROUND_ONLY_PHASES
    )


def foreground_phase_attributes() -> tuple[str, ...]:
    """Phases a priority tick actually runs, in order.

    ``GodModeToolPhase`` is excluded here: on a priority tick it runs only
    when the turn's intent is SKILL or TASK, so it belongs to "conditional
    tool execution" rather than to the always-on set.
    """
    return tuple(
        attribute
        for attribute in _KERNEL_PIPELINE_ATTRIBUTE_ORDER
        if _ATTRIBUTE_TO_PHASE_CLASS.get(attribute) not in BACKGROUND_ONLY_PHASES
    )


def pipeline_rate_report() -> dict[str, Any]:
    """The two-rate split, from the code that enforces it.

    Exists so a description of Aura's pipeline can be checked instead of
    recalled. "29 phases per turn" was written from the blueprint length and
    was wrong by roughly the number of phases that make a turn slow.
    """
    foreground = foreground_phase_attributes()
    background = background_only_phase_names()
    return {
        "blueprint_phases": len(_KERNEL_PIPELINE_ATTRIBUTE_ORDER),
        "foreground_phases": len(foreground),
        "background_only_phases": len(background),
        "foreground_order": list(foreground),
        "suppressed_on_priority_tick": list(background),
        "conditional_on_priority_tick": ["GodModeToolPhase"],
        "note": (
            "A priority tick runs the foreground set; the suppressed phases run "
            "on MindTick's kernel.tick(priority=False) and commit to shared "
            "state. One runtime, two rates — not 29 phases per turn."
        ),
    }


def resolve_phase_instances(target: Any, attribute_order: tuple[str, ...]) -> list[Any]:
    instances: list[Any] = []
    for attribute_name in attribute_order:
        phase = getattr(target, attribute_name, None)
        if phase is None:
            raise AttributeError(
                f"Expected phase attribute '{attribute_name}' to be initialized before pipeline resolution."
            )
        instances.append(phase)
    return instances
