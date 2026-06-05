# Aura Subsystem and Service Ownership Manifest

This file outlines every registered service, its source code location, registration origin, failure policy, and operational requirements.

| Service | Owner File | Registered By | Required For | Failure Policy |
|---|---|---|---|---|
| `absorbed_voices` | `core/consciousness/system.py` | `core/consciousness/system.py` | boot | `fail-closed` |
| `actor_bus` | `aura_main.py` | `aura_main.py` | boot | `fail-closed` |
| `adaptive_mood` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `aesthetic_engine` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | optional features | `degrade_with_receipt` |
| `affect_coordinator` | `core/orchestrator/boot.py` | `core/orchestrator/boot.py` | boot | `fail-closed` |
| `affect_engine` | `core/orchestrator/mixins/boot/boot_cognitive.py` | `core/orchestrator/mixins/boot/boot_cognitive.py` | boot | `fail-closed` |
| `affect_facade` | `core/orchestrator/boot.py` | `core/orchestrator/boot.py` | boot | `fail-closed` |
| `affect_manager` | `core/orchestrator/mixins/boot/boot_cognitive.py` | `core/orchestrator/mixins/boot/boot_cognitive.py` | boot | `fail-closed` |
| `affective_steering` | `core/consciousness/system.py` | `core/consciousness/system.py` | boot | `fail-closed` |
| `affective_steering_engine` | `core/consciousness/affective_steering.py` | `core/consciousness/affective_steering.py` | optional features | `degrade_with_receipt` |
| `agency_coordinator` | `core/orchestrator/boot.py` | `core/orchestrator/boot.py` | boot | `fail-closed` |
| `agency_core` | `core/orchestrator/boot.py` | `core/orchestrator/boot.py` | boot | `fail-closed` |
| `agency_facade` | `core/orchestrator/boot.py` | `core/orchestrator/boot.py` | boot | `fail-closed` |
| `agent_delegator` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `agent_workspace` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `alignment` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | optional features | `degrade_with_receipt` |
| `alignment_engine` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | boot | `fail-closed` |
| `anomaly_detector` | `core/cybernetics/ice_layer.py` | `core/cybernetics/ice_layer.py` | boot | `fail-closed` |
| `api_adapter` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `architecture_governor` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `attention_schema` | `core/consciousness/system.py` | `core/consciousness/system.py` | boot | `fail-closed` |
| `audit` | `core/orchestrator/initializers/core_baseline.py` | `core/orchestrator/initializers/core_baseline.py` | boot | `fail-closed` |
| `aura_kernel` | `core/kernel/kernel_interface.py` | `core/kernel/kernel_interface.py` | boot | `fail-closed` |
| `aura_now` | `core/being/runtime.py` | `core/being/runtime.py` | optional features | `degrade_with_receipt` |
| `aura_now_runtime` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `aura_runtime` | `aura_main.py` | `aura_main.py` | optional features | `degrade_with_receipt` |
| `aura_workspace` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `authority_gateway` | `core/executive/authority_gateway.py` | `core/executive/authority_gateway.py` | optional features | `degrade_with_receipt` |
| `autonomous_architecture_governor` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `autonomous_brain` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | optional features | `degrade_with_receipt` |
| `backup_manager` | `core/orchestrator/initializers/core_baseline.py` | `core/orchestrator/initializers/core_baseline.py` | boot | `fail-closed` |
| `backup_system` | `core/safety/self_preservation_safe.py` | `core/safety/self_preservation_safe.py` | boot | `fail-closed` |
| `being_runtime` | `core/being/runtime.py` | `core/being/runtime.py` | optional features | `degrade_with_receipt` |
| `belief_authority` | `core/constitution.py` | `core/constitution.py` | optional features | `degrade_with_receipt` |
| `belief_challenger` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `branch_manager` | `core/consciousness/parallel_branches.py` | `core/consciousness/parallel_branches.py` | boot | `fail-closed` |
| `bryan_model` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `canonical_self` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `canonical_self_engine` | `core/service_registration.py` | `core/service_registration.py` | boot | `fail-closed` |
| `capability_engine` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | boot | `fail-closed` |
| `cellular_turnover` | `core/consciousness/system.py` | `core/consciousness/system.py` | boot | `fail-closed` |
| `closed_causal_loop` | `core/consciousness/closed_loop.py` | `core/consciousness/closed_loop.py` | boot | `fail-closed` |
| `code_refiner` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `cognition` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | optional features | `degrade_with_receipt` |
| `cognitive_engine` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `cognitive_integration` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `cognitive_kernel` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `cognitive_ledger` | `core/kernel/aura_kernel.py` | `core/kernel/aura_kernel.py` | boot | `fail-closed` |
| `cognitive_manager` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `cognitive_router` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | boot | `fail-closed` |
| `composer_node` | `core/orchestrator/mixins/boot/boot_cognitive.py` | `core/orchestrator/mixins/boot/boot_cognitive.py` | boot | `fail-closed` |
| `concept_linker` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `conscious_substrate` | `core/consciousness/system.py` | `core/consciousness/system.py` | boot | `fail-closed` |
| `consciousness` | `core/orchestrator/mixins/boot/boot_cognitive.py` | `core/orchestrator/mixins/boot/boot_cognitive.py` | boot | `fail-closed` |
| `consciousness_bridge` | `core/consciousness/system.py` | `core/consciousness/system.py` | boot | `fail-closed` |
| `consciousness_core` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | boot | `fail-closed` |
| `consciousness_evidence` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | optional features | `degrade_with_receipt` |
| `consciousness_integration` | `core/orchestrator/mixins/boot/boot_cognitive.py` | `core/orchestrator/mixins/boot/boot_cognitive.py` | boot | `fail-closed` |
| `consciousness_system` | `core/orchestrator/mixins/boot/boot_cognitive.py` | `core/orchestrator/mixins/boot/boot_cognitive.py` | boot | `fail-closed` |
| `constitutional_core` | `core/constitution.py` | `core/constitution.py` | optional features | `degrade_with_receipt` |
| `context_manager` | `core/orchestrator/mixins/boot/boot_cognitive.py` | `core/orchestrator/mixins/boot/boot_cognitive.py` | boot | `fail-closed` |
| `continuity` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `continuous_experience_stream` | `core/consciousness/continuous_experience.py` | `core/consciousness/continuous_experience.py` | optional features | `degrade_with_receipt` |
| `continuous_learner` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | optional features | `degrade_with_receipt` |
| `conversation_reflector` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `counterfactual_engine` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | optional features | `degrade_with_receipt` |
| `credit_assignment` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | optional features | `degrade_with_receipt` |
| `critic_engine` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `curiosity_engine` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | boot | `fail-closed` |
| `database_coordinator` | `core/orchestrator/boot.py` | `core/orchestrator/boot.py` | boot | `fail-closed` |
| `deliberator` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `dlq` | `core/orchestrator/initializers/core_baseline.py` | `core/orchestrator/initializers/core_baseline.py` | boot | `fail-closed` |
| `dreamer_v2` | `core/providers/memory_provider.py` | `core/providers/memory_provider.py` | optional features | `degrade_with_receipt` |
| `drift_monitor` | `core/orchestrator/mixins/boot/boot_identity.py` | `core/orchestrator/mixins/boot/boot_identity.py` | boot | `fail-closed` |
| `drive_engine` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | boot | `fail-closed` |
| `dynamic_router` | `core/service_registration.py` | `core/service_registration.py` | boot | `fail-closed` |
| `embodied_interoception` | `core/consciousness/consciousness_bridge.py` | `core/consciousness/consciousness_bridge.py` | boot | `fail-closed` |
| `emergent_goal_engine` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `emotional_coloring` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `episodic_memory` | `core/orchestrator/mixins/boot/boot_resilience.py` | `core/orchestrator/mixins/boot/boot_resilience.py` | boot | `fail-closed` |
| `epistemic_state` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | optional features | `degrade_with_receipt` |
| `epistemic_tracker` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `event_bus` | `core/orchestrator/initializers/core_baseline.py` | `core/orchestrator/initializers/core_baseline.py` | boot | `fail-closed` |
| `evidence_mode` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `executive_authority` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | optional features | `degrade_with_receipt` |
| `executive_closure` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | optional features | `degrade_with_receipt` |
| `executive_core` | `core/executive/executive_core.py` | `core/executive/executive_core.py` | optional features | `degrade_with_receipt` |
| `feedback_processor` | `core/somatic/action_feedback.py` | `core/somatic/action_feedback.py` | optional features | `degrade_with_receipt` |
| `free_energy_engine` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | boot | `fail-closed` |
| `global_workspace` | `core/orchestrator/mixins/boot/boot_cognitive.py` | `core/orchestrator/mixins/boot/boot_cognitive.py` | boot | `fail-closed` |
| `goal_belief_manager` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `goal_engine` | `core/service_registration.py` | `core/service_registration.py` | boot | `fail-closed` |
| `goal_hierarchy` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `goal_manager` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `goal_memory` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `growth_ladder` | `core/orchestrator/mixins/boot/boot_identity.py` | `core/orchestrator/mixins/boot/boot_identity.py` | boot | `fail-closed` |
| `health_monitor` | `core/providers/ops_provider.py` | `core/providers/ops_provider.py` | boot | `fail-closed` |
| `hearing` | `core/providers/sensory_provider.py` | `core/providers/sensory_provider.py` | optional features | `degrade_with_receipt` |
| `hemispheric_split` | `core/consciousness/system.py` | `core/consciousness/system.py` | boot | `fail-closed` |
| `hephaestus_engine` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | boot | `fail-closed` |
| `hierarchical_phi` | `core/consciousness/system.py` | `core/consciousness/system.py` | boot | `fail-closed` |
| `homeostasis` | `core/orchestrator/mixins/boot/boot_cognitive.py` | `core/orchestrator/mixins/boot/boot_cognitive.py` | boot | `fail-closed` |
| `homeostatic_coupling` | `core/consciousness/system.py` | `core/consciousness/system.py` | boot | `fail-closed` |
| `identity` | `core/orchestrator/boot.py` | `core/orchestrator/boot.py` | boot | `fail-closed` |
| `identity_anchor` | `core/service_registration.py` | `core/service_registration.py` | boot | `fail-closed` |
| `identity_chronicle` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `identity_service` | `core/orchestrator/boot.py` | `core/orchestrator/boot.py` | boot | `fail-closed` |
| `inference_gate` | `core/orchestrator/boot.py` | `core/orchestrator/boot.py` | boot | `fail-closed` |
| `inhibition_manager` | `core/ops/resilient_boot.py` | `core/ops/resilient_boot.py` | boot | `fail-closed` |
| `initiative_arbiter` | `core/service_registration.py` | `core/service_registration.py` | boot | `fail-closed` |
| `inner_monologue` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `inquiry_engine` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `insight_journal` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `integrity_guard` | `core/orchestrator/mixins/boot/boot_resilience.py` | `core/orchestrator/mixins/boot/boot_resilience.py` | boot | `fail-closed` |
| `intent_router` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | boot | `fail-closed` |
| `interaction_signals` | `core/providers/sensory_provider.py` | `core/providers/sensory_provider.py` | optional features | `degrade_with_receipt` |
| `internal_simulator` | `core/service_registration.py` | `core/service_registration.py` | boot | `fail-closed` |
| `joy_social` | `skills/joy_social_integration.py` | `skills/joy_social_integration.py` | optional features | `degrade_with_receipt` |
| `keep_awake_controller` | `core/runtime/keep_awake.py` | `core/runtime/keep_awake.py` | optional features | `degrade_with_receipt` |
| `kernel_interface` | `core/kernel/kernel_interface.py` | `core/kernel/kernel_interface.py` | boot | `fail-closed` |
| `knowledge_graph` | `core/providers/memory_provider.py` | `core/providers/memory_provider.py` | optional features | `degrade_with_receipt` |
| `language_center` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `life_trace` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `lineage_manager` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `liquid_neural_network` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | optional features | `degrade_with_receipt` |
| `liquid_state` | `core/consciousness/system.py` | `core/consciousness/system.py` | boot | `fail-closed` |
| `liquid_substrate` | `core/orchestrator/mixins/boot/boot_resilience.py` | `core/orchestrator/mixins/boot/boot_resilience.py` | boot | `fail-closed` |
| `llm_interface` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `llm_router` | `core/orchestrator/boot.py` | `core/orchestrator/boot.py` | boot | `fail-closed` |
| `loop_monitor` | `core/service_registration.py` | `core/service_registration.py` | boot | `fail-closed` |
| `markdown_workspace` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `memory` | `core/providers/memory_provider.py` | `core/providers/memory_provider.py` | boot | `fail-closed` |
| `memory_coordinator` | `core/orchestrator/boot.py` | `core/orchestrator/boot.py` | boot | `fail-closed` |
| `memory_facade` | `core/orchestrator/boot.py` | `core/orchestrator/boot.py` | boot | `fail-closed` |
| `memory_guard` | `core/orchestrator/mixins/boot/boot_resilience.py` | `core/orchestrator/mixins/boot/boot_resilience.py` | boot | `fail-closed` |
| `memory_manager` | `core/providers/memory_provider.py` | `core/providers/memory_provider.py` | boot | `fail-closed` |
| `memory_monitor` | `aura_main.py` | `aura_main.py` | optional features | `degrade_with_receipt` |
| `memory_subsystem` | `core/providers/memory_provider.py` | `core/providers/memory_provider.py` | optional features | `degrade_with_receipt` |
| `memory_synthesizer` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `memory_vector` | `core/providers/memory_provider.py` | `core/providers/memory_provider.py` | optional features | `degrade_with_receipt` |
| `mesh_cognition` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `meta_cognition_loop` | `core/service_registration.py` | `core/service_registration.py` | boot | `fail-closed` |
| `meta_cognition_shard` | `core/orchestrator/mixins/boot/boot_background.py` | `core/orchestrator/mixins/boot/boot_background.py` | boot | `fail-closed` |
| `meta_evolution` | `core/orchestrator/mixins/boot/boot_background.py` | `core/orchestrator/mixins/boot/boot_background.py` | boot | `fail-closed` |
| `metabolic_coordinator` | `core/providers/ops_provider.py` | `core/providers/ops_provider.py` | boot | `fail-closed` |
| `metabolic_monitor` | `core/providers/ops_provider.py` | `core/providers/ops_provider.py` | boot | `fail-closed` |
| `metabolism` | `core/providers/ops_provider.py` | `core/providers/ops_provider.py` | optional features | `degrade_with_receipt` |
| `metabolism_state` | `core/providers/ops_provider.py` | `core/providers/ops_provider.py` | optional features | `degrade_with_receipt` |
| `metacognition` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | boot | `fail-closed` |
| `metrics` | `core/orchestrator/initializers/core_baseline.py` | `core/orchestrator/initializers/core_baseline.py` | boot | `fail-closed` |
| `mind_model` | `core/orchestrator/mixins/boot/boot_cognitive.py` | `core/orchestrator/mixins/boot/boot_cognitive.py` | boot | `fail-closed` |
| `minimal_selfhood` | `core/consciousness/system.py` | `core/consciousness/system.py` | boot | `fail-closed` |
| `morphogenetic_runtime` | `core/morphogenesis/integration.py` | `core/morphogenesis/integration.py` | optional features | `degrade_with_receipt` |
| `motivation_engine` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | boot | `fail-closed` |
| `motor_cortex` | `core/somatic/motor_cortex.py` | `core/somatic/motor_cortex.py` | optional features | `degrade_with_receipt` |
| `multimodal_orchestrator` | `core/orchestrator/mixins/boot/boot_sensory.py` | `core/orchestrator/mixins/boot/boot_sensory.py` | boot | `fail-closed` |
| `mycelial_network` | `core/service_registration.py` | `core/service_registration.py` | boot | `fail-closed` |
| `mycelium` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `narrative_thread` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `narrator` | `core/brain/narrator.py` | `core/brain/narrator.py` | boot | `fail-closed` |
| `native_system2` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `nethack_adapter` | `aura_main.py` | `aura_main.py` | optional features | `degrade_with_receipt` |
| `neural_intent_router` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `neural_mesh` | `core/consciousness/consciousness_bridge.py` | `core/consciousness/consciousness_bridge.py` | boot | `fail-closed` |
| `neurochemical_system` | `core/consciousness/consciousness_bridge.py` | `core/consciousness/consciousness_bridge.py` | boot | `fail-closed` |
| `nucleus` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | optional features | `degrade_with_receipt` |
| `octopus_federation` | `core/consciousness/system.py` | `core/consciousness/system.py` | boot | `fail-closed` |
| `orchestrator` | `core/orchestrator/boot.py` | `core/orchestrator/boot.py` | boot | `fail-closed` |
| `oscillatory_binding` | `core/consciousness/consciousness_bridge.py` | `core/consciousness/consciousness_bridge.py` | boot | `fail-closed` |
| `output_gate` | `core/orchestrator/boot.py` | `core/orchestrator/boot.py` | boot | `fail-closed` |
| `paraconsistent_engine` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | optional features | `degrade_with_receipt` |
| `permission_setup` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `persistence` | `core/orchestrator/initializers/core_baseline.py` | `core/orchestrator/initializers/core_baseline.py` | boot | `fail-closed` |
| `persistent_state` | `core/providers/memory_provider.py` | `core/providers/memory_provider.py` | optional features | `degrade_with_receipt` |
| `personality` | `core/orchestrator/mixins/boot/boot_identity.py` | `core/orchestrator/mixins/boot/boot_identity.py` | boot | `fail-closed` |
| `personality_bridge` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | optional features | `degrade_with_receipt` |
| `personality_engine` | `core/orchestrator/mixins/boot/boot_identity.py` | `core/orchestrator/mixins/boot/boot_identity.py` | boot | `fail-closed` |
| `phenomenological_experiencer` | `core/consciousness/integration.py` | `core/consciousness/integration.py` | boot | `fail-closed` |
| `phi_core` | `core/consciousness/system.py` | `core/consciousness/system.py` | boot | `fail-closed` |
| `plasticity_controller` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `platform_root` | `core/service_registration.py` | `core/service_registration.py` | boot | `fail-closed` |
| `pre_linguistic` | `core/cognition/pre_linguistic.py` | `core/cognition/pre_linguistic.py` | optional features | `degrade_with_receipt` |
| `precognitive_engine` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | optional features | `degrade_with_receipt` |
| `predictive_engine` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | optional features | `degrade_with_receipt` |
| `prompt_compiler` | `core/brain/llm/compiler.py` | `core/brain/llm/compiler.py` | boot | `fail-closed` |
| `qualia_engine` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | boot | `fail-closed` |
| `qualia_synthesizer` | `core/orchestrator/mixins/boot/boot_cognitive.py` | `core/orchestrator/mixins/boot/boot_cognitive.py` | boot | `fail-closed` |
| `react_loop` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | optional features | `degrade_with_receipt` |
| `recursive_tom` | `core/consciousness/system.py` | `core/consciousness/system.py` | boot | `fail-closed` |
| `reimplementation_lab` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `resilience` | `core/providers/ops_provider.py` | `core/providers/ops_provider.py` | optional features | `degrade_with_receipt` |
| `resilience_engine` | `core/orchestrator/mixins/boot/boot_resilience.py` | `core/orchestrator/mixins/boot/boot_resilience.py` | boot | `fail-closed` |
| `resource_stakes` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `runtime_hygiene` | `core/orchestrator/boot.py` | `core/orchestrator/boot.py` | boot | `fail-closed` |
| `scheduler` | `core/scheduler.py` | `core/scheduler.py` | optional features | `degrade_with_receipt` |
| `scratchpad_engine` | `core/orchestrator/mixins/boot/boot_cognitive.py` | `core/orchestrator/mixins/boot/boot_cognitive.py` | boot | `fail-closed` |
| `self_awareness_suite` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `self_model` | `core/orchestrator/boot.py` | `core/orchestrator/boot.py` | boot | `fail-closed` |
| `self_modification_engine` | `core/providers/ops_provider.py` | `core/providers/ops_provider.py` | boot | `fail-closed` |
| `self_prediction` | `core/orchestrator/mixins/boot/boot_cognitive.py` | `core/orchestrator/mixins/boot/boot_cognitive.py` | boot | `fail-closed` |
| `self_report_engine` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | boot | `fail-closed` |
| `semantic_memory` | `core/providers/memory_provider.py` | `core/providers/memory_provider.py` | optional features | `degrade_with_receipt` |
| `sentience_engine` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | optional features | `degrade_with_receipt` |
| `shutdown_coordinator` | `aura_main.py` | `aura_main.py` | optional features | `degrade_with_receipt` |
| `simulation_well` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `singularity_monitor` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | boot | `fail-closed` |
| `skill_evolution` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `skill_manager` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | boot | `fail-closed` |
| `skill_registry` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | optional features | `degrade_with_receipt` |
| `skill_router` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | boot | `fail-closed` |
| `sme` | `core/providers/ops_provider.py` | `core/providers/ops_provider.py` | optional features | `degrade_with_receipt` |
| `soma` | `core/orchestrator/mixins/boot/boot_resilience.py` | `core/orchestrator/mixins/boot/boot_resilience.py` | boot | `fail-closed` |
| `soma_subsystem` | `core/providers/sensory_provider.py` | `core/providers/sensory_provider.py` | optional features | `degrade_with_receipt` |
| `somatic_marker_gate` | `core/consciousness/consciousness_bridge.py` | `core/consciousness/consciousness_bridge.py` | boot | `fail-closed` |
| `sovereign_ears` | `core/ops/resilient_boot.py` | `core/ops/resilient_boot.py` | boot | `fail-closed` |
| `sovereign_pruner` | `core/orchestrator/mixins/boot/boot_resilience.py` | `core/orchestrator/mixins/boot/boot_resilience.py` | boot | `fail-closed` |
| `sovereign_scanner` | `core/orchestrator/mixins/boot/boot_resilience.py` | `core/orchestrator/mixins/boot/boot_resilience.py` | boot | `fail-closed` |
| `sovereign_watchdog` | `core/orchestrator/mixins/boot/boot_resilience.py` | `core/orchestrator/mixins/boot/boot_resilience.py` | boot | `fail-closed` |
| `spine` | `core/orchestrator/mixins/boot/boot_identity.py` | `core/orchestrator/mixins/boot/boot_identity.py` | boot | `fail-closed` |
| `stability_guardian` | `core/orchestrator/mixins/boot/boot_resilience.py` | `core/orchestrator/mixins/boot/boot_resilience.py` | boot | `fail-closed` |
| `state_machine` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | boot | `fail-closed` |
| `state_repo` | `core/orchestrator/mixins/boot/boot_resilience.py` | `core/orchestrator/mixins/boot/boot_resilience.py` | boot | `fail-closed` |
| `state_repository` | `core/orchestrator/mixins/boot/boot_resilience.py` | `core/orchestrator/mixins/boot/boot_resilience.py` | boot | `fail-closed` |
| `stream_of_being` | `core/consciousness/stream_of_being.py` | `core/consciousness/stream_of_being.py` | boot | `fail-closed` |
| `structural_mutator` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `substrate_authority` | `core/consciousness/system.py` | `core/consciousness/system.py` | boot | `fail-closed` |
| `substrate_evolution` | `core/consciousness/consciousness_bridge.py` | `core/consciousness/consciousness_bridge.py` | boot | `fail-closed` |
| `substrate_voice_engine` | `core/voice/substrate_voice_engine.py` | `core/voice/substrate_voice_engine.py` | boot | `fail-closed` |
| `subsystem_audit` | `core/orchestrator/mixins/boot/boot_cognitive.py` | `core/orchestrator/mixins/boot/boot_cognitive.py` | boot | `fail-closed` |
| `supervisor` | `aura_main.py` | `aura_main.py` | boot | `fail-closed` |
| `system2_search` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | optional features | `degrade_with_receipt` |
| `system_governor` | `core/orchestrator/mixins/boot/boot_resilience.py` | `core/orchestrator/mixins/boot/boot_resilience.py` | boot | `fail-closed` |
| `system_monitor` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `task_supervisor` | `aura_main.py` | `aura_main.py` | optional features | `degrade_with_receipt` |
| `task_tracker` | `aura_main.py` | `aura_main.py` | optional features | `degrade_with_receipt` |
| `temporal_atlas_factory` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `temporal_binding` | `core/orchestrator/mixins/boot/boot_cognitive.py` | `core/orchestrator/mixins/boot/boot_cognitive.py` | boot | `fail-closed` |
| `tension_engine` | `core/service_registration.py` | `core/service_registration.py` | boot | `fail-closed` |
| `theory_arbitration` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | optional features | `degrade_with_receipt` |
| `theory_of_mind` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | optional features | `degrade_with_receipt` |
| `time_dilation` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | optional features | `degrade_with_receipt` |
| `tool_orchestrator` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `tts_stream` | `core/providers/sensory_provider.py` | `core/providers/sensory_provider.py` | optional features | `degrade_with_receipt` |
| `unified_field` | `core/consciousness/consciousness_bridge.py` | `core/consciousness/consciousness_bridge.py` | boot | `fail-closed` |
| `unified_will` | `core/governance/will.py` | `core/governance/will.py` | optional features | `degrade_with_receipt` |
| `unity_runtime` | `core/unity/runtime.py` | `core/unity/runtime.py` | optional features | `degrade_with_receipt` |
| `unity_workspace_frame` | `core/unity/runtime.py` | `core/unity/runtime.py` | optional features | `degrade_with_receipt` |
| `vector_memory` | `core/providers/memory_provider.py` | `core/providers/memory_provider.py` | optional features | `degrade_with_receipt` |
| `vector_memory_engine` | `core/providers/memory_provider.py` | `core/providers/memory_provider.py` | optional features | `degrade_with_receipt` |
| `vision` | `core/providers/sensory_provider.py` | `core/providers/sensory_provider.py` | optional features | `degrade_with_receipt` |
| `voice_engine` | `core/orchestrator/mixins/boot/boot_sensory.py` | `core/orchestrator/mixins/boot/boot_sensory.py` | boot | `fail-closed` |
