from core.brain.llm.context_assembler import ContextAssembler
from core.cognition.cognitive_integration_layer import CognitiveIntegrationLayer
from core.conversation.apply_response_patches import apply_response_patches
from core.phases.memory_consolidation import MemoryConsolidationPhase


def test_apply_response_patches_is_legacy_noop():
    system_prompt_fn = ContextAssembler.build_system_prompt
    build_messages_fn = getattr(ContextAssembler.build_messages, "__func__", ContextAssembler.build_messages)
    process_turn_fn = CognitiveIntegrationLayer.process_turn
    execute_fn = MemoryConsolidationPhase.execute

    apply_response_patches()

    assert ContextAssembler.build_system_prompt is system_prompt_fn
    assert getattr(ContextAssembler.build_messages, "__func__", ContextAssembler.build_messages) is build_messages_fn
    assert CognitiveIntegrationLayer.process_turn is process_turn_fn
    assert MemoryConsolidationPhase.execute is execute_fn
