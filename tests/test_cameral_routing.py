import pytest
from types import SimpleNamespace

from core.brain.cognitive_engine import CognitiveEngine
from core.config import get_config
from core.state.aura_state import AuraState


class AsyncCallRecorder:
    def __init__(self, result=None):
        self.result = result
        self.calls = []

    async def __call__(self, *args, **kwargs):
        self.calls.append(SimpleNamespace(args=args, kwargs=kwargs))
        return self.result


class InMemoryStateRepository:
    def __init__(self, state):
        self.state = state
        self.commits = []

    async def get_current(self):
        return self.state

    async def commit(self, state, cause=None):
        self.state = state
        self.commits.append(SimpleNamespace(state=state, cause=cause))

@pytest.fixture
def config():
    cfg = get_config()
    cfg.llm.deep_model = "gemini-2.5-pro"
    cfg.llm.gemini_api_key = "test" + "_key_" + "123"
    cfg.llm.fast_model = "qwen3:8b"
    return cfg

@pytest.fixture
def engine():
    from core.container import get_container
    container = get_container()
    container.reset()
    
    router = SimpleNamespace(
        think=AsyncCallRecorder(result="Router response."),
        last_tier="unknown",
    )
    container.register_instance("llm_router", router)
    
    state = AuraState.default()
    state.cognition.working_memory = []
    state.cognition.last_thought_at = 1710450000.0 # Some timestamp
    
    # Affect attributes need to be numbers for AffectUpdatePhase
    state.affect.valence = 0.0
    state.affect.arousal = 0.5
    state.affect.curiosity = 0.5
    state.affect.social_hunger = 0.5
    
    # CognitiveRoutingPhase sets cognition.current_mode
    state.cognition.current_mode = None
    
    repo = InMemoryStateRepository(state)
    container.register_instance("state_repository", repo)
    
    eng = CognitiveEngine()
    eng.setup() # Initialize phases
    
    # Store refs for easy assertion
    eng._test_router = router
    eng._test_state = state
    
    return eng
