from core.container import ServiceContainer
from core.state.state_authority import (
    StateAuthority,
    TruthTier,
    get_state_authority,
    register_state_authority,
)
from core.values.prime_directives import PRIME_DIRECTIVES


def setup_function():
    ServiceContainer.clear()


def teardown_function():
    ServiceContainer.clear()


class KnowledgeSource:
    def query_knowledge(self, topic):
        if topic == "continuity":
            return "continuity is tracked by the state repository"
        return None

    def recall(self, topic):
        if topic == "fallback":
            return "recalled fallback fact"
        return None


class VectorSource:
    def retrieve_context(self, topic, top_k=1):
        if topic == "semantic":
            return [{"content": "semantic memory result"}]
        return []


def test_truth_prefers_prime_directive_over_runtime_context():
    """Runtime context cannot demote kin.

    This asserted the literal string "Bryan is kin." — which was not the
    constitution, it was the hardcoded stub the authority fell back to when
    `from core.values.prime_directives import PRIME_DIRECTIVES` raised
    ImportError, because that name had never existed. The test therefore
    passed only while the directive loader was broken, and would have failed
    the moment it was fixed. Assert the property instead: whatever the
    constitution says about Bryan is what comes back, at IMMUTABLE tier,
    regardless of what the caller's context claims.
    """
    authority = StateAuthority()

    truth, tier = authority.get_truth("bryan", context={"bryan": "ordinary user"})

    assert tier is TruthTier.IMMUTABLE
    assert "ordinary user" not in truth
    assert truth == PRIME_DIRECTIVES["bryan"]
    assert "Bryan" in truth


def test_truth_reads_registered_knowledge_source():
    ServiceContainer.register_instance("memory", KnowledgeSource())
    authority = StateAuthority()

    truth, tier = authority.get_truth("continuity")

    assert truth == "continuity is tracked by the state repository"
    assert tier is TruthTier.HARD_FACT


def test_truth_reads_registered_vector_source_after_context():
    ServiceContainer.register_instance("vector_memory", VectorSource())
    authority = StateAuthority()

    truth, tier = authority.get_truth("semantic")

    assert truth == "semantic memory result"
    assert tier is TruthTier.INFERENCE


def test_register_state_authority_is_idempotent():
    register_state_authority()
    first = get_state_authority()
    register_state_authority()
    second = get_state_authority()

    assert first is second
