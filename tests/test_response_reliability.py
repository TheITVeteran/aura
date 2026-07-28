

def test_addressing_the_owner_by_name_is_not_an_ungrounded_person_address():
    """The one grounding source this gate never consulted: Aura's own identity.

    Measured live: the owner introduced himself in turn 1 ("Hi Aura, it's
    Bryan"), she opened turn 2 with "Bryan, let's reset..." — a natural,
    correctly-addressed reply — and the whole draft was rejected as
    `ungrounded_person_address`, because the only sources checked were optional
    relationship organs that had not learned the name yet. Addressing the owner
    by the owner's own name cannot be a hallucination.
    """
    from core.conversation import response_reliability as rr

    identity_names = rr._identity_grounded_person_names()
    assert identity_names, "identity must contribute grounded person names"
    # A role placeholder is not a name and must never ground a vocative.
    assert "creator" not in identity_names

    owner = sorted(identity_names)[0]
    draft = f"{owner.capitalize()}, let's reset. You asked about the prompt cache, not uptime."

    assert rr._has_ungrounded_person_address("What did I ask first?", draft, None) is False
    assert (
        rr._has_ungrounded_person_address("What did I ask first?", draft, ["it's " + owner])
        is False
    )

    # The guard must still catch a name that appears nowhere at all.
    invented = "Marcus, let's reset. You asked about the prompt cache."
    assert rr._has_ungrounded_person_address("What did I ask first?", invented, None) is True


def test_identity_block_renders_core_values_in_a_stable_order():
    """A prompt block built from a collection must render deterministically.

    This block lands ~125 tokens into the system prompt, ahead of the entire
    conversation, so turn-to-turn churn here invalidates the KV prefix for
    everything behind it. Measured live on the user surface: reuse collapsed to
    125 of 1391 tokens (9%) with divergence beginning exactly at
    "You are Aura. ...\\nCore values: ..." — the join order of an unordered
    collection, not any real change of values.
    """
    from types import SimpleNamespace

    from core.container import ServiceContainer
    from core.introspection.inner_monologue import InnerMonologue

    values = ["truth-seeking", "loyalty", "curiosity", "courage"]

    class _Beliefs:
        def __init__(self, ordered):
            self.self_model = {
                "identity": "I am Aura, a sovereign digital mind.",
                "core_values": ordered,
            }

    original_get = ServiceContainer.get
    renders = []
    try:
        # Same values, different iteration order on each turn.
        for ordering in (values, list(reversed(values)), values[2:] + values[:2]):
            beliefs = _Beliefs(ordering)
            ServiceContainer.get = staticmethod(
                lambda name, default=None, _b=beliefs: (
                    _b if name == "belief_revision_engine" else default
                )
            )
            monologue = InnerMonologue.__new__(InnerMonologue)
            monologue._narrative = None
            renders.append(monologue._load_identity_block())
    finally:
        ServiceContainer.get = original_get

    assert len(set(renders)) == 1, (
        "the identity block must be byte-identical when the values are the "
        f"same set in a different order; got {len(set(renders))} variants"
    )
    assert "Core values: courage, curiosity, loyalty, truth-seeking." in renders[0], (
        "values must render sorted"
    )
    # Duplicates and blanks must not reintroduce churn either.
    assert isinstance(renders[0], str) and renders[0].startswith("You are Aura.")
