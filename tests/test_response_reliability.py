

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
