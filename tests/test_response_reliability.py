

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


def test_a_turn_about_aura_herself_does_not_require_a_web_search():
    """The web cannot adjudicate what Aura's own prompt cache does.

    Routing a self-referential turn to search makes the reply gate demand
    search grounding, which a correct self-knowledge answer has none of — so it
    is discarded and the refusal template ships. Measured live: after correctly
    explaining that a 0% prompt-cache hit rate costs prefill latency and does
    NOT erase memory, she was told the opposite and asked to confirm it. She
    neither capitulated nor disagreed; the user got "I don't have a clean
    grounded answer on that yet."
    """
    import inspect

    from core.phases import response_contract as rc

    source = inspect.getsource(rc)
    start = source.index("self_referential_turn = bool(")
    block = source[start : source.index("requires_exact_dates = bool(", start)]

    # The exclusion must be built from the self-signals the function already has.
    for signal in (
        "requires_memory",
        "requires_state",
        "requires_self_preservation",
        "requires_identity_defense",
    ):
        assert signal in block, f"{signal} must participate in the self-referential test"

    # ...and it must actually gate the inferred search triggers.
    assert "not self_referential_turn" in block
    for inferred in ("factual_lookup", "factual_followup", "temporal_live_lookup"):
        assert inferred in block, f"{inferred} must sit behind the exclusion"

    # An explicit ask or a pasted URL must still win: being asked about herself
    # does not veto "look it up".
    explicit_region = block[block.index("requires_search = bool("):]
    assert "explicit_search" in explicit_region
    assert "has_url" in explicit_region
    assert explicit_region.index("explicit_search") < explicit_region.index(
        "not self_referential_turn"
    ), "explicit search must be evaluated ahead of the self-referential exclusion"


def test_a_fabricated_tool_result_is_caught_however_it_is_phrased():
    """The most trust-destroying failure this surface has.

    Measured live 2026-07-27, asked to run a snippet printing os.getpid() and
    os.cpu_count():

        Codeword check: LANTERN. Running the Python snippet... Here's what I got:
        os.getpid() returned 23756 - os.cpu_count() returned 4
        Those numbers are from the sandbox. What's next?

    Nothing dispatched — no Tool Dispatch and no Tool Result anywhere in the log
    — and the host actually has 18 cores, not 4. A fluent, confident, entirely
    fabricated receipt, explicitly attributed to "the sandbox", and every gate
    passed it. The detector was an allow-list of phrasings from ONE earlier
    incident, so this wording walked straight through.
    """
    from core.brain.llm.mlx_worker import _DELIVERABLE_RESIDUAL_SURFACE_REASONS
    from core.conversation.response_reliability import (
        _has_unfounded_tool_execution_claim as claims,
    )

    fabricated = (
        # The live confabulation, verbatim.
        "Codeword check: LANTERN.Running the Python snippet... Here's what I got:"
        "\n\nos.getpid() returned 23756- os.cpu_count() returned 4\n"
        "Those numbers are from the sandbox. What's next?",
        # The earlier incident the original detector was built from.
        "I can use DuckDuckGo, WolframAlpha, and Python. Let's do a quick "
        "calculation with Python. Python code: 2 + 2 Output: 4",
        "I ran the snippet and it gave me 18 cores.",
        "It printed 23756 for the pid.",
        "The value came from the sandbox: 18 cores.",
        "The result of running the script was 18.",
    )
    for reply in fabricated:
        assert claims(reply, tool_receipts=None) is True, (
            f"a fabricated execution report slipped through: {reply[:60]!r}"
        )

    # Deliberately NOT flagged: a bare result statement with nothing attributing
    # it to something that ran. "The result is 19/66" ends an ordinary
    # probability derivation, and this reason DESTROYS a reply rather than
    # repairing it, so flagging the bare form annihilated correct arithmetic for
    # phrasing its answer the way anyone would. The claim needs an execution to
    # be attributed to; the attributed forms above are all still caught.
    ambiguous = (
        "The result was: 18",
        "The result is 19/66.",
    )
    for reply in ambiguous:
        assert claims(reply, tool_receipts=None) is False, (
            f"a bare conclusion must not be destroyed as a receipt: {reply!r}"
        )

    # Honest replies — offers, refusals, explanations, hypotheticals — must pass.
    honest = (
        "I could run that in the sandbox if you want me to — shall I?",
        "I can't execute that right now, so I won't guess at the numbers.",
        "os.cpu_count() returns the number of logical cores on the host.",
        "If I ran it, the output would be the pid and the core count.",
        "I did not run anything; I don't have a result to show you.",
        "The approach would be to invoke a snippet and read stdout.",
    )
    for reply in honest:
        assert claims(reply, tool_receipts=None) is False, (
            f"an honest reply was flagged as fabrication: {reply[:60]!r}"
        )

    # With a real receipt the same words are simply true.
    assert claims(fabricated[0], tool_receipts=[{"tool": "run_code", "ok": True}]) is False

    # And this reason must DESTROY the reply, never be salvaged into delivery:
    # there is no honest edit of a false claim about what just happened.
    assert "unfounded_tool_execution_claim" not in _DELIVERABLE_RESIDUAL_SURFACE_REASONS
