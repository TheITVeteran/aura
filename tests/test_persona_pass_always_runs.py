"""Her voice was silenced by a boot race, 2,270 times.

The chat route shapes every reply through the personality engine, looked up in
the ServiceContainer. That entry is written during boot_identity — and the
route can serve a turn before that lands, in which case the reply shipped the
base model's register with a warning nobody reads:

    [DEGRADATION] chat (warning): RuntimeError: personality_engine absent;
    reply shaped by nothing

2,270 of those in one log. The last arrived 79 seconds before a restart after
which there were none, which is the shape of a race rather than of a missing
organ.

The engine is a module singleton and never needed the container to exist.
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def chat_module():
    from interface.routes import chat

    return chat


def test_the_persona_pass_runs_with_an_empty_container(chat_module, monkeypatch):
    from core.container import ServiceContainer

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: default),
    )
    registered: dict[str, object] = {}
    monkeypatch.setattr(
        ServiceContainer,
        "register_instance",
        staticmethod(lambda name, value: registered.__setitem__(name, value)),
    )

    shaped = chat_module._apply_aura_voice_shaping("I am here.", "hi")
    assert isinstance(shaped, str) and shaped.strip()
    assert "personality_engine" in registered, (
        "the singleton must be adopted so the next turn takes the fast path"
    )


def test_it_reaches_for_the_singleton_rather_than_warning(chat_module):
    """The old behaviour recorded a degradation and served the draft unshaped.
    Warning about a race is not handling it."""
    import inspect

    source = inspect.getsource(chat_module._apply_aura_voice_shaping)
    fallback = source.index("get_personality_engine")
    warning = source.index("personality_engine absent")
    assert fallback < warning, (
        "the singleton fallback must be tried before the absence is recorded"
    )


def test_the_degradation_still_exists_for_a_real_absence(chat_module):
    """If even the singleton cannot be built, that is worth saying."""
    import inspect

    source = inspect.getsource(chat_module._apply_aura_voice_shaping)
    assert "personality_engine absent; reply shaped by nothing" in source
