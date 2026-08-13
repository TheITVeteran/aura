"""core/affect/emotion_engine.py — retired shim over a second affect engine.

Aura had two affect engines. One of them ran.

    core/affect/damasio_v2.py::AffectEngineV2   the canonical engine. Registered
                                                as the `affect_engine` service
                                                (required=True), constructed by
                                                consciousness_provider and by
                                                boot_cognitive, read by the
                                                metabolic coordinator, vector
                                                memory, self_object, runtime
                                                tools, spiking active inference.

    core/affect/__init__.py::AffectEngine       a PAD engine documented as the
                                                "lightweight fallback when
                                                DamasioV2 is unavailable". Its
                                                only constructor in the entire
                                                tree was this module, and this
                                                module had ZERO importers. The
                                                fallback had no path to it and
                                                could never have run.

So the second engine was not a redundancy, it was an unreachable one — and
worse, it was scored as covered: two tests in tests/test_e2e_pipeline.py
exercised its decay, which is coverage spent on an engine with no construction
path. They now exercise AffectEngineV2's decay instead.

Wiring the fallback up was the other option and it is the wrong one, by this
codebase's own rule. CP126 established that "no engine must never look like a
calm engine": when affect is unavailable the facade must SAY unavailable rather
than emit plausible neutral numbers. A silent fallback to a different affect
model does exactly what that rule forbids — it would answer with confident PAD
values from a model nothing else in the system reads, and no caller could tell
which engine produced them. Unavailability is the honest output; a second
opinion presented as the first is not.

This module also constructed its engine at import time
(``emotion_engine = EmotionEngine()`` at module scope), so importing it for any
reason would have instantiated the shadow engine as a side effect.

Retirement follows the pattern core/global_workspace.py used over the duplicate
workspace: keep the module importable, re-export the canonical names, let one
implementation exist. tests/test_one_canonical_affect_engine.py pins it.
"""

from __future__ import annotations

from typing import Any

from core.affect import AffectState
from core.affect.damasio_v2 import AffectEngineV2

__all__ = ["AffectEngineV2", "AffectState", "get_emotion_engine"]


def get_emotion_engine() -> Any:
    """The one affect engine, from the container rather than a fresh instance.

    Constructing an engine here is what created the second one. Callers that
    want affect want the registered singleton the rest of the runtime reads,
    and if it is not registered they want to know that rather than be handed a
    private engine whose state nothing else shares.
    """
    from core.container import ServiceContainer

    return ServiceContainer.get("affect_engine", default=None)
