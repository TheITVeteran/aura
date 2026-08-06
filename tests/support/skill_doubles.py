"""Test skills that satisfy the runtime's actual contract.

`CapabilityEngine._prepare_skill_instance` refuses an implementation that is not
a `BaseSkill` subclass, whose `name` disagrees with the catalog, or which has no
`execute` / `safe_execute`. That check is correct and load-bearing: a skill that
does not satisfy the contract cannot be governed, timed out or receipted, and
letting one through would put an ungoverned callable on the tool path.

Several tests predate the check and build toy classes —
``class _Skill: timeout_seconds = 57.0`` — then assert the engine returns
``ok: True``. They cannot pass without the engine abandoning a contract it
should keep, so the fixture is what has to change.

`make_test_skill` builds a real subclass, so those tests exercise the real
preflight path instead of a hole in it.
"""

from __future__ import annotations

from typing import Any

from core.skills.base_skill import BaseSkill


def make_test_skill(
    name: str,
    *,
    result: dict[str, Any] | None = None,
    timeout_seconds: float = 30.0,
    description: str = "test double",
    **attributes: Any,
) -> type[BaseSkill]:
    """A real BaseSkill subclass named ``name``.

    The name must match the catalog entry — the engine compares them, because a
    class claiming to be one skill while registered as another is exactly the
    substitution the source-identity checks exist to catch.
    """
    payload = dict(result or {"ok": True})

    # `execute` is defined in the class BODY, not assigned afterwards. ABCMeta
    # computes __abstractmethods__ at class creation, so a later assignment
    # leaves the class abstract and uninstantiable while looking complete.
    class _TestSkill(BaseSkill):
        async def execute(self, params: Any, context: dict[str, Any]) -> dict[str, Any]:
            return dict(payload)

    _TestSkill.name = name
    _TestSkill.description = description
    _TestSkill.timeout_seconds = timeout_seconds
    for key, value in attributes.items():
        setattr(_TestSkill, key, value)
    _TestSkill.__name__ = f"TestSkill_{name}"
    _TestSkill.__qualname__ = _TestSkill.__name__
    return _TestSkill


__all__ = ["make_test_skill"]
