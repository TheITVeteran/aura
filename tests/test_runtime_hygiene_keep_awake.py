"""Aura's own keep-awake assertion is not a rogue child process.

`core.runtime.keep_awake` spawns `caffeinate` through the subprocess gateway
and owns its lifecycle. Because it is named for the macOS binary rather than
for Aura, it matched none of the worker name tags in the hygiene summary and
was reported as "1 unregistered child process(es) detected" on EVERY boot — a
permanent DEGRADED card in the user's neural feed for a process the runtime
deliberately started.
"""
from __future__ import annotations


class _FakeProc:
    def __init__(self, name: str, cmdline: list[str]):
        self._name = name
        self._cmdline = list(cmdline)
        self.info = {"name": name, "cmdline": list(cmdline)}

    def name(self) -> str:
        return self._name

    def cmdline(self) -> list[str]:
        return list(self._cmdline)


def test_auras_own_sleep_assertion_is_recognised():
    from core.runtime.runtime_hygiene import _is_keep_awake_assertion_process

    assert _is_keep_awake_assertion_process(
        _FakeProc("caffeinate", ["caffeinate", "-i", "-m", "-s"])
    ) is True


def test_an_unrelated_caffeinate_is_not_adopted():
    from core.runtime.runtime_hygiene import _is_keep_awake_assertion_process

    # A timer-style caffeinate a person started themselves carries none of the
    # assertion flags Aura uses, and must still be reported.
    assert _is_keep_awake_assertion_process(
        _FakeProc("caffeinate", ["caffeinate", "-t", "3600"])
    ) is False


def test_worker_children_are_unaffected():
    from core.runtime.runtime_hygiene import _is_keep_awake_assertion_process

    assert _is_keep_awake_assertion_process(
        _FakeProc("Python", ["python", "-c", "from multiprocessing.spawn import x"])
    ) is False


def test_the_flags_aura_actually_uses_are_covered():
    from core.runtime.keep_awake import MacKeepAwakeController
    from core.runtime.runtime_hygiene import _is_keep_awake_assertion_process

    controller = MacKeepAwakeController()
    for display in (False, True):
        for ac_power in (False, True):
            command = controller.build_command(
                keep_display_awake=display, require_ac_power=ac_power
            )
            assert _is_keep_awake_assertion_process(
                _FakeProc(command[0], list(command))
            ) is True, f"hygiene would flag Aura's own assertion: {command}"
