"""CP126 contracts for core/adapters/nethack_adapter.py.

Fourteen findings in 144 lines, and they shared one shape: the adapter acted
on the world and then described the action rather than the outcome. It
answered "Destroy old game?" with `y` on a save it could not attribute, wrote
into the human's home directory, sent whatever string it was handed to a live
process, read one chunk of a frame and called it an observation, treated the
game ending as a quiet moment, and killed the child without asking it to save
or checking that it died.

These tests drive the adapter against a fake pexpect child, so the grammar,
the refusals, the drain loop and the shutdown ladder are exercised without a
game installed.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from core.adapters import nethack_adapter as module
from core.adapters.nethack_adapter import (
    NetHackActionRefused,
    NetHackAdapter,
    NetHackExistingSaveError,
    NetHackSessionError,
    NetHackUnavailable,
    support_status,
)


class _FakeTimeout(Exception):
    pass


class _FakeEOF(Exception):
    pass


class _FakeChild:
    """A pexpect child that hands out scripted frames, one read at a time."""

    def __init__(self, frames: list[str], *, alive: bool = True) -> None:
        self._frames = list(frames)
        self.sent: list[str] = []
        self.lines: list[str] = []
        self._alive = alive
        self.terminate_calls: list[bool] = []
        self.closed = False
        self.exitstatus: int | None = None
        self.winsize: tuple[int, int] | None = None
        self.dies_after_sends = None

    def setwinsize(self, rows: int, cols: int) -> None:
        self.winsize = (rows, cols)

    def read_nonblocking(self, size: int = 0, timeout: float = 0.0) -> str:
        if not self._frames:
            raise _FakeTimeout("no more output")
        frame = self._frames.pop(0)
        if frame is _EOF_SENTINEL:
            raise _FakeEOF("terminal closed")
        return frame

    def send(self, payload: str) -> None:
        self.sent.append(payload)
        if self.dies_after_sends is not None and len(self.sent) >= self.dies_after_sends:
            self._alive = False

    def sendline(self, payload: str) -> None:
        self.lines.append(payload)

    def isalive(self) -> bool:
        return self._alive

    def terminate(self, force: bool = False) -> None:
        self.terminate_calls.append(force)
        self._alive = False

    def close(self, force: bool = False) -> None:
        self.closed = True


_EOF_SENTINEL = "<<EOF>>"


class _FakeScreen:
    def __init__(self, columns: int, rows: int) -> None:
        self.columns = columns
        self.rows = rows
        self.display = [""]


class _FakeStream:
    """Accumulates like a terminal does.

    Replacing the display on every feed made the fixture lie: draining to
    quiet would erase the "Destroy old game?" prompt with whatever frame
    followed it, and the adapter would look like it had never seen it.
    """

    def __init__(self, screen: _FakeScreen) -> None:
        self._screen = screen
        self._lines: list[str] = []

    def feed(self, text: str) -> None:
        self._lines.extend(text.split("\n"))
        self._screen.display = self._lines[-screen_rows(self._screen) :]


def screen_rows(screen: _FakeScreen) -> int:
    return screen.rows


@pytest.fixture
def fake_terminal(monkeypatch, tmp_path):
    """A spawnable, drainable terminal with no game installed."""
    spawned: list[_FakeChild] = []
    script: dict[str, Any] = {"frames": ["welcome"]}

    def _spawn(executable, argv, **kwargs):
        child = _FakeChild(list(script["frames"]))
        child.spawn_args = (executable, argv, kwargs)
        spawned.append(child)
        return child

    fake_pexpect = type(
        "pexpect",
        (),
        {"spawn": staticmethod(_spawn), "TIMEOUT": _FakeTimeout, "EOF": _FakeEOF},
    )
    fake_pyte = type(
        "pyte", (), {"Screen": _FakeScreen, "Stream": _FakeStream}
    )
    monkeypatch.setattr(module, "pexpect", fake_pexpect)
    monkeypatch.setattr(module, "pyte", fake_pyte)

    executable = tmp_path / "nethack"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)

    adapter = NetHackAdapter(str(executable), state_root=tmp_path / "sessions")
    return adapter, spawned, script


class TestAnExistingSaveIsNotDestroyedOnItsOwn:
    def test_the_destroy_prompt_refuses_by_default(self, fake_terminal) -> None:
        """The critical finding. It used to answer `y`."""
        adapter, spawned, script = fake_terminal
        script["frames"] = ["Destroy old game?"]

        with pytest.raises(NetHackExistingSaveError, match="destroy_existing_save=True"):
            adapter.start("Aura")

        assert "y" not in spawned[0].sent, "no confirmation may be sent"

    def test_refusing_does_not_leave_the_child_running(self, fake_terminal) -> None:
        adapter, spawned, script = fake_terminal
        script["frames"] = ["Destroy old game?"]

        with pytest.raises(NetHackExistingSaveError):
            adapter.start("Aura")

        assert spawned[0].isalive() is False
        assert adapter.is_alive() is False

    def test_an_explicit_request_does_destroy_it(self, fake_terminal) -> None:
        """Refusing must not mean the capability is gone — only that it is asked for."""
        adapter, spawned, script = fake_terminal
        script["frames"] = ["Destroy old game?", "you begin"]

        result = adapter.start("Aura", destroy_existing_save=True)

        assert result["destroyed_existing_save"] is True
        assert "y" in spawned[0].sent

    def test_a_clean_start_sends_nothing(self, fake_terminal) -> None:
        adapter, spawned, _script = fake_terminal
        adapter.start("Aura")
        assert spawned[0].sent == []

    def test_the_refusal_path_does_not_answer_the_prompt_while_quitting(
        self, fake_terminal
    ) -> None:
        """The clean-quit ladder confirms itself with "y", and at THIS prompt
        "y" is the destroy. Stopping here must send no keys at all."""
        adapter, spawned, script = fake_terminal
        script["frames"] = ["Destroy old game?"]

        with pytest.raises(NetHackExistingSaveError):
            adapter.start("Aura")

        assert spawned[0].sent == []
        assert spawned[0].lines == []


class TestNothingIsWrittenIntoTheHumansHome:
    def test_the_rc_lives_in_the_session_directory(self, fake_terminal, tmp_path) -> None:
        adapter, spawned, _script = fake_terminal
        result = adapter.start("Aura")

        directory = Path(result["directory"])
        assert directory.is_relative_to(tmp_path / "sessions")
        assert (directory / "nethackrc").is_file()

    def test_the_child_home_is_the_session_directory(self, fake_terminal) -> None:
        """So the only save that can be at the prompt is one of hers."""
        adapter, spawned, _script = fake_terminal
        result = adapter.start("Aura")

        _executable, _argv, kwargs = spawned[0].spawn_args
        assert kwargs["env"]["HOME"] == result["directory"]
        assert kwargs["env"]["NETHACKOPTIONS"].endswith("nethackrc")

    def test_two_sessions_do_not_share_a_directory(self, fake_terminal) -> None:
        adapter, _spawned, _script = fake_terminal
        first = adapter.start("Aura")
        adapter.stop()
        second = adapter.start("Aura")
        assert first["directory"] != second["directory"]


class TestTheExecutableAndNameAreVerified:
    def test_an_argv_list_is_used_not_a_command_string(self, fake_terminal) -> None:
        adapter, spawned, _script = fake_terminal
        adapter.start("Aura")

        executable, argv, _kwargs = spawned[0].spawn_args
        assert argv == ["-u", "Aura"]
        assert " " not in str(executable)

    @pytest.mark.parametrize(
        "name", ["", "-u", "a b", "Aura; rm -rf /", "../escape", "x" * 40]
    )
    def test_a_name_outside_the_grammar_is_refused(self, fake_terminal, name) -> None:
        adapter, _spawned, _script = fake_terminal
        with pytest.raises(NetHackSessionError, match="player name"):
            adapter.start(name)

    def test_a_missing_executable_is_refused_before_spawning(self, fake_terminal, tmp_path) -> None:
        adapter, spawned, _script = fake_terminal
        adapter.nethack_path = str(tmp_path / "not-installed")
        with pytest.raises(NetHackUnavailable, match="not found|not a file"):
            adapter.start("Aura")
        assert spawned == []

    def test_a_non_executable_file_is_refused(self, fake_terminal, tmp_path) -> None:
        adapter, _spawned, _script = fake_terminal
        blob = tmp_path / "plain"
        blob.write_text("not a program")
        blob.chmod(0o644)
        adapter.nethack_path = str(blob)
        with pytest.raises(NetHackUnavailable, match="not executable"):
            adapter.start("Aura")


class TestStartDoesNotOrphanARunningGame:
    def test_a_second_start_is_refused(self, fake_terminal) -> None:
        adapter, spawned, _script = fake_terminal
        adapter.start("Aura")
        with pytest.raises(NetHackSessionError, match="still running"):
            adapter.start("Aura")
        assert len(spawned) == 1, "the running child must not be replaced silently"

    def test_replacing_stops_the_previous_child(self, fake_terminal) -> None:
        adapter, spawned, _script = fake_terminal
        adapter.start("Aura")
        adapter.start("Aura", replace_running_session=True)

        assert len(spawned) == 2
        assert spawned[0].isalive() is False
        assert spawned[0].closed is True


class TestTheActionGrammar:
    def test_a_movement_key_is_accepted(self, fake_terminal) -> None:
        adapter, spawned, _script = fake_terminal
        adapter.start("Aura")
        receipt = adapter.send_action("h")
        assert receipt["accepted"] is True
        assert spawned[0].sent == ["h"]

    def test_an_arbitrary_string_is_refused(self, fake_terminal) -> None:
        """send_action used to forward whatever it was handed."""
        adapter, spawned, _script = fake_terminal
        adapter.start("Aura")
        with pytest.raises(NetHackActionRefused):
            adapter.send_action("go north and then quaff everything")
        assert spawned[0].sent == []

    def test_a_known_extended_command_is_accepted(self, fake_terminal) -> None:
        adapter, spawned, _script = fake_terminal
        adapter.start("Aura")
        adapter.send_action("#pray")
        assert spawned[0].sent == ["#pray\n"]

    def test_an_unknown_extended_command_is_refused(self, fake_terminal) -> None:
        adapter, _spawned, _script = fake_terminal
        adapter.start("Aura")
        with pytest.raises(NetHackActionRefused, match="not permitted"):
            adapter.send_action("#wizmap")

    def test_quit_is_not_reachable_as_an_action(self, fake_terminal) -> None:
        """Ending the game is stop()'s job, with a receipt. Not a keystroke
        the model can emit into a live session."""
        adapter, _spawned, _script = fake_terminal
        adapter.start("Aura")
        with pytest.raises(NetHackActionRefused):
            adapter.send_action("#quit")

    def test_an_empty_action_is_refused(self, fake_terminal) -> None:
        adapter, _spawned, _script = fake_terminal
        adapter.start("Aura")
        with pytest.raises(NetHackActionRefused):
            adapter.send_action("")

    def test_escape_and_return_are_in_the_grammar(self, fake_terminal) -> None:
        adapter, spawned, _script = fake_terminal
        adapter.start("Aura")
        adapter.send_action("\x1b")
        adapter.send_action("\n")
        assert spawned[0].sent == ["\x1b", "\n"]


class TestObservationsCarryTheirReceipt:
    def test_the_observation_places_itself_in_a_causal_chain(self, fake_terminal) -> None:
        """Text, a timestamp and the string "nethack" could not tell two
        different games apart."""
        adapter, _spawned, script = fake_terminal
        script["frames"] = ["first", "second"]
        started = adapter.start("Aura")

        observation = adapter.get_observation()
        meta = observation["metadata"]

        assert meta["session_id"] == started["session_id"]
        assert meta["player"] == "Aura"
        assert meta["process_state"] == "running"
        assert meta["frame_seq"] >= 1
        assert (meta["screen_columns"], meta["screen_rows"]) == (80, 24)

    def test_the_last_action_is_named_in_the_next_observation(self, fake_terminal) -> None:
        adapter, _spawned, script = fake_terminal
        script["frames"] = ["a", "b", "c"]
        adapter.start("Aura")

        receipt = adapter.send_action("h")
        observation = adapter.get_observation()

        assert observation["metadata"]["last_action_id"] == receipt["action_id"]

    def test_an_observation_before_any_session_is_refused(self, fake_terminal) -> None:
        adapter, _spawned, _script = fake_terminal
        with pytest.raises(NetHackSessionError, match="no NetHack session"):
            adapter.get_observation()


class TestTheScreenIsDrainedNotSampled:
    def test_every_pending_frame_is_consumed(self, fake_terminal) -> None:
        """One bounded read left the rest in the pty, so an action could be
        chosen against a screen the game had already replaced."""
        adapter, _spawned, script = fake_terminal
        script["frames"] = ["one", "two", "three", "final screen"]

        started = adapter.start("Aura")

        assert started["frame_seq"] == 4, "every pending frame must be consumed"
        assert adapter.get_screen_text().splitlines()[-1] == "final screen"

    def test_the_frame_counter_advances_once_per_frame(self, fake_terminal) -> None:
        adapter, _spawned, script = fake_terminal
        script["frames"] = ["a", "b", "c"]
        started = adapter.start("Aura")
        assert started["frame_seq"] == 3


class TestEndOfFileIsTheGameEnding:
    def test_eof_moves_the_adapter_to_dead(self, fake_terminal) -> None:
        """It used to be a debug line, and the adapter kept answering."""
        adapter, spawned, script = fake_terminal
        script["frames"] = ["playing", _EOF_SENTINEL]

        adapter.start("Aura")

        assert adapter.get_observation()["metadata"]["process_state"] == "dead"

    def test_an_action_into_a_dead_game_is_reported_not_sent(self, fake_terminal) -> None:
        adapter, spawned, script = fake_terminal
        script["frames"] = ["playing"]
        adapter.start("Aura")
        spawned[0]._alive = False

        receipt = adapter.send_action("h")

        assert receipt["accepted"] is False
        assert receipt["process_state"] == "dead"
        assert spawned[0].sent == []


class TestShutdownSaysWhatHappened:
    def test_a_clean_quit_is_attempted_before_termination(self, fake_terminal) -> None:
        adapter, spawned, _script = fake_terminal
        adapter.start("Aura")
        spawned[0].dies_after_sends = 2

        result = adapter.stop()

        assert result["disposition"] == "saved_and_exited"
        assert "#quit\n" in spawned[0].sent
        assert spawned[0].terminate_calls == [], "no kill was needed"

    def test_a_game_that_will_not_quit_is_terminated_and_said_so(self, fake_terminal) -> None:
        adapter, spawned, _script = fake_terminal
        adapter.start("Aura")

        result = adapter.stop(deadline_s=0.5)

        assert result["reaped"] is True
        assert result["disposition"] in {"terminated", "killed_state_unknown"}

    def test_stopping_without_a_session_is_not_an_error(self, fake_terminal) -> None:
        adapter, _spawned, _script = fake_terminal
        assert adapter.stop()["stopped"] is False

    def test_the_pty_is_closed(self, fake_terminal) -> None:
        adapter, spawned, _script = fake_terminal
        adapter.start("Aura")
        adapter.stop()
        assert spawned[0].closed is True

    def test_skipping_the_save_is_possible_and_named(self, fake_terminal) -> None:
        adapter, spawned, _script = fake_terminal
        adapter.start("Aura")
        adapter.stop(save=False)
        assert "#quit\n" not in spawned[0].sent


class TestTheContractsCallersRelyOn:
    def test_is_alive_returns_a_bool_with_no_session(self, fake_terminal) -> None:
        """It used to return None, because `None and ...` is None."""
        adapter, _spawned, _script = fake_terminal
        assert adapter.is_alive() is False

    def test_is_alive_returns_a_bool_with_a_session(self, fake_terminal) -> None:
        adapter, _spawned, _script = fake_terminal
        adapter.start("Aura")
        assert adapter.is_alive() is True

    def test_missing_game_dependencies_are_a_state_not_an_import_error(self, monkeypatch) -> None:
        """pexpect and pyte were imported at module scope, so a host without
        them could not enumerate the adapter at all."""
        monkeypatch.setattr(module, "pexpect", None)
        monkeypatch.setattr(module, "_PEXPECT_IMPORT_ERROR", "No module named 'pexpect'")

        status = support_status()
        assert status["available"] is False
        assert any("pexpect" in item for item in status["missing"])

        with pytest.raises(NetHackUnavailable, match="pexpect"):
            NetHackAdapter().start("Aura")

    def test_the_real_module_imports_without_the_game_installed(self) -> None:
        """This test file is proof: it imported the module at the top."""
        assert isinstance(support_status()["available"], bool)


class TestTheBlockingWorkIsOffTheLoop:
    def test_the_async_pair_exists_for_every_blocking_entry_point(self) -> None:
        for name in ("start", "send_action", "get_observation", "get_screen_text", "stop"):
            assert hasattr(NetHackAdapter, f"{name}_async"), name

    def test_an_async_action_round_trips(self, fake_terminal) -> None:
        adapter, spawned, script = fake_terminal
        script["frames"] = ["a", "b"]

        async def _drive() -> dict:
            await adapter.start_async("Aura")
            return await adapter.send_action_async("j")

        receipt = asyncio.run(_drive())
        assert receipt["accepted"] is True
        assert spawned[0].sent == ["j"]

    def test_the_skill_never_calls_the_blocking_api(self) -> None:
        """The skill runs on the conversation loop; a one-second pty wait
        there is a one-second stall for every other turn in flight."""
        source = Path("core/skills/nethack.py").read_text(encoding="utf-8")
        for blocking in (
            "adapter.start(",
            "adapter.send_action(",
            "adapter.get_screen_text(",
            "adapter.get_observation(",
        ):
            assert blocking not in source, blocking


class TestTheSkillAndTheGrammarAgree:
    """A key the skill can emit that the adapter refuses is a dead action.

    ``PRAY`` mapped to a bare ``'#'`` — the extended-command PREFIX, which
    leaves the game at a half-typed command line waiting for a word. Under
    the old adapter it was sent anyway and looked like it had worked.
    """

    def test_every_named_key_the_skill_offers_is_in_the_grammar(self) -> None:
        from core.skills.nethack import SPECIAL_KEYS

        refused = {}
        for name, key in SPECIAL_KEYS.items():
            try:
                NetHackAdapter.resolve_action(key)
            except NetHackActionRefused as exc:
                refused[name] = f"{key!r}: {exc}"
        assert refused == {}, refused

    def test_the_skill_refuses_a_multi_character_action_instead_of_truncating(self) -> None:
        """It used to take action[0] and send that — "quaff" became "q",
        a different command, executed as though it had been asked for."""
        source = Path("core/skills/nethack.py").read_text(encoding="utf-8")
        assert "action[0] if action else None" not in source
