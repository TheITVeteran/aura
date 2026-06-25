"""General terminal parser: NetHack's glyph/threat idea generalized to any terminal."""
from __future__ import annotations

import pytest

from core.perception.general_terminal_parser import GeneralTerminalParser
from core.perception.environment_parser import EnvironmentState


@pytest.fixture
def parser():
    return GeneralTerminalParser()


def test_returns_environment_state_with_terminal_domain(parser):
    state = parser.parse("bryan@mac ~/proj $ ls\nfile1  file2")
    assert isinstance(state, EnvironmentState)
    assert state.domain == "terminal"


def test_extracts_cwd_user_and_command_from_prompt(parser):
    state = parser.parse("(venv) bryan@mac:~/aura/live-source$ pytest -q")
    assert state.self_state["cwd"] == "~/aura/live-source"
    assert state.self_state["user"] == "bryan"
    assert state.self_state["venv"] == "venv"
    assert state.self_state["current_command"] == "pytest -q"


def test_destructive_command_scores_high_threat(parser):
    safe = parser.threat_score("bryan@mac ~ $ ls -la")
    danger = parser.threat_score("bryan@mac ~ $ rm -rf /")
    assert danger >= 0.9
    assert danger > safe


def test_traceback_and_exception_flagged_as_threats(parser):
    out = (
        "Traceback (most recent call last):\n"
        '  File "x.py", line 3, in <module>\n'
        "ValueError: bad input\n"
    )
    state = parser.parse(out)
    labels = {e["label"] for e in state.entities}
    assert "python_traceback" in labels
    assert "exception" in labels
    assert state.self_state["threat_score"] >= 0.7


def test_password_prompt_detected_as_blocking(parser):
    state = parser.parse("Connecting...\nuser@host's password: ")
    assert "password_prompt" in state.active_prompts
    assert state.self_state["prompt_blocking"] is True
    assert state.has_active_prompt()


def test_confirmation_prompt_detected(parser):
    state = parser.parse("This will delete everything. Continue? [y/N] ")
    assert "confirmation_prompt" in state.active_prompts


def test_clean_output_is_low_threat(parser):
    state = parser.parse("bryan@mac ~/proj $ echo hello\nhello")
    assert state.self_state["threat_score"] < 0.3
    assert all(not e.get("hostile") for e in state.entities)


def test_empty_input_is_low_confidence(parser):
    state = parser.parse("")
    assert state.confidence < 0.5
    assert state.self_state["threat_score"] == 0.0


def test_dangerous_command_at_prompt_raises_frame_threat(parser):
    # even with no error output yet, a dangerous command typed at the prompt is a threat
    state = parser.parse("bryan@mac ~ $ sudo dd if=/dev/zero of=/dev/sda")
    assert state.self_state["threat_score"] >= 0.9
    assert state.self_state["current_command"].startswith("sudo dd")


def test_parser_for_domain_dispatches_nethack_vs_general():
    from core.perception.environment_parser import parser_for_domain
    from core.perception.nethack_parser import NetHackParser

    assert isinstance(parser_for_domain("nethack"), NetHackParser)
    assert isinstance(parser_for_domain("shell"), GeneralTerminalParser)
    assert isinstance(parser_for_domain("unknown_env"), GeneralTerminalParser)
