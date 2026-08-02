"""Final Blocker: No raw or legacy bypass of the canonical kernel.

No skill, old runtime, or direct adapter call may send environment actions
outside the canonical EnvironmentKernel path.
"""
import ast
import pytest
from pathlib import Path


# Raw keystroke sinks, identified by what they ARE rather than by a
# substring.
#
# This scan used to grep file CONTENT for ".send(" — the most overloaded
# method name in Python. socket.send, queue.send, bus.send and Apple
# Messages automation all matched, so the allowlist grew to fifteen
# directory prefixes (core/skills/, core/orchestrator/, core/consciousness/,
# core/bus/, core/state/ ...) to suppress the noise. At that point the check
# covered almost nothing: a genuine keystroke sink added under any of those
# trees would have passed silently, which is the opposite of a final blocker.
#
# Matching is now AST-based. A module-level sink is a known dangerous
# function; a `.send()` counts only when its receiver is a terminal child.
# Result: three real adapters instead of half of core/, and the allowlist
# below is short enough to read.

#: (module, attribute) pairs that inject keystrokes or spawn a pty.
MODULE_KEY_SINKS = {
    ("pyautogui", "press"),
    ("pyautogui", "typewrite"),
    ("pyautogui", "write"),
    ("keyboard", "write"),
    ("keyboard", "press"),
    ("keyboard", "send"),
    ("pexpect", "spawn"),
}

#: Receiver names that mark a `.send()` as writing to a terminal/pty child
#: rather than to a socket, a queue, or a messaging API.
TERMINAL_RECEIVER_HINTS = ("child", "pty", "term", "console", "spawn", "shell", "tty")

#: Bare call names that are unambiguous on their own.
BARE_CALL_SINKS = {"send_action"}

#: Modules where raw key sinks are ALLOWED. Every entry is a terminal
#: adapter that legitimately drives a pty.
ALLOWED_RAW_KEY_MODULES = {
    "core/environment/adapter.py",
    "core/environment/command.py",
    "core/environment/generic_command_handlers.py",
    "core/adapters/nethack_adapter.py",
    "core/embodiment/games/nethack/env.py",
    "core/environments/terminal_grid/nethack_adapter.py",
}

REPO_ROOT = Path(__file__).resolve().parents[3]


class TestNoRawBypass:
    """Raw keystroke sinks must not appear outside approved adapter/compiler modules."""

    @staticmethod
    def _keystroke_sinks_in(path: Path) -> list[str]:
        """Genuine keystroke/pty sinks in one file, by AST."""
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            return []
        found: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name):
                if func.id in BARE_CALL_SINKS:
                    found.append(f"{func.id}()")
                continue
            if not isinstance(func, ast.Attribute):
                continue
            attr = func.attr
            base = getattr(func.value, "id", "") or getattr(func.value, "attr", "")
            if (base, attr) in MODULE_KEY_SINKS:
                found.append(f"{base}.{attr}()")
            elif attr == "send" and any(
                hint in base.lower() for hint in TERMINAL_RECEIVER_HINTS
            ):
                found.append(f"{base}.send()")
        return sorted(set(found))

    def test_static_scan_raw_key_sinks(self):
        """Genuine keystroke sinks must not appear outside approved adapters."""
        violations = []
        core_dir = REPO_ROOT / "core"
        assert core_dir.exists(), "core directory not found"

        for py_file in core_dir.rglob("*.py"):
            rel = str(py_file.relative_to(REPO_ROOT))
            if any(allowed in rel for allowed in ALLOWED_RAW_KEY_MODULES):
                continue
            if "__pycache__" in rel:
                continue
            for sink in self._keystroke_sinks_in(py_file):
                violations.append(f"{rel}: raw key sink {sink}")

        if violations:
            msg = "Raw key sinks found outside approved modules:\n" + "\n".join(violations)
            pytest.fail(msg)

    def test_the_scan_can_still_see_a_real_sink(self):
        """A check that finds nothing everywhere would pass vacuously.

        The allowed adapters genuinely drive a pty; if the scanner stops
        seeing them, it has stopped working and the test above is decorative.
        """
        adapter = REPO_ROOT / "core" / "adapters" / "nethack_adapter.py"
        if not adapter.exists():
            pytest.skip("nethack adapter not present in this checkout")
        found = self._keystroke_sinks_in(adapter)
        assert found, "scanner found no sinks in a known pty adapter — it is broken"

    def test_ordinary_send_calls_are_not_keystroke_sinks(self, tmp_path):
        """socket.send and a Messages automation are not keyboard bypasses."""
        sample = tmp_path / "ordinary.py"
        sample.write_text(
            "import socket\n"
            "def f(sock, bus, messages_app, payload):\n"
            "    sock.send(payload)\n"
            "    bus.send(payload)\n"
            "    messages_app.send(payload)\n"
            "    queue.send(payload)\n",
            encoding="utf-8",
        )
        assert self._keystroke_sinks_in(sample) == []

    def test_no_direct_adapter_execute_from_policy(self):
        """Policy modules must not call adapter.execute directly."""
        policy_dir = REPO_ROOT / "core" / "environment" / "policy"
        assert policy_dir.exists(), "policy directory not found"

        violations = []
        for py_file in policy_dir.rglob("*.py"):
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            if "adapter.execute" in content or "adapter.send" in content:
                violations.append(str(py_file.relative_to(REPO_ROOT)))

        assert not violations, f"Policy modules must not call adapter directly: {violations}"

    def test_policy_returns_action_intent_not_raw_key(self):
        """Policy output must be ActionIntent, not raw string keys."""
        from core.environment.policy.policy_orchestrator import PolicyOrchestrator
        from core.environment.command import ActionIntent
        from core.environment.parsed_state import ParsedState
        from core.environment.belief_graph import EnvironmentBeliefGraph
        from core.environment.homeostasis import Homeostasis

        orch = PolicyOrchestrator()
        parsed = ParsedState(
            environment_id="test",
            context_id="test",
            sequence_id=0,
            self_state={"hp": 20, "max_hp": 20},
        )
        belief = EnvironmentBeliefGraph()
        homeo = Homeostasis()
        intent = orch.select_action(
            parsed_state=parsed,
            belief=belief,
            homeostasis=homeo,
            episode=None,
            recent_frames=[],
        )
        assert isinstance(intent, ActionIntent), f"Policy returned {type(intent)}, not ActionIntent"
        # Must not be a single raw key character
        assert len(intent.name) > 1 or intent.name in ("i",), f"Policy returned raw key: {intent.name}"

    def test_command_compiler_rejects_unknown_intent(self):
        """CommandCompiler must fail closed on unknown intents."""
        from core.environment.command import CommandCompiler, ActionIntent
        compiler = CommandCompiler("test")
        # Don't register any handlers
        with pytest.raises(ValueError, match="unknown_intent"):
            compiler.compile(ActionIntent(name="totally_made_up_action"))
