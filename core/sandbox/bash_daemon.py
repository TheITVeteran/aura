"""
Persistent Bash Daemon — Wholesale Addition

Provides a persistent Bash session for the Shell Skill. Enables stateful
interactions (e.g., maintaining `cd`, `export` variables, and activating 
virtual environments) across consecutive shell commands.
"""

import asyncio
import logging
import os

from core.runtime.errors import record_degradation
from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Aura.BashDaemon")
_BASH_READ_ERRORS = (RuntimeError, AttributeError, TypeError, ValueError, UnicodeError)
_BASH_WRITE_ERRORS = (BrokenPipeError, ConnectionError, OSError)


class BashDaemonNotStartedError(RuntimeError):
    """Raised when a persistent bash read is attempted before startup."""


class PersistentBashSession:
    def __init__(self, cwd: str):
        self.cwd = cwd
        self._process = None
        self._delimiter = f"---AURA_CMD_DELIM_{os.urandom(4).hex()}---"
        self._lock = asyncio.Lock()

    async def _start(self) -> None:
        env = os.environ.copy()
        # Start bash and immediately set it to print our delimiter after every command
        self._process = await get_subprocess_gateway().spawn_async(
            ["bash", "--noprofile", "--norc"],
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=self.cwd,
            env=env,
            source="core.sandbox.bash_daemon.persistent_bash",
        )
        
        # Setup bash to echo the delimiter and the exit code
        setup_cmd = f"export PS1=''\nPROMPT_COMMAND='echo \"\n{self._delimiter}:$?\"'\n"
        self._process.stdin.write(setup_cmd.encode('utf-8'))
        await self._process.stdin.drain()
        
        # Read until first delimiter
        await self._read_until_delimiter()

    async def _read_until_delimiter(self) -> tuple[str, int]:
        """Reads stdout until the delimiter is found. Returns (output, exit_code)."""
        if self._process is None or self._process.stdout is None:
            raise BashDaemonNotStartedError("bash daemon has no stdout pipe")
        output: list[str] = []
        line = await self._process.stdout.readline()
        while line:
            try:
                line_str = line.decode('utf-8', errors='replace')
                if line_str.startswith(self._delimiter):
                    # Extract exit code
                    parts = line_str.strip().split(":")
                    exit_code = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
                    return "".join(output).strip(), exit_code
                output.append(line_str)
                line = await self._process.stdout.readline()
            except _BASH_READ_ERRORS as e:
                record_degradation('bash_daemon', e)
                logger.error("Error reading from bash daemon: %s", e)
                break
        return "".join(output).strip(), -1

    async def execute(
        self,
        cmd: str,
        timeout_s: float = 10.0,
        **legacy_options: object,
    ) -> tuple[bool, str]:
        if "timeout" in legacy_options:
            timeout_s = float(legacy_options.pop("timeout"))
        if legacy_options:
            unknown = ", ".join(sorted(legacy_options))
            raise TypeError(f"unknown bash daemon execution options: {unknown}")

        async with self._lock:
            if self._process is None or self._process.returncode is not None:
                await self._start()

            # Write command
            try:
                self._process.stdin.write(f"{cmd}\n".encode())
                await self._process.stdin.drain()
            except _BASH_WRITE_ERRORS as e:
                record_degradation('bash_daemon', e)
                return False, f"Failed to write to daemon: {e}"

            # Wait for output up to timeout
            try:
                output, exit_code = await asyncio.wait_for(self._read_until_delimiter(), timeout=timeout_s)
                return exit_code == 0, output
            except TimeoutError:
                return False, f"Command timed out after {timeout_s}s."
            except (RuntimeError, AttributeError) as e:
                record_degradation('bash_daemon', e)
                return False, f"Execution failed: {e}"

    async def kill(self) -> None:
        if self._process and self._process.returncode is None:
            self._process.kill()
            await self._process.wait()

class BashDaemonManager:
    """Manages persistent sessions per conversation or agent."""
    def __init__(self) -> None:
        self.sessions: dict[str, PersistentBashSession] = {}

    def get_session(self, session_id: str, cwd: str) -> PersistentBashSession:
        if session_id not in self.sessions:
            self.sessions[session_id] = PersistentBashSession(cwd)
        return self.sessions[session_id]

    async def shutdown(self) -> None:
        for s in self.sessions.values():
            await s.kill()
        self.sessions.clear()

# Global singleton
bash_manager = BashDaemonManager()
