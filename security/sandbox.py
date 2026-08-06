"""
Sandbox for autonomous code execution.

Enforcement: command allowlisting per security level, rlimits on child
processes, workdir containment, env-var stripping.

Limitations: no filesystem or network namespace isolation. True sandboxing
on macOS requires sandbox-exec or a container runtime. This is defense-in-depth,
not a hard security boundary.
"""
import logging
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation
from core.runtime.subprocess_gateway import AcceleratorCapability, get_subprocess_gateway

HAS_UNIX = os.name == "posix"
_SANDBOX_EXECUTION_ERRORS = (
    OSError,
    subprocess.SubprocessError,
    UnicodeError,
    ValueError,
)
_RESOURCE_LIMIT_ERRORS = (OSError, ValueError)

logger = logging.getLogger("security.sandbox")


class SecurityLevel(Enum):
    """Security isolation levels"""
    UNTRUSTED = auto()     # Maximum restrictions
    RESTRICTED = auto()    # Restricted FS access, no network
    TRUSTED = auto()       # Controlled access with logging
    PRIVILEGED = auto()    # Full access (internal only)


# Command allowlists per security level
_ALLOWED_COMMANDS = {
    SecurityLevel.UNTRUSTED: frozenset(),  # Nothing allowed
    SecurityLevel.RESTRICTED: frozenset({
        "python", "python3",
    }),
    SecurityLevel.TRUSTED: frozenset({
        "python", "python3", "git", "pip",
    }),
    SecurityLevel.PRIVILEGED: None,  # All commands (internal use only)
}


@dataclass
class ResourceLimits:
    """Resource limits for sandbox"""
    cpu_time_seconds: float = 30.0
    memory_mb: int = 512
    max_processes: int = 50
    max_open_files: int = 100
    max_file_size_mb: int = 10
    wall_clock_seconds: float = 60.0

    def to_rlimit_args(self) -> dict[int, tuple[int, int]]:
        """Convert to resource limit arguments"""
        limits = {}

        if not HAS_UNIX:
            return limits

        # CPU time (seconds)
        limits[resource.RLIMIT_CPU] = (
            int(self.cpu_time_seconds),
            int(self.cpu_time_seconds) + 1
        )

        # Memory (bytes)
        memory_bytes = self.memory_mb * 1024 * 1024
        limits[resource.RLIMIT_AS] = (memory_bytes, memory_bytes)

        # Processes/Threads
        try:
            limits[resource.RLIMIT_NPROC] = (self.max_processes, self.max_processes)
        except ValueError:
            pass

        # File descriptors
        try:
            limits[resource.RLIMIT_NOFILE] = (self.max_open_files, self.max_open_files)
        except ValueError:
            pass

        # File size (bytes)
        file_size_bytes = self.max_file_size_mb * 1024 * 1024
        limits[resource.RLIMIT_FSIZE] = (file_size_bytes, file_size_bytes)

        return limits


@dataclass
class ExecutionResult:
    """Result of sandboxed execution"""
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    execution_time: float
    memory_used_mb: float
    security_violations: list[str]
    metrics: dict[str, Any]


class SecurityViolationError(Exception):
    """Security policy violation"""


SecurityViolation = SecurityViolationError


class SecureSandbox:
    """Execution environment with resource limits and command allowlisting.

    Enforces:
    - Command allowlisting based on security level
    - Workdir containment (child process cwd)
    - rlimits on CPU time, memory, file descriptors, file size
    - Stdout/stderr size caps (1MB)
    - Sensitive env-var stripping
    
    macOS Hardening:
    - Automatically injects `sandbox-exec` with a strict version 1 profile
    - Denies network access entirely
    - Restricts file-write strictly to the workdir/tmp
    """

    MAX_OUTPUT_BYTES = 1024 * 1024  # 1MB output cap

    def __init__(
        self,
        security_level: SecurityLevel = SecurityLevel.RESTRICTED,
        workdir: Path | None = None,
        allowed_paths: list[Path] | None = None,
        allowed_commands: list[str] | None = None
    ):
        self.security_level = security_level
        self.allowed_paths = [p.resolve() for p in (allowed_paths or [])]
        self.allowed_commands = set(allowed_commands or [])

        # Merge with level-based allowlist
        level_commands = _ALLOWED_COMMANDS.get(security_level)
        if level_commands is not None:
            self.allowed_commands = self.allowed_commands | set(level_commands)
        else:
            self.allowed_commands = None  # None = all allowed (PRIVILEGED)

        # Create isolated workspace
        if workdir:
            self.workdir = Path(workdir).resolve()
            self.workdir.mkdir(parents=True, exist_ok=True)
            self._cleanup_workdir = False
        else:
            self.workdir = Path(tempfile.mkdtemp(prefix="sandbox_")).resolve()
            self._cleanup_workdir = True

        self.resource_limits = ResourceLimits()
        self.violations: list[str] = []
        self.execution_history: list[ExecutionResult] = []

        logger.info(
            "Sandbox initialized at %s (level: %s)", self.workdir, security_level.name
        )

    def _validate_command(self, cmd: list[str]) -> list[str]:
        """Validate command against allowlist."""
        if not cmd:
            raise SecurityViolationError("Empty command")

        binary_path = Path(cmd[0])
        binary = binary_path.name  # Basename only
        is_python_executable = (
            binary == os.path.basename(sys.executable)
            or (binary.startswith("python") and all(c.isdigit() or c == "." for c in binary[6:]))
        )
        allowed_list = self.allowed_commands
        if allowed_list is not None:
            is_allowed = binary in allowed_list
            if not is_allowed and ("python" in allowed_list or "python3" in allowed_list):
                if is_python_executable:
                    is_allowed = True
            if not is_allowed:
                raise SecurityViolationError(
                    f"Command '{binary}' not in allowlist: {self.allowed_commands}"
                )

        if self.security_level != SecurityLevel.PRIVILEGED and binary_path.parent != Path("."):
            self._validate_canonical_binary(binary_path, binary)

        # No metacharacter filtering — we use subprocess.Popen with a list
        # (no shell=True), so shell metacharacters have no special meaning.
        # The allowlist above is the actual security boundary.

        return cmd

    def _validate_canonical_binary(self, binary_path: Path, binary: str) -> None:
        """Reject path-based allowlist bypasses for restricted commands."""
        allowed_targets = {os.path.realpath(sys.executable)}
        discovered = shutil.which(binary)
        if discovered:
            allowed_targets.add(os.path.realpath(discovered))

        try:
            candidate = os.path.realpath(binary_path)
        except OSError as exc:
            raise SecurityViolationError(
                f"Command path could not be resolved: {binary_path}"
            ) from exc

        if candidate not in allowed_targets:
            raise SecurityViolationError(
                "Command path is not an approved runtime binary: "
                f"{binary_path}"
            )

    @staticmethod
    def _sandbox_profile_literal(path: Path) -> str:
        """Escape paths embedded in a sandbox-exec string literal."""
        return str(path.absolute()).replace("\\", "\\\\").replace('"', '\\"')

    def execute_command(
        self,
        cmd: list[str],
        timeout: float = 30.0,
        input_data: str | None = None
    ) -> ExecutionResult:
        """Execute command with resource limits, allowlisting, and monitoring."""
        start_time = time.time()
        violations = []

        # Validate command before execution
        try:
            cmd = self._validate_command(cmd)
        except SecurityViolationError as sv:
            violations.append(str(sv))
            logger.warning("Sandbox blocked command: %s", sv)
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=str(sv),
                execution_time=0.0,
                memory_used_mb=0.0,
                security_violations=violations,
                metrics={}
            )

        try:
            # Build environment with restricted vars
            env = os.environ.copy()
            # Strip with the SAME classifier the subprocess gateway enforces
            # with. This list used to be its own — TOKEN/SECRET/PASSWORD/KEY/
            # CREDENTIAL/AUTH — and the gateway's is broader (session_id,
            # cookie, cert, bearer, passphrase, signature, ssn). So the
            # sandbox stripped what it knew about, the gateway then refused
            # the spawn over what it did not, and the sandbox could not launch
            # at all: "untrusted_code may not hold secrets; environment
            # carries CLAUDE_CODE_HOST_SESSION_ID, CLAUDE_CODE_SESSION_ID,
            # OLDPWD, PWD". Two definitions of "sensitive", disagreeing.
            #
            # Sharing the classifier means stripping is exactly what passing
            # is, and anything added to one is honoured by both.
            try:
                from core.security.structural_redaction import is_sensitive_key
            except ImportError:  # pragma: no cover - keep the sandbox launchable
                def is_sensitive_key(key: str) -> bool:
                    return any(
                        marker in key.upper()
                        for marker in ("TOKEN", "SECRET", "PASSWORD", "KEY", "CREDENTIAL", "AUTH")
                    )
            for key in list(env.keys()):
                if is_sensitive_key(key):
                    del env[key]

            # macOS strict sandbox-exec injection
            if sys.platform == "darwin" and self.security_level != SecurityLevel.PRIVILEGED:
                profile_path = self.workdir / ".sandbox_profile.sb"
                workdir_literal = self._sandbox_profile_literal(self.workdir)
                atomic_write_text(profile_path, f'''(version 1)
(deny default)
(allow process-exec*)
(allow process-fork)
(allow file-read*)
(allow file-write*
    (subpath "{workdir_literal}")
    (subpath "/private/tmp")
    (subpath "/private/var/folders")
    (literal "/dev/null")
)
(deny network*)
(allow sysctl-read)
(allow ipc-posix-shm)
''', encoding="utf-8")
                profile_path.chmod(0o600)
                cmd = ["sandbox-exec", "-f", str(profile_path)] + cmd

            process = get_subprocess_gateway().spawn(
                cmd,
                stdin=subprocess.PIPE if input_data else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self.workdir),
                env=env,
                preexec_fn=self._set_resource_limits if HAS_UNIX else None,
                source="security.sandbox.execute_command",
                accelerator_capability=AcceleratorCapability.NONE,
            )

            try:
                stdout, stderr = process.communicate(
                    input=input_data,
                    timeout=timeout
                )
                # Cap output size
                if len(stdout) > self.MAX_OUTPUT_BYTES:
                    stdout = stdout[:self.MAX_OUTPUT_BYTES] + "\n[OUTPUT TRUNCATED]"
                if len(stderr) > self.MAX_OUTPUT_BYTES:
                    stderr = stderr[:self.MAX_OUTPUT_BYTES] + "\n[OUTPUT TRUNCATED]"
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                violations.append("Execution timeout")

            exit_code = process.returncode
            if exit_code != 0:
                violations.append(f"Non-zero exit code: {exit_code}")

            execution_time = time.time() - start_time

            return ExecutionResult(
                success=exit_code == 0 and not violations,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                execution_time=execution_time,
                memory_used_mb=0.0,
                security_violations=violations,
                metrics={
                    "start_time": start_time,
                    "end_time": time.time(),
                    "security_level": self.security_level.name,
                }
            )
        except _SANDBOX_EXECUTION_ERRORS as e:
            record_degradation("sandbox", e)
            logger.exception("Sandbox execution failed before completion")
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                execution_time=time.time() - start_time,
                memory_used_mb=0.0,
                security_violations=[str(e)],
                metrics={}
            )

    def _set_resource_limits(self) -> None:
        """Set resource limits for child process"""
        if not HAS_UNIX:
            return

        for resource_id, limits in self.resource_limits.to_rlimit_args().items():
            try:
                resource.setrlimit(resource_id, limits)
            except _RESOURCE_LIMIT_ERRORS:
                continue  # Non-critical fallback inside the child process.

    def cleanup(self):
        """Clean up the sandbox workdir if we created it."""
        if self._cleanup_workdir and self.workdir.exists():
            try:
                shutil.rmtree(self.workdir)
                logger.debug("Sandbox workdir cleaned: %s", self.workdir)
            except OSError as e:
                record_degradation("sandbox.cleanup", e)
                logger.warning("Failed to clean sandbox workdir: %s", e)
