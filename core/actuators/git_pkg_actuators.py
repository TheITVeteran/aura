"""core/actuators/git_pkg_actuators.py
===================================
Actuators for Git and package installation operations.
"""

import subprocess
import sys
import re
from typing import Any
from core.actuators.actuator_registry import BaseActuator, ActuatorResult
from core.runtime.subprocess_gateway import get_subprocess_gateway


class GitActuator(BaseActuator):
    requires_authority = True

    @property
    def name(self) -> str:
        return "git_operation"

    @property
    def description(self) -> str:
        return "Perform git operations (status, diff, branch, commit, checkout, clone) within the codebase."

    def validate_params(self, params: dict[str, Any]) -> bool:
        if not isinstance(params, dict) or "action" not in params:
            return False
        action = params["action"]
        if action not in ("status", "diff", "branch", "commit", "checkout", "clone", "log"):
            return False
        if action == "clone" and "url" not in params:
            return False
        if action == "commit" and "message" not in params:
            return False
        if action in {"branch", "commit", "checkout"} and not bool(params.get("allow_mutation")):
            return False
        if action == "clone" and not bool(params.get("allow_external_clone")):
            return False
        return True

    def execute(self, params: dict[str, Any]) -> ActuatorResult:
        if not params.get("_aura_authorized"):
            return ActuatorResult(False, "Git operations require ActuatorRegistry/AuthorityGateway authorization.", {})
        if not self.validate_params(params):
            return ActuatorResult(False, "Invalid git action or missing parameters.", {})

        action = params["action"]
        import os
        cwd = params.get("cwd") or os.getcwd()

        cmd = ["git"]
        if action == "clone":
            cmd.extend(["clone", params["url"]])
            if "path" in params:
                cmd.append(params["path"])
        elif action == "commit":
            cmd.extend(["commit", "-m", params["message"]])
        elif action == "checkout":
            cmd.extend(["checkout", params["branch_or_commit"]])
        elif action == "diff":
            cmd.extend(["diff"])
            if "target" in params:
                cmd.append(params["target"])
        elif action == "branch":
            cmd.extend(["branch"])
            if "name" in params:
                cmd.append(params["name"])
        elif action == "status":
            cmd.extend(["status"])
        elif action == "log":
            cmd.extend(["log", "-n", str(params.get("n", 5))])

        try:
            res = get_subprocess_gateway().run(
                cmd,
                cwd=cwd,
                timeout=30.0,
                source="git_actuator",
            )
            success = res.returncode == 0
            msg = f"Git {action} executed with exit code {res.returncode}."
            return ActuatorResult(
                success=success,
                message=msg,
                updates={
                    "exit_code": res.returncode,
                    "stdout": res.stdout,
                    "stderr": res.stderr
                }
            )
        except (subprocess.SubprocessError, OSError, ValueError) as e:
            return ActuatorResult(False, f"Git execution failed: {e}", {})


class PackageInstallActuator(BaseActuator):
    requires_authority = True

    @property
    def name(self) -> str:
        return "package_install"

    @property
    def description(self) -> str:
        return "Install a Python package into the virtual environment using pip."

    def validate_params(self, params: dict[str, Any]) -> bool:
        if not isinstance(params, dict) or "package_name" not in params:
            return False
        pkg = params["package_name"]
        if not isinstance(pkg, str) or not pkg.strip():
            return False
        if not re.match(r"^[a-zA-Z0-9_\-\[\]>=<.,]+$", pkg):
            return False
        if not bool(params.get("allow_install")):
            return False
        return True

    def execute(self, params: dict[str, Any]) -> ActuatorResult:
        if not params.get("_aura_authorized"):
            return ActuatorResult(False, "Package installation requires ActuatorRegistry/AuthorityGateway authorization.", {})
        if not self.validate_params(params):
            return ActuatorResult(False, "Safety validation failed: package name or explicit install approval is invalid.", {})

        pkg = params["package_name"]
        cmd = [sys.executable, "-m", "pip", "install", pkg]

        try:
            res = get_subprocess_gateway().run(
                cmd,
                timeout=120.0,
                source="package_install_actuator",
            )
            success = res.returncode == 0
            msg = f"Package '{pkg}' installation finished with exit code {res.returncode}."
            return ActuatorResult(
                success=success,
                message=msg,
                updates={
                    "exit_code": res.returncode,
                    "stdout": res.stdout,
                    "stderr": res.stderr
                }
            )
        except (subprocess.SubprocessError, OSError, ValueError) as e:
            return ActuatorResult(False, f"Package installation failed: {e}", {})
