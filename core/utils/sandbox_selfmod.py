# core/utils/sandbox_selfmod.py
"""
Safe Self-Modification Sandbox:
- create ephemeral worktree (git)
- apply provided patch (unified diff)
- run tests (timeout)
- return result, logs, and optionally a patch verdict
"""

from core.runtime.errors import record_degradation
import tempfile
import os
import shutil
import logging
import time
from subprocess import TimeoutExpired
from typing import Dict, Any, Optional

logger = logging.getLogger("aura.sandbox_selfmod")


def _run_cmd(cmd, cwd=None, timeout=60):
    logger.debug("Running command: %s in %s", cmd, cwd)
    import shlex
    cmd_list = shlex.split(cmd) if isinstance(cmd, str) else cmd
    try:
        from core.runtime.subprocess_gateway import get_subprocess_gateway

        proc = get_subprocess_gateway().run(
            cmd_list,
            cwd=cwd,
            timeout=timeout,
            capture_output=True,
            source="sandbox_selfmod.run_cmd",
        )
        return proc.returncode, proc.stdout, proc.stderr
    except TimeoutExpired:
        raise RuntimeError(f"Command timeout: {cmd} (cwd={cwd})")


def test_patch_in_sandbox(repo_root: str, patch_text: str, test_cmd: str = "pytest -q", timeout: int = 120) -> Dict[str, Any]:
    """
    repo_root: path to repo root (where .git exists)
    patch_text: unified diff to apply (git apply compatible)
    test_cmd: command to run tests in sandbox
    returns: dict {ok: bool, rc, out, err, sandbox_path}
    """
    if not os.path.isdir(os.path.join(repo_root, ".git")):
        raise RuntimeError("Not a git repo root: " + repo_root)

    tmpdir = tempfile.mkdtemp(prefix="aura-sandbox-")
    logger.info("Creating sandbox worktree at %s", tmpdir)

    try:
        # create a new worktree branch
        branch = f"sandbox-{int(time.time())}"
        # 1) clone minimal repo via git worktree
        rc, out, err = _run_cmd(f"git worktree add {tmpdir} -b {branch}", cwd=repo_root, timeout=30)
        if rc != 0:
            raise RuntimeError(f"git worktree failed: {err}")

        # 2) apply patch
        patch_file = os.path.join(tmpdir, "patch.diff")
        from core.runtime.file_write_gateway import get_file_write_gateway

        get_file_write_gateway().write_text(
            patch_file,
            patch_text,
            encoding="utf-8",
            source="sandbox_selfmod.write_patch",
        )

        rc, out, err = _run_cmd(f"git apply {patch_file}", cwd=tmpdir, timeout=30)
        if rc != 0:
            # gather apply error and return
            logger.error("Patch apply failed: %s", err)
            return {"ok": False, "stage": "apply", "rc": rc, "out": out, "err": err, "sandbox": tmpdir}

        # 3) run tests
        rc, out, err = _run_cmd(test_cmd, cwd=tmpdir, timeout=timeout)
        ok = rc == 0
        return {"ok": ok, "stage": "test", "rc": rc, "out": out, "err": err, "sandbox": tmpdir}
    except (OSError, IOError) as e:
        record_degradation('sandbox_selfmod', e)
        logger.exception("Sandbox exception")
        return {"ok": False, "stage": "exception", "err": str(e), "sandbox": tmpdir}
    finally:
        # cleanup: remove worktree and branch safely
        try:
            _run_cmd(f"git worktree remove {tmpdir} --force", cwd=repo_root, timeout=10)
        except (RuntimeError, AttributeError, TypeError, ValueError) as _e:
            record_degradation('sandbox_selfmod', _e)
            logger.debug('Ignored Exception in sandbox_selfmod.py: %s', _e)
        # Remove any remaining dir
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)
