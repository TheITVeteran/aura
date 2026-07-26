"""`--reboot` must replace the running instance, not run beside it.

`aura_cleanup.py` refuses to touch a VERIFIED live runtime — correctly, because
an unasked-for launch must never kill a healthy instance someone is using. The
only override is `AURA_CLEANUP_FORCE`, and nothing set it.

So `./launch_aura.sh --reboot`, whose documented purpose is "Replace an existing
Aura runtime", logged

    Verified live Aura runtime detected (PID: 19849); skipping aggressive
    pre-launch process cleanup.

and then started a SECOND desktop runtime beside the first: two 32B models at
roughly 20GB each on a 64GB host. That is the duplicate-runtime memory cascade,
arriving through the one flag whose entire job was to prevent it. Observed
repeatedly while rebooting onto fixes, 2026-07-26.

Explicitly asking to reboot IS the authorization the guard was waiting for.
"""
from __future__ import annotations

import re
from pathlib import Path

LAUNCHER = Path("launch_aura.sh")
CLEANUP = Path("scripts/one_off/aura_cleanup.py")


def _launcher() -> str:
    return LAUNCHER.read_text(encoding="utf-8")


def test_reboot_authorizes_the_cleanup() -> None:
    src = _launcher()
    assert re.search(
        r'if \[ "\$REBOOT_MODE" = "1" \]; then\s*\n\s*export AURA_CLEANUP_FORCE=1',
        src,
    ), "--reboot must set the force flag the cleanup checks"


def test_the_authorization_precedes_the_cleanup_run() -> None:
    """Exporting it after cleanup has already run would change nothing."""
    src = _launcher()
    assert src.index("export AURA_CLEANUP_FORCE=1") < src.index(
        '"$PYTHON_CMD" aura_cleanup.py'
    )


def test_a_plain_launch_still_protects_a_live_runtime() -> None:
    """The guard must remain: only an explicit reboot may replace an instance."""
    src = _launcher()
    force_lines = [
        line for line in src.splitlines() if "AURA_CLEANUP_FORCE" in line and "export" in line
    ]
    assert len(force_lines) == 1, "exactly one place may authorize replacement"
    # …and it is inside the reboot branch, not unconditional.
    guard = src[src.index('if [ "$REBOOT_MODE" = "1" ]') : src.index("export AURA_CLEANUP_FORCE=1")]
    assert "REBOOT_MODE" in guard


def test_the_cleanup_still_honours_the_force_flag() -> None:
    """The launcher's half is useless if the cleanup stops reading it."""
    cleanup = CLEANUP.read_text(encoding="utf-8")
    assert '_truthy_env("AURA_CLEANUP_FORCE")' in cleanup
    branch = cleanup[cleanup.index("def _kill_stale_processes") :][:400]
    assert "AURA_CLEANUP_FORCE" in branch, (
        "the force flag must short-circuit the live-runtime guard"
    )
