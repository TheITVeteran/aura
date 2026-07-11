from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dispatcher_cancellation_cannot_pin_asyncio_runner() -> None:
    script = """
import asyncio
from core.state.state_registry import UnifiedStateRegistry

async def main() -> None:
    registry = UnifiedStateRegistry()
    task = asyncio.create_task(registry._notification_dispatcher())
    await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

asyncio.run(main())
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=3.0,
        check=False,
    )

    assert result.returncode == 0, result.stderr
