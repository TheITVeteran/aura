#!/usr/bin/env python3
"""
Setup DeepSeek R1 Model
Explicitly pulls the required model to fix 404 errors.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime.subprocess_gateway import get_subprocess_gateway  # noqa: E402

MODEL = "deepseek-r1:14b"
_OLLAMA_RECOVERABLE_ERRORS = (OSError, RuntimeError, TimeoutError, ValueError)


def setup():
    print(f"🚀 Setting up DeepSeek R1 ({MODEL})...")
    
    # 1. Check if installed
    try:
        res = get_subprocess_gateway().run(
            ["ollama", "list"],
            cwd=ROOT,
            timeout=30,
            read_only=True,
            source="maintenance_tooling:setup_deepseek_list",
        )
        if MODEL in res.stdout:
            print(f"✅ {MODEL} is already installed.")
            return
    except _OLLAMA_RECOVERABLE_ERRORS as exc:
        print(f"❌ Failed to check ollama list: {type(exc).__name__}: {exc}")
        return

    # 2. Pull if missing
    print(f"📥 Pulling {MODEL}... (This may take a few minutes)")
    try:
        result = get_subprocess_gateway().run(
            ["ollama", "pull", MODEL],
            cwd=ROOT,
            timeout=3600,
            capture_output=False,
            offline_tooling=True,
            source="maintenance_tooling:setup_deepseek_pull",
        )
        if result.returncode == 0:
            print(f"✅ Successfully pulled {MODEL}")
        else:
            print(f"❌ Failed to pull {MODEL}")
            raise SystemExit(1)
            
    except _OLLAMA_RECOVERABLE_ERRORS as exc:
        print(f"❌ Error during pull: {type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc

if __name__ == "__main__":
    setup()
