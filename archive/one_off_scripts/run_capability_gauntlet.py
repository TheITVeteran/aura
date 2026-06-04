#!/usr/bin/env python3
import os
import sys
import time
from pathlib import Path
from subprocess import TimeoutExpired

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime.subprocess_gateway import get_subprocess_gateway  # noqa: E402

GAUNTLET_TESTS = {
    "Closure Gauntlet": "tests/test_audit_chain.py",
    "Activation Audit": "tests/test_causal_exclusion.py",
    "Headless Environment Stress": "tests/test_headless_live_boot.py",
    "Replay Learning Improvement": "tests/test_canary_replay_real.py",
    "Abstraction Transfer": "tests/test_grounding_and_plasticity.py",
    "Self-Mod Rollback Drill": "tests/test_restore_drill.py",
    "Production 32B CAA Validation": "tests/steering/test_caa_32b.py",
    "Long-Run Stability Trace": "tests/test_long_run_model.py",
    "External Task Performance": "tests/test_agent_workspace_integrations.py"
}
_RUN_RECOVERABLE_ERRORS = (OSError, RuntimeError, TimeoutExpired, ValueError)


def run_gauntlet():
    root_dir = ROOT
    os.chdir(root_dir)
    
    results_file = root_dir / "tests" / "CAPABILITY_GAUNTLET_RESULTS.md"
    
    with open(results_file, "w") as f:
        f.write("# Aura Capability Gauntlet Results\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("**Status:** Execution Complete\n\n")
        
        f.write("## Execution Summary\n\n")
        
        for name, path in GAUNTLET_TESTS.items():
            print(f"Running: {name} ({path})...")
            
            start_time = time.time()
            try:
                # We use pytest to run each file. We capture output.
                # Adding -v to get more verbose output but limiting to stdout.
                # If a test fails because of missing dependencies or long runtimes, we still capture it.
                # Find correct pytest path
                pytest_cmd = ["pytest"]
                if (root_dir / ".venv" / "bin" / "pytest").exists():
                    pytest_cmd = [str(root_dir / ".venv" / "bin" / "pytest")]

                result = get_subprocess_gateway().run(
                    pytest_cmd + ["-v", "--tb=short", path],
                    cwd=root_dir,
                    capture_output=True,
                    timeout=300, # 5 minute timeout per suite
                    offline_tooling=True,
                    source="maintenance_tooling:capability_gauntlet",
                )
                duration = time.time() - start_time
                status = "✅ PASS" if result.returncode == 0 else "❌ FAIL"
                
                f.write(f"### {name}\n")
                f.write(f"- **File:** `{path}`\n")
                f.write(f"- **Status:** {status}\n")
                f.write(f"- **Duration:** {duration:.2f}s\n\n")
                
                f.write("```text\n")
                # Truncate output to avoid massive files
                output = result.stdout + "\n" + result.stderr
                if len(output) > 2000:
                    output = output[:1000] + "\n...[TRUNCATED]...\n" + output[-1000:]
                f.write(output)
                f.write("\n```\n\n")
                
                print(f"  -> {status} in {duration:.2f}s")
                
            except TimeoutExpired:
                f.write(f"### {name}\n")
                f.write(f"- **File:** `{path}`\n")
                f.write("- **Status:** ⚠️ TIMEOUT (>300s)\n\n")
                print("  -> ⚠️ TIMEOUT")
            except _RUN_RECOVERABLE_ERRORS as exc:
                f.write(f"### {name}\n")
                f.write(f"- **File:** `{path}`\n")
                f.write(f"- **Status:** 💥 ERROR ({type(exc).__name__}: {exc})\n\n")
                print(f"  -> 💥 ERROR: {type(exc).__name__}: {exc}")

    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    run_gauntlet()
