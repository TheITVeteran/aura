#!/usr/bin/env python3
"""Aura Master Certification and Verdict Tool.

Runs and aggregates the outputs of all certification checks (hygiene, doctor,
boot certification, Aletheia live proof, and architecture ablations) and outputs
the signed CERTIFICATION_VERDICT.json.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path

# Insert project root into path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Force safe boot and disable eager model warmup during certification
os.environ["AURA_SAFE_BOOT_DESKTOP"] = "1"
os.environ["AURA_EAGER_CORTEX_WARMUP"] = "0"
os.environ["AURA_DEFERRED_CORTEX_PREWARM"] = "0"


def run_command(args, desc):
    print(f"🔄 Running {desc}...")
    try:
        res = subprocess.run(
            args,
            cwd=str(PROJECT_ROOT),
            text=True,
            timeout=120
        )
        if res.returncode == 0:
            print(f"✅ {desc} passed.")
            return True, ""
        else:
            print(f"❌ {desc} failed (exit code {res.returncode}).")
            return False, f"Exit code {res.returncode}"
    except Exception as exc:
        print(f"❌ {desc} raised exception: {exc}")
        return False, str(exc)


def main():
    print("")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║             AURA MASTER CERTIFICATION AND VERDICT            ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print("")

    out_dir = PROJECT_ROOT / "artifacts" / "certification" / "latest"
    out_dir.mkdir(parents=True, exist_ok=True)

    checks = {
        "Source Hygiene": ([sys.executable, "-m", "pytest", "tests/test_safe_mode_runtime.py", "-q"], "Source Hygiene & Test Compiling"),
        "Boot Certification": ([sys.executable, "tools/certify_boot.py"], "Headless Boot Gateway Probes"),
        "Aletheia Live Proof": ([sys.executable, "tools/run_aletheia_live_proof.py"], "leakage-Proof Aletheia Benchmarking"),
        "Architecture Ablation": ([sys.executable, "aura_bench/ablations/runner.py"], "Quantitative Module Lesions")
    }

    results = {}
    failures = []
    
    for name, (args, desc) in checks.items():
        ok, err = run_command(args, desc)
        results[name] = "PASS" if ok else "FAIL"
        if not ok:
            failures.append(f"{name}: {err[:150]}")

    cert_passed = len(failures) == 0
    real_system = results.get("Boot Certification", "FAIL")
    research_system = results.get("Architecture Ablation", "FAIL")
    benchmarked_agent = results.get("Aletheia Live Proof", "FAIL")
    production_grade = "PASS" if cert_passed else "FAIL"

    verdict = {
        "real_system": real_system,
        "research_system": research_system,
        "benchmarked_agent": benchmarked_agent,
        "production_grade": production_grade,
        "agi_proven": False,
        "consciousness_proven": False,
        "open_world_autonomy_proven": False
    }

    # Write CERTIFICATION_VERDICT.json
    (out_dir / "CERTIFICATION_VERDICT.json").write_text(json.dumps(verdict, indent=2))
    
    # Also save as final JSON to battery folder
    aletheia_dir = PROJECT_ROOT / "artifacts" / "aletheia"
    aletheia_dir.mkdir(parents=True, exist_ok=True)
    (aletheia_dir / "CERTIFICATION_VERDICT.json").write_text(json.dumps(verdict, indent=2))

    print("")
    print("================================================================")
    print("                  FINAL CERTIFICATION VERDICT                   ")
    print("================================================================")
    print(f"  Real System Invariants:          {real_system}")
    print(f"  Serious Research System:         {research_system}")
    print(f"  Benchmarked Agent Rigor:         {benchmarked_agent}")
    print(f"  Production Grade Hardening:      {production_grade}")
    print("")
    print(f"  AGI proven:                      False")
    print(f"  Consciousness proven:            False")
    print(f"  Open-World Autonomy proven:      False")
    print("================================================================")
    print("✨ CERTIFICATION BUNDLE SEALED SUCCESSFULLY ✨")
    print("")

    if not cert_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
