#!/usr/bin/env python3
"""Audit record_degradation() calls to identify log-and-limp anti-patterns.

Classifies each call as:
- CRITICAL: In a path that MUST succeed (model loading, boot, response generation)
- ADVISORY: In an optional integration (background tasks, telemetry, metrics)

For CRITICAL calls, checks if the error is properly handled (raise/return/abort).
"""

import ast
import re
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {"__pycache__", ".git", ".venv", "venv"}

# Paths considered critical — failures here break the user experience
CRITICAL_MODULES = {
    "inference_gate", "response_generation", "unitary",
    "mlx_client", "mlx_worker", "cognitive_engine",
    "orchestrator", "kernel", "boot",
    "state_machine", "conversation_support",
}

# Functions considered critical
CRITICAL_FUNCTIONS = {
    "generate", "think", "route", "process_message",
    "handle_incoming", "execute", "run", "start",
    "load_model", "spawn_worker",
}


def find_python_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def analyze_file(filepath: Path) -> list[dict]:
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
    except (OSError, UnicodeDecodeError):
        return []

    if "record_degradation" not in content:
        return []

    results = []
    lines = content.split("\n")
    
    # Determine if this is a critical module
    rel = str(filepath.relative_to(ROOT))
    is_critical_module = any(cm in rel.lower() for cm in CRITICAL_MODULES)
    
    for i, line in enumerate(lines):
        if "record_degradation(" not in line:
            continue
        
        # Check if next 3 lines have raise/return/break
        has_failclose = False
        for j in range(i + 1, min(i + 4, len(lines))):
            check = lines[j].strip()
            if check.startswith("raise") or check.startswith("return") or check == "break":
                has_failclose = True
                break
        
        # Check enclosing function name
        func_name = "unknown"
        for j in range(i, -1, -1):
            match = re.match(r'\s+(?:async\s+)?def\s+(\w+)', lines[j])
            if match:
                func_name = match.group(1)
                break
        
        is_critical_func = any(cf in func_name.lower() for cf in CRITICAL_FUNCTIONS)
        
        severity = "CRITICAL" if (is_critical_module and is_critical_func) else (
            "HIGH" if is_critical_module or is_critical_func else "ADVISORY"
        )
        
        if severity in ("CRITICAL", "HIGH") and not has_failclose:
            results.append({
                "file": rel,
                "line": i + 1,
                "function": func_name,
                "severity": severity,
                "has_failclose": has_failclose,
                "code": line.strip()[:100],
            })
    
    return results


def main():
    all_files = find_python_files(ROOT / "core")
    all_files.extend(find_python_files(ROOT / "skills"))
    
    issues = []
    for f in all_files:
        issues.extend(analyze_file(f))
    
    critical = [i for i in issues if i["severity"] == "CRITICAL"]
    high = [i for i in issues if i["severity"] == "HIGH"]
    
    print(f"Total CRITICAL log-and-limp calls: {len(critical)}")
    print(f"Total HIGH log-and-limp calls: {len(high)}")
    
    if critical:
        print(f"\n{'='*80}")
        print("CRITICAL: These fail silently in user-facing code paths")
        print(f"{'='*80}\n")
        for issue in sorted(critical, key=lambda x: x["file"]):
            print(f"  {issue['file']}:{issue['line']} in {issue['function']}()")
            print(f"    {issue['code']}")
    
    if high:
        print(f"\n{'='*80}")
        print(f"HIGH: These fail silently in important modules ({len(high)} total)")
        print(f"{'='*80}\n")
        
        by_file = Counter(i["file"] for i in high)
        for fname, count in by_file.most_common(15):
            print(f"  {fname}: {count} calls")
            file_issues = [i for i in high if i["file"] == fname]
            for issue in file_issues[:3]:
                print(f"    L{issue['line']} {issue['function']}(): {issue['code'][:80]}")


if __name__ == "__main__":
    main()
