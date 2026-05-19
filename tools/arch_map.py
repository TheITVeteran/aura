#!/usr/bin/env python3
"""Generate Aura's true runtime architecture dependency map.

Outputs:
1. A Mermaid diagram of subsystem dependencies
2. A report of ServiceContainer.get() usage (cross-wiring audit)
3. core/ directory structure analysis (consolidation candidates)
4. record_degradation() usage audit (log-and-limp vs fail-closed)
5. Non-runtime file identification (research/proof artifacts)
"""

import ast
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
SKIP_DIRS = {"__pycache__", ".git", "node_modules", ".venv", "venv"}


def find_python_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def get_subsystem(filepath: Path) -> str:
    """Map a file to its top-level subsystem under core/."""
    try:
        rel = filepath.relative_to(CORE)
        parts = rel.parts
        if len(parts) <= 1:
            return "core_root"
        return parts[0]
    except ValueError:
        return "other"


def analyze_imports(filepath: Path) -> list[str]:
    """Extract all core.X imports from a file."""
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(content, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("core."):
                parts = node.module.split(".")
                if len(parts) >= 2:
                    imports.append(parts[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("core."):
                    parts = alias.name.split(".")
                    if len(parts) >= 2:
                        imports.append(parts[1])
    return imports


def analyze_service_container_usage(filepath: Path) -> list[dict]:
    """Find ServiceContainer.get() calls with their service names."""
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
    except (OSError, UnicodeDecodeError):
        return []

    results = []
    for i, line in enumerate(content.split("\n"), 1):
        match = re.search(r'ServiceContainer\.get\(["\'](\w+)["\']', line)
        if match:
            results.append({
                "service": match.group(1),
                "file": str(filepath.relative_to(ROOT)),
                "line": i,
            })
        match2 = re.search(r'ServiceContainer\.register[_instance]*\(["\'](\w+)["\']', line)
        if match2:
            results.append({
                "service": match2.group(1),
                "file": str(filepath.relative_to(ROOT)),
                "line": i,
                "type": "register",
            })
    return results


def analyze_degradation_usage(filepath: Path) -> list[dict]:
    """Find record_degradation() calls and check if they're log-and-limp."""
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
    except (OSError, UnicodeDecodeError):
        return []

    results = []
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if "record_degradation(" in line:
            # Check surrounding context for re-raise or return
            context_after = "\n".join(lines[i:i+5])
            is_limp = not ("raise" in context_after or "return" in context_after.split("\n")[1] if len(context_after.split("\n")) > 1 else True)
            results.append({
                "file": str(filepath.relative_to(ROOT)),
                "line": i + 1,
                "limp_on": is_limp,
                "context": line.strip(),
            })
    return results


def identify_non_runtime(filepath: Path) -> bool:
    """Check if a file is research/proof/narrative rather than runtime."""
    try:
        rel = str(filepath.relative_to(ROOT))
    except ValueError:
        return False
    
    name = filepath.name.lower()
    content = filepath.read_text(encoding="utf-8", errors="ignore")[:500].lower()
    
    indicators = 0
    if "proof" in name or "narrative" in name or "scoping" in name:
        indicators += 2
    if "research" in name or "experiment" in name:
        indicators += 2
    if "validation" in content and "production" not in content:
        indicators += 1
    if "proof of concept" in content or "poc" in content:
        indicators += 2
    if filepath.stat().st_size < 500 and "class " not in content and "def " not in content:
        indicators += 1
    
    return indicators >= 2


def main():
    all_files = find_python_files(CORE)
    skills_files = find_python_files(ROOT / "skills")
    
    # ─── 1. SUBSYSTEM DEPENDENCY MAP ───
    subsystem_deps = defaultdict(set)
    subsystem_files = defaultdict(list)
    
    for f in all_files:
        sub = get_subsystem(f)
        subsystem_files[sub].append(f)
        for imp in analyze_imports(f):
            if imp != sub:  # Skip self-imports
                subsystem_deps[sub].add(imp)
    
    # ─── 2. SERVICE CONTAINER AUDIT ───
    sc_gets = []
    sc_registers = []
    for f in all_files + skills_files:
        for usage in analyze_service_container_usage(f):
            if usage.get("type") == "register":
                sc_registers.append(usage)
            else:
                sc_gets.append(usage)
    
    # ─── 3. DEGRADATION AUDIT ───
    degradation_calls = []
    for f in all_files + skills_files:
        degradation_calls.extend(analyze_degradation_usage(f))
    
    # ─── 4. NON-RUNTIME FILES ───
    non_runtime = []
    for f in all_files:
        if identify_non_runtime(f):
            non_runtime.append(str(f.relative_to(ROOT)))
    
    # ─── 5. CORE DIRECTORY STRUCTURE ───
    dir_stats = {}
    for sub in sorted(subsystem_files.keys()):
        files = subsystem_files[sub]
        total_lines = 0
        total_bytes = 0
        for f in files:
            total_bytes += f.stat().st_size
            try:
                total_lines += len(f.read_text(encoding="utf-8", errors="ignore").split("\n"))
            except OSError:
                pass
        dir_stats[sub] = {
            "files": len(files),
            "lines": total_lines,
            "bytes": total_bytes,
            "deps_out": len(subsystem_deps[sub]),
            "deps_in": sum(1 for other in subsystem_deps if sub in subsystem_deps[other]),
        }
    
    # ─── OUTPUT ───
    print("=" * 80)
    print("AURA ARCHITECTURE DEPENDENCY MAP")
    print("=" * 80)
    
    # Mermaid diagram
    print("\n## Subsystem Dependency Graph (Mermaid)\n")
    print("```mermaid")
    print("graph TD")
    
    # Sort by importance (deps_in)
    sorted_subs = sorted(dir_stats.keys(), key=lambda s: dir_stats[s]["deps_in"], reverse=True)
    
    for sub in sorted_subs:
        stats = dir_stats[sub]
        print(f'    {sub}["{sub}<br/>{stats["files"]} files, {stats["lines"]} lines"]')
    
    for sub in sorted_subs:
        for dep in sorted(subsystem_deps[sub]):
            if dep in dir_stats:
                print(f"    {sub} --> {dep}")
    
    print("```")
    
    # Directory stats
    print("\n## Core Subsystem Stats\n")
    print(f"{'Subsystem':<30} {'Files':>6} {'Lines':>8} {'Bytes':>10} {'Deps Out':>9} {'Deps In':>8}")
    print("-" * 80)
    for sub in sorted(dir_stats.keys(), key=lambda s: dir_stats[s]["lines"], reverse=True):
        s = dir_stats[sub]
        print(f"{sub:<30} {s['files']:>6} {s['lines']:>8} {s['bytes']:>10} {s['deps_out']:>9} {s['deps_in']:>8}")
    
    print(f"\nTotal subsystems: {len(dir_stats)}")
    print(f"Total files: {sum(s['files'] for s in dir_stats.values())}")
    print(f"Total lines: {sum(s['lines'] for s in dir_stats.values())}")
    
    # ServiceContainer audit
    service_get_counts = Counter(u["service"] for u in sc_gets)
    service_reg_counts = Counter(u["service"] for u in sc_registers)
    
    print("\n## ServiceContainer Cross-Wiring Audit\n")
    print(f"Total .get() calls: {len(sc_gets)}")
    print(f"Total .register() calls: {len(sc_registers)}")
    print(f"Unique services retrieved: {len(service_get_counts)}")
    print(f"Unique services registered: {len(service_reg_counts)}")
    
    # Services retrieved but never registered (potential missing registrations)
    missing = set(service_get_counts.keys()) - set(service_reg_counts.keys())
    if missing:
        print(f"\n⚠️  Services GET'd but never REGISTER'd ({len(missing)}):")
        for s in sorted(missing):
            print(f"    - {s} (get'd {service_get_counts[s]}x)")
    
    # Top cross-wired services
    print(f"\nTop 20 most-fetched services:")
    for svc, count in service_get_counts.most_common(20):
        reg_count = service_reg_counts.get(svc, 0)
        print(f"    {svc:<40} get={count:>3}  register={reg_count}")
    
    # Degradation audit
    limp_count = sum(1 for d in degradation_calls if d["limp_on"])
    fail_count = sum(1 for d in degradation_calls if not d["limp_on"])
    print(f"\n## record_degradation() Audit\n")
    print(f"Total calls: {len(degradation_calls)}")
    print(f"  Log-and-limp (no raise/return after): {limp_count}")
    print(f"  Fail-closed (raise/return follows): {fail_count}")
    
    if limp_count > 0:
        print(f"\n  Top 10 limp-on files:")
        limp_by_file = Counter(d["file"] for d in degradation_calls if d["limp_on"])
        for fname, count in limp_by_file.most_common(10):
            print(f"    {fname}: {count}")
    
    # Non-runtime
    if non_runtime:
        print(f"\n## Non-Runtime Files ({len(non_runtime)})\n")
        for f in non_runtime:
            print(f"    {f}")
    
    # Consolidation candidates (subsystems with < 3 files)
    small_subs = {sub: stats for sub, stats in dir_stats.items() if stats["files"] <= 2 and sub != "core_root"}
    if small_subs:
        print(f"\n## Consolidation Candidates ({len(small_subs)} small subsystems with ≤2 files)\n")
        for sub in sorted(small_subs.keys()):
            s = small_subs[sub]
            print(f"    {sub}/: {s['files']} files, {s['lines']} lines")


if __name__ == "__main__":
    main()
