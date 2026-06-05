"""core/factory/repo_cartographer.py — Repository Analysis and Mapping.

Parses codebases, constructs dependency trees, identifies test suites,
finds weaknesses, and builds a structural map for the software factory.
"""
from __future__ import annotations

import ast
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("Aura.RepoCartographer")


class RepoCartographer:
    """Maps repository structure, dependencies, and quality signals."""

    async def map_repo(self, repo_path: str) -> dict[str, Any]:
        """Produce a structural map of the repository."""
        root = Path(repo_path).resolve()
        if not root.exists():
            return {"error": "repo_not_found", "file_count": 0, "module_count": 0}

        py_files: list[str] = []
        test_files: list[str] = []
        modules: set[str] = set()
        total_lines = 0
        syntax_errors: list[str] = []

        skip_dirs = {".git", ".venv", "__pycache__", "node_modules", ".pytest_cache", ".ruff_cache", "dist", "build"}

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]
            rel_dir = os.path.relpath(dirpath, root)

            for fname in filenames:
                if not fname.endswith(".py"):
                    continue
                rel_path = os.path.join(rel_dir, fname)
                full_path = os.path.join(dirpath, fname)

                py_files.append(rel_path)
                if fname.startswith("test_") or "/tests/" in rel_path:
                    test_files.append(rel_path)

                # Count lines and detect syntax errors
                try:
                    content = Path(full_path).read_text(encoding="utf-8", errors="ignore")
                    total_lines += content.count("\n")
                    ast.parse(content, filename=rel_path)
                except SyntaxError:
                    syntax_errors.append(rel_path)

                # Extract module path
                if rel_dir != ".":
                    modules.add(rel_dir.replace(os.sep, "."))

        return {
            "repo_root": str(root),
            "file_count": len(py_files),
            "test_file_count": len(test_files),
            "module_count": len(modules),
            "modules": sorted(modules),
            "files": py_files,
            "total_lines": total_lines,
            "syntax_errors": syntax_errors,
            "has_tests": len(test_files) > 0,
            "test_coverage_ratio": len(test_files) / max(1, len(py_files)),
        }

    async def find_weaknesses(self, repo_map: dict[str, Any]) -> list[dict[str, Any]]:
        """Identify weaknesses: modules without tests, syntax errors, low coverage."""
        weaknesses = []
        if repo_map.get("syntax_errors"):
            weaknesses.append({
                "type": "syntax_errors",
                "severity": "high",
                "files": repo_map["syntax_errors"],
            })
        if repo_map.get("test_coverage_ratio", 0) < 0.1:
            weaknesses.append({
                "type": "low_test_coverage",
                "severity": "medium",
                "ratio": repo_map.get("test_coverage_ratio", 0),
            })
        return weaknesses
