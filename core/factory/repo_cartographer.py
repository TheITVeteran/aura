"""core/factory/repo_cartographer.py — Repository Dependency Cartographer.
"""
from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Dict, List, Set

logger = logging.getLogger("Aura.RepoCartographer")


class RepoCartographer:
    """Parses codebase code structures, imports, and builds dependency trees."""

    def __init__(self, root_dir: str = ".") -> None:
        self.root_dir = Path(root_dir).resolve()
        self.dependencies: Dict[str, Set[str]] = {}

    def map_repository(self) -> Dict[str, Any]:
        """Scans the repository folder, parsing Python files for import chains."""
        logger.info("🏭 Cartographer scanning codebase at: %s", self.root_dir)
        py_files = []
        # Simple walk to find files (excluding .venv, build, etc.)
        for path in self.root_dir.rglob("*.py"):
            if not any(part in path.parts for part in (".venv", ".git", "build", "dist")):
                py_files.append(path)

        for filepath in py_files:
            rel_path = filepath.relative_to(self.root_dir).as_posix()
            self.dependencies[rel_path] = self._extract_imports(filepath)

        logger.info("🏭 Indexed %d codebase files and imports.", len(py_files))
        return {
            "files_count": len(py_files),
            "files": list(self.dependencies.keys()),
        }

    def _extract_imports(self, filepath: Path) -> Set[str]:
        imports = set()
        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(content, filename=filepath.name)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imports.add(name.name)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module)
        except Exception as e:
            logger.debug("Failed parsing imports for %s: %s", filepath, e)
        return imports
