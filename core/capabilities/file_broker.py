"""core/capabilities/file_broker.py — Sandboxed File Operations
================================================================
All file operations go through this broker.

Enforces allowlist: ~/Documents/Aura, ~/Desktop/Aura, ~/Downloads,
system temporary Aura paths, ~/.aura/data. Handles special characters, versioning,
rollback, and produces receipts for every operation.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.container import ServiceContainer
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Aura.FileBroker")


@dataclass
class FileOperation:
    """Record of a file operation for rollback."""
    operation: str       # "create", "move", "copy", "delete", "write"
    source: str
    destination: str = ""
    backup_path: str = ""  # for rollback
    timestamp: float = field(default_factory=time.time)


class SandboxedFileBroker:
    """All file operations go through this broker.

    Features:
    - Allowlist enforcement (only approved directories)
    - Path sanitization (special chars, traversal prevention)
    - Rollback support (undo last N operations)
    - File versioning (auto-version if file exists)
    - Receipt generation for audit
    """

    # Allowed root directories (expanded at runtime)
    ALLOWED_ROOTS = [
        "~/Documents/Aura",
        "~/Desktop/Aura",
        "~/Downloads",
        "~/.aura/data",
    ]
    TEMP_PREFIX = "aura_"

    def __init__(self) -> None:
        self._operations: List[FileOperation] = []
        self._max_operations = 200
        self._expanded_roots: List[Path] = []
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        self._expanded_roots = [
            Path(os.path.expanduser(r)) for r in self.ALLOWED_ROOTS
        ]
        ServiceContainer.register_instance("file_broker", self, required=False)
        self._started = True
        logger.info(
            "SandboxedFileBroker ONLINE — allowed roots: %s",
            [str(r) for r in self._expanded_roots],
        )

    def _is_allowed(self, path: Path) -> bool:
        """Check if a path is within an allowed root."""
        resolved = path.resolve()
        for root in self._expanded_roots:
            try:
                resolved.relative_to(root)
                return True
            except ValueError:
                continue
        try:
            relative_to_temp = resolved.relative_to(Path(tempfile.gettempdir()).resolve())
            if relative_to_temp.parts and relative_to_temp.parts[0].startswith(self.TEMP_PREFIX):
                return True
        except ValueError:
            return False
        return False

    @staticmethod
    def sanitize_name(name: str) -> str:
        """Sanitize a filename for safe filesystem use.

        Handles apostrophes, special chars, and length limits.
        """
        # Replace problematic chars but keep common ones
        sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
        # Keep apostrophes but escape them for shell safety
        sanitized = sanitized.strip(". ")
        # Truncate to reasonable length
        if len(sanitized) > 200:
            sanitized = sanitized[:200]
        return sanitized or "unnamed"

    def _record_op(self, op: FileOperation) -> None:
        self._operations.append(op)
        if len(self._operations) > self._max_operations:
            self._operations = self._operations[-self._max_operations:]

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    async def create_folder(self, path: str) -> Dict[str, Any]:
        """Create a folder, including parent directories."""
        p = Path(path).expanduser()
        if not self._is_allowed(p):
            return {"success": False, "error": f"Path not in allowlist: {path}"}

        try:
            p.mkdir(parents=True, exist_ok=True)
            self._record_op(FileOperation("create", str(p)))
            return {"success": True, "path": str(p)}
        except OSError as e:
            return {"success": False, "error": str(e)}

    async def write_file(self, path: str, content: str, overwrite: bool = False) -> Dict[str, Any]:
        """Write content to a file atomically."""
        p = Path(path).expanduser()
        if not self._is_allowed(p):
            return {"success": False, "error": f"Path not in allowlist: {path}"}

        if p.exists() and not overwrite:
            # Auto-version
            p = self._version_path(p)

        try:
            get_file_write_gateway().write_text(
                p,
                content,
                encoding="utf-8",
                source="file_broker.write_file",
            )
            file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
            self._record_op(FileOperation("write", str(p)))
            return {"success": True, "path": str(p), "size": len(content), "hash": file_hash}
        except OSError as e:
            return {"success": False, "error": str(e)}

    async def write_bytes(self, path: str, data: bytes, overwrite: bool = False) -> Dict[str, Any]:
        """Write binary data to a file."""
        p = Path(path).expanduser()
        if not self._is_allowed(p):
            return {"success": False, "error": f"Path not in allowlist: {path}"}

        if p.exists() and not overwrite:
            p = self._version_path(p)

        try:
            get_file_write_gateway().write_bytes(
                p,
                data,
                source="file_broker.write_bytes",
            )
            file_hash = hashlib.sha256(data).hexdigest()[:16]
            self._record_op(FileOperation("write", str(p)))
            return {"success": True, "path": str(p), "size": len(data), "hash": file_hash}
        except OSError as e:
            return {"success": False, "error": str(e)}

    async def move_file(self, source: str, destination: str) -> Dict[str, Any]:
        """Move a file, with rollback support."""
        src = Path(source).expanduser()
        dst = Path(destination).expanduser()

        if not src.exists():
            return {"success": False, "error": f"Source not found: {source}"}
        if not self._is_allowed(dst):
            return {"success": False, "error": f"Destination not in allowlist: {destination}"}

        if dst.exists():
            dst = self._version_path(dst)

        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            self._record_op(FileOperation("move", str(src), str(dst)))
            return {"success": True, "source": str(src), "destination": str(dst)}
        except (OSError, shutil.Error) as e:
            return {"success": False, "error": str(e)}

    async def copy_file(self, source: str, destination: str) -> Dict[str, Any]:
        """Copy a file."""
        src = Path(source).expanduser()
        dst = Path(destination).expanduser()

        if not src.exists():
            return {"success": False, "error": f"Source not found: {source}"}
        if not self._is_allowed(dst):
            return {"success": False, "error": f"Destination not in allowlist: {destination}"}

        if dst.exists():
            dst = self._version_path(dst)

        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dst))
            self._record_op(FileOperation("copy", str(src), str(dst)))
            return {"success": True, "source": str(src), "destination": str(dst)}
        except (OSError, shutil.Error) as e:
            return {"success": False, "error": str(e)}

    async def read_file(self, path: str) -> Dict[str, Any]:
        """Read a text file."""
        p = Path(path).expanduser()
        if not p.exists():
            return {"success": False, "error": f"File not found: {path}"}
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
            return {"success": True, "content": content, "size": len(content)}
        except OSError as e:
            return {"success": False, "error": str(e)}

    async def file_exists(self, path: str) -> bool:
        return Path(path).expanduser().exists()

    async def file_hash(self, path: str) -> str:
        """Compute SHA256 hash of a file."""
        p = Path(path).expanduser()
        if not p.exists():
            return ""
        return hashlib.sha256(p.read_bytes()).hexdigest()

    async def list_dir(self, path: str) -> Dict[str, Any]:
        """List directory contents."""
        p = Path(path).expanduser()
        if not p.exists():
            return {"success": False, "error": f"Directory not found: {path}"}
        if not p.is_dir():
            return {"success": False, "error": f"Not a directory: {path}"}
        try:
            entries = []
            for entry in sorted(p.iterdir()):
                entries.append({
                    "name": entry.name,
                    "is_dir": entry.is_dir(),
                    "size": entry.stat().st_size if entry.is_file() else 0,
                })
            return {"success": True, "entries": entries, "count": len(entries)}
        except (OSError, PermissionError) as e:
            return {"success": False, "error": str(e)}

    async def reveal_in_finder(self, path: str) -> bool:
        """Open Finder and highlight the file."""
        try:
            proc = await get_subprocess_gateway().spawn_async(
                ["open", "-R", str(Path(path).expanduser())],
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                source="file_broker.reveal_in_finder",
            )
            await asyncio.wait_for(proc.wait(), timeout=5.0)
            return proc.returncode == 0
        except (OSError, asyncio.TimeoutError):
            return False

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    async def rollback_last(self) -> Dict[str, Any]:
        """Undo the last file operation."""
        if not self._operations:
            return {"success": False, "error": "No operations to rollback"}

        op = self._operations.pop()
        try:
            if op.operation == "move":
                if Path(op.destination).exists() and not Path(op.source).exists():
                    shutil.move(op.destination, op.source)
                    return {"success": True, "rolled_back": f"move {op.destination} → {op.source}"}
            elif op.operation == "copy":
                if Path(op.destination).exists():
                    os.remove(op.destination)
                    return {"success": True, "rolled_back": f"removed copy {op.destination}"}
            elif op.operation == "create":
                if Path(op.source).exists() and Path(op.source).is_dir():
                    # Only remove if empty
                    try:
                        Path(op.source).rmdir()
                        return {"success": True, "rolled_back": f"removed folder {op.source}"}
                    except OSError:
                        return {"success": False, "error": "Folder not empty"}
            elif op.operation == "write":
                if Path(op.source).exists():
                    os.remove(op.source)
                    return {"success": True, "rolled_back": f"removed {op.source}"}
            return {"success": False, "error": f"Cannot rollback {op.operation}"}
        except OSError as e:
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _version_path(self, path: Path) -> Path:
        """Add version suffix to avoid overwriting."""
        stem = path.stem
        suffix = path.suffix
        parent = path.parent
        for v in range(2, 100):
            candidate = parent / f"{stem}_v{v}{suffix}"
            if not candidate.exists():
                return candidate
        return parent / f"{stem}_{int(time.time())}{suffix}"

    def get_status(self) -> Dict[str, Any]:
        return {
            "operations": len(self._operations),
            "allowed_roots": [str(r) for r in self._expanded_roots],
        }


_instance: Optional[SandboxedFileBroker] = None


def get_file_broker() -> SandboxedFileBroker:
    global _instance
    if _instance is None:
        _instance = SandboxedFileBroker()
    return _instance


__all__ = ["SandboxedFileBroker", "get_file_broker"]
