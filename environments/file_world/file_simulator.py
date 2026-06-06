"""environments/file_world/file_simulator.py
Simulated file world environment with virtual filesystem capabilities.
"""
from typing import Dict, Any, List


class VirtualFileWorld:
    """Simulates a secure virtual file sandbox for testing file actuators."""

    def __init__(self):
        # Maps virtual_path -> content
        self._files: Dict[str, str] = {
            "root/readme.txt": "Welcome to virtual file sandbox.",
            "root/src/index.py": "print('hello')"
        }

    def write_virtual_file(self, path: str, content: str) -> None:
        self._files[path] = content

    def read_virtual_file(self, path: str) -> str:
        return self._files.get(path, "")

    def delete_virtual_file(self, path: str) -> bool:
        if path in self._files:
            del self._files[path]
            return True
        return False

    def list_virtual_files(self) -> List[str]:
        return list(self._files.keys())
