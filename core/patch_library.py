import logging
import re
import subprocess
import sys
from abc import ABC, abstractmethod

from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Optimizer.PatchLibrary")

class PatchStrategy(ABC):
    name = "base_patch"
    
    def match(self, failure_reason: str) -> bool:
        return False
        
    @abstractmethod
    async def apply(self, failure_reason: str) -> bool:
        """Applies the fix. Returns True if successful, False otherwise."""
        raise NotImplementedError

class GitInitPatch(PatchStrategy):
    name = "git_init_fix"
    
    def match(self, failure_reason: str) -> bool:
        return "not a git repository" in failure_reason.lower()
        
    async def apply(self, failure_reason: str) -> bool:
        logger.warning("⚙️ Autonomic Core engaging 'git init' self-repair...")
        try:
            await get_subprocess_gateway().run_async(
                ["git", "init"],
                check=True,
                capture_output=True,
                offline_tooling=True,
                source="maintenance_tooling:patch_library.git_init",
            )
            await get_subprocess_gateway().run_async(
                ["git", "add", "."],
                check=True,
                capture_output=True,
                offline_tooling=True,
                source="maintenance_tooling:patch_library.git_add",
            )
            await get_subprocess_gateway().run_async(
                ["git", "commit", "-m", "Auto-Healer: Re-init corrupted repository"],
                check=True,
                capture_output=True,
                offline_tooling=True,
                source="maintenance_tooling:patch_library.git_commit",
            )
            logger.info("✅ Autonomic Core successfully repaired local Git repository.")
            return True
        except subprocess.CalledProcessError as e:
            logger.error("❌ Git repair failed: %s", e.stderr.decode() if e.stderr else e)
            return False

class PipInstallPatch(PatchStrategy):
    name = "pip_install_fix"
    
    def match(self, failure_reason: str) -> bool:
        return "modulenotfounderror" in failure_reason.lower()
        
    async def apply(self, failure_reason: str) -> bool:
        # Extract module name matches "No module named 'xyz'"
        match = re.search(r"No module named '(\w+)'", failure_reason)
        if match:
            module = match.group(1)
            
            import_to_pip: dict[str, str] = {
                "aiohttp": "aiohttp",
                "google": "google-generativeai",
                "pydantic": "pydantic",
                "structlog": "structlog",
                "psutil": "psutil",
                "webrtcvad": "webrtcvad",
                "pyaudio": "PyAudio",
                "numpy": "numpy",
            }
            
            if module not in import_to_pip:
                logger.error("🛑 SECURITY: Blocked autonomous installation of '%s'", module)
                return False
            
            pip_package = import_to_pip[module]
            logger.warning("⚙️ Autonomic Core attempting to install missing module: %s (as %s)", module, pip_package)
            try:
                await get_subprocess_gateway().run_async(
                    [sys.executable, "-m", "pip", "install", pip_package],
                    check=True,
                    capture_output=True,
                    offline_tooling=True,
                    source="maintenance_tooling:patch_library.pip_install",
                )
                logger.info("✅ Autonomic Core successfully installed missing package '%s'", pip_package)
                return True
            except subprocess.CalledProcessError as e:
                logger.error("❌ Autonomic Core failed to install '%s': %s", pip_package, e.stderr.decode() if e.stderr else e)
                return False
        return False

# Registry
def get_patches() -> list[PatchStrategy]:
    return [GitInitPatch(), PipInstallPatch()]
