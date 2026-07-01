"""Model download + lifecycle manager.

Gives a launcher/UI (and boot) a single, observable view of Aura's local model
set: which role models are present, where they live, what's missing, how much
disk a fresh fetch needs, and a safe, resumable, integrity-checked download
orchestration. Previously the only path was ``scripts/fetch_models.py`` (a
print-only CLI) plus an implicit "MLX auto-downloads a repo id on first load"
fallback — neither is queryable, so a fresh install would silently stall on the
first generation while a multi-GB model downloaded with no surfaced progress.

Design:
  - Presence/source resolution reuses the registry's own ``get_model_path`` so
    we never diverge from what the runtime actually loads. A model is "present"
    if the resolved artifact exists OR the canonical ``models/<name>`` base dir
    is populated (the latter covers the fine-tuned/fused active model, whose
    resolved path differs from the downloadable base).
  - The downloader is injected, so orchestration (missing-set, bounded retries,
    re-verification, progress, status) is fully unit-testable without network.
    The default downloader wraps ``huggingface_hub.snapshot_download`` (which
    resumes partial downloads by default).
"""
from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("Aura.ModelLifecycle")

_LIFECYCLE_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    KeyError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)

# Canonical HF repos for the downloadable base models (incl. the -Q4 alias the
# deep solver uses). Source of truth for "where does a missing model come from".
DEFAULT_REPO_MAP: dict[str, str] = {
    "Qwen2.5-1.5B-Instruct-4bit": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "Qwen2.5-7B-Instruct-4bit": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "Qwen2.5-14B-Instruct-4bit": "mlx-community/Qwen2.5-14B-Instruct-4bit",
    "Qwen2.5-32B-Instruct-4bit": "mlx-community/Qwen2.5-32B-Instruct-4bit",
    "Qwen2.5-32B-Instruct-8bit": "mlx-community/Qwen2.5-32B-Instruct-8bit",
    "Qwen2.5-72B-Instruct-4bit": "mlx-community/Qwen2.5-72B-Instruct-4bit",
    "Qwen2.5-72B-Instruct-Q4": "mlx-community/Qwen2.5-72B-Instruct-4bit",
    "QwQ-32B-4bit": "mlx-community/QwQ-32B-4bit",
    "DeepSeek-R1-Distill-Qwen-32B-4bit": "mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit",
    "DeepSeek-R1-Distill-Qwen-32B-8bit": "mlx-community/DeepSeek-R1-Distill-Qwen-32B-MLX-8Bit",
}

# Rough on-disk sizes (bytes) for disk preflight. Clearly an ESTIMATE — the real
# size is known only after the HF metadata fetch; this is for a pre-download
# "do you have room?" check, not an exact figure.
_APPROX_SIZE_GB: dict[str, float] = {
    "Qwen2.5-1.5B-Instruct-4bit": 1.0,
    "Qwen2.5-7B-Instruct-4bit": 4.5,
    "Qwen2.5-14B-Instruct-4bit": 8.5,
    "Qwen2.5-32B-Instruct-4bit": 18.0,
    "Qwen2.5-32B-Instruct-8bit": 35.0,
    "Qwen2.5-72B-Instruct-4bit": 41.0,
    "Qwen2.5-72B-Instruct-Q4": 41.0,
    "QwQ-32B-4bit": 18.0,
    "DeepSeek-R1-Distill-Qwen-32B-4bit": 18.0,
    "DeepSeek-R1-Distill-Qwen-32B-8bit": 35.0,
}

_GB = float(1024**3)


@dataclass
class ModelStatus:
    role: str
    name: str
    present: bool
    location: str
    source_repo: str | None
    size_bytes: int = 0
    approx_download_gb: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "name": self.name,
            "present": self.present,
            "location": self.location,
            "source_repo": self.source_repo,
            "size_bytes": self.size_bytes,
            "approx_download_gb": self.approx_download_gb,
        }


@dataclass
class DiskPreflight:
    target: str
    free_bytes: int
    required_bytes: int

    @property
    def ok(self) -> bool:
        # Require the estimate plus a 5GB working margin.
        return self.free_bytes >= self.required_bytes + int(5 * _GB)

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "free_gb": round(self.free_bytes / _GB, 1),
            "required_gb": round(self.required_bytes / _GB, 1),
            "ok": self.ok,
        }


def _dir_is_populated(path: Path) -> bool:
    try:
        return path.is_dir() and any(path.iterdir())
    except OSError:
        return False


def _dir_size_bytes(path: Path, *, cap_entries: int = 100_000) -> int:
    total = 0
    seen = 0
    try:
        for root, _dirs, files in os.walk(path):
            for fname in files:
                seen += 1
                if seen > cap_entries:
                    return total
                try:
                    total += (Path(root) / fname).stat().st_size
                except OSError:
                    continue
    except OSError:
        return total
    return total


class ModelLifecycleManager:
    """Inventory, disk preflight, and resumable download of Aura's model set."""

    def __init__(
        self,
        *,
        plan: dict[str, str] | None = None,
        resolver: Callable[[str], str] | None = None,
        repo_map: dict[str, str] | None = None,
        base_dir: Path | str | None = None,
    ):
        self._plan = plan or self._default_plan()
        self._resolver = resolver or self._default_resolver()
        self._repo_map = dict(repo_map or DEFAULT_REPO_MAP)
        self._base_dir = Path(base_dir) if base_dir is not None else self._default_base_dir()
        self._models_dir = self._base_dir / "models"

    # ---- defaults pulled lazily from the registry (kept injectable for tests) ----

    @staticmethod
    def _default_plan() -> dict[str, str]:
        from core.brain.llm import model_registry as r

        return {
            "fallback": r.FALLBACK_MODEL,
            "brainstem": r.BRAINSTEM_MODEL,
            "cortex": r.ACTIVE_MODEL,
            "solver": r.DEEP_MODEL,
        }

    @staticmethod
    def _default_resolver() -> Callable[[str], str]:
        from core.brain.llm.model_registry import get_model_path

        return get_model_path

    @staticmethod
    def _default_base_dir() -> Path:
        from core.brain.llm import model_registry as r

        return Path(getattr(r, "BASE_DIR", Path(__file__).resolve().parents[3]))

    # ---- inventory -----------------------------------------------------------

    def _resolve_repo(self, name: str, resolved: str, present_resolved: bool) -> str | None:
        if present_resolved:
            return None
        mapped = self._repo_map.get(name)
        if mapped:
            return mapped
        # get_model_path returns the HF repo id (non-absolute) when a model is
        # missing and has a known fallback.
        return resolved if not Path(resolved).is_absolute() else None

    def status_for(self, role: str, name: str) -> ModelStatus:
        resolved = str(self._resolver(name))
        resolved_path = Path(resolved)
        present_resolved = (
            resolved_path.is_absolute()
            and resolved_path.exists()
            and _dir_is_populated(resolved_path)
        )
        base_path = self._models_dir / name
        present_base = _dir_is_populated(base_path)
        present = present_resolved or present_base

        if present_resolved:
            location = str(resolved_path)
        elif present_base:
            location = str(base_path)
        else:
            location = str(base_path)  # where a fetch would place it

        size = 0
        if present_resolved:
            size = _dir_size_bytes(resolved_path)
        elif present_base:
            size = _dir_size_bytes(base_path)

        return ModelStatus(
            role=role,
            name=name,
            present=present,
            location=location,
            source_repo=self._resolve_repo(name, resolved, present_resolved),
            size_bytes=size,
            approx_download_gb=_APPROX_SIZE_GB.get(name, 0.0),
        )

    def inventory(self) -> list[ModelStatus]:
        return [self.status_for(role, name) for role, name in self._plan.items()]

    def missing(self) -> list[ModelStatus]:
        """Models that are absent AND have a known download source."""
        return [s for s in self.inventory() if not s.present and s.source_repo]

    def all_present(self) -> bool:
        return all(s.present for s in self.inventory())

    def active_model_present(self) -> bool:
        name = self._plan.get("cortex")
        if not name:
            return True
        return self.status_for("cortex", name).present

    # ---- disk preflight ------------------------------------------------------

    def estimated_download_bytes(self, statuses: list[ModelStatus] | None = None) -> int:
        statuses = statuses if statuses is not None else self.missing()
        return int(sum(s.approx_download_gb for s in statuses) * _GB)

    def disk_preflight(self, required_bytes: int | None = None) -> DiskPreflight:
        required = (
            required_bytes if required_bytes is not None else self.estimated_download_bytes()
        )
        # Check the volume that will actually hold the models.
        probe = self._models_dir
        anchor = probe
        while not anchor.exists() and anchor != anchor.parent:
            anchor = anchor.parent
        try:
            free = shutil.disk_usage(anchor).free
        except OSError:
            free = 0
        return DiskPreflight(target=str(probe), free_bytes=int(free), required_bytes=int(required))

    # ---- download orchestration ---------------------------------------------

    def _default_downloader(self) -> Callable[[str, str], None]:
        def _download(repo: str, target_dir: str) -> None:
            from huggingface_hub import snapshot_download

            Path(target_dir).mkdir(parents=True, exist_ok=True)
            # snapshot_download resumes partial files by default.
            snapshot_download(repo_id=repo, local_dir=target_dir)

        return _download

    def ensure_present(
        self,
        *,
        downloader: Callable[[str, str], None] | None = None,
        progress: Callable[[dict[str, Any]], None] | None = None,
        retries: int = 2,
        check_disk: bool = True,
    ) -> dict[str, Any]:
        """Download every missing model with a known source. Bounded + resumable.

        Returns a structured report. ``downloader(repo, target_dir)`` is injected
        for tests; the default uses huggingface_hub. ``progress`` receives one
        event dict per model lifecycle step.
        """
        download = downloader or self._default_downloader()
        missing = self.missing()
        report: dict[str, Any] = {
            "requested": [s.name for s in missing],
            "downloaded": [],
            "failed": [],
            "skipped_disk": False,
        }

        if not missing:
            self._emit(progress, {"event": "all_present"})
            return report

        if check_disk:
            pf = self.disk_preflight(self.estimated_download_bytes(missing))
            if not pf.ok:
                report["skipped_disk"] = True
                report["disk"] = pf.as_dict()
                logger.error(
                    "Insufficient disk for model fetch: need ~%.1fGB, free %.1fGB at %s",
                    pf.required_bytes / _GB,
                    pf.free_bytes / _GB,
                    pf.target,
                )
                self._emit(progress, {"event": "disk_insufficient", **pf.as_dict()})
                return report

        for status in missing:
            target_dir = str(self._models_dir / status.name)
            ok = False
            last_error = ""
            for attempt in range(1, max(1, retries) + 1):
                self._emit(
                    progress,
                    {
                        "event": "download_start",
                        "name": status.name,
                        "repo": status.source_repo,
                        "attempt": attempt,
                    },
                )
                try:
                    download(status.source_repo or "", target_dir)
                    # Verify the artifact actually landed before declaring success.
                    if _dir_is_populated(Path(target_dir)) or self.status_for(
                        status.role, status.name
                    ).present:
                        ok = True
                        break
                    last_error = "downloader returned but target is empty"
                except _LIFECYCLE_RECOVERABLE_ERRORS as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    logger.warning(
                        "Model download attempt %d/%d failed for %s: %s",
                        attempt,
                        retries,
                        status.name,
                        last_error,
                    )
            if ok:
                report["downloaded"].append(status.name)
                self._emit(progress, {"event": "download_ok", "name": status.name})
            else:
                report["failed"].append({"name": status.name, "error": last_error})
                self._emit(
                    progress,
                    {"event": "download_failed", "name": status.name, "error": last_error},
                )
        return report

    @staticmethod
    def _emit(progress: Callable[[dict[str, Any]], None] | None, event: dict[str, Any]) -> None:
        if progress is None:
            return
        try:
            progress(event)
        except _LIFECYCLE_RECOVERABLE_ERRORS as exc:
            logger.debug("Model lifecycle progress callback failed: %s", exc)

    def report(self) -> dict[str, Any]:
        inv = self.inventory()
        return {
            "models": [s.as_dict() for s in inv],
            "all_present": all(s.present for s in inv),
            "missing": [s.name for s in inv if not s.present and s.source_repo],
            "active_present": self.active_model_present(),
        }


_SHARED_MANAGER: ModelLifecycleManager | None = None


def get_model_lifecycle_manager() -> ModelLifecycleManager:
    global _SHARED_MANAGER
    if _SHARED_MANAGER is None:
        _SHARED_MANAGER = ModelLifecycleManager()
    return _SHARED_MANAGER
