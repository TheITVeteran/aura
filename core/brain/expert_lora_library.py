"""Expert-LoRA library — the capacity loophole.

A 32B's weights hold a bounded amount of information (a real limit). But total
*reachable* expertise is not bounded by RAM — only *resident* expertise is. So we
keep a library of domain-specialist LoRA adapters on disk (you have ~813 GB free),
select the right one per task, and load only a few into memory at a time. Effective
capacity becomes ``base_model + on-disk adapter library``, accessed on demand.

This pairs with the self-improvement flywheel: promoted adapters (verifier-clean
reasoning distilled into a domain LoRA) register here, and the library serves them
back per task. The organism's expertise grows on disk without growing its RAM.

Scope (honest): this module owns the *registry, selection, and RAM-budgeted
residency* — the real, testable substrate. The actual MLX attach/detach of adapter
weights is delegated to a pluggable ``AdapterApplier`` so it can be wired to the live
worker later without this module ever touching model state itself. Default-off
(``AURA_EXPERT_LORA_LIBRARY``) so it never changes generation behavior implicitly.
"""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.ExpertLoRALibrary")

_WORD_RE = re.compile(r"[a-z0-9]+")


def _flag_on(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "on", "yes", "enabled"}


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(str(text or "").lower()))


@dataclass
class LoRAAdapter:
    name: str
    path: str
    base_model: str = ""
    task_types: set[str] = field(default_factory=set)
    keywords: set[str] = field(default_factory=set)
    size_mb: float = 0.0
    quality: float = 0.5     # promotion score; higher wins ties
    source: str = ""         # e.g. "self_improvement", "manual"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "base_model": self.base_model,
            "task_types": sorted(self.task_types),
            "keywords": sorted(self.keywords),
            "size_mb": round(float(self.size_mb), 2),
            "quality": round(float(self.quality), 4),
            "source": self.source,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LoRAAdapter":
        return cls(
            name=str(data.get("name", "")),
            path=str(data.get("path", "")),
            base_model=str(data.get("base_model", "")),
            task_types=set(data.get("task_types", []) or []),
            keywords=set(data.get("keywords", []) or []),
            size_mb=float(data.get("size_mb", 0.0) or 0.0),
            quality=float(data.get("quality", 0.5) or 0.5),
            source=str(data.get("source", "")),
            created_at=float(data.get("created_at", time.time()) or time.time()),
        )


@runtime_checkable
class AdapterApplier(Protocol):
    """Attaches/detaches adapter weights to/from the live model. MLX-specific."""

    def load(self, adapter: LoRAAdapter) -> bool: ...
    def unload(self, adapter: LoRAAdapter) -> bool: ...


class NoopApplier:
    """Safe default — records intent without touching model state."""

    def load(self, adapter: LoRAAdapter) -> bool:
        logger.info("📎 [ExpertLoRA] (noop) would load adapter '%s' from %s", adapter.name, adapter.path)
        return True

    def unload(self, adapter: LoRAAdapter) -> bool:
        logger.info("📎 [ExpertLoRA] (noop) would unload adapter '%s'", adapter.name)
        return True


@runtime_checkable
class AsyncAdapterApplier(Protocol):
    """Async applier for live seams (worker IPC) that must never block a loop."""

    async def load(self, adapter: LoRAAdapter) -> bool: ...
    async def unload(self, adapter: LoRAAdapter) -> bool: ...


class ExpertLoRALibrary:
    """Registry + per-task selection + RAM-budgeted residency for domain LoRAs."""

    _ERRORS = (OSError, ValueError, TypeError, json.JSONDecodeError, UnicodeDecodeError)

    def __init__(
        self,
        manifest_path: str | Path | None = None,
        *,
        max_resident: int | None = None,
        applier: AdapterApplier | None = None,
    ) -> None:
        self._manifest_path = Path(
            manifest_path or os.path.expanduser("~/.aura/data/adapters/library.json")
        )
        self._max_resident = int(max_resident if max_resident is not None
                                 else os.getenv("AURA_EXPERT_LORA_MAX_RESIDENT", "2"))
        self._max_resident = max(1, self._max_resident)
        self._applier = applier or NoopApplier()
        self._lock = threading.RLock()
        self._adapters: dict[str, LoRAAdapter] = {}
        self._resident: "OrderedDict[str, float]" = OrderedDict()  # name -> last_used
        self._load_manifest()

    # ── registry ────────────────────────────────────────────────────────────
    def register(self, adapter: LoRAAdapter) -> bool:
        if not adapter.name or not adapter.path:
            return False
        with self._lock:
            self._adapters[adapter.name] = adapter
            self._persist()
        logger.info("📚 [ExpertLoRA] registered '%s' (task_types=%s)", adapter.name, sorted(adapter.task_types))
        return True

    def unregister(self, name: str) -> bool:
        with self._lock:
            existed = self._adapters.pop(name, None) is not None
            if name in self._resident:
                self._evict(name)
            if existed:
                self._persist()
            return existed

    def list(self) -> list[LoRAAdapter]:
        with self._lock:
            return list(self._adapters.values())

    def get(self, name: str) -> LoRAAdapter | None:
        with self._lock:
            return self._adapters.get(name)

    # ── selection ─────────────────────────────────────────────────────────────
    def select_for(self, objective: str, task_type: str, *, base_model: str = "") -> LoRAAdapter | None:
        """Best adapter for a task: task_type must match; rank by keyword overlap × quality."""
        tt = str(task_type or "").strip().lower()
        obj_tokens = _tokens(objective)
        best: tuple[float, LoRAAdapter] | None = None
        with self._lock:
            for adapter in self._adapters.values():
                if base_model and adapter.base_model and adapter.base_model != base_model:
                    continue  # never apply an adapter trained on a different base
                if adapter.task_types and tt not in adapter.task_types:
                    continue
                overlap = len(obj_tokens & adapter.keywords) if adapter.keywords else 0
                # task_type match alone is worth a small base score so a tagged
                # specialist still wins over nothing even with no keyword overlap.
                relevance = (1.0 + overlap) * max(0.05, adapter.quality)
                if best is None or relevance > best[0]:
                    best = (relevance, adapter)
        return best[1] if best else None

    # ── RAM-budgeted residency ──────────────────────────────────────────────
    def activate(self, name: str) -> bool:
        """Make an adapter resident (load on demand, LRU-evict over budget)."""
        with self._lock:
            adapter = self._adapters.get(name)
            if adapter is None:
                return False
            if name in self._resident:
                self._resident.move_to_end(name)
                self._resident[name] = time.time()
                return True
            while len(self._resident) >= self._max_resident:
                lru_name = next(iter(self._resident))
                self._evict(lru_name)
            ok = bool(self._applier.load(adapter))
            if ok:
                self._resident[name] = time.time()
            return ok

    def _evict(self, name: str) -> None:
        # Caller holds lock.
        self._resident.pop(name, None)
        adapter = self._adapters.get(name)
        if adapter is not None:
            try:
                self._applier.unload(adapter)
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation("expert_lora_unload", exc)

    def resident(self) -> list[str]:
        with self._lock:
            return list(self._resident.keys())

    def select_and_activate(self, objective: str, task_type: str, *, base_model: str = "") -> LoRAAdapter | None:
        """One call for the generation path: pick the specialist and make it resident.

        Returns the activated adapter, or None when disabled / no match. Default-off
        via AURA_EXPERT_LORA_LIBRARY so it never alters generation implicitly.
        """
        if not _flag_on("AURA_EXPERT_LORA_LIBRARY"):
            return None
        adapter = self.select_for(objective, task_type, base_model=base_model)
        if adapter is None:
            return None
        return adapter if self.activate(adapter.name) else None

    # ── async residency (live worker seam) ────────────────────────────────────
    async def activate_async(self, name: str, applier: "AsyncAdapterApplier") -> bool:
        """Make an adapter resident through an ASYNC applier (worker IPC).

        Same residency contract as ``activate`` but the attach/detach I/O is
        awaited without holding the registry lock, so a multi-second worker
        swap can never stall other registry readers. Residency maps update
        only from actual applier outcomes — ``resident()`` never claims an
        adapter the worker refused.
        """
        with self._lock:
            adapter = self._adapters.get(name)
            if adapter is None:
                return False
            if name in self._resident:
                self._resident.move_to_end(name)
                self._resident[name] = time.time()
                return True
            evictees = []
            overflow = len(self._resident) - self._max_resident + 1
            for lru_name in list(self._resident.keys())[:max(0, overflow)]:
                lru = self._adapters.get(lru_name)
                if lru is not None:
                    evictees.append(lru)
                self._resident.pop(lru_name, None)
        for lru in evictees:
            try:
                await applier.unload(lru)
            except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
                record_degradation("expert_lora_unload", exc)
        try:
            ok = bool(await applier.load(adapter))
        except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
            record_degradation("expert_lora_load", exc)
            return False
        if ok:
            with self._lock:
                self._resident[name] = time.time()
        return ok

    async def select_and_activate_async(
        self,
        objective: str,
        task_type: str,
        applier: "AsyncAdapterApplier",
        *,
        base_model: str = "",
    ) -> LoRAAdapter | None:
        """Async twin of ``select_and_activate`` for the live generation path."""
        if not _flag_on("AURA_EXPERT_LORA_LIBRARY"):
            return None
        adapter = self.select_for(objective, task_type, base_model=base_model)
        if adapter is None:
            return None
        return adapter if await self.activate_async(adapter.name, applier) else None

    # ── disk discovery ────────────────────────────────────────────────────────
    def scan(self, directory: str | Path, *, base_model: str = "", source: str = "scan") -> int:
        """Register adapters found under ``directory`` (dirs with adapter_config.json /
        adapters.safetensors). Returns count newly registered."""
        root = Path(os.path.expanduser(str(directory)))
        if not root.exists():
            return 0
        found = 0
        markers = ("adapter_config.json", "adapters.safetensors", "adapters.npz")
        for cfg in root.rglob("*"):
            try:
                if cfg.is_dir() and any((cfg / m).exists() for m in markers):
                    name = cfg.name
                    if name in self._adapters:
                        continue
                    size_mb = sum(
                        f.stat().st_size for f in cfg.glob("*") if f.is_file()
                    ) / (1024 * 1024)
                    self.register(
                        LoRAAdapter(
                            name=name,
                            path=str(cfg),
                            base_model=base_model,
                            task_types=set(),
                            keywords=_tokens(name),
                            size_mb=size_mb,
                            source=source,
                        )
                    )
                    found += 1
            except self._ERRORS as exc:
                record_degradation("expert_lora_scan", exc)
        return found

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "adapters": len(self._adapters),
                "resident": list(self._resident.keys()),
                "max_resident": self._max_resident,
                "enabled": _flag_on("AURA_EXPERT_LORA_LIBRARY"),
            }

    # ── persistence ───────────────────────────────────────────────────────────
    def _load_manifest(self) -> None:
        if not self._manifest_path.exists():
            return
        try:
            raw = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            items = raw.get("adapters", {}) if isinstance(raw, dict) else {}
            with self._lock:
                for name, data in items.items():
                    try:
                        self._adapters[name] = LoRAAdapter.from_dict(data)
                    except self._ERRORS:
                        continue
        except self._ERRORS as exc:
            record_degradation("expert_lora_load", exc)

    def _persist(self) -> None:
        try:
            self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": 1,
                "saved_at": time.time(),
                "adapters": {n: a.to_dict() for n, a in self._adapters.items()},
            }
            fd, tmp = tempfile.mkstemp(prefix=".lora_lib_", suffix=".json", dir=str(self._manifest_path.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, ensure_ascii=False)
                os.replace(tmp, self._manifest_path)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        except self._ERRORS as exc:
            record_degradation("expert_lora_persist", exc)


_singleton: ExpertLoRALibrary | None = None
_singleton_lock = threading.Lock()


def get_expert_lora_library() -> ExpertLoRALibrary:
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = ExpertLoRALibrary()
    return _singleton


def reset_expert_lora_library() -> None:
    global _singleton
    with _singleton_lock:
        _singleton = None
