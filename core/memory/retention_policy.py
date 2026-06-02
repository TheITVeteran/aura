"""Memory retention policy helpers.

The memory system needs large enough durable capacity for long-lived identity
and work continuity, while still protecting boot time, RAM, and save latency on
smaller machines.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any


def _env_int(name: str, default: int, *, low: int, high: int) -> int:
    try:
        value = int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))


def _env_float(name: str, default: float, *, low: float, high: float) -> float:
    try:
        value = float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))


def physical_ram_gb() -> float:
    try:
        import psutil

        return float(psutil.virtual_memory().total) / (1024**3)
    except (ImportError, RuntimeError, OSError, AttributeError, ValueError):
        try:
            pages = float(os.sysconf("SC_PHYS_PAGES"))
            page_size = float(os.sysconf("SC_PAGE_SIZE"))
            return (pages * page_size) / (1024**3)
        except (RuntimeError, OSError, AttributeError, ValueError):
            return 16.0


@dataclass(frozen=True)
class MemoryRetentionPolicy:
    max_items: int
    prune_keep_fraction: float
    basis: str

    def keep_count(self, current_count: int) -> int:
        if current_count <= self.max_items:
            return max(0, int(current_count))
        return max(1, min(self.max_items, int(current_count * self.prune_keep_fraction)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def black_hole_retention_policy(*, ram_gb: float | None = None) -> MemoryRetentionPolicy:
    """Return the default durable vault retention policy.

    Defaults intentionally exceed the old 5k cap.  The vault still has a hard
    cap because it currently serializes as one encrypted JSON document.
    """
    if "AURA_BLACK_HOLE_MAX_MEMORIES" in os.environ:
        max_items = _env_int("AURA_BLACK_HOLE_MAX_MEMORIES", 50_000, low=1_000, high=1_000_000)
        basis = "env:AURA_BLACK_HOLE_MAX_MEMORIES"
    else:
        ram = physical_ram_gb() if ram_gb is None else float(ram_gb)
        if ram >= 48.0:
            max_items = 100_000
            basis = "ram>=48gb"
        elif ram >= 32.0:
            max_items = 75_000
            basis = "ram>=32gb"
        elif ram >= 16.0:
            max_items = 50_000
            basis = "ram>=16gb"
        elif ram >= 8.0:
            max_items = 25_000
            basis = "ram>=8gb"
        else:
            max_items = 10_000
            basis = "ram<8gb"
    keep_fraction = _env_float("AURA_BLACK_HOLE_PRUNE_KEEP_FRACTION", 0.90, low=0.50, high=0.98)
    return MemoryRetentionPolicy(
        max_items=max_items,
        prune_keep_fraction=keep_fraction,
        basis=basis,
    )


def long_term_retention_policy(*, ram_gb: float | None = None) -> MemoryRetentionPolicy:
    if "AURA_LTM_MAX_MEMORIES" in os.environ:
        max_items = _env_int("AURA_LTM_MAX_MEMORIES", 50_000, low=1_000, high=1_000_000)
        basis = "env:AURA_LTM_MAX_MEMORIES"
    else:
        ram = physical_ram_gb() if ram_gb is None else float(ram_gb)
        if ram >= 48.0:
            max_items = 75_000
            basis = "ram>=48gb"
        elif ram >= 32.0:
            max_items = 50_000
            basis = "ram>=32gb"
        elif ram >= 16.0:
            max_items = 30_000
            basis = "ram>=16gb"
        elif ram >= 8.0:
            max_items = 15_000
            basis = "ram>=8gb"
        else:
            max_items = 10_000
            basis = "ram<8gb"
    keep_fraction = _env_float("AURA_LTM_PRUNE_KEEP_FRACTION", 0.92, low=0.50, high=0.98)
    return MemoryRetentionPolicy(
        max_items=max_items,
        prune_keep_fraction=keep_fraction,
        basis=basis,
    )


def episodic_retention_policy(*, ram_gb: float | None = None) -> MemoryRetentionPolicy:
    if "AURA_EPISODIC_MAX_EPISODES" in os.environ:
        max_items = _env_int("AURA_EPISODIC_MAX_EPISODES", 50_000, low=1_000, high=1_000_000)
        basis = "env:AURA_EPISODIC_MAX_EPISODES"
    else:
        ram = physical_ram_gb() if ram_gb is None else float(ram_gb)
        if ram >= 48.0:
            max_items = 100_000
            basis = "ram>=48gb"
        elif ram >= 32.0:
            max_items = 75_000
            basis = "ram>=32gb"
        elif ram >= 16.0:
            max_items = 50_000
            basis = "ram>=16gb"
        elif ram >= 8.0:
            max_items = 25_000
            basis = "ram>=8gb"
        else:
            max_items = 10_000
            basis = "ram<8gb"
    keep_fraction = _env_float("AURA_EPISODIC_PRUNE_KEEP_FRACTION", 0.95, low=0.50, high=0.99)
    return MemoryRetentionPolicy(
        max_items=max_items,
        prune_keep_fraction=keep_fraction,
        basis=basis,
    )


def hybrid_memory_retention_policy(*, ram_gb: float | None = None) -> MemoryRetentionPolicy:
    if "AURA_HYBRID_MEMORY_MAX_ENTRIES" in os.environ:
        max_items = _env_int(
            "AURA_HYBRID_MEMORY_MAX_ENTRIES",
            50_000,
            low=1_000,
            high=1_000_000,
        )
        basis = "env:AURA_HYBRID_MEMORY_MAX_ENTRIES"
    else:
        ram = physical_ram_gb() if ram_gb is None else float(ram_gb)
        if ram >= 48.0:
            max_items = 100_000
            basis = "ram>=48gb"
        elif ram >= 32.0:
            max_items = 75_000
            basis = "ram>=32gb"
        elif ram >= 16.0:
            max_items = 50_000
            basis = "ram>=16gb"
        elif ram >= 8.0:
            max_items = 25_000
            basis = "ram>=8gb"
        else:
            max_items = 10_000
            basis = "ram<8gb"
    keep_fraction = _env_float(
        "AURA_HYBRID_MEMORY_PRUNE_KEEP_FRACTION",
        0.95,
        low=0.50,
        high=0.99,
    )
    return MemoryRetentionPolicy(
        max_items=max_items,
        prune_keep_fraction=keep_fraction,
        basis=basis,
    )


def working_history_retention_policy(
    env_name: str,
    *,
    ram_gb: float | None = None,
) -> MemoryRetentionPolicy:
    """Return a bounded in-process trace/history retention policy.

    These buffers are not durable memory, but they carry continuity evidence
    during long autonomous runs.  The old fixed 500-entry defaults were small
    enough to erase useful receipts on healthy machines, while still needing a
    hard cap to avoid unbounded RAM growth.
    """
    if env_name in os.environ:
        max_items = _env_int(env_name, 5_000, low=100, high=100_000)
        basis = f"env:{env_name}"
    else:
        ram = physical_ram_gb() if ram_gb is None else float(ram_gb)
        if ram >= 48.0:
            max_items = 10_000
            basis = "ram>=48gb"
        elif ram >= 32.0:
            max_items = 7_500
            basis = "ram>=32gb"
        elif ram >= 16.0:
            max_items = 5_000
            basis = "ram>=16gb"
        elif ram >= 8.0:
            max_items = 2_500
            basis = "ram>=8gb"
        else:
            max_items = 1_000
            basis = "ram<8gb"
    return MemoryRetentionPolicy(
        max_items=max_items,
        prune_keep_fraction=0.95,
        basis=basis,
    )


def state_log_retention_policy(*, ram_gb: float | None = None) -> MemoryRetentionPolicy:
    """Return the SQLite state-version log retention policy.

    State snapshots can be substantially larger than lightweight decision or
    thought records, so this policy raises the legacy 500-row cap without
    treating state versions like cheap semantic memories.
    """
    if "AURA_STATE_LOG_MAX_ROWS" in os.environ:
        max_items = _env_int("AURA_STATE_LOG_MAX_ROWS", 5_000, low=500, high=100_000)
        basis = "env:AURA_STATE_LOG_MAX_ROWS"
    else:
        ram = physical_ram_gb() if ram_gb is None else float(ram_gb)
        if ram >= 48.0:
            max_items = 5_000
            basis = "ram>=48gb"
        elif ram >= 32.0:
            max_items = 3_000
            basis = "ram>=32gb"
        elif ram >= 16.0:
            max_items = 2_000
            basis = "ram>=16gb"
        elif ram >= 8.0:
            max_items = 1_000
            basis = "ram>=8gb"
        else:
            max_items = 500
            basis = "ram<8gb"
    return MemoryRetentionPolicy(
        max_items=max_items,
        prune_keep_fraction=0.90,
        basis=basis,
    )


def sovereign_pruner_target_retention() -> float:
    return _env_float(
        "AURA_SOVEREIGN_PRUNER_TARGET_RETENTION",
        0.75,
        low=0.10,
        high=0.98,
    )
