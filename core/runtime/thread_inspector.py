"""Live thread inventory — observability for thread-leak diagnosis.

A histogram of live threads grouped by a normalized name prefix turns
"104 threads piled up" from a mystery into a named pool. Pure read-only
enumeration; safe to call on any running process.
"""
from __future__ import annotations

import re
import threading
from collections import Counter
from typing import Any

# Trailing worker indices that distinguish members of one pool:
#   "Aura.Events_3" / "ThreadPoolExecutor-2_0" / "AuraVision-1" all collapse
#   to their pool so the histogram counts pools, not individual workers.
_WORKER_SUFFIX_RE = re.compile(r"[-_]\d+(?:_\d+)*$")


def normalize_thread_name(name: str) -> str:
    """Collapse a thread's worker index so siblings share one bucket."""
    base = str(name or "unnamed").strip() or "unnamed"
    prev = None
    # Strip repeated trailing "-N"/"_N" segments (pool + worker indices).
    while prev != base:
        prev = base
        base = _WORKER_SUFFIX_RE.sub("", base)
    return base or "unnamed"


def thread_summary(*, top: int = 15) -> dict[str, Any]:
    """Return a histogram of live threads grouped by normalized name.

    Keys: total, daemon, non_daemon, distinct_groups, groups (sorted
    name->count, largest first, truncated to `top`).
    """
    threads = list(threading.enumerate())
    histogram: Counter[str] = Counter()
    daemon = 0
    for thread in threads:
        histogram[normalize_thread_name(thread.name)] += 1
        if thread.daemon:
            daemon += 1
    ordered = histogram.most_common(max(1, top))
    return {
        "total": len(threads),
        "daemon": daemon,
        "non_daemon": len(threads) - daemon,
        "distinct_groups": len(histogram),
        "groups": {name: count for name, count in ordered},
    }
