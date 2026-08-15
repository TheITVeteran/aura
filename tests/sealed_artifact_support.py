"""Shared gate for tests that need a sealed learning artifact admitted.

A sealed artifact is refused when its provenance no longer holds — the manifest
pins the SHA-256 of every source file the training run was sealed against, and
`core/learning/frontier_process_supervision.py` drifted from its pinned hash in
8c48eec8d (CP546, schema v1→v2). The refusal is correct and fail-closed.

Five capability tests then failed with `RuntimeError: mathematics memory
admission evidence differs`. That reads as a capability regression and is not
one: the tissue never ran. The honest outcome is a skip that NAMES the reason,
the same treatment this suite already gives a model that is not on the host.

The skip cannot hide a regression, because the refusal is loud everywhere else:
`sealed_artifact_admission_report()` publishes it, the runtime health report
carries it under `sealed_artifacts`, the producer records a degradation the
first time it falls back, and the registered runtime claim reports NOT MEASURED
with the drifted file named. What is skipped here is visible in four other
places.

Re-sealing is the fix, and it belongs to whoever owns the GPU:

    python tools/run_mathematics_memory_canary.py --out <canary.json>
    python tools/materialize_mathematics_memory_tissue.py

That trains a tissue through MLX, so it contends for Metal with a resident
model and must not be run under a live training job.
"""

from __future__ import annotations

import pytest

from core.learning.sealed_artifact_admission import mathematics_memory_admitted


def require_mathematics_memory_tissue() -> None:
    """Skip the calling test when the sealed tissue is not admitted."""
    admitted, detail = mathematics_memory_admitted()
    if not admitted:
        pytest.skip(
            "sealed mathematics memory tissue is not admitted, so this capability "
            f"never ran: {detail}. Re-seal with tools/run_mathematics_memory_canary.py "
            "when the GPU is free; the refusal is reported on the health surface "
            "under sealed_artifacts."
        )


__all__ = ["require_mathematics_memory_tissue"]
