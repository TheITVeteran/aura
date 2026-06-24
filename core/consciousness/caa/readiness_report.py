"""CAA readiness verification — is affective steering at design capacity?

The critique's fourth point: whether the steering vectors were genuinely *extracted*
from contrastive activation differences in the fused 32B model, or merely *derived at
runtime*, is an operational fact that determines the readiness level and therefore how
much the steering alpha is damped. If the vectors are runtime-derived, the system's
affective self-expression runs below design capacity — and that was silent.

This module verifies it from ground truth: it reads each steering-vector file's
provenance (``source`` / ``extracted`` / ``derived_at``) directly off disk, ties it to
the active fused model (``training/fused-model/active.json``), classifies the readiness
level, estimates the resulting steering capacity, and warns loudly when steering is
running below design capacity. "Are the Zenith vectors registered?" becomes a queryable,
surfaced fact instead of a guess.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.CAA.Readiness")

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Estimated steering capacity (alpha fraction of design) per readiness level.
_CAPACITY = {"bootstrap": 0.3, "mixed": 0.6, "validated": 0.85, "production": 1.0}


def scan_vector_files(vectors_dir: Path) -> dict[str, Any]:
    """Read provenance off every steering-vector ``.npz`` on disk."""
    extracted = 0
    runtime_derived = 0
    fallback = 0
    other = 0
    newest_derived_at = 0.0
    files = 0
    by_source: dict[str, int] = {}
    try:
        for npz in sorted(vectors_dir.glob("*.npz")):
            files += 1
            try:
                d = np.load(npz, allow_pickle=True)
                source = str(d["source"]) if "source" in d else "unknown"
                is_extracted = bool(d["extracted"]) if "extracted" in d else False
                derived_at = float(d["derived_at"]) if "derived_at" in d else 0.0
            except (OSError, ValueError, KeyError) as exc:
                record_degradation("caa_readiness_report", exc)
                continue
            by_source[source] = by_source.get(source, 0) + 1
            newest_derived_at = max(newest_derived_at, derived_at)
            if is_extracted:
                extracted += 1
            elif source == "runtime_derived_caa":
                runtime_derived += 1
            elif source == "fallback_random":
                fallback += 1
            else:
                other += 1
    except OSError as exc:
        record_degradation("caa_readiness_report", exc)
    return {
        "files": files,
        "extracted": extracted,
        "runtime_derived": runtime_derived,
        "fallback": fallback,
        "other": other,
        "by_source": by_source,
        "newest_derived_at": newest_derived_at,
    }


def _active_model(fused_model_dir: Path) -> dict[str, Any]:
    try:
        aj = fused_model_dir / "active.json"
        if aj.exists():
            data = json.loads(aj.read_text(encoding="utf-8"))
            return {
                "path": data.get("active_model_path"),
                "fused_at": float(data.get("fused_at", 0.0) or 0.0),
            }
    except (OSError, ValueError, TypeError) as exc:
        record_degradation("caa_readiness_report", exc)
    return {"path": None, "fused_at": 0.0}


def verify_readiness(
    *,
    vectors_dir: Path | None = None,
    fused_model_dir: Path | None = None,
) -> dict[str, Any]:
    """Classify CAA readiness from on-disk provenance + the active model."""
    vectors_dir = vectors_dir or (_REPO_ROOT / "training" / "vectors")
    fused_model_dir = fused_model_dir or (_REPO_ROOT / "training" / "fused-model")
    scan = scan_vector_files(vectors_dir)
    active = _active_model(fused_model_dir)
    total = scan["files"] or 0
    extracted_ratio = (scan["extracted"] / total) if total else 0.0

    if total == 0:
        level, detail = "bootstrap", "no steering vectors present"
    elif extracted_ratio < 0.5:
        level = "bootstrap"
        detail = (
            f"{scan['runtime_derived']} runtime-derived / {scan['fallback']} fallback vectors — "
            "NOT extracted from the fused model"
        )
    elif extracted_ratio < 1.0:
        level, detail = "mixed", f"{scan['extracted']}/{total} vectors extracted; rest runtime/nearest"
    else:
        level, detail = "production", "all vectors extracted from the model"

    capacity = _CAPACITY.get(level, 0.3)
    return {
        "level": level,
        "detail": detail,
        "extracted_ratio": round(extracted_ratio, 3),
        "steering_capacity_pct": round(capacity * 100, 1),
        "below_design_capacity": capacity < 1.0,
        "active_model": active["path"],
        "vectors": scan,
    }


def audit(**kwargs: Any) -> dict[str, Any]:
    """Verify readiness and warn loudly when steering runs below design capacity."""
    report = verify_readiness(**kwargs)
    if report["below_design_capacity"]:
        logger.warning(
            "🎚️ [CAA] steering BELOW design capacity (%.0f%%): readiness=%s — %s. "
            "Extract CAA vectors from the fused model to reach production steering.",
            report["steering_capacity_pct"], report["level"], report["detail"],
        )
        try:
            from core.observability.metrics import get_metrics

            get_metrics().increment_counter("caa_below_design_capacity_total")
        except (ImportError, AttributeError, RuntimeError, TypeError):
            pass
    return report


def governance_signal() -> dict[str, Any]:
    return verify_readiness()
