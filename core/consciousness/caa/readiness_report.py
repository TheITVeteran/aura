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


def _parse_vector_stem(stem: str) -> tuple[str, int]:
    import re

    match = re.match(r"^(?P<dimension>.+)_layer(?P<layer>\d+)$", stem)
    if match:
        return match.group("dimension"), int(match.group("layer"))
    return stem, -1


def _runtime_expected_keys() -> list[str]:
    try:
        from core.consciousness.affective_steering import AFFECTIVE_DIMENSIONS

        return [str(spec["key"]) for spec in AFFECTIVE_DIMENSIONS if spec.get("key")]
    except (ImportError, AttributeError, RuntimeError, TypeError) as exc:
        record_degradation("caa_readiness_report", exc)
        return ["valence_positive", "arousal", "curiosity", "frustration", "energy"]


def _target_layers_for_active_model(active_model_path: str | None) -> list[int]:
    if not active_model_path:
        return []
    try:
        cfg = json.loads((Path(active_model_path) / "config.json").read_text(encoding="utf-8"))
        n_layers = int(cfg.get("num_hidden_layers") or cfg.get("n_layer") or cfg.get("num_layers") or 0)
    except (OSError, ValueError, TypeError) as exc:
        record_degradation("caa_readiness_report", exc)
        return []
    if n_layers <= 0:
        return []
    lo = int(n_layers * 0.40)
    hi = int(n_layers * 0.65)
    span = hi - lo
    if span <= 2:
        return [lo]
    if span <= 5:
        return [lo, lo + span // 2]
    return [lo, lo + span // 3, lo + 2 * span // 3]


def scan_vector_files(vectors_dir: Path) -> dict[str, Any]:
    """Read provenance off every steering-vector ``.npz`` on disk."""
    extracted = 0
    runtime_derived = 0
    fallback = 0
    other = 0
    newest_derived_at = 0.0
    files = 0
    by_source: dict[str, int] = {}
    details: list[dict[str, Any]] = []
    try:
        for npz in sorted(vectors_dir.glob("*.npz")):
            files += 1
            dimension, layer = _parse_vector_stem(npz.stem)
            try:
                d = np.load(npz, allow_pickle=True)
                source = str(d["source"]) if "source" in d else "unknown"
                is_extracted = bool(d["extracted"]) if "extracted" in d else False
                derived_at = float(d["derived_at"]) if "derived_at" in d else 0.0
                vector_dim = int(np.asarray(d["v"] if "v" in d else d[d.files[0]]).reshape(-1).shape[0]) if d.files else 0
            except (OSError, ValueError, KeyError) as exc:
                record_degradation("caa_readiness_report", exc)
                continue
            details.append(
                {
                    "path": str(npz),
                    "dimension": dimension,
                    "layer": layer,
                    "source": source,
                    "extracted": is_extracted,
                    "derived_at": derived_at,
                    "vector_dim": vector_dim,
                }
            )
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
        "details": details,
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
    expected_keys = _runtime_expected_keys()
    expected_layers = _target_layers_for_active_model(active.get("path"))
    expected_total = len(expected_keys) * len(expected_layers)
    details = list(scan.get("details") or [])
    expected_files = [
        item
        for item in details
        if item.get("dimension") in expected_keys and item.get("layer") in expected_layers
    ]
    expected_extracted = [
        item
        for item in expected_files
        if item.get("extracted") and str(item.get("source", "")).startswith("extracted")
    ]
    expected_ratio = (len(expected_extracted) / expected_total) if expected_total else 0.0
    missing_expected = []
    present_pairs = {(item.get("dimension"), item.get("layer")) for item in expected_files}
    for key in expected_keys:
        for layer in expected_layers:
            if (key, layer) not in present_pairs:
                missing_expected.append({"dimension": key, "layer": layer})
    ignored_files = [
        item
        for item in details
        if item.get("dimension") not in expected_keys or item.get("layer") not in expected_layers
    ]

    if expected_total and len(expected_extracted) == expected_total:
        level, detail = "production", "all runtime target vectors extracted from the active model"
        readiness_ratio = 1.0
    elif expected_total and expected_files:
        readiness_ratio = expected_ratio
        if expected_ratio < 0.5:
            level = "bootstrap"
            detail = (
                f"{len(expected_extracted)}/{expected_total} runtime target vectors extracted; "
                "runtime steering will still rely on derived/nearest vectors"
            )
        else:
            level = "mixed"
            detail = (
                f"{len(expected_extracted)}/{expected_total} runtime target vectors extracted; "
                "missing exact production coverage"
            )
    elif total == 0:
        level, detail = "bootstrap", "no steering vectors present"
        readiness_ratio = 0.0
    elif extracted_ratio < 0.5:
        level = "bootstrap"
        detail = (
            f"{scan['runtime_derived']} runtime-derived / {scan['fallback']} fallback vectors — "
            "NOT extracted from the fused model"
        )
        readiness_ratio = extracted_ratio
    elif extracted_ratio < 1.0:
        level, detail = "mixed", f"{scan['extracted']}/{total} vectors extracted; rest runtime/nearest"
        readiness_ratio = extracted_ratio
    else:
        level, detail = "production", "all vectors extracted from the model"
        readiness_ratio = extracted_ratio

    capacity = _CAPACITY.get(level, 0.3)
    return {
        "level": level,
        "detail": detail,
        "extracted_ratio": round(readiness_ratio, 3),
        "all_files_extracted_ratio": round(extracted_ratio, 3),
        "steering_capacity_pct": round(capacity * 100, 1),
        "below_design_capacity": capacity < 1.0,
        "active_model": active["path"],
        "runtime_contract": {
            "expected_keys": expected_keys,
            "expected_layers": expected_layers,
            "expected_total": expected_total,
            "expected_extracted": len(expected_extracted),
            "missing_expected": missing_expected,
            "ignored_file_count": len(ignored_files),
        },
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
