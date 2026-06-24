"""CRSM→LoRA loop closure monitor — make the training loop *verifiable*.

The CRSM-LoRA bridge captures high-salience moments and accumulates them into a
JSONL dataset (``data/synthetic_training/lora_dataset.jsonl``). What it could not
answer — the critique's gap — is whether that dataset is actually *consumed* by LoRA
training and whether the resulting weights *persist* into the next session. The
architecture supports it; whether it is running was invisible and "required active
reading."

This monitor closes that observability gap. It compares three timestamps —
captured-dataset growth, the newest fused-model artifact, and the active-model
pointer — to classify the loop as CLOSED, OPEN (captures accumulating but not trained
in), or IDLE, and surfaces a governance signal + a loud warning when it is open. It
also exposes ``mark_dataset_consumed`` so a training run records exactly how much of
the dataset it ingested and which model it produced — turning "is it running?" into a
verified, queryable fact.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.CRSMLoop")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_UNCONSUMED_WARN = 25          # captures accumulated past this without training → warn
_STALE_AFTER_S = 7 * 24 * 3600  # dataset newer than the model by this long → stale


class CRSMLoopMonitor:
    def __init__(
        self,
        *,
        dataset_path: Path | None = None,
        fused_model_dir: Path | None = None,
        marker_path: Path | None = None,
    ) -> None:
        self.dataset_path = dataset_path or (_REPO_ROOT / "data" / "synthetic_training" / "lora_dataset.jsonl")
        self.fused_model_dir = fused_model_dir or (_REPO_ROOT / "training" / "fused-model")
        self.marker_path = marker_path or (self.dataset_path.parent / ".crsm_consumed.json")

    # ── pipeline observations ─────────────────────────────────────────────

    def dataset_state(self) -> dict[str, Any]:
        try:
            if not self.dataset_path.exists():
                return {"exists": False, "lines": 0, "mtime": 0.0, "size": 0}
            lines = 0
            with open(self.dataset_path, "r", encoding="utf-8") as fh:
                for _ in fh:
                    lines += 1
            st = self.dataset_path.stat()
            return {"exists": True, "lines": lines, "mtime": st.st_mtime, "size": st.st_size}
        except OSError as exc:
            record_degradation("crsm_loop_monitor", exc)
            return {"exists": False, "lines": 0, "mtime": 0.0, "size": 0}

    def latest_training_artifact(self) -> dict[str, Any]:
        """Newest fused-model directory + the active-model pointer's fuse time."""
        newest_mtime = 0.0
        newest_name = None
        try:
            if self.fused_model_dir.exists():
                for child in self.fused_model_dir.iterdir():
                    if child.is_dir():
                        m = child.stat().st_mtime
                        if m > newest_mtime:
                            newest_mtime, newest_name = m, child.name
        except OSError as exc:
            record_degradation("crsm_loop_monitor", exc)
        active_fused_at = 0.0
        active_path = None
        try:
            active_json = self.fused_model_dir / "active.json"
            if active_json.exists():
                data = json.loads(active_json.read_text(encoding="utf-8"))
                active_fused_at = float(data.get("fused_at", 0.0) or 0.0)
                active_path = data.get("active_model_path")
        except (OSError, ValueError, TypeError) as exc:
            record_degradation("crsm_loop_monitor", exc)
        return {
            "newest_model": newest_name,
            "newest_mtime": newest_mtime,
            "active_fused_at": active_fused_at,
            "active_model_path": active_path,
        }

    def _consumed_marker(self) -> dict[str, Any]:
        try:
            if self.marker_path.exists():
                return json.loads(self.marker_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            record_degradation("crsm_loop_monitor", exc)
        return {}

    # ── loop closure ──────────────────────────────────────────────────────

    def mark_dataset_consumed(
        self,
        *,
        model_path: str | None = None,
        lines_consumed: int | None = None,
        accepted_lines: int | None = None,
        rejected_lines: int | None = None,
        manifest_path: str | None = None,
        source: str | None = None,
    ) -> None:
        """Record that a training run ingested the dataset — call after LoRA training.

        Writes how many dataset lines were consumed and which model resulted, so loop
        closure is a verified fact rather than an inference.
        """
        if lines_consumed is None:
            lines_consumed = int(self.dataset_state().get("lines", 0))
        accepted = int(accepted_lines if accepted_lines is not None else lines_consumed)
        rejected = int(rejected_lines if rejected_lines is not None else max(0, lines_consumed - accepted))
        payload = {
            "lines_consumed": int(lines_consumed),
            "accepted_lines": max(0, accepted),
            "rejected_lines": max(0, rejected),
            "consumed_at": time.time(),
            "model_path": model_path,
            "manifest_path": manifest_path,
            "source": source or "unspecified",
        }
        try:
            self.marker_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.marker_path.with_name(self.marker_path.name + ".tmp")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            tmp.replace(self.marker_path)
            logger.info("🔁 [CRSMLoop] dataset consumed: %d lines → %s", lines_consumed, model_path)
        except OSError as exc:
            record_degradation("crsm_loop_monitor", exc)

    def loop_state(self) -> dict[str, Any]:
        ds = self.dataset_state()
        art = self.latest_training_artifact()
        marker = self._consumed_marker()
        lines = int(ds.get("lines", 0))
        consumed = int(marker.get("lines_consumed", 0))
        unconsumed = max(0, lines - consumed)
        last_train = max(float(art.get("newest_mtime", 0.0)), float(art.get("active_fused_at", 0.0)))
        ds_mtime = float(ds.get("mtime", 0.0))

        accepted = int(marker.get("accepted_lines", consumed) or 0)
        rejected = int(marker.get("rejected_lines", max(0, consumed - accepted)) or 0)

        if lines == 0:
            state, reason = "idle", "no captured moments yet"
        elif consumed >= lines and last_train >= ds_mtime:
            if rejected > 0:
                state, reason = (
                    "closed",
                    f"{accepted} eligible captures trained and {rejected} retired by the training gate",
                )
            else:
                state, reason = "closed", "dataset trained in and weights persisted"
        elif last_train >= ds_mtime and unconsumed <= _UNCONSUMED_WARN:
            state, reason = "closed", "latest model is newer than the captured data"
        elif unconsumed > _UNCONSUMED_WARN or (ds_mtime - last_train) > _STALE_AFTER_S:
            state, reason = "open", f"{unconsumed} captures accumulated but not trained in"
        else:
            state, reason = "pending", "captures awaiting the next training run"

        return {
            "state": state,
            "reason": reason,
            "dataset_lines": lines,
            "unconsumed": unconsumed,
            "accepted_lines": accepted,
            "rejected_lines": rejected,
            "last_training_at": last_train,
            "active_model": art.get("active_model_path"),
            "dataset_mtime": ds_mtime,
            "consumption_marker": marker,
        }

    def audit(self) -> dict[str, Any]:
        """Evaluate the loop and log loudly if it is open (the previously-silent gap)."""
        state = self.loop_state()
        if state["state"] == "open":
            logger.warning(
                "🔁 [CRSMLoop] LOOP OPEN: %s — captured experience is not being "
                "crystallized into weights. Run LoRA training on the synthetic dataset.",
                state["reason"],
            )
            try:
                from core.observability.metrics import get_metrics

                get_metrics().increment_counter("crsm_loop_open_total")
            except (ImportError, AttributeError, RuntimeError, TypeError):
                pass
        return state

    def governance_signal(self) -> dict[str, Any]:
        return self.loop_state()


_monitor: CRSMLoopMonitor | None = None


def get_crsm_loop_monitor() -> CRSMLoopMonitor:
    global _monitor
    if _monitor is None:
        _monitor = CRSMLoopMonitor()
    return _monitor
