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

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway

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
        integration_manifest_path: Path | None = None,
        training_state_path: Path | None = None,
        training_data_dir: Path | None = None,
    ) -> None:
        self.dataset_path = dataset_path or (_REPO_ROOT / "data" / "synthetic_training" / "lora_dataset.jsonl")
        self.fused_model_dir = fused_model_dir or (_REPO_ROOT / "training" / "fused-model")
        self.marker_path = marker_path or (self.dataset_path.parent / ".crsm_consumed.json")
        self.training_data_dir = training_data_dir or (_REPO_ROOT / "training" / "data")
        self.integration_manifest_path = integration_manifest_path or (
            self.training_data_dir / "crsm_integration_manifest.json"
        )
        self.training_state_path = training_state_path or (
            _REPO_ROOT / "training" / "adapters" / "aura-personality" / "training_state.json"
        )

    # ── pipeline observations ─────────────────────────────────────────────

    def dataset_state(self) -> dict[str, Any]:
        try:
            if not self.dataset_path.exists():
                return {"exists": False, "lines": 0, "mtime": 0.0, "size": 0}
            lines = 0
            with open(self.dataset_path, encoding="utf-8") as fh:
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

    def _jsonl_file_state(self, path: Path, expected: dict[str, Any] | None = None) -> dict[str, Any]:
        expected = expected or {}
        try:
            if not path.exists():
                return {"exists": False, "path": str(path), "matches_expected": False}
            digest = hashlib.sha256()
            lines = 0
            with path.open("rb") as fh:
                for raw in fh:
                    lines += 1
                    digest.update(raw)
            stat = path.stat()
            actual = {
                "exists": True,
                "path": str(path),
                "lines": lines,
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "sha256": digest.hexdigest(),
            }
            expected_hash = str(expected.get("sha256") or "")
            expected_lines = int(expected.get("lines", -1) or -1)
            expected_size = int(expected.get("size", -1) or -1)
            actual["matches_expected"] = bool(
                expected_hash
                and actual["sha256"] == expected_hash
                and expected_lines == lines
                and expected_size == stat.st_size
            )
            return actual
        except (OSError, ValueError, TypeError) as exc:
            record_degradation("crsm_loop_monitor", exc)
            return {"exists": False, "path": str(path), "matches_expected": False, "error": f"{type(exc).__name__}: {exc}"}

    def integration_manifest_state(self) -> dict[str, Any]:
        try:
            if not self.integration_manifest_path.exists():
                return {"exists": False, "current_for_dataset": False}
            manifest = json.loads(self.integration_manifest_path.read_text(encoding="utf-8"))
            ds = self.dataset_state()
            source_lines = int(manifest.get("source_lines", 0) or 0)
            source_mtime = float(manifest.get("source_mtime", 0.0) or 0.0)
            dataset_mtime = float(ds.get("mtime", 0.0) or 0.0)
            output = dict(manifest.get("output") or {})
            train_expected = dict(output.get("train") or {})
            valid_expected = dict(output.get("valid") or {})
            train_path = Path(str(train_expected.get("path") or (self.training_data_dir / "train.jsonl")))
            valid_path = Path(str(valid_expected.get("path") or (self.training_data_dir / "valid.jsonl")))
            train_state = self._jsonl_file_state(train_path, train_expected)
            valid_state = self._jsonl_file_state(valid_path, valid_expected)
            expected_total = int(output.get("total_examples", 0) or 0)
            actual_total = int(train_state.get("lines", 0) or 0) + int(valid_state.get("lines", 0) or 0)
            source_current = (
                source_lines == int(ds.get("lines", 0) or 0)
                and source_mtime + 1.0 >= dataset_mtime
            )
            corpus_current = bool(
                output
                and train_state.get("matches_expected")
                and valid_state.get("matches_expected")
                and expected_total == actual_total
            )
            return {
                "exists": True,
                "path": str(self.integration_manifest_path),
                "source_lines": source_lines,
                "accepted": int(manifest.get("accepted", 0) or 0),
                "deduplicated": int(manifest.get("deduplicated", 0) or 0),
                "rejected_by_reason": dict(manifest.get("rejected_by_reason") or {}),
                "source_mtime": source_mtime,
                "output_integrity": {
                    "expected_total": expected_total,
                    "actual_total": actual_total,
                    "train": train_state,
                    "valid": valid_state,
                    "corpus_current": corpus_current,
                },
                "current_for_dataset": source_current and corpus_current,
            }
        except (OSError, ValueError, TypeError) as exc:
            record_degradation("crsm_loop_monitor", exc)
            return {"exists": False, "current_for_dataset": False, "error": f"{type(exc).__name__}: {exc}"}

    def training_state(self) -> dict[str, Any]:
        try:
            if not self.training_state_path.exists():
                return {"exists": False}
            state = json.loads(self.training_state_path.read_text(encoding="utf-8"))
            return {
                "exists": True,
                "path": str(self.training_state_path),
                "phase": state.get("phase"),
                "last_iter": int(state.get("last_iter", 0) or 0),
                "last_checkpoint_path": state.get("last_checkpoint_path"),
                "last_pipeline_rc": state.get("last_pipeline_rc"),
                "last_resume_rc": state.get("last_resume_rc"),
                "last_signal": state.get("last_signal"),
                "last_heartbeat": state.get("last_heartbeat"),
            }
        except (OSError, ValueError, TypeError) as exc:
            record_degradation("crsm_loop_monitor", exc)
            return {"exists": False, "error": f"{type(exc).__name__}: {exc}"}

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
            get_file_write_gateway().write_text(
                self.marker_path,
                json.dumps(payload),
                encoding="utf-8",
                source="training:crsm_loop_monitor",
            )
            logger.info("🔁 [CRSMLoop] dataset consumed: %d lines → %s", lines_consumed, model_path)
        except OSError as exc:
            record_degradation("crsm_loop_monitor", exc)

    def loop_state(self) -> dict[str, Any]:
        ds = self.dataset_state()
        art = self.latest_training_artifact()
        marker = self._consumed_marker()
        manifest = self.integration_manifest_state()
        training_state = self.training_state()
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
            state = "open"
            if manifest.get("current_for_dataset"):
                reason = (
                    f"{unconsumed} captures are integrated into the LoRA corpus "
                    "but have not been trained/fused into the active model"
                )
            else:
                reason = f"{unconsumed} captures accumulated but not trained in"
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
            "integration_manifest": manifest,
            "training_state": training_state,
            "next_action": self.next_action(state, manifest, training_state),
            "consumption_marker": marker,
        }

    def next_action(
        self,
        state: str | None = None,
        manifest: dict[str, Any] | None = None,
        training_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = state or self.loop_state().get("state")
        manifest = manifest if manifest is not None else self.integration_manifest_state()
        training_state = training_state if training_state is not None else self.training_state()
        command = [
            "python",
            "training/train_and_fuse.py",
            "--crsm-delta",
            "--tag",
            "crsm-closeout",
        ]
        preflight_command = [
            "python",
            "training/train_and_fuse.py",
            "--crsm-delta",
            "--preflight-only",
            "--tag",
            "crsm-closeout",
        ]
        if state == "closed":
            return {"required": False, "reason": "CRSM captures already consumed by active training marker"}
        if not manifest.get("current_for_dataset"):
            return {
                "required": True,
                "phase": "prepare_dataset",
                "command": ["python", "training/build_dataset_v3.py"],
                "reason": "CRSM integration manifest is missing or stale",
            }
        return {
            "required": True,
            "phase": "crsm_delta_train_fuse_publish",
            "command": command,
            "preflight_command": preflight_command,
            "reason": (
                "Current CRSM captures are in the LoRA corpus, but proof closure "
                "requires a bounded real CRSM delta train/fuse marker from "
                "training/train_and_fuse.py"
            ),
            "last_training_phase": training_state.get("phase"),
            "last_training_rc": training_state.get("last_pipeline_rc"),
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
