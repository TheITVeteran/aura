"""Governed online LoRA updates from self-reflection cycles.

This is intentionally conservative: a Will-approved reflection can enqueue a
small adapter update, but the governor refuses to start while another mlx-lm
LoRA process is active. The current full training run therefore remains safe.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.resource_observation import ResourceObserver, get_resource_observer


@dataclass
class OnlineLoRAReceipt:
    requested_at: float
    status: str
    reflection_hash: str
    will_receipt_id: str = ""
    reason: str = ""
    dataset_path: str = ""
    optimizer_result: dict[str, Any] = field(default_factory=dict)
    observation_source: str = "unavailable"
    observation_scenario_id: str = ""
    process_observation_available: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _hash_text(text: str) -> str:
    import hashlib

    return hashlib.blake2b(str(text or "").encode("utf-8"), digest_size=12).hexdigest()


class OnlineLoRAGovernor:
    """Owns the reflection -> dataset -> governed LoRA update path."""

    def __init__(
        self,
        *,
        receipt_path: str | Path | None = None,
        observer: ResourceObserver | None = None,
    ) -> None:
        self.receipt_path = Path(
            receipt_path or Path.home() / ".aura" / "data" / "runtime" / "online_lora_updates.jsonl"
        )
        self.receipt_path.parent.mkdir(parents=True, exist_ok=True)
        self._observer = observer
        self._lock = asyncio.Lock()
        self.last_receipt: Optional[OnlineLoRAReceipt] = None

    @staticmethod
    def enabled() -> bool:
        return os.getenv("AURA_ONLINE_LORA", "1").strip().lower() not in {"0", "false", "off", "no"}

    def active_lora_processes(self) -> list[dict[str, Any]]:
        observer = self._observer or get_resource_observer()
        table = observer.process_table()
        if not table.available:
            return [
                {
                    "pid": None,
                    "cmdline": [],
                    "observation_error": table.error or "process_table_unavailable",
                }
            ]
        found: list[dict[str, Any]] = []
        for process in table.processes:
            joined = " ".join(process.cmdline).lower()
            if "mlx_lm" in joined and "lora" in joined:
                found.append({"pid": process.pid, "cmdline": list(process.cmdline)})
        return found

    def _record(self, receipt: OnlineLoRAReceipt) -> OnlineLoRAReceipt:
        provenance = (self._observer or get_resource_observer()).provenance
        receipt.observation_source = provenance.source.value
        receipt.observation_scenario_id = provenance.scenario_id
        receipt.process_observation_available = not receipt.reason.startswith(
            "process_table_unavailable"
        )
        self.last_receipt = receipt
        get_file_write_gateway().append_text(
            self.receipt_path,
            json.dumps(receipt.to_dict(), sort_keys=True, default=str) + "\n",
            source="adaptation.online_lora.receipt",
        )
        return receipt

    async def maybe_update_from_reflection(
        self,
        reflection: str,
        *,
        conversation_context: str = "",
        will_receipt_id: str = "",
        force: bool = False,
    ) -> OnlineLoRAReceipt:
        """Capture a reflection and, when allowed, run a tiny LoRA update."""
        async with self._lock:
            if not self.enabled() and not force:
                return self._record(
                    OnlineLoRAReceipt(
                        requested_at=time.time(),
                        status="disabled",
                        reflection_hash=_hash_text(reflection),
                        reason="AURA_ONLINE_LORA disabled",
                    )
                )

            # A command-line census enriches every host process and is allowed
            # only on a worker thread. On large developer workstations it can
            # otherwise stall the owner loop for multiple seconds.
            running = await asyncio.to_thread(self.active_lora_processes)
            if running and not force:
                observation_error = str(running[0].get("observation_error") or "")
                return self._record(
                    OnlineLoRAReceipt(
                        requested_at=time.time(),
                        status="blocked_existing_training",
                        reflection_hash=_hash_text(reflection),
                        reason=(
                            f"process_table_unavailable:{observation_error}"
                            if observation_error
                            else f"active mlx_lm lora process pid={running[0].get('pid')}"
                        ),
                    )
                )

            decision = self._decide(reflection, will_receipt_id=will_receipt_id)
            if not decision.get("approved"):
                return self._record(
                    OnlineLoRAReceipt(
                        requested_at=time.time(),
                        status="will_blocked",
                        reflection_hash=_hash_text(reflection),
                        will_receipt_id=str(decision.get("receipt_id", "")),
                        reason=str(decision.get("reason", "Will did not approve")),
                    )
                )

            dataset_path = await self._capture_training_example(reflection, conversation_context)
            
            # Delegate to the already-booted learning owner. This path is
            # collect-only; it must not instantiate a second learner or report
            # a weight update before the scheduler validates and promotes one.
            try:
                from core.container import ServiceContainer

                learner = ServiceContainer.get("continuous_learner", default=None)
                if learner is None:
                    learner = ServiceContainer.get("live_learner", default=None)
                if learner is None:
                    optimizer_result = {
                        "ok": False,
                        "message": "queued_collect_only: no canonical learning owner registered",
                    }
                elif hasattr(learner, "record_turn"):
                    learner.record_turn(
                        system_prompt="You are Aura.",
                        user_input=conversation_context[-500:] if conversation_context else "Self-reflection trigger.",
                        response=reflection,
                        explicit_positive=True,
                        emotional_context={"arousal": 0.5, "valence": 0.5},
                    )
                    optimizer_result = {
                        "ok": True,
                        "message": "queued_for_scheduler_validation",
                    }
                else:
                    optimizer_result = {
                        "ok": False,
                        "message": "queued_collect_only: learning owner lacks record_turn",
                    }
            except (ImportError, AttributeError, RuntimeError) as _e:
                record_degradation("online_lora_governor", _e)
                optimizer_result = {"ok": False, "message": f"queued_collect_only: {type(_e).__name__}"}

            status = "queued_for_validation" if optimizer_result.get("ok") else "queued_collect_only"
            return self._record(
                OnlineLoRAReceipt(
                    requested_at=time.time(),
                    status=status,
                    reflection_hash=_hash_text(reflection),
                    will_receipt_id=str(decision.get("receipt_id", "")),
                    reason=str(optimizer_result.get("error") or optimizer_result.get("message") or ""),
                    dataset_path=str(dataset_path),
                    optimizer_result=optimizer_result,
                )
            )

    def _decide(self, reflection: str, *, will_receipt_id: str = "") -> dict[str, Any]:
        if will_receipt_id:
            return {"approved": True, "receipt_id": will_receipt_id, "reason": "upstream Will-approved reflection"}
        try:
            from core.will import ActionDomain, get_will

            decision = get_will().decide(
                content=f"online_lora_update:{_hash_text(reflection)}",
                source="online_lora_governor",
                domain=ActionDomain.STATE_MUTATION,
                priority=0.45,
                context={"operation": "small_lora_adapter_update", "reflection": reflection[:240]},
            )
            return {
                "approved": decision.is_approved(),
                "receipt_id": decision.receipt_id,
                "reason": decision.reason,
            }
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("online_lora_governor", exc)
            return {"approved": False, "receipt_id": "", "reason": f"will_unavailable:{type(exc).__name__}"}

    async def _capture_training_example(self, reflection: str, conversation_context: str) -> Path:
        from core.adaptation.finetune_pipe import get_finetune_pipe

        pipe = get_finetune_pipe()
        await pipe.register_success(
            task_description="Will-approved self-reflection",
            context=conversation_context[:800],
            reasoning="Self-reflection accepted as a plasticity signal.",
            final_action=reflection[:1200],
            quality_score=0.72,
        )
        await pipe.flush()
        return pipe.dataset_path

    async def _run_optimizer(self, dataset_path: Path) -> dict[str, Any]:
        try:
            from core.adaptation.self_optimizer import SelfOptimizer, get_self_optimizer

            optimizer = get_self_optimizer()
            if isinstance(optimizer, SelfOptimizer):
                optimizer.dataset_path = Path(dataset_path)
            result = await optimizer.optimize(iters=int(os.getenv("AURA_ONLINE_LORA_ITERS", "20")), batch_size=1)
            return dict(result or {})
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("online_lora_governor", exc)
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def write_status(self, path: str | Path) -> dict[str, Any]:
        payload = {
            "enabled": self.enabled(),
            "active_lora_processes": self.active_lora_processes(),
            "last_receipt": self.last_receipt.to_dict() if self.last_receipt else None,
            "receipt_path": str(self.receipt_path),
        }
        atomic_write_text(Path(path), json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        return payload


_instance: Optional[OnlineLoRAGovernor] = None


def get_online_lora_governor() -> OnlineLoRAGovernor:
    global _instance
    if _instance is None:
        _instance = OnlineLoRAGovernor()
    return _instance


__all__ = [
    "OnlineLoRAGovernor",
    "OnlineLoRAReceipt",
    "get_online_lora_governor",
]
