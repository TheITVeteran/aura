"""Versioned execution contract shared by recurrence-native training and RLC.

The v1 trainer described a few hyperparameters while silently executing a
different graph from the live engine. This spec makes the causal object
serializable and hashable: slot construction, branch topology, recurrence,
layer boundaries, adapter scope, and explicitly supported post-transforms.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any

from core.brain.llm.latent_cortex.branches import BRANCH_ROLES
from core.brain.llm.latent_cortex.types import WorkspaceConfig

RLC_EXECUTION_SPEC_SCHEMA = "aura.rlc_execution_spec.v1"


@dataclass(frozen=True)
class RLCExecutionSpec:
    """Frozen computation contract for one recurrence-native model family."""

    schema: str = RLC_EXECUTION_SPEC_SCHEMA
    prelude_frac: float = 0.25
    coda_frac: float = 0.25
    n_slots: int = 16
    slot_seed: int = 0
    slot_roles: tuple[str, ...] = WorkspaceConfig().roles
    anchor_scale: float = 0.05
    branch_roles: tuple[str, ...] = BRANCH_ROLES[:2]
    exchange_interval: int = 1
    exchange_gamma: float = 0.35
    comm_slot: int = 0
    exchange_source_policy: str = (
        "bounded_private_reasoning_mean_excluding_mailbox_and_context_v1"
    )
    exchange_source_slot_limit: int = 16
    collapse_cos_threshold: float = 0.98
    jitter_scale: float = 0.02
    recurrent_steps: int = 2
    alpha: float = 0.5
    alpha_schedule: str = "constant"
    rms_clip_ratio: float = 3.0
    decode_bridge_policy: str = "none"
    adapter_scope: str = "latent_slots_only"
    adaptive_halting: bool = False
    latent_opt_mode: str = "disabled"
    fast_weights_mode: str = "disabled"
    branch_training_objective: str = "mean_answer_ce"

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.schema != RLC_EXECUTION_SPEC_SCHEMA:
            problems.append("unsupported execution spec schema")
        for name, value in (
            ("prelude_frac", self.prelude_frac),
            ("coda_frac", self.coda_frac),
        ):
            if not math.isfinite(value) or not 0.0 < value < 0.5:
                problems.append(f"{name} must be finite and inside (0, 0.5)")
        if self.prelude_frac + self.coda_frac >= 1.0:
            problems.append("prelude and coda leave no recurrent window")
        if type(self.n_slots) is not int or not 2 <= self.n_slots <= 128:
            problems.append("n_slots must be inside [2, 128]")
        if type(self.slot_seed) is not int:
            problems.append("slot_seed must be an integer")
        if not self.slot_roles or any(
            not isinstance(role, str) or not role.strip() for role in self.slot_roles
        ):
            problems.append("slot_roles must contain non-empty strings")
        if not math.isfinite(self.anchor_scale) or not 0.0 <= self.anchor_scale <= 1.0:
            problems.append("anchor_scale must be finite and inside [0, 1]")
        if not 1 <= len(self.branch_roles) <= 8 or any(
            not isinstance(role, str) or not role.strip() for role in self.branch_roles
        ):
            problems.append("branch_roles must contain one to eight non-empty strings")
        if type(self.exchange_interval) is not int or self.exchange_interval < 1:
            problems.append("exchange_interval must be positive")
        if not math.isfinite(self.exchange_gamma) or not 0.0 <= self.exchange_gamma <= 1.0:
            problems.append("exchange_gamma must be finite and inside [0, 1]")
        if type(self.comm_slot) is not int or not 0 <= self.comm_slot < self.n_slots:
            problems.append("comm_slot must identify a workspace slot")
        if self.exchange_source_policy != (
            "bounded_private_reasoning_mean_excluding_mailbox_and_context_v1"
        ):
            problems.append("exchange source policy is unsupported")
        if self.exchange_source_slot_limit != 16:
            problems.append("exchange source slot limit must be 16")
        if (
            not math.isfinite(self.collapse_cos_threshold)
            or not -1.0 <= self.collapse_cos_threshold <= 1.0
        ):
            problems.append("collapse_cos_threshold must be inside [-1, 1]")
        if not math.isfinite(self.jitter_scale) or not 0.0 <= self.jitter_scale <= 1.0:
            problems.append("jitter_scale must be finite and inside [0, 1]")
        if type(self.recurrent_steps) is not int or not 1 <= self.recurrent_steps <= 64:
            problems.append("recurrent_steps must be inside [1, 64]")
        if not math.isfinite(self.alpha) or not 0.0 < self.alpha <= 1.0:
            problems.append("alpha must be finite and inside (0, 1]")
        if self.alpha_schedule not in {"constant", "cosine"}:
            problems.append("alpha_schedule must be constant or cosine")
        if not math.isfinite(self.rms_clip_ratio) or self.rms_clip_ratio < 1.0:
            problems.append("rms_clip_ratio must be finite and at least 1")
        if self.decode_bridge_policy not in {"none", "assistant_answer"}:
            problems.append("decode_bridge_policy is unsupported")
        if self.adapter_scope != "latent_slots_only":
            problems.append("adapter_scope must be latent_slots_only")
        if self.adaptive_halting is not False:
            problems.append("v2 training requires fixed-depth recurrence")
        if self.latent_opt_mode != "disabled":
            problems.append("latent optimization is not certified for v2 training")
        if self.fast_weights_mode != "disabled":
            problems.append("fast weights are not certified for v2 training")
        if self.branch_training_objective != "mean_answer_ce":
            problems.append("branch_training_objective must be mean_answer_ce")
        return problems

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["slot_roles"] = list(self.slot_roles)
        payload["branch_roles"] = list(self.branch_roles)
        return payload

    @property
    def sha256(self) -> str:
        raw = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        return hashlib.sha256(raw).hexdigest()

    def with_depth(self, recurrent_steps: int) -> RLCExecutionSpec:
        candidate = replace(self, recurrent_steps=recurrent_steps)
        problems = candidate.validate()
        if problems:
            raise ValueError(f"invalid execution spec: {problems}")
        return candidate

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RLCExecutionSpec:
        if not isinstance(payload, Mapping):
            raise TypeError("execution spec must be a mapping")
        expected = set(cls.__dataclass_fields__)
        unknown = set(payload) - expected
        missing = expected - set(payload)
        if unknown or missing:
            raise ValueError(
                f"execution spec fields differ: missing={sorted(missing)} "
                f"unknown={sorted(unknown)}"
            )
        values = dict(payload)
        if isinstance(values.get("slot_roles"), list):
            values["slot_roles"] = tuple(values["slot_roles"])
        if isinstance(values.get("branch_roles"), list):
            values["branch_roles"] = tuple(values["branch_roles"])
        spec = cls(**values)
        problems = spec.validate()
        if problems:
            raise ValueError(f"invalid execution spec: {problems}")
        return spec


__all__ = ["RLC_EXECUTION_SPEC_SCHEMA", "RLCExecutionSpec"]
