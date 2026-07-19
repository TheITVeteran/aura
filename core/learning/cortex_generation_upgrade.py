"""core/learning/cortex_generation_upgrade.py

Cortex generation upgrades — the governed path to frontier-grade compiled
understanding in her OWN weights.

The Compiled Understanding Layer closes the assimilation gap with machinery;
this pipeline closes it at the substrate: replacing the base checkpoint with
a newer generation (e.g. Qwen2.5→Qwen3 class) whose weights carry richer
conceptual machinery — while preserving identity, and never deciding alone.

The pipeline is complete, tested software; the DECISION is deliberately not
software. Swapping the mind's base model is an identity-level act, so
activation hard-requires an explicit operator authorization string and a
PASS evaluation receipt. That gate is the design, not a limitation.

Stages, each receipted:

  evaluate   — run the capability battery (breadth cloze probes, verifiable
               reasoning micro-tasks, identity behavior snapshot) on the
               current and candidate models, behind a MEMORY GUARD that
               refuses to load a candidate the host cannot afford beside
               the live processes (a second 32B during a training run is a
               memory incident, not an experiment).
  plan       — enumerate the identity artifacts a generation swap must
               migrate (fused persona/CRSM deltas, CAA steering vectors,
               expert adapters), with per-step lane: what is automatic and
               what requires an operator-launched training run.
  stage      — write the staged activation pointer + a byte-exact rollback
               copy of the current pointer. Nothing live changes.
  activate   — flip training/fused-model/active.json to the staged target
               (governed write). Requires authorization + PASS evaluation.
               Takes effect at the next boot — the running mind is never
               hot-swapped.
  rollback   — restore the rollback copy, byte-exact, verified.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.CortexGenerationUpgrade")

EVALUATION_SCHEMA = "aura.cortex_upgrade.evaluation.v1"
MIGRATION_PLAN_SCHEMA = "aura.cortex_upgrade.migration_plan.v1"
STAGING_SCHEMA = "aura.cortex_upgrade.staging.v1"
ACTIVATION_SCHEMA = "aura.cortex_upgrade.activation.v1"

STAGED_POINTER_NAME = "active.json.staged"
ROLLBACK_POINTER_NAME = "active.json.rollback"

# Memory guard: candidate projected RSS = on-disk weight bytes × this factor
# (activation buffers, cache); the host must retain this many GB free AFTER
# the load or the guard refuses.
_LOAD_OVERHEAD_FACTOR = 1.3
_FREE_MARGIN_GB = 8.0

# Breadth probes: factual cloze items with acceptable-answer alternates.
# Deterministic greedy decoding, case-insensitive containment scoring. These
# measure compiled knowledge, not retrieval — no tools, no context.
BREADTH_PROBES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("The chemical symbol for gold is", ("au",)),
    ("The powerhouse of the cell is the", ("mitochondri",)),
    ("The speed of light in vacuum is approximately", ("3", "300,000", "299")),
    ("The author of 'On the Origin of Species' was", ("darwin",)),
    ("The capital of Japan is", ("tokyo",)),
    ("Water's chemical formula is", ("h2o",)),
    ("The largest planet in the solar system is", ("jupiter",)),
    ("The derivative of x squared is", ("2x",)),
    ("DNA stands for", ("deoxyribonucleic",)),
    ("The French Revolution began in the year", ("1789",)),
    ("In computing, CPU stands for", ("central processing",)),
    ("The square root of 144 is", ("12",)),
    ("The theory of general relativity was published by", ("einstein",)),
    ("The smallest prime number is", ("2",)),
    ("Photosynthesis converts carbon dioxide and water into", ("glucose", "sugar", "oxygen")),
    ("The longest river in Africa is the", ("nile",)),
    ("An algorithm's O(n log n) sorting example is", ("merge", "heap", "quick")),
    ("The currency of the United Kingdom is the", ("pound", "sterling")),
    ("Shakespeare wrote the tragedy of Prince Hamlet of", ("denmark",)),
    ("The boiling point of water at sea level in Celsius is", ("100",)),
    ("The human heart has this many chambers:", ("4", "four")),
    ("The most abundant gas in Earth's atmosphere is", ("nitrogen",)),
    ("In SQL, the command to retrieve rows is", ("select",)),
    ("The Pythagorean theorem relates the sides of a", ("right", "triangle")),
)


def _greedy_decode(model, tokenizer, prompt: str, *, max_tokens: int = 10) -> str:
    """Minimal deterministic decode for battery probes (no cache reuse)."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.models.cache import KVCache

    try:
        tokens = list(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=True,
            )
        )
    except (AttributeError, TypeError, ValueError):
        tokens = list(tokenizer.encode(prompt))
    eos: set[int] = set()
    eid = getattr(tokenizer, "eos_token_id", None)
    if eid is not None:
        eos.add(int(eid))
    for extra in getattr(tokenizer, "eos_token_ids", None) or ():
        eos.add(int(extra))

    inner = model.model
    cache = [KVCache() for _ in inner.layers]
    h = inner.embed_tokens(mx.array([tokens]))
    mask = create_attention_mask(h, cache)
    for index, layer in enumerate(inner.layers):
        h = layer(h, mask, cache[index])
    h = inner.norm(h[:, -1:, :])
    head = getattr(model, "lm_head", None)
    logits = (head(h) if head is not None else inner.embed_tokens.as_linear(h))[0, -1]
    out: list[int] = []
    token = int(mx.argmax(logits))
    for _ in range(max_tokens):
        if token in eos:
            break
        out.append(token)
        h = inner.embed_tokens(mx.array([[token]]))
        mask = create_attention_mask(h, cache)
        for index, layer in enumerate(inner.layers):
            h = layer(h, mask, cache[index])
        h = inner.norm(h)
        logits = (head(h) if head is not None else inner.embed_tokens.as_linear(h))[0, -1]
        token = int(mx.argmax(logits))
    try:
        return str(tokenizer.decode(out))
    except (TypeError, ValueError, KeyError):
        return ""


def capability_battery(model, tokenizer, *, label: str = "model") -> dict[str, Any]:
    """Deterministic capability measurement: breadth + reasoning + identity.

    Breadth: factual cloze accuracy (compiled knowledge, closed book).
    Reasoning: verifiable micro-tasks from the falsification generators.
    Identity: the natural-probe behavior snapshot (for migration DELTAS —
    identity is compared across models, never pass/failed in isolation).
    """
    from core.brain.llm.latent_cortex.experiments import modular_chain, nested_boolean
    from core.learning.interference_battery import (
        natural_stability_probes,
        snapshot_probe_behavior,
    )

    started = time.monotonic()
    breadth_hits = 0
    breadth_rows: list[dict[str, Any]] = []
    for prompt, accepted in BREADTH_PROBES:
        answer = _greedy_decode(model, tokenizer, prompt, max_tokens=10).lower()
        hit = any(option in answer for option in accepted)
        breadth_hits += int(hit)
        breadth_rows.append({"prompt": prompt, "answer": answer[:60], "hit": hit})

    reasoning_hits = 0
    reasoning_total = 0
    for seed in range(6):
        for task in (modular_chain(3, seed=seed), nested_boolean(3, seed=seed)):
            reasoning_total += 1
            answer = _greedy_decode(model, tokenizer, task.prompt, max_tokens=48)
            reasoning_hits += int(task.verify(answer))

    try:
        identity_snapshot = snapshot_probe_behavior(
            model, natural_stability_probes(tokenizer)
        )
        identity_digests = [row["digest"] for row in identity_snapshot]
    except (ValueError, AttributeError, TypeError, RuntimeError) as exc:
        record_degradation(
            "cortex_upgrade",
            exc,
            action="recorded capability battery without an identity snapshot",
        )
        identity_digests = []

    return {
        "schema": EVALUATION_SCHEMA,
        "label": label,
        "breadth_accuracy": round(breadth_hits / len(BREADTH_PROBES), 4),
        "breadth_hits": breadth_hits,
        "breadth_total": len(BREADTH_PROBES),
        "breadth_rows": breadth_rows,
        "reasoning_accuracy": round(reasoning_hits / max(1, reasoning_total), 4),
        "reasoning_hits": reasoning_hits,
        "reasoning_total": reasoning_total,
        "identity_digests": identity_digests,
        "elapsed_s": round(time.monotonic() - started, 3),
    }


def compare_batteries(
    current: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """The upgrade verdict: candidate must WIN breadth and NOT LOSE reasoning."""
    breadth_delta = candidate["breadth_accuracy"] - current["breadth_accuracy"]
    reasoning_delta = candidate["reasoning_accuracy"] - current["reasoning_accuracy"]
    identity_changed = current.get("identity_digests") != candidate.get(
        "identity_digests"
    )
    verdict = "PASS" if breadth_delta > 0 and reasoning_delta >= -0.05 else "FAIL"
    return {
        "schema": EVALUATION_SCHEMA,
        "current_label": current["label"],
        "candidate_label": candidate["label"],
        "breadth_delta": round(breadth_delta, 4),
        "reasoning_delta": round(reasoning_delta, 4),
        "identity_behavior_changed": identity_changed,
        "identity_note": (
            "a new generation ALWAYS changes identity behavior — migration "
            "(persona retrain + steering re-extraction) is what restores it; "
            "this field feeds the migration plan, it does not gate the verdict"
        ),
        "verdict": verdict,
        "compared_at": time.time(),
    }


class MemoryGuard:
    """Refuses candidate loads the host cannot afford beside live processes."""

    def __init__(
        self,
        *,
        overhead_factor: float = _LOAD_OVERHEAD_FACTOR,
        free_margin_gb: float = _FREE_MARGIN_GB,
    ) -> None:
        self.overhead_factor = float(overhead_factor)
        self.free_margin_gb = float(free_margin_gb)

    @staticmethod
    def _weights_bytes(model_dir: Path) -> int:
        total = 0
        for pattern in ("*.safetensors", "*.npz", "*.gguf"):
            for file in Path(model_dir).glob(pattern):
                total += file.stat().st_size
        return total

    @staticmethod
    def _resident_giants_gb(threshold_gb: float = 6.0) -> list[dict[str, Any]]:
        """Python processes holding model-scale RSS (live app, training runs)."""
        try:
            import psutil
        except ImportError:
            return []
        giants = []
        for proc in psutil.process_iter(["pid", "name", "memory_info"]):
            try:
                info = proc.info
                rss_gb = (info["memory_info"].rss if info["memory_info"] else 0) / 1024**3
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            if rss_gb >= threshold_gb:
                giants.append({"pid": info["pid"], "name": info["name"], "rss_gb": round(rss_gb, 1)})
        return giants

    @staticmethod
    def _available_gb() -> float:
        try:
            import psutil

            return psutil.virtual_memory().available / 1024**3
        except ImportError:
            return 0.0

    def admit(self, model_dir: Path | str) -> dict[str, Any]:
        model_dir = Path(model_dir)
        weights_gb = self._weights_bytes(model_dir) / 1024**3
        projected_gb = weights_gb * self.overhead_factor
        available_gb = self._available_gb()
        giants = self._resident_giants_gb()
        admitted = (
            weights_gb > 0
            and available_gb - projected_gb >= self.free_margin_gb
        )
        receipt = {
            "model_dir": str(model_dir),
            "weights_gb": round(weights_gb, 2),
            "projected_load_gb": round(projected_gb, 2),
            "available_gb": round(available_gb, 2),
            "free_margin_gb": self.free_margin_gb,
            "resident_giants": giants,
            "admitted": admitted,
        }
        if not admitted:
            reason = (
                "no weight files found"
                if weights_gb == 0
                else "insufficient memory headroom beside resident processes"
            )
            receipt["refusal_reason"] = reason
        return receipt


@dataclass
class MigrationStep:
    name: str
    artifact: str
    exists: bool
    lane: str  # automatic | operator_training_run | operator_review
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "artifact": self.artifact,
            "exists": self.exists,
            "lane": self.lane,
            "detail": self.detail,
        }


def build_migration_plan(
    *,
    fused_model_dir: Path | str | None = None,
    data_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Enumerate what a generation swap must carry across, honestly laned.

    A new base is NOT Aura until her identity artifacts are rebuilt on it:
    the persona/CRSM delta must be RETRAINED (deltas are basis-specific),
    steering vectors RE-EXTRACTED, and expert adapters retrained or retired.
    Each step names its lane; nothing here pretends migration is a copy.
    """
    if fused_model_dir is None:
        from core.brain.llm.model_registry import BASE_DIR

        fused_model_dir = Path(BASE_DIR) / "training" / "fused-model"
    if data_dir is None:
        from core.config import DATA_DIR

        data_dir = Path(DATA_DIR)
    fused_model_dir = Path(fused_model_dir)
    data_dir = Path(data_dir)

    pointer = fused_model_dir / "active.json"
    steering = data_dir / "steering_vectors"
    adapters = data_dir / "expert_adapters"
    steps = [
        MigrationStep(
            name="activation_pointer",
            artifact=str(pointer),
            exists=pointer.is_file(),
            lane="automatic",
            detail="staged/activated/rolled back by this pipeline",
        ),
        MigrationStep(
            name="persona_crsm_delta",
            artifact=str(fused_model_dir),
            exists=fused_model_dir.is_dir(),
            lane="operator_training_run",
            detail=(
                "retrain the CRSM/persona delta against the NEW base "
                "(training/train_and_fuse.py --crsm-delta) and fuse a new "
                "artifact; low-rank deltas are basis-specific and never copy "
                "across generations"
            ),
        ),
        MigrationStep(
            name="caa_steering_vectors",
            artifact=str(steering),
            exists=steering.is_dir(),
            lane="operator_training_run",
            detail=(
                "re-extract CAA steering vectors from the new fused model; "
                "directions from the old activation basis do not transfer"
            ),
        ),
        MigrationStep(
            name="expert_adapters",
            artifact=str(adapters),
            exists=adapters.is_dir(),
            lane="operator_review",
            detail=(
                "retrain or retire domain expert LoRAs; each adapter's "
                "capture data can retrain against the new base through the "
                "existing compounding lanes"
            ),
        ),
        MigrationStep(
            name="recurrence_native_adapter",
            artifact="artifacts/closeout/latent_cortex/",
            exists=True,
            lane="operator_training_run",
            detail=(
                "rerun the recurrence-native curriculum on the new base so "
                "the RLC's trained recurrent mode carries into the next "
                "generation"
            ),
        ),
    ]
    return {
        "schema": MIGRATION_PLAN_SCHEMA,
        "steps": [step.to_dict() for step in steps],
        "automatic_steps": [s.name for s in steps if s.lane == "automatic"],
        "operator_steps": [s.name for s in steps if s.lane != "automatic"],
        "built_at": time.time(),
    }


def _read_pointer(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "active_model_path" not in payload:
        raise ValueError(f"activation pointer at {path} is not schema v2")
    return payload


def _governed_write(path: Path, payload: bytes, *, source: str) -> None:
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    gateway = get_file_write_gateway()
    with local_internal_governed_scope("cortex_generation_upgrade"):
        gateway.ensure_directory(path.parent, source=source)
        gateway.write_bytes(path, payload, source=source)


def stage_upgrade(
    *,
    candidate_model_path: Path | str,
    base_model_path: Path | str,
    tag: str,
    fused_model_dir: Path | str | None = None,
    evaluation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the staged pointer + byte-exact rollback. Nothing live changes."""
    if fused_model_dir is None:
        from core.brain.llm.model_registry import BASE_DIR

        fused_model_dir = Path(BASE_DIR) / "training" / "fused-model"
    fused_model_dir = Path(fused_model_dir)
    candidate = Path(candidate_model_path).expanduser()
    if not candidate.is_dir():
        raise ValueError(f"candidate model directory missing: {candidate}")

    pointer_path = fused_model_dir / "active.json"
    current_bytes = pointer_path.read_bytes()
    current = _read_pointer(pointer_path)

    staged_payload = {
        "active_model_path": str(candidate),
        "base_model": str(base_model_path),
        "fused_at": int(time.time()),
        "schema_version": 2,
        "size": current.get("size", "32B"),
        "tag": str(tag),
    }
    staged_bytes = (
        json.dumps(staged_payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _governed_write(
        fused_model_dir / ROLLBACK_POINTER_NAME,
        current_bytes,
        source="cortex_upgrade.stage",
    )
    _governed_write(
        fused_model_dir / STAGED_POINTER_NAME,
        staged_bytes,
        source="cortex_upgrade.stage",
    )
    receipt = {
        "schema": STAGING_SCHEMA,
        "staged_pointer": str(fused_model_dir / STAGED_POINTER_NAME),
        "rollback_pointer": str(fused_model_dir / ROLLBACK_POINTER_NAME),
        "current_active_model": current["active_model_path"],
        "staged_active_model": str(candidate),
        "staged_sha256": hashlib.sha256(staged_bytes).hexdigest(),
        "rollback_sha256": hashlib.sha256(current_bytes).hexdigest(),
        "evaluation_verdict": (evaluation or {}).get("verdict"),
        "staged_at": time.time(),
    }
    logger.info("🧬 Cortex upgrade STAGED: %s → %s", current["active_model_path"], candidate)
    return receipt


def activate_upgrade(
    *,
    fused_model_dir: Path | str | None = None,
    authorized_by: str,
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    """Flip the activation pointer to the staged target. Boot-time effect only.

    Hard gates, no overrides: a real operator authorization string and a
    PASS comparison verdict. The running mind is never hot-swapped — the
    new cortex exists only after the operator restarts.
    """
    if not isinstance(authorized_by, str) or len(authorized_by.strip()) < 3:
        raise PermissionError(
            "cortex activation requires an explicit operator authorization string"
        )
    if not isinstance(evaluation, dict) or evaluation.get("verdict") != "PASS":
        raise PermissionError(
            "cortex activation requires a PASS capability-comparison verdict"
        )
    if fused_model_dir is None:
        from core.brain.llm.model_registry import BASE_DIR

        fused_model_dir = Path(BASE_DIR) / "training" / "fused-model"
    fused_model_dir = Path(fused_model_dir)
    staged_path = fused_model_dir / STAGED_POINTER_NAME
    rollback_path = fused_model_dir / ROLLBACK_POINTER_NAME
    if not staged_path.is_file() or not rollback_path.is_file():
        raise ValueError("nothing staged: run stage_upgrade first")
    staged_bytes = staged_path.read_bytes()
    _read_pointer(staged_path)  # schema check before the flip
    _governed_write(
        fused_model_dir / "active.json", staged_bytes, source="cortex_upgrade.activate"
    )
    receipt = {
        "schema": ACTIVATION_SCHEMA,
        "activated_model": _read_pointer(staged_path)["active_model_path"],
        "active_sha256": hashlib.sha256(staged_bytes).hexdigest(),
        "authorized_by": authorized_by.strip(),
        "evaluation_verdict": "PASS",
        "effective": "next_boot",
        "activated_at": time.time(),
    }
    logger.info(
        "🧬 Cortex upgrade ACTIVATED by %s (effective at next boot)",
        authorized_by.strip(),
    )
    return receipt


def rollback_upgrade(*, fused_model_dir: Path | str | None = None) -> dict[str, Any]:
    """Restore the rollback pointer byte-exactly, verified by digest."""
    if fused_model_dir is None:
        from core.brain.llm.model_registry import BASE_DIR

        fused_model_dir = Path(BASE_DIR) / "training" / "fused-model"
    fused_model_dir = Path(fused_model_dir)
    rollback_path = fused_model_dir / ROLLBACK_POINTER_NAME
    if not rollback_path.is_file():
        raise ValueError("no rollback pointer exists")
    rollback_bytes = rollback_path.read_bytes()
    _read_pointer(rollback_path)
    _governed_write(
        fused_model_dir / "active.json", rollback_bytes, source="cortex_upgrade.rollback"
    )
    restored = (fused_model_dir / "active.json").read_bytes()
    exact = restored == rollback_bytes
    if not exact:
        record_degradation(
            "cortex_upgrade",
            RuntimeError("rollback restore was not byte-exact"),
            action="flagged rollback receipt; operator must inspect the pointer",
            severity="critical",
        )
    receipt = {
        "schema": ACTIVATION_SCHEMA,
        "restored_model": _read_pointer(rollback_path)["active_model_path"],
        "byte_exact": exact,
        "restored_sha256": hashlib.sha256(restored).hexdigest(),
        "rolled_back_at": time.time(),
    }
    logger.info("🧬 Cortex upgrade ROLLED BACK (byte_exact=%s)", exact)
    return receipt


__all__ = [
    "ACTIVATION_SCHEMA",
    "BREADTH_PROBES",
    "EVALUATION_SCHEMA",
    "MIGRATION_PLAN_SCHEMA",
    "MemoryGuard",
    "MigrationStep",
    "ROLLBACK_POINTER_NAME",
    "STAGED_POINTER_NAME",
    "STAGING_SCHEMA",
    "activate_upgrade",
    "build_migration_plan",
    "capability_battery",
    "compare_batteries",
    "rollback_upgrade",
    "stage_upgrade",
]
