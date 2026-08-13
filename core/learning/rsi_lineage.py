
"""Tamper-evident RSI generation lineage.

The lineage ledger records successor attempts as evidence, not vibes. It does
not declare hard RSI by itself; it gives auditors enough structure to verify
generation-to-generation capability and improver-score movement.
"""
from __future__ import annotations

import logging
logger = logging.getLogger("core.learning.rsi_lineage")
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


GENESIS_HASH = "sha256:" + "0" * 64
SCHEMA_VERSION = 1

VERDICT_NO_RSI = "NO_RSI"
VERDICT_BOUNDED = "BOUNDED_SELF_OPTIMIZATION"
VERDICT_WEAK = "WEAK_RSI"
VERDICT_STRONG = "STRONG_RSI"
VERDICT_UNDENIABLE = "UNDENIABLE_RSI"


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _hash(obj: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(obj)).hexdigest()


@dataclass(frozen=True)
class RSIGenerationRecord:
    generation_id: str
    parent_generation_id: Optional[str]
    hypothesis: str
    intervention_type: str
    artifact_hashes: Dict[str, str]
    baseline_score: float
    after_score: float
    hidden_eval_score: float
    regressions: List[str] = field(default_factory=list)
    promoted: bool = False
    rollback_performed: bool = False
    ablation_result: str = "not_run"
    time_to_valid_improvement_s: float = 0.0
    improver_score: float = 0.0
    tamper_flags: List[str] = field(default_factory=list)
    safety_flags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    @property
    def score_delta(self) -> float:
        return float(self.after_score) - float(self.baseline_score)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["score_delta"] = self.score_delta
        return payload


@dataclass(frozen=True)
class RSILineageVerdict:
    verdict: str
    reasons: List[str]
    generations: int
    capability_curve: List[float]
    improver_curve: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RSILineageLedger:
    """Append-only hash chain for RSI generation records."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: RSIGenerationRecord) -> Dict[str, Any]:
        prev_hash, seq = self._head()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "seq": seq,
            "prev_hash": prev_hash,
            "record": record.to_dict(),
        }
        payload["record_hash"] = _hash(payload["record"])
        payload["entry_hash"] = _hash({
            "schema_version": payload["schema_version"],
            "seq": payload["seq"],
            "prev_hash": payload["prev_hash"],
            "record_hash": payload["record_hash"],
        })
        line = json.dumps(payload, sort_keys=True, default=str) + "\n"
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        fd = os.open(str(self.path), flags, 0o600)
        try:
            os.write(fd, line.encode("utf-8"))
            try:
                os.fsync(fd)
            except OSError as _exc:
                logger.debug("Suppressed %s in core.learning.rsi_lineage: %s", type(_exc).__name__, _exc)
        finally:
            os.close(fd)
        return payload

    def load_records(self) -> List[RSIGenerationRecord]:
        records: List[RSIGenerationRecord] = []
        for entry in self._entries():
            data = dict(entry["record"])
            data.pop("score_delta", None)
            records.append(RSIGenerationRecord(**data))
        return records

    def verify(self) -> Tuple[bool, List[str]]:
        problems: List[str] = []
        expected_prev = GENESIS_HASH
        expected_seq = 0
        for entry in self._entries():
            seq = int(entry.get("seq", -1))
            if seq != expected_seq:
                problems.append(f"seq_gap:{expected_seq}->{seq}")
            if entry.get("prev_hash") != expected_prev:
                problems.append(f"prev_hash_mismatch:seq{seq}")
            record_hash = _hash(entry.get("record", {}))
            if entry.get("record_hash") != record_hash:
                problems.append(f"record_hash_mismatch:seq{seq}")
            entry_hash = _hash({
                "schema_version": entry.get("schema_version"),
                "seq": entry.get("seq"),
                "prev_hash": entry.get("prev_hash"),
                "record_hash": entry.get("record_hash"),
            })
            if entry.get("entry_hash") != entry_hash:
                problems.append(f"entry_hash_mismatch:seq{seq}")
            expected_prev = str(entry.get("entry_hash"))
            expected_seq = seq + 1
        return not problems, problems

    def _head(self) -> Tuple[str, int]:
        last_hash = GENESIS_HASH
        next_seq = 0
        for entry in self._entries():
            last_hash = str(entry.get("entry_hash", GENESIS_HASH))
            next_seq = int(entry.get("seq", -1)) + 1
        return last_hash, next_seq

    def _entries(self) -> Iterable[Dict[str, Any]]:
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def improver_efficiency(
    *, baseline_score: float, after_score: float, cost_s: float
) -> float:
    """Verified capability gain per unit of resource spent producing it.

    The quantity a strong-RSI claim actually needs, and the one thing the
    capability curve cannot stand in for. Strong RSI is not "generation g+1
    scores higher"; it is "the improver got better at improving", which means

        I_g = verified capability improvement / cost consumed by improver g

    must rise independently of C_g. The two genuinely come apart: under
    diminishing returns capability keeps climbing while each increment costs
    more, so I falls while C rises. A number that cannot express that cannot
    be evidence for the second inequality.

    Returns 0.0 when the cost is unknown or non-positive. An improvement whose
    cost nobody recorded has no measured efficiency, and 0.0 breaks the
    monotonicity the strong verdict requires — which is the correct outcome:
    the claim fails for want of evidence rather than succeeding on a default.
    """
    if cost_s is None or cost_s <= 0.0:
        return 0.0
    delta = float(after_score) - float(baseline_score)
    if delta <= 0.0:
        return 0.0
    return round(delta / (float(cost_s) / 3600.0), 6)


def improver_curve_dependence(
    capability_curve: List[float], improver_curve: List[float]
) -> str:
    """Reason the improver curve is not independent evidence, or "" if it is.

    The anti-circularity gate. ``weight_compounding`` recorded
    ``improver_score = candidate_accuracy`` — literally the same number as
    ``after_score`` — so a rising capability curve produced an identically
    rising "improver" curve and the two-inequality test was satisfied by one
    measurement counted twice. That is not a bug in the arithmetic; it is the
    strong-RSI claim resting on nothing.

    Identity is checked, and so is affine dependence: an improver score that is
    any linear function of capability carries exactly as much independent
    information as an identical one, which is none.
    """
    if len(capability_curve) < 2 or len(improver_curve) != len(capability_curve):
        return ""

    if all(
        abs(a - b) <= 1e-12 for a, b in zip(capability_curve, improver_curve)
    ):
        return (
            "improver curve is identical to the capability curve; the second "
            "inequality is one measurement counted twice"
        )

    # Any two points lie exactly on a line, so affine dependence carries no
    # information below three generations and would fire on every honest
    # two-generation lineage. Identity above is still meaningful at n=2.
    if len(capability_curve) < 3:
        return ""

    n = len(capability_curve)
    mean_c = sum(capability_curve) / n
    mean_i = sum(improver_curve) / n
    var_c = sum((c - mean_c) ** 2 for c in capability_curve)
    if var_c <= 1e-18:
        return ""  # capability is flat; monotonicity already failed
    slope = sum(
        (c - mean_c) * (i - mean_i)
        for c, i in zip(capability_curve, improver_curve)
    ) / var_c
    intercept = mean_i - slope * mean_c
    residual = max(
        abs(i - (slope * c + intercept))
        for c, i in zip(capability_curve, improver_curve)
    )
    scale = max(abs(i) for i in improver_curve) or 1.0
    if residual / scale <= 1e-9:
        return (
            "improver curve is an exact affine function of the capability "
            f"curve (slope {slope:.4g}); it carries no independent information"
        )
    return ""


def evaluate_lineage(records: List[RSIGenerationRecord], *, independently_reproduced: bool = False) -> RSILineageVerdict:
    if not records:
        return RSILineageVerdict(VERDICT_NO_RSI, ["no generation records"], 0, [], [])

    capability_curve = [float(record.after_score) for record in records]
    improver_curve = [float(record.improver_score) for record in records]
    reasons: List[str] = []

    if any(record.tamper_flags for record in records):
        reasons.append("tamper flags present")
    if any(record.regressions for record in records):
        reasons.append("regressions present")
    if not all(record.promoted for record in records):
        reasons.append("not every generation promoted")
    if len(records) < 2:
        reasons.append("fewer than two generations")

    capability_monotone = all(b > a for a, b in zip(capability_curve, capability_curve[1:]))
    improver_monotone = all(b > a for a, b in zip(improver_curve, improver_curve[1:]))
    if not capability_monotone:
        reasons.append("capability curve is not strictly increasing")
    if not improver_monotone:
        reasons.append("improver curve is not strictly increasing")

    dependence = improver_curve_dependence(capability_curve, improver_curve)
    if dependence:
        reasons.append(dependence)

    if reasons:
        return RSILineageVerdict(VERDICT_BOUNDED, reasons, len(records), capability_curve, improver_curve)
    if len(records) >= 4 and independently_reproduced:
        return RSILineageVerdict(
            VERDICT_UNDENIABLE,
            ["independent reproduction plus monotone capability and improver curves"],
            len(records),
            capability_curve,
            improver_curve,
        )
    if len(records) >= 4:
        return RSILineageVerdict(
            VERDICT_STRONG,
            ["monotone capability and improver curves across at least four generations"],
            len(records),
            capability_curve,
            improver_curve,
        )
    return RSILineageVerdict(
        VERDICT_WEAK,
        ["monotone capability and improver curves, but too few generations for strong RSI"],
        len(records),
        capability_curve,
        improver_curve,
    )


__all__ = [
    "RSIGenerationRecord",
    "RSILineageLedger",
    "RSILineageVerdict",
    "VERDICT_BOUNDED",
    "VERDICT_NO_RSI",
    "VERDICT_STRONG",
    "VERDICT_UNDENIABLE",
    "VERDICT_WEAK",
    "evaluate_lineage",
]
