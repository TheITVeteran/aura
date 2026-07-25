"""Can the parameters we may train actually fix the failures we observe?

A training run answers a question of the form "which weights reduce this
loss". It cannot answer it usefully if the weights it is allowed to touch
have no causal path to the thing that is failing. That is not a tuning
problem — no learning rate, group size or reward shaping repairs it — and it
is invisible from inside the optimizer, which simply sees a flat objective.

This has a measured cost in this repository. Seven consecutive resident
recurrent-GRPO campaigns (cp259, 271, 273, 285, 291, 294, 305) ran with
``adapter_scope = latent_slots_only`` — trainable parameters confined to the
recurrent slot window. Their receipts record where the episodes actually
failed:

    score_reasons    {"unparseable": 36}                 36 of 36
    contract_reasons {"no_marker": 32, "marker_line_has_no_object": 4}

Every failure is an OUTPUT-CONTRACT failure in the decode path: the model
burned its whole token budget without ever emitting the answer marker. CP226
had already localised this — at recurrent depth 4 and 8 the answered rate
collapses to 4% and 0% while every state stays finite, because the coda
layers have never seen a state that passed through the middle block that
many times and the output distribution degrades.

The decode path is not in ``latent_slots_only``. So each campaign spent
~86 minutes optimising a component that, by construction, could not repair
the measured failure; every reward was zero, every group was degenerate,
every advantage was a vector of zeros, and no gradient existed to follow.

This module makes that mismatch a preflight verdict instead of a discovery
made after the compute is gone. It is deliberately conservative: it refuses
only when the observed evidence is strong and clearly attributable, and
otherwise returns "unknown" and lets the run proceed. A guard that blocks
legitimate work is worse than the failure it prevents.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SCHEMA = "aura.scope_reachability.v1"

# Sites in the forward path that a failure can be attributed to.
SITE_RECURRENT_SLOTS = "recurrent_slots"
SITE_DECODE = "decode_path"
SITE_PROMPT = "prompt_or_task"
SITE_UNKNOWN = "unknown"

# What each trainable scope can actually change. A scope absent from this
# map is treated as unknown rather than assumed omnipotent — assuming reach
# is how this defect arises in the first place.
SCOPE_REACHES: dict[str, frozenset[str]] = {
    # The recurrent slot window only: the adapter is dark outside it, which
    # CP227 already proved empirically when an accuracy gate read as VOID
    # because the adapter never activated beyond its scope.
    "latent_slots_only": frozenset({SITE_RECURRENT_SLOTS}),
    # A full-model LoRA or fine-tune reaches the decode path too.
    "full_model": frozenset({SITE_RECURRENT_SLOTS, SITE_DECODE}),
    "coda_and_late_layers": frozenset({SITE_DECODE}),
    "decode_path": frozenset({SITE_DECODE}),
}

# Where an observed failure reason lives. These identifiers are the ones the
# runtime actually emits (answer_contract.py, verifiable_tasks.py), not
# invented labels.
FAILURE_SITES: dict[str, str] = {
    # Output-contract failures: the answer never took the required shape.
    # Fixing these means changing what the model EMITS, which is the decode
    # path — the coda layers and output head.
    "no_marker": SITE_DECODE,
    "marker_line_has_no_object": SITE_DECODE,
    "unparseable": SITE_DECODE,
    "unparseable_output": SITE_DECODE,
    "token_limit": SITE_DECODE,
    "empty_output": SITE_DECODE,
    # A wrong-but-well-formed answer is a reasoning failure: the recurrent
    # computation produced the wrong content in the right shape, which is
    # exactly what the slot window can learn to fix.
    "incorrect": SITE_RECURRENT_SLOTS,
    "incorrect_lenient": SITE_RECURRENT_SLOTS,
    "wrong_answer": SITE_RECURRENT_SLOTS,
    # Task-side problems no weight update can fix.
    "task_malformed": SITE_PROMPT,
    "prompt_too_long": SITE_PROMPT,
}

# Below this many attributed observations the sample says nothing, and the
# verdict is unknown rather than a refusal.
MIN_OBSERVATIONS = 12
# Share of attributable failures that must sit OUTSIDE the trainable scope
# before the run is refused. Set high: a run must be clearly futile, not
# merely difficult, before a guard stops it.
UNREACHABLE_SHARE = 0.9

REACHABLE = "reachable"
UNREACHABLE = "unreachable"
UNKNOWN = "unknown"


# A refusal that does not say what would work is only half a finding.
_REMEDIES: dict[str, str] = {
    SITE_DECODE: (
        "the failures are in the decode path, so the fix must reach it: "
        "widen the trainable scope to the coda/late layers, or run a "
        "distillation pass first (tools/latent_consolidation_train.py, "
        "core/learning/latent_adapter_distillation.py) to teach those layers "
        "to accept recurrent states. Distillation gives a dense per-token "
        "signal and does not depend on a nonzero base reward rate, which is "
        "what RL here does not have"
    ),
    SITE_PROMPT: (
        "the failures are in the tasks or prompts, which no weight update "
        "repairs; fix the task generator or the prompt contract first"
    ),
}


def _remedy_for(sites: list[str]) -> str:
    parts = [_REMEDIES[site] for site in sites if site in _REMEDIES]
    return "; ".join(parts)


@dataclass
class ReachabilityVerdict:
    """Whether training under this scope can affect the observed failures."""

    verdict: str
    scope: str
    reachable_sites: tuple[str, ...] = ()
    observations: int = 0
    attributed: int = 0
    in_scope: int = 0
    out_of_scope: int = 0
    by_site: dict[str, int] = field(default_factory=dict)
    unrecognized: dict[str, int] = field(default_factory=dict)
    detail: str = ""
    remedy: str = ""

    @property
    def should_refuse(self) -> bool:
        """Only a positive UNREACHABLE verdict refuses. Unknown proceeds."""
        return self.verdict == UNREACHABLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "verdict": self.verdict,
            "scope": self.scope,
            "reachable_sites": list(self.reachable_sites),
            "observations": self.observations,
            "attributed": self.attributed,
            "in_scope": self.in_scope,
            "out_of_scope": self.out_of_scope,
            "by_site": dict(self.by_site),
            "unrecognized": dict(self.unrecognized),
            "detail": self.detail,
            "remedy": self.remedy,
        }


def assess(
    failure_reasons: dict[str, int] | None,
    *,
    adapter_scope: str,
    min_observations: int = MIN_OBSERVATIONS,
    unreachable_share: float = UNREACHABLE_SHARE,
) -> ReachabilityVerdict:
    """Judge whether ``adapter_scope`` can reach the observed failures.

    ``failure_reasons`` is a count per reason, exactly the shape the GRPO
    receipts already record as ``score_reasons`` and ``contract_reasons``.
    """
    counts = {
        str(reason): int(count)
        for reason, count in (failure_reasons or {}).items()
        if isinstance(count, int) and not isinstance(count, bool) and count > 0
    }
    observations = sum(counts.values())

    reaches = SCOPE_REACHES.get(str(adapter_scope or ""))
    if reaches is None:
        return ReachabilityVerdict(
            UNKNOWN,
            str(adapter_scope or ""),
            observations=observations,
            detail=(
                f"scope {adapter_scope!r} is not declared in SCOPE_REACHES; "
                "reach cannot be assumed"
            ),
        )

    by_site: dict[str, int] = {}
    unrecognized: dict[str, int] = {}
    for reason, count in counts.items():
        site = FAILURE_SITES.get(reason)
        if site is None:
            unrecognized[reason] = count
            continue
        by_site[site] = by_site.get(site, 0) + count

    attributed = sum(by_site.values())
    in_scope = sum(count for site, count in by_site.items() if site in reaches)
    out_of_scope = attributed - in_scope

    base = ReachabilityVerdict(
        UNKNOWN,
        str(adapter_scope),
        reachable_sites=tuple(sorted(reaches)),
        observations=observations,
        attributed=attributed,
        in_scope=in_scope,
        out_of_scope=out_of_scope,
        by_site=by_site,
        unrecognized=unrecognized,
    )

    if attributed < min_observations:
        base.detail = (
            f"only {attributed} attributable observation(s); "
            f"{min_observations} needed before refusing a run"
        )
        return base

    if in_scope > 0:
        base.verdict = REACHABLE
        base.detail = (
            f"{in_scope} of {attributed} failures are at sites this scope can "
            f"change ({', '.join(sorted(reaches))})"
        )
        return base

    # Measured against ALL observations, not just the attributable ones.
    # Dividing by `attributed` would make this ratio 1.0 whenever nothing is
    # in scope — the threshold would never bind, and a profile that is half
    # unclassifiable would be refused on the strength of the half we happen
    # to recognise. Unrecognized failures are exactly the case where we do
    # not know enough to stop someone's run.
    share = out_of_scope / observations if observations else 0.0
    if share >= unreachable_share:
        outside = sorted(site for site in by_site if site not in reaches)
        base.verdict = UNREACHABLE
        base.remedy = _remedy_for(sorted(site for site in by_site if site not in reaches))
        base.detail = (
            f"{out_of_scope} of {attributed} failures ({share:.0%}) are at "
            f"{', '.join(outside)}, which scope {adapter_scope!r} cannot "
            f"change (it reaches {', '.join(sorted(reaches))}). Training "
            "cannot reduce a loss it has no causal path to."
        )
        return base

    base.detail = (
        f"{out_of_scope} of {observations} observed failures are out of scope "
        f"({share:.0%}), below the {unreachable_share:.0%} refusal threshold; "
        f"{sum(unrecognized.values())} unrecognized"
    )
    return base


def merge_reason_counts(*sources: dict[str, int] | None) -> dict[str, int]:
    """Combine several reason dicts (score_reasons, contract_reasons, ...)."""
    merged: dict[str, int] = {}
    for source in sources:
        for reason, count in (source or {}).items():
            if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                continue
            merged[str(reason)] = merged.get(str(reason), 0) + int(count)
    return merged


__all__ = [
    "FAILURE_SITES",
    "MIN_OBSERVATIONS",
    "REACHABLE",
    "SCHEMA",
    "SCOPE_REACHES",
    "UNKNOWN",
    "UNREACHABLE",
    "UNREACHABLE_SHARE",
    "ReachabilityVerdict",
    "assess",
    "merge_reason_counts",
]
