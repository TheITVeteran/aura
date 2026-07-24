"""Bounded search over real recurrent latent/KV states.

The search controller owns topology and proof, while the engine owns neural
execution.  Private snapshots never enter a receipt: callbacks restore an
exact parent, execute one metered recurrent transition, and return only the
new snapshot plus its public state/KV hashes and bounded verifier evidence.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from core.brain.llm.latent_cortex.loop_core import canonical_sha256
from core.brain.llm.latent_cortex.resource_accounting import RESOURCE_COUNTERS
from core.brain.llm.latent_cortex.verified_best import (
    VerifierObservation,
    validate_observation,
)

LATENT_TREE_SEARCH_SCHEMA = "aura.rlc.latent_tree_search.v1"
LATENT_TREE_TRANSACTION_SCHEMA = "aura.rlc.latent_tree_search.transaction.v1"
LATENT_TREE_NODE_SCHEMA = "aura.rlc.latent_tree_search.node.v1"

DISABLED = "disabled"
ACTIVE = "active"
STRATEGIES = ("uct", "beam", "bfs")
MAX_NODES = 32
MAX_DEPTH = 6
MAX_BRANCHING = 8


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite(value: Any, *, name: str, minimum: float, maximum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not minimum <= float(value) <= maximum
    ):
        raise ValueError(f"{name} must be finite and inside [{minimum}, {maximum}]")
    return float(value)


def _ordinal_inventory(value: Any, *, name: str) -> list[int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or any(type(ordinal) is not int or ordinal < 0 for ordinal in value)
    ):
        raise ValueError(f"{name} is invalid")
    normalized = list(value)
    if normalized != sorted(set(normalized)):
        raise ValueError(f"{name} must be sorted and unique")
    return normalized


def _validate_resource_window(*, spent_layer_apps: Any, resource_delta: Any) -> None:
    if (
        type(spent_layer_apps) is not int
        or spent_layer_apps < 0
        or not isinstance(resource_delta, Mapping)
        or set(resource_delta) != set(RESOURCE_COUNTERS)
        or any(type(amount) is not int or amount < 0 for amount in resource_delta.values())
        or spent_layer_apps < resource_delta["transformer_layer_apps"]
    ):
        raise ValueError("latent tree resource window is invalid")


def _resource_snapshot(budget: Any) -> dict[str, int]:
    ledger = getattr(budget, "resource_ledger", None)
    totals = ledger.totals() if ledger is not None else None
    if not isinstance(totals, Mapping) or set(totals) != set(RESOURCE_COUNTERS):
        raise ValueError("latent tree resource ledger is unavailable")
    normalized: dict[str, int] = {}
    for name in RESOURCE_COUNTERS:
        value = totals[name]
        if type(value) is not int or value < 0:
            raise ValueError("latent tree resource counter is invalid")
        normalized[name] = value
    return normalized


def _resource_delta(before: Mapping[str, int], after: Mapping[str, int]) -> dict[str, int]:
    if set(before) != set(RESOURCE_COUNTERS) or set(after) != set(RESOURCE_COUNTERS):
        raise ValueError("latent tree resource windows differ")
    delta = {name: int(after[name]) - int(before[name]) for name in RESOURCE_COUNTERS}
    if any(value < 0 for value in delta.values()):
        raise ValueError("latent tree resource counters regressed")
    return delta


def _validate_probe_accounting(value: Any) -> dict[str, Any]:
    required = {
        "spent_layer_apps",
        "resource_delta",
        "probe_tokens_sha256",
        "probe_token_count",
        "target_branch",
        "probe_cache_hit",
        "probe_cache_key_sha256",
        "probe_cache_layer_apps_saved",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("latent tree probe accounting fields differ")
    row = dict(value)
    delta = row["resource_delta"]
    if (
        type(row["spent_layer_apps"]) is not int
        or type(row["probe_token_count"]) is not int
        or type(row["target_branch"]) is not int
        or row["spent_layer_apps"] < 0
        or row["probe_token_count"] <= 0
        or row["target_branch"] < 0
        or type(row["probe_cache_hit"]) is not bool
        or type(row["probe_cache_layer_apps_saved"]) is not int
        or row["probe_cache_layer_apps_saved"] < 0
        or not _is_sha256(row["probe_tokens_sha256"])
        or not isinstance(delta, Mapping)
        or set(delta) != set(RESOURCE_COUNTERS)
        or any(type(amount) is not int or amount < 0 for amount in delta.values())
        or row["spent_layer_apps"] < delta["transformer_layer_apps"]
        or delta["verifier_calls"] < 1
        or delta["verifier_input_bytes"] <= 0
        or delta["verifier_output_bytes"] <= 0
    ):
        raise ValueError("latent tree probe accounting is incomplete")
    if row["probe_cache_hit"]:
        if (
            not _is_sha256(row["probe_cache_key_sha256"])
            or row["probe_cache_layer_apps_saved"] <= 0
            or delta["transformer_layer_apps"] != 0
            or delta["output_head_tokens"] != 0
        ):
            raise ValueError("latent tree cached probe accounting differs")
    elif (
        (
            row["probe_cache_key_sha256"] not in {"", None}
            and not _is_sha256(row["probe_cache_key_sha256"])
        )
        or row["probe_cache_layer_apps_saved"] != 0
        or delta["transformer_layer_apps"] <= 0
        or delta["output_head_tokens"] != row["probe_token_count"]
    ):
        raise ValueError("latent tree uncached probe accounting differs")
    row["resource_delta"] = dict(delta)
    return row


def _observation(value: Any) -> VerifierObservation:
    if isinstance(value, Mapping) and "observation_sha256" in value:
        validated = validate_observation(value)
        return VerifierObservation.from_value(
            {
                key: validated[key]
                for key in (
                    "schema",
                    "score",
                    "lower_bound",
                    "upper_bound",
                    "sample_count",
                    "basis",
                    "independent",
                    "evidence_sha256",
                )
            }
        )
    return VerifierObservation.from_value(value)


def validate_branch_boundaries(value: Any) -> list[dict[str, Any]]:
    """Normalize the public, tensor-free membership of one ensemble state."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError("latent tree branch boundaries are invalid")
    normalized: list[dict[str, Any]] = []
    required = {
        "index",
        "state_sha256",
        "kv_boundary_sha256",
        "steps",
        "operator",
        "halted",
    }
    for expected_index, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise ValueError("latent tree branch boundary fields differ")
        row = dict(raw)
        if (
            row["index"] != expected_index
            or not _is_sha256(row["state_sha256"])
            or not _is_sha256(row["kv_boundary_sha256"])
            or type(row["steps"]) is not int
            or row["steps"] < 0
            or not isinstance(row["operator"], str)
            or not row["operator"]
            or type(row["halted"]) is not bool
        ):
            raise ValueError("latent tree branch boundary identity is invalid")
        normalized.append(row)
    return normalized


def ensemble_identity_from_boundaries(value: Any) -> tuple[str, str]:
    boundaries = validate_branch_boundaries(value)
    return (
        canonical_sha256(boundaries),
        canonical_sha256(
            [
                {
                    "index": row["index"],
                    "kv_boundary_sha256": row["kv_boundary_sha256"],
                }
                for row in boundaries
            ]
        ),
    )


@dataclass(frozen=True, slots=True)
class LatentTreeSearchConfig:
    mode: str = ACTIVE
    strategy: str = "uct"
    max_nodes: int = 5
    max_depth: int = 2
    branching_factor: int = 2
    exploration_weight: float = math.sqrt(2.0)
    min_verifier_margin: float = 0.01
    seed: int = 0

    def __post_init__(self) -> None:
        if self.mode not in {DISABLED, ACTIVE}:
            raise ValueError("latent tree mode is invalid")
        if self.strategy not in STRATEGIES:
            raise ValueError("latent tree strategy is invalid")
        if type(self.max_nodes) is not int or not 2 <= self.max_nodes <= MAX_NODES:
            raise ValueError(f"latent tree max_nodes must be inside [2, {MAX_NODES}]")
        if type(self.max_depth) is not int or not 1 <= self.max_depth <= MAX_DEPTH:
            raise ValueError(f"latent tree max_depth must be inside [1, {MAX_DEPTH}]")
        if (
            type(self.branching_factor) is not int
            or not 1 <= self.branching_factor <= MAX_BRANCHING
        ):
            raise ValueError(f"latent tree branching_factor must be inside [1, {MAX_BRANCHING}]")
        _finite(
            self.exploration_weight,
            name="latent tree exploration_weight",
            minimum=0.0,
            maximum=8.0,
        )
        _finite(
            self.min_verifier_margin,
            name="latent tree min_verifier_margin",
            minimum=0.0,
            maximum=0.25,
        )
        if type(self.seed) is not int or not -(2**63) <= self.seed <= 2**63 - 1:
            raise ValueError("latent tree seed must be signed 64-bit")

    @classmethod
    def from_value(cls, value: Mapping[str, Any] | None) -> LatentTreeSearchConfig:
        raw = dict(value or {})
        allowed = {
            "mode",
            "strategy",
            "max_nodes",
            "max_depth",
            "branching_factor",
            "exploration_weight",
            "min_verifier_margin",
            "seed",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"latent tree has unknown keys: {sorted(unknown)}")
        return cls(
            mode=raw.get("mode", ACTIVE),
            strategy=raw.get("strategy", "uct"),
            max_nodes=raw.get("max_nodes", 5),
            max_depth=raw.get("max_depth", 2),
            branching_factor=raw.get("branching_factor", 2),
            exploration_weight=raw.get("exploration_weight", math.sqrt(2.0)),
            min_verifier_margin=raw.get("min_verifier_margin", 0.01),
            seed=raw.get("seed", 0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "strategy": self.strategy,
            "max_nodes": self.max_nodes,
            "max_depth": self.max_depth,
            "branching_factor": self.branching_factor,
            "exploration_weight": round(float(self.exploration_weight), 10),
            "min_verifier_margin": round(float(self.min_verifier_margin), 10),
            "seed": self.seed,
        }


@dataclass(slots=True)
class _Node:
    index: int
    parent: int | None
    depth: int
    action: str
    state_sha256: str
    kv_boundary_sha256: str
    observation: VerifierObservation
    branch_boundaries: list[dict[str, Any]]
    snapshot: Any = field(repr=False)
    visits: int = 1
    value_sum: float = 0.0
    tried_actions: list[str] = field(default_factory=list)
    duplicate_of: int | None = None

    @property
    def mean_value(self) -> float:
        return self.value_sum / max(1, self.visits)


def _node_identity(*, episode_id: str, objective_sha256: str, node: _Node) -> str:
    return canonical_sha256(
        {
            "episode_id": episode_id,
            "objective_sha256": objective_sha256,
            "node_index": node.index,
            "parent_index": node.parent,
            "depth": node.depth,
            "action": node.action,
            "state_sha256": node.state_sha256,
            "kv_boundary_sha256": node.kv_boundary_sha256,
        }
    )


def _select_node(
    nodes: Sequence[_Node],
    *,
    config: LatentTreeSearchConfig,
    action_count: int,
) -> tuple[_Node, float]:
    expandable = [
        node
        for node in nodes
        if node.duplicate_of is None
        and node.depth < config.max_depth
        and len(node.tried_actions) < min(config.branching_factor, action_count)
    ]
    if not expandable:
        raise LookupError("latent tree has no expandable node")
    if config.strategy == "bfs":
        selected = min(expandable, key=lambda node: (node.depth, node.index))
        return selected, 0.0
    if config.strategy == "beam":
        selected = max(
            expandable,
            key=lambda node: (
                node.observation.lower_bound,
                node.observation.score,
                -node.depth,
                -node.index,
            ),
        )
        return selected, selected.observation.lower_bound

    def uct(node: _Node) -> float:
        parent_visits = (
            sum(item.visits for item in nodes) if node.parent is None else nodes[node.parent].visits
        )
        return node.mean_value + float(config.exploration_weight) * math.sqrt(
            math.log(max(1, parent_visits) + 1.0) / max(1, node.visits)
        )

    selected = max(expandable, key=lambda node: (uct(node), -node.depth, -node.index))
    return selected, uct(selected)


def _public_node(
    node: _Node,
    *,
    episode_id: str,
    objective_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": LATENT_TREE_NODE_SCHEMA,
        "index": node.index,
        "parent": node.parent,
        "depth": node.depth,
        "action": node.action,
        "node_sha256": _node_identity(
            episode_id=episode_id,
            objective_sha256=objective_sha256,
            node=node,
        ),
        "state_sha256": node.state_sha256,
        "kv_boundary_sha256": node.kv_boundary_sha256,
        "observation": node.observation.to_dict(),
        "branch_boundaries": [dict(row) for row in node.branch_boundaries],
        "visits": node.visits,
        "value_sum": round(node.value_sum, 10),
        "mean_value": round(node.mean_value, 10),
        "tried_actions": list(node.tried_actions),
        "duplicate_of": node.duplicate_of,
    }


def build_empty_latent_tree_receipt(
    *,
    episode_id: str,
    objective_sha256: str,
    config: LatentTreeSearchConfig,
    status: str = "not_invoked",
    reason: str = "branch_action_not_selected",
) -> dict[str, Any]:
    if not isinstance(episode_id, str) or not episode_id or not _is_sha256(objective_sha256):
        raise ValueError("latent tree identity is invalid")
    if not isinstance(config, LatentTreeSearchConfig):
        raise TypeError("latent tree config is invalid")
    if status not in {"disabled", "not_invoked", "unavailable"}:
        raise ValueError("latent tree empty status is invalid")
    payload = {
        "schema": LATENT_TREE_SEARCH_SCHEMA,
        "episode_id": episode_id,
        "objective_sha256": objective_sha256,
        "config": config.to_dict(),
        "status": status,
        "reason": reason,
        "transactions": [],
        "private_snapshots_disclosed": False,
        "answer_text_stored": False,
    }
    return {**payload, "receipt_sha256": canonical_sha256(payload)}


def append_transaction(receipt: dict[str, Any], transaction: Mapping[str, Any]) -> None:
    """Append one validated transaction and refresh the aggregate digest."""

    if not isinstance(receipt, dict) or receipt.get("schema") != LATENT_TREE_SEARCH_SCHEMA:
        raise ValueError("latent tree aggregate receipt is invalid")
    validated = validate_latent_tree_transaction(transaction)
    receipt["transactions"].append(validated)
    receipt["status"] = "executed"
    receipt["reason"] = "search_transactions_recorded"
    payload = dict(receipt)
    payload.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = canonical_sha256(payload)


def run_latent_tree_search(
    *,
    episode_id: str,
    objective_sha256: str,
    action_step: int,
    root_snapshot: Any,
    root_state_sha256: str,
    root_kv_boundary_sha256: str,
    root_branch_boundaries: Sequence[Mapping[str, Any]],
    root_observation: Any,
    root_evaluation: Mapping[str, Any],
    authority_observation: Any | None = None,
    actions: Sequence[str],
    config: LatentTreeSearchConfig,
    budget: Any,
    restore_snapshot: Callable[[Any], tuple[str, str]],
    expand: Callable[[str, int, int], Mapping[str, Any]],
    recurrent_call_inventory: Callable[[], Sequence[int]],
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Search, then atomically leave either the proven winner or exact root live."""

    if (
        not isinstance(episode_id, str)
        or not episode_id
        or not _is_sha256(objective_sha256)
        or type(action_step) is not int
        or action_step < 0
        or not _is_sha256(root_state_sha256)
        or not _is_sha256(root_kv_boundary_sha256)
        or not isinstance(config, LatentTreeSearchConfig)
    ):
        raise ValueError("latent tree transaction identity is invalid")
    normalized_actions = tuple(actions)
    if (
        not normalized_actions
        or len(normalized_actions) != len(set(normalized_actions))
        or any(not isinstance(action, str) or not action or len(action) > 80 for action in actions)
    ):
        raise ValueError("latent tree action inventory is invalid")
    root_obs = _observation(root_observation)
    if not root_obs.authoritative:
        raise ValueError("latent tree root observation is not authoritative")
    authority_obs = (
        root_obs if authority_observation is None else _observation(authority_observation)
    )
    if not authority_obs.authoritative:
        raise ValueError("latent tree authority observation is not authoritative")
    root_accounting = _validate_probe_accounting(root_evaluation)
    root_boundaries = validate_branch_boundaries(root_branch_boundaries)
    if ensemble_identity_from_boundaries(root_boundaries) != (
        root_state_sha256,
        root_kv_boundary_sha256,
    ):
        raise ValueError("latent tree root boundary commitment differs")
    if (
        not callable(restore_snapshot)
        or not callable(expand)
        or not callable(recurrent_call_inventory)
    ):
        raise TypeError("latent tree callbacks are invalid")
    inventory_before = _ordinal_inventory(
        recurrent_call_inventory(),
        name="latent tree recurrent-call inventory before search",
    )

    root = _Node(
        index=0,
        parent=None,
        depth=0,
        action="root",
        state_sha256=root_state_sha256,
        kv_boundary_sha256=root_kv_boundary_sha256,
        observation=root_obs,
        branch_boundaries=root_boundaries,
        snapshot=root_snapshot,
        value_sum=root_obs.score,
    )
    nodes = [root]
    selections: list[dict[str, Any]] = []
    expansions: list[dict[str, Any]] = []
    seen = {(root_state_sha256, root_kv_boundary_sha256): 0}
    cancelled = False
    failure = ""

    while len(nodes) < config.max_nodes:
        if cancel_check is not None and cancel_check():
            cancelled = True
            break
        try:
            parent, selection_value = _select_node(
                nodes,
                config=config,
                action_count=len(normalized_actions),
            )
        except LookupError:
            break
        action_offset = (config.seed + parent.index + len(parent.tried_actions)) % len(
            normalized_actions
        )
        action = next(
            normalized_actions[(action_offset + offset) % len(normalized_actions)]
            for offset in range(len(normalized_actions))
            if normalized_actions[(action_offset + offset) % len(normalized_actions)]
            not in parent.tried_actions
        )
        parent.tried_actions.append(action)
        selections.append(
            {
                "ordinal": len(selections),
                "parent": parent.index,
                "strategy": config.strategy,
                "selection_value": round(selection_value, 10),
                "parent_visits_before": parent.visits,
            }
        )
        restored_state, restored_kv = restore_snapshot(parent.snapshot)
        if restored_state != parent.state_sha256 or restored_kv != parent.kv_boundary_sha256:
            failure = "parent_restore_mismatch"
            break
        before_resources = _resource_snapshot(budget)
        spent_before = int(budget.spent_layer_apps)
        try:
            result = expand(action, parent.index, len(nodes))
            after_resources = _resource_snapshot(budget)
            spent_after = int(budget.spent_layer_apps)
            required = {
                "snapshot",
                "state_sha256",
                "kv_boundary_sha256",
                "observation",
                "transition_sha256",
                "target_branch",
                "probe_tokens_sha256",
                "probe_token_count",
                "branch_boundaries",
                "probe_cache_hit",
                "probe_cache_key_sha256",
                "probe_cache_layer_apps_saved",
                "recurrent_kv_call_ordinals",
            }
            if not isinstance(result, Mapping) or set(result) != required:
                raise ValueError("latent tree expansion fields differ")
            if (
                not _is_sha256(result["state_sha256"])
                or not _is_sha256(result["kv_boundary_sha256"])
                or not _is_sha256(result["transition_sha256"])
            ):
                raise ValueError("latent tree expansion identity is invalid")
            if type(result["target_branch"]) is not int or result["target_branch"] < 0:
                raise ValueError("latent tree expansion branch is invalid")
            observation = _observation(result["observation"])
            if not observation.authoritative:
                raise ValueError("latent tree expansion observation is not authoritative")
            resource_delta = _resource_delta(before_resources, after_resources)
            layer_apps = spent_after - spent_before
            _validate_probe_accounting(
                {
                    "spent_layer_apps": layer_apps,
                    "resource_delta": resource_delta,
                    "probe_tokens_sha256": result["probe_tokens_sha256"],
                    "probe_token_count": result["probe_token_count"],
                    "target_branch": result["target_branch"],
                    "probe_cache_hit": result["probe_cache_hit"],
                    "probe_cache_key_sha256": result["probe_cache_key_sha256"],
                    "probe_cache_layer_apps_saved": result["probe_cache_layer_apps_saved"],
                }
            )
            child_boundaries = validate_branch_boundaries(result["branch_boundaries"])
            recurrent_kv_call_ordinals = _ordinal_inventory(
                result["recurrent_kv_call_ordinals"],
                name="latent tree expansion recurrent-call ordinals",
            )
            if not recurrent_kv_call_ordinals:
                raise ValueError("latent tree expansion made no recurrent transition")
            if ensemble_identity_from_boundaries(child_boundaries) != (
                result["state_sha256"],
                result["kv_boundary_sha256"],
            ):
                raise ValueError("latent tree child boundary commitment differs")
        except (RuntimeError, TypeError, ValueError) as exc:
            after_resources = _resource_snapshot(budget)
            spent_after = int(budget.spent_layer_apps)
            restore_snapshot(parent.snapshot)
            failure = f"expansion_failed:{type(exc).__name__}"
            expansions.append(
                {
                    "ordinal": len(expansions),
                    "parent": parent.index,
                    "action": action,
                    "status": "failed",
                    "reason": failure,
                    "spent_layer_apps": spent_after - spent_before,
                    "resource_delta": _resource_delta(before_resources, after_resources),
                }
            )
            break
        key = (str(result["state_sha256"]), str(result["kv_boundary_sha256"]))
        duplicate_of = seen.get(key)
        child = _Node(
            index=len(nodes),
            parent=parent.index,
            depth=parent.depth + 1,
            action=action,
            state_sha256=key[0],
            kv_boundary_sha256=key[1],
            observation=observation,
            branch_boundaries=child_boundaries,
            snapshot=result["snapshot"],
            value_sum=observation.score,
            duplicate_of=duplicate_of,
        )
        nodes.append(child)
        if duplicate_of is None:
            seen[key] = child.index
        expansions.append(
            {
                "ordinal": len(expansions),
                "parent": parent.index,
                "child": child.index,
                "action": action,
                "status": "duplicate" if duplicate_of is not None else "expanded",
                "duplicate_of": duplicate_of,
                "transition_sha256": result["transition_sha256"],
                "target_branch": result["target_branch"],
                "probe_tokens_sha256": result["probe_tokens_sha256"],
                "probe_token_count": result["probe_token_count"],
                "probe_cache_hit": result["probe_cache_hit"],
                "probe_cache_key_sha256": result["probe_cache_key_sha256"],
                "probe_cache_layer_apps_saved": result["probe_cache_layer_apps_saved"],
                "recurrent_kv_call_ordinals": recurrent_kv_call_ordinals,
                "spent_layer_apps": layer_apps,
                "resource_delta": resource_delta,
            }
        )
        value = observation.score
        cursor: _Node | None = parent
        while cursor is not None:
            cursor.visits += 1
            cursor.value_sum += value
            cursor = None if cursor.parent is None else nodes[cursor.parent]

    candidates = [
        node
        for node in nodes[1:]
        if node.duplicate_of is None
        and node.observation.lower_bound
        > max(root.observation.upper_bound, authority_obs.upper_bound)
        + float(config.min_verifier_margin)
    ]
    winner = max(
        candidates,
        key=lambda node: (node.observation.lower_bound, node.observation.score, -node.index),
        default=None,
    )
    committed = winner is not None and not cancelled and not failure
    target = winner if committed else root
    final_state, final_kv = restore_snapshot(target.snapshot)
    restore_exact = final_state == target.state_sha256 and final_kv == target.kv_boundary_sha256
    if not restore_exact:
        raise RuntimeError("latent tree final restore mismatch")

    inventory_after = _ordinal_inventory(
        recurrent_call_inventory(),
        name="latent tree recurrent-call inventory after search",
    )
    before_set = set(inventory_before)
    after_set = set(inventory_after)
    if not before_set <= after_set:
        raise ValueError("latent tree recurrent-call inventory regressed")
    new_recurrent_ordinals = sorted(after_set - before_set)
    claimed_recurrent_ordinals = {
        ordinal
        for expansion in expansions
        for ordinal in expansion.get("recurrent_kv_call_ordinals", ())
    }
    if not claimed_recurrent_ordinals <= set(new_recurrent_ordinals):
        raise ValueError("latent tree expansion claimed an external recurrent call")
    committed_nodes = {0}
    if committed:
        cursor = target
        while cursor.parent is not None:
            committed_nodes.add(cursor.index)
            cursor = nodes[cursor.parent]
    committed_recurrent_ordinals = sorted(
        ordinal
        for expansion in expansions
        if expansion.get("child") in committed_nodes
        for ordinal in expansion.get("recurrent_kv_call_ordinals", ())
    )
    discarded_recurrent_ordinals = sorted(
        set(new_recurrent_ordinals) - set(committed_recurrent_ordinals)
    )

    public_nodes = [
        _public_node(node, episode_id=episode_id, objective_sha256=objective_sha256)
        for node in nodes
    ]
    status = "committed" if committed else "cancelled" if cancelled else "restored"
    reason = (
        "confidence_bound_winner"
        if committed
        else "cancelled_exact_root_restore"
        if cancelled
        else failure
        if failure
        else "no_candidate_dominated_root"
    )
    payload = {
        "schema": LATENT_TREE_TRANSACTION_SCHEMA,
        "episode_id": episode_id,
        "objective_sha256": objective_sha256,
        "action_step": action_step,
        "config": config.to_dict(),
        "action_inventory": list(normalized_actions),
        "root_node": 0,
        "authority_observation": authority_obs.to_dict(),
        "root_evaluation": root_accounting,
        "nodes": public_nodes,
        "selections": selections,
        "expansions": expansions,
        "recurrent_kv_new_call_ordinals": new_recurrent_ordinals,
        "committed_recurrent_kv_call_ordinals": committed_recurrent_ordinals,
        "discarded_recurrent_kv_call_ordinals": discarded_recurrent_ordinals,
        "duplicate_count": sum(node.duplicate_of is not None for node in nodes),
        "winner_node": winner.index if committed and winner is not None else None,
        "committed_node": target.index,
        "winner_dominates_root": committed,
        "final_state_sha256": final_state,
        "final_kv_boundary_sha256": final_kv,
        "restore_exact": restore_exact,
        "cancelled": cancelled,
        "status": status,
        "reason": reason,
        "private_snapshots_disclosed": False,
        "answer_text_stored": False,
    }
    return validate_latent_tree_transaction(
        {**payload, "receipt_sha256": canonical_sha256(payload)}
    )


def validate_latent_tree_transaction(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("latent tree transaction is not a mapping")
    receipt = dict(value)
    digest = receipt.pop("receipt_sha256", None)
    if not _is_sha256(digest) or canonical_sha256(receipt) != digest:
        raise ValueError("latent tree transaction digest differs")
    required = {
        "schema",
        "episode_id",
        "objective_sha256",
        "action_step",
        "config",
        "action_inventory",
        "root_node",
        "authority_observation",
        "root_evaluation",
        "nodes",
        "selections",
        "expansions",
        "recurrent_kv_new_call_ordinals",
        "committed_recurrent_kv_call_ordinals",
        "discarded_recurrent_kv_call_ordinals",
        "duplicate_count",
        "winner_node",
        "committed_node",
        "winner_dominates_root",
        "final_state_sha256",
        "final_kv_boundary_sha256",
        "restore_exact",
        "cancelled",
        "status",
        "reason",
        "private_snapshots_disclosed",
        "answer_text_stored",
    }
    if set(receipt) != required or receipt["schema"] != LATENT_TREE_TRANSACTION_SCHEMA:
        raise ValueError("latent tree transaction fields differ")
    config = LatentTreeSearchConfig.from_value(receipt["config"])
    authority_observation = _observation(receipt["authority_observation"])
    if not authority_observation.authoritative:
        raise ValueError("latent tree authority observation is not authoritative")
    _validate_probe_accounting(receipt["root_evaluation"])
    action_inventory = receipt["action_inventory"]
    if (
        not isinstance(action_inventory, list)
        or not action_inventory
        or len(action_inventory) != len(set(action_inventory))
        or any(
            not isinstance(action, str) or not action or len(action) > 80
            for action in action_inventory
        )
    ):
        raise ValueError("latent tree action inventory is invalid")
    nodes = receipt["nodes"]
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= config.max_nodes:
        raise ValueError("latent tree node inventory is invalid")
    normalized: list[dict[str, Any]] = []
    identities: dict[tuple[str, str], int] = {}
    for index, raw in enumerate(nodes):
        if not isinstance(raw, Mapping):
            raise ValueError("latent tree node is invalid")
        node = dict(raw)
        if node.get("schema") != LATENT_TREE_NODE_SCHEMA or node.get("index") != index:
            raise ValueError("latent tree node identity differs")
        parent = node.get("parent")
        if (index == 0 and parent is not None) or (
            index > 0 and (type(parent) is not int or not 0 <= parent < index)
        ):
            raise ValueError("latent tree topology is cyclic or disconnected")
        expected_depth = 0 if index == 0 else normalized[parent]["depth"] + 1
        if node.get("depth") != expected_depth or expected_depth > config.max_depth:
            raise ValueError("latent tree depth differs")
        if not _is_sha256(node.get("state_sha256")) or not _is_sha256(
            node.get("kv_boundary_sha256")
        ):
            raise ValueError("latent tree node state identity is invalid")
        _observation(node.get("observation"))
        boundaries = validate_branch_boundaries(node.get("branch_boundaries"))
        if ensemble_identity_from_boundaries(boundaries) != (
            node["state_sha256"],
            node["kv_boundary_sha256"],
        ):
            raise ValueError("latent tree node boundary commitment differs")
        duplicate_of = node.get("duplicate_of")
        key = (node["state_sha256"], node["kv_boundary_sha256"])
        expected_duplicate = identities.get(key)
        if duplicate_of != expected_duplicate:
            raise ValueError("latent tree duplicate declaration differs")
        if expected_duplicate is None:
            identities[key] = index
        identity_node = _Node(
            index=index,
            parent=parent,
            depth=expected_depth,
            action=str(node.get("action")),
            state_sha256=node["state_sha256"],
            kv_boundary_sha256=node["kv_boundary_sha256"],
            observation=_observation(node["observation"]),
            branch_boundaries=boundaries,
            snapshot=None,
        )
        if node.get("node_sha256") != _node_identity(
            episode_id=str(receipt["episode_id"]),
            objective_sha256=str(receipt["objective_sha256"]),
            node=identity_node,
        ):
            raise ValueError("latent tree node digest differs")
        normalized.append(node)
    selections = receipt["selections"]
    expansions = receipt["expansions"]
    if not isinstance(selections, list) or not isinstance(expansions, list):
        raise ValueError("latent tree trace inventories are invalid")
    successful = [row for row in expansions if isinstance(row, Mapping) and "child" in row]
    failed = [row for row in expansions if isinstance(row, Mapping) and "child" not in row]
    if len(successful) != len(nodes) - 1 or len(selections) != len(expansions) or len(failed) > 1:
        raise ValueError("latent tree expansion cardinality differs")
    reconstructed: list[_Node] = []
    for node in normalized:
        reconstructed.append(
            _Node(
                index=node["index"],
                parent=node["parent"],
                depth=node["depth"],
                action=node["action"],
                state_sha256=node["state_sha256"],
                kv_boundary_sha256=node["kv_boundary_sha256"],
                observation=_observation(node["observation"]),
                branch_boundaries=validate_branch_boundaries(node["branch_boundaries"]),
                snapshot=None,
                visits=1,
                value_sum=_observation(node["observation"]).score,
                duplicate_of=node["duplicate_of"],
            )
        )
    live: list[_Node] = [reconstructed[0]]
    next_child = 1
    for ordinal, (selection_raw, expansion_raw) in enumerate(
        zip(selections, expansions, strict=True)
    ):
        if not isinstance(selection_raw, Mapping) or not isinstance(expansion_raw, Mapping):
            raise ValueError("latent tree trace row is invalid")
        selection = dict(selection_raw)
        expansion = dict(expansion_raw)
        expected_parent, expected_value = _select_node(
            live,
            config=config,
            action_count=len(action_inventory),
        )
        if (
            selection.get("ordinal") != ordinal
            or selection.get("parent") != expected_parent.index
            or selection.get("strategy") != config.strategy
            or selection.get("parent_visits_before") != expected_parent.visits
            or not math.isclose(
                float(selection.get("selection_value")),
                expected_value,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or expansion.get("ordinal") != ordinal
            or expansion.get("parent") != expected_parent.index
        ):
            raise ValueError("latent tree selection trace differs")
        action_offset = (
            config.seed + expected_parent.index + len(expected_parent.tried_actions)
        ) % len(action_inventory)
        expected_action = next(
            action_inventory[(action_offset + offset) % len(action_inventory)]
            for offset in range(len(action_inventory))
            if action_inventory[(action_offset + offset) % len(action_inventory)]
            not in expected_parent.tried_actions
        )
        if expansion.get("action") != expected_action:
            raise ValueError("latent tree action schedule differs")
        expected_parent.tried_actions.append(expected_action)
        if "child" not in expansion:
            if (
                set(expansion)
                != {
                    "ordinal",
                    "parent",
                    "action",
                    "status",
                    "reason",
                    "spent_layer_apps",
                    "resource_delta",
                }
                or ordinal != len(expansions) - 1
                or expansion.get("status") != "failed"
                or not isinstance(expansion.get("reason"), str)
                or not expansion["reason"].startswith("expansion_failed:")
            ):
                raise ValueError("latent tree failed expansion differs")
            _validate_resource_window(
                spent_layer_apps=expansion["spent_layer_apps"],
                resource_delta=expansion["resource_delta"],
            )
            continue
        expected_expansion_fields = {
            "ordinal",
            "parent",
            "child",
            "action",
            "status",
            "duplicate_of",
            "transition_sha256",
            "target_branch",
            "probe_tokens_sha256",
            "probe_token_count",
            "probe_cache_hit",
            "probe_cache_key_sha256",
            "probe_cache_layer_apps_saved",
            "recurrent_kv_call_ordinals",
            "spent_layer_apps",
            "resource_delta",
        }
        if set(expansion) != expected_expansion_fields:
            raise ValueError("latent tree successful expansion fields differ")
        if expansion.get("child") != next_child:
            raise ValueError("latent tree child order differs")
        if not _is_sha256(expansion.get("transition_sha256")):
            raise ValueError("latent tree resource proof differs")
        _validate_probe_accounting(
            {
                "spent_layer_apps": expansion.get("spent_layer_apps"),
                "resource_delta": expansion.get("resource_delta"),
                "probe_tokens_sha256": expansion.get("probe_tokens_sha256"),
                "probe_token_count": expansion.get("probe_token_count"),
                "target_branch": expansion.get("target_branch"),
                "probe_cache_hit": expansion.get("probe_cache_hit"),
                "probe_cache_key_sha256": expansion.get("probe_cache_key_sha256"),
                "probe_cache_layer_apps_saved": expansion.get("probe_cache_layer_apps_saved"),
            }
        )
        _ordinal_inventory(
            expansion["recurrent_kv_call_ordinals"],
            name="latent tree expansion recurrent-call ordinals",
        )
        if not expansion["recurrent_kv_call_ordinals"]:
            raise ValueError("latent tree expansion made no recurrent transition")
        child = reconstructed[next_child]
        expected_status = "duplicate" if child.duplicate_of is not None else "expanded"
        if (
            expansion.get("status") != expected_status
            or expansion.get("duplicate_of") != child.duplicate_of
        ):
            raise ValueError("latent tree duplicate trace differs")
        live.append(child)
        next_child += 1
        cursor: _Node | None = expected_parent
        while cursor is not None:
            cursor.visits += 1
            cursor.value_sum += child.observation.score
            cursor = None if cursor.parent is None else live[cursor.parent]
    for public, rebuilt in zip(normalized, reconstructed, strict=True):
        if (
            public.get("visits") != rebuilt.visits
            or not math.isclose(
                float(public.get("value_sum")), rebuilt.value_sum, rel_tol=0.0, abs_tol=1e-9
            )
            or not math.isclose(
                float(public.get("mean_value")),
                rebuilt.mean_value,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or public.get("tried_actions") != rebuilt.tried_actions
        ):
            raise ValueError("latent tree visit/value reconstruction differs")
    if receipt["duplicate_count"] != sum(node["duplicate_of"] is not None for node in normalized):
        raise ValueError("latent tree duplicate count differs")
    if receipt["root_node"] != 0 or receipt["committed_node"] not in range(len(nodes)):
        raise ValueError("latent tree committed node is invalid")
    committed = receipt["status"] == "committed"
    winner = receipt["winner_node"]
    if committed:
        if type(winner) is not int or not 1 <= winner < len(nodes):
            raise ValueError("latent tree winner is invalid")
        root_obs = _observation(nodes[0]["observation"])
        winner_obs = _observation(nodes[winner]["observation"])
        if not (
            winner_obs.lower_bound
            > max(root_obs.upper_bound, authority_observation.upper_bound)
            + float(config.min_verifier_margin)
            and receipt["committed_node"] == winner
            and receipt["winner_dominates_root"] is True
        ):
            raise ValueError("latent tree winner lacks confidence-bound authority")
    elif winner is not None or receipt["committed_node"] != 0:
        raise ValueError("latent tree non-winner did not restore root")
    new_recurrent_ordinals = _ordinal_inventory(
        receipt["recurrent_kv_new_call_ordinals"],
        name="latent tree new recurrent-call ordinals",
    )
    committed_recurrent_ordinals = _ordinal_inventory(
        receipt["committed_recurrent_kv_call_ordinals"],
        name="latent tree committed recurrent-call ordinals",
    )
    discarded_recurrent_ordinals = _ordinal_inventory(
        receipt["discarded_recurrent_kv_call_ordinals"],
        name="latent tree discarded recurrent-call ordinals",
    )
    committed_nodes = {0}
    if committed:
        cursor_index = int(receipt["committed_node"])
        while cursor_index != 0:
            committed_nodes.add(cursor_index)
            cursor_index = int(normalized[cursor_index]["parent"])
    expected_committed_ordinals = sorted(
        ordinal
        for expansion in expansions
        if expansion.get("child") in committed_nodes
        for ordinal in expansion.get("recurrent_kv_call_ordinals", ())
    )
    if (
        committed_recurrent_ordinals != expected_committed_ordinals
        or set(committed_recurrent_ordinals) & set(discarded_recurrent_ordinals)
        or sorted(committed_recurrent_ordinals + discarded_recurrent_ordinals)
        != new_recurrent_ordinals
    ):
        raise ValueError("latent tree recurrent-call partition differs")
    claimed_ordinals = {
        ordinal
        for expansion in expansions
        for ordinal in expansion.get("recurrent_kv_call_ordinals", ())
    }
    if not claimed_ordinals <= set(new_recurrent_ordinals):
        raise ValueError("latent tree expansion recurrent-call proof differs")
    final = nodes[receipt["committed_node"]]
    if (
        receipt["final_state_sha256"] != final["state_sha256"]
        or receipt["final_kv_boundary_sha256"] != final["kv_boundary_sha256"]
        or receipt["restore_exact"] is not True
        or receipt["private_snapshots_disclosed"] is not False
        or receipt["answer_text_stored"] is not False
    ):
        raise ValueError("latent tree final state proof differs")
    receipt["receipt_sha256"] = digest
    return receipt


def validate_latent_tree_receipt(
    value: Mapping[str, Any],
    *,
    episode_id: str,
    objective_sha256: str,
    expected_config: LatentTreeSearchConfig,
    kv_state_tree: Mapping[str, Any] | None = None,
    cognitive_action_trace: Sequence[Mapping[str, Any]] | None = None,
    resource_accounting: Mapping[str, Any] | None = None,
    loop_stability: Mapping[str, Any] | None = None,
    require_external_bindings: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("latent tree receipt is not a mapping")
    receipt = dict(value)
    digest = receipt.pop("receipt_sha256", None)
    if not _is_sha256(digest) or canonical_sha256(receipt) != digest:
        raise ValueError("latent tree aggregate digest differs")
    if (
        receipt.get("schema") != LATENT_TREE_SEARCH_SCHEMA
        or receipt.get("episode_id") != episode_id
        or receipt.get("objective_sha256") != objective_sha256
        or receipt.get("config") != expected_config.to_dict()
        or receipt.get("private_snapshots_disclosed") is not False
        or receipt.get("answer_text_stored") is not False
    ):
        raise ValueError("latent tree aggregate identity differs")
    transactions = receipt.get("transactions")
    if not isinstance(transactions, list):
        raise ValueError("latent tree transactions are invalid")
    prior_step = -1
    for raw in transactions:
        transaction = validate_latent_tree_transaction(raw)
        if (
            transaction["episode_id"] != episode_id
            or transaction["objective_sha256"] != objective_sha256
            or transaction["config"] != expected_config.to_dict()
            or transaction["action_step"] <= prior_step
        ):
            raise ValueError("latent tree transaction binding differs")
        prior_step = transaction["action_step"]
    if transactions and receipt.get("status") != "executed":
        raise ValueError("latent tree aggregate status differs")
    if not transactions and receipt.get("status") not in {
        "disabled",
        "not_invoked",
        "unavailable",
    }:
        raise ValueError("latent tree empty status differs")
    if require_external_bindings and transactions:
        if not isinstance(kv_state_tree, Mapping):
            raise ValueError("latent tree KV tree binding is absent")
        kv_nodes = kv_state_tree.get("nodes")
        if not isinstance(kv_nodes, list):
            raise ValueError("latent tree KV node inventory is absent")
        known_kv = {
            row.get("node_sha256")
            for row in kv_nodes
            if isinstance(row, Mapping) and _is_sha256(row.get("node_sha256"))
        }
        searched_kv = {
            boundary["kv_boundary_sha256"]
            for transaction in transactions
            for node in transaction["nodes"]
            for boundary in node["branch_boundaries"]
        }
        if not searched_kv or not searched_kv <= known_kv:
            raise ValueError("latent tree searched an unknown KV boundary")
        if not isinstance(cognitive_action_trace, Sequence) or isinstance(
            cognitive_action_trace, (str, bytes)
        ):
            raise ValueError("latent tree action trace binding is absent")
        trace_by_step = {
            row.get("transition", {}).get("step_index"): row
            for row in cognitive_action_trace
            if isinstance(row, Mapping) and isinstance(row.get("transition"), Mapping)
        }
        for transaction in transactions:
            row = trace_by_step.get(transaction["action_step"])
            if (
                not isinstance(row, Mapping)
                or row.get("decision", {}).get("action") != "branch"
                or not str(row.get("transition", {}).get("outcome", "")).startswith("latent_tree_")
            ):
                raise ValueError("latent tree action trace binding differs")
        if not isinstance(resource_accounting, Mapping):
            raise ValueError("latent tree resource accounting binding is absent")
        totals = resource_accounting.get("totals")
        if not isinstance(totals, Mapping) or set(totals) != set(RESOURCE_COUNTERS):
            raise ValueError("latent tree resource totals are invalid")
        claimed = {name: 0 for name in RESOURCE_COUNTERS}
        for transaction in transactions:
            windows = [transaction["root_evaluation"]] + [
                row for row in transaction["expansions"] if "resource_delta" in row
            ]
            for window in windows:
                for name in RESOURCE_COUNTERS:
                    claimed[name] += int(window["resource_delta"][name])
        if any(
            type(totals[name]) is not int or totals[name] < 0 or claimed[name] > totals[name]
            for name in RESOURCE_COUNTERS
        ):
            raise ValueError("latent tree resource claims exceed the episode ledger")
        if not isinstance(loop_stability, Mapping):
            raise ValueError("latent tree loop-stability binding is absent")
        kv_bound = loop_stability.get("kv_bound")
        kv_calls = kv_bound.get("calls") if isinstance(kv_bound, Mapping) else None
        if not isinstance(kv_calls, list):
            raise ValueError("latent tree recurrent KV call ledger is absent")
        call_by_ordinal = {
            row.get("ordinal"): row
            for row in kv_calls
            if isinstance(row, Mapping) and type(row.get("ordinal")) is int
        }
        new_ordinals = {
            ordinal
            for transaction in transactions
            for ordinal in transaction["recurrent_kv_new_call_ordinals"]
        }
        if any(
            ordinal not in call_by_ordinal or call_by_ordinal[ordinal].get("persist") is not False
            for ordinal in new_ordinals
        ):
            raise ValueError("latent tree recurrent KV call binding differs")
        expected_exclusions = sorted(
            {
                ordinal
                for transaction in transactions
                for ordinal in transaction["discarded_recurrent_kv_call_ordinals"]
            }
        )
        if loop_stability.get("excluded_speculative_kv_call_ordinals") != expected_exclusions:
            raise ValueError("latent tree speculative KV exclusions differ")
    receipt["receipt_sha256"] = digest
    return receipt
