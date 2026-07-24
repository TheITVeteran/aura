from __future__ import annotations

import copy
import hashlib

import pytest

from core.brain.llm.latent_cortex.latent_tree_search import (
    LATENT_TREE_SEARCH_SCHEMA,
    LatentTreeSearchConfig,
    append_transaction,
    build_empty_latent_tree_receipt,
    ensemble_identity_from_boundaries,
    run_latent_tree_search,
    validate_latent_tree_receipt,
    validate_latent_tree_transaction,
)
from core.brain.llm.latent_cortex.loop_core import canonical_sha256
from core.brain.llm.latent_cortex.resource_accounting import RESOURCE_COUNTERS
from core.brain.llm.latent_cortex.types import ComputeBudget
from core.brain.llm.latent_cortex.verified_best import VerifierObservation


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _exact(score: float, label: str) -> dict:
    return VerifierObservation(
        score=score,
        lower_bound=score,
        upper_bound=score,
        sample_count=1,
        basis="deterministic_exact",
        independent=True,
        evidence_sha256=_digest(label),
    ).to_dict()


class _Harness:
    def __init__(self, budget: ComputeBudget, scores: list[float]):
        self.budget = budget
        self.scores = list(scores)
        root_boundaries = [
            {
                "index": 0,
                "state_sha256": _digest("root-branch-state"),
                "kv_boundary_sha256": _digest("root-branch-kv"),
                "steps": 0,
                "operator": "direct_derivation",
                "halted": False,
            }
        ]
        state, kv = ensemble_identity_from_boundaries(root_boundaries)
        self.live = {"state": state, "kv": kv, "boundaries": root_boundaries}
        self.restores: list[tuple[str, str]] = []
        self.expansions = 0
        self.kv_ordinals: list[int] = []

    @property
    def root(self) -> dict:
        return copy.deepcopy(self.live)

    def restore(self, snapshot: dict) -> tuple[str, str]:
        self.live = copy.deepcopy(snapshot)
        identity = (self.live["state"], self.live["kv"])
        self.restores.append(identity)
        return identity

    def expand(self, action: str, parent: int, child: int) -> dict:
        score = self.scores[self.expansions]
        self.expansions += 1
        self.budget.charge(
            4,
            3,
            operation="test_tree_transition",
            attention_pairs=16,
            output_head_tokens=4,
        )
        self.budget.charge_verifier(
            "test_tree_verifier",
            input_bytes=32,
            output_bytes=32,
            host_scalar_ops=16,
        )
        self.kv_ordinals.append(len(self.kv_ordinals))
        state = _digest(f"state:{parent}:{child}:{action}:{score}")
        branch_kv = _digest(f"kv:{parent}:{child}:{action}:{score}")
        boundaries = [
            {
                "index": 0,
                "state_sha256": state,
                "kv_boundary_sha256": branch_kv,
                "steps": child,
                "operator": action,
                "halted": False,
            }
        ]
        state, kv = ensemble_identity_from_boundaries(boundaries)
        self.live = {"state": state, "kv": kv, "boundaries": boundaries}
        return {
            "snapshot": copy.deepcopy(self.live),
            "state_sha256": state,
            "kv_boundary_sha256": kv,
            "observation": _exact(score, f"obs:{parent}:{child}:{score}"),
            "transition_sha256": _digest(f"transition:{parent}:{child}:{action}"),
            "target_branch": 0,
            "probe_tokens_sha256": _digest(f"probe:{parent}:{child}:{action}"),
            "probe_token_count": 4,
            "branch_boundaries": copy.deepcopy(boundaries),
            "probe_cache_hit": False,
            "probe_cache_key_sha256": "",
            "probe_cache_layer_apps_saved": 0,
            "recurrent_kv_call_ordinals": [self.kv_ordinals[-1]],
        }


def _root_evaluation(budget: ComputeBudget) -> dict:
    before = budget.resource_ledger.totals()
    spent_before = budget.spent_layer_apps
    budget.charge(
        4,
        3,
        operation="test_tree_root_probe",
        attention_pairs=16,
        output_head_tokens=4,
    )
    budget.charge_verifier(
        "test_tree_root_verifier",
        input_bytes=32,
        output_bytes=32,
        host_scalar_ops=16,
    )
    after = budget.resource_ledger.totals()
    return {
        "spent_layer_apps": budget.spent_layer_apps - spent_before,
        "resource_delta": {name: after[name] - before[name] for name in RESOURCE_COUNTERS},
        "probe_tokens_sha256": _digest("root-probe"),
        "probe_token_count": 4,
        "target_branch": 0,
        "probe_cache_hit": False,
        "probe_cache_key_sha256": "",
        "probe_cache_layer_apps_saved": 0,
    }


def _run(*, strategy: str = "uct", scores: list[float] | None = None, cancel=None):
    budget = ComputeBudget(max_layer_apps=10_000, wall_clock_s=30.0)
    harness = _Harness(budget, scores or [0.9, 0.5, 0.6, 0.7])
    root = harness.root
    config = LatentTreeSearchConfig(
        strategy=strategy,
        max_nodes=5,
        max_depth=2,
        branching_factor=2,
        seed=7,
    )
    receipt = run_latent_tree_search(
        episode_id="episode-tree",
        objective_sha256="a" * 64,
        action_step=3,
        root_snapshot=root,
        root_state_sha256=root["state"],
        root_kv_boundary_sha256=root["kv"],
        root_branch_boundaries=root["boundaries"],
        root_observation=_exact(0.2, "root"),
        root_evaluation=_root_evaluation(budget),
        actions=("decompose", "falsify", "simulate"),
        config=config,
        budget=budget,
        restore_snapshot=harness.restore,
        expand=harness.expand,
        recurrent_call_inventory=lambda: harness.kv_ordinals,
        cancel_check=cancel,
    )
    return receipt, harness, config


@pytest.mark.parametrize("strategy", ["uct", "beam", "bfs"])
def test_search_strategies_commit_only_confidence_bound_winner(strategy):
    receipt, harness, _ = _run(strategy=strategy)

    assert receipt["status"] == "committed"
    assert receipt["winner_dominates_root"] is True
    assert receipt["winner_node"] == receipt["committed_node"]
    winner = receipt["nodes"][receipt["winner_node"]]
    assert harness.live["state"] == winner["state_sha256"]
    assert harness.live["kv"] == winner["kv_boundary_sha256"]
    assert receipt["restore_exact"] is True
    assert receipt["private_snapshots_disclosed"] is False
    assert receipt["answer_text_stored"] is False
    validate_latent_tree_transaction(receipt)


def test_no_dominating_candidate_restores_exact_root():
    receipt, harness, _ = _run(scores=[0.2, 0.1, 0.2, 0.1])

    assert receipt["status"] == "restored"
    assert receipt["winner_node"] is None
    assert receipt["committed_node"] == 0
    assert receipt["reason"] == "no_candidate_dominated_root"
    assert harness.live == harness.root


def test_cancellation_before_expansion_restores_root_without_compute():
    receipt, harness, _ = _run(cancel=lambda: True)

    assert receipt["status"] == "cancelled"
    assert receipt["cancelled"] is True
    assert receipt["expansions"] == []
    assert receipt["committed_node"] == 0
    assert harness.expansions == 0
    assert harness.live == harness.root


def test_duplicate_state_and_kv_are_pruned_and_counted():
    budget = ComputeBudget(max_layer_apps=10_000, wall_clock_s=30.0)
    harness = _Harness(budget, [0.9, 0.8])
    root = harness.root

    def duplicate_expand(action: str, parent: int, child: int):
        result = harness.expand(action, parent, child)
        result["state_sha256"] = root["state"]
        result["kv_boundary_sha256"] = root["kv"]
        result["branch_boundaries"] = copy.deepcopy(root["boundaries"])
        result["snapshot"] = copy.deepcopy(root)
        return result

    receipt = run_latent_tree_search(
        episode_id="episode-tree",
        objective_sha256="a" * 64,
        action_step=3,
        root_snapshot=root,
        root_state_sha256=root["state"],
        root_kv_boundary_sha256=root["kv"],
        root_branch_boundaries=root["boundaries"],
        root_observation=_exact(0.2, "root"),
        root_evaluation=_root_evaluation(budget),
        actions=("decompose", "falsify"),
        config=LatentTreeSearchConfig(max_nodes=3, branching_factor=2),
        budget=budget,
        restore_snapshot=harness.restore,
        expand=duplicate_expand,
        recurrent_call_inventory=lambda: harness.kv_ordinals,
    )

    assert receipt["duplicate_count"] == 2
    assert all(node["duplicate_of"] == 0 for node in receipt["nodes"][1:])
    assert receipt["status"] == "restored"
    assert receipt["committed_node"] == 0


def test_failed_expansion_is_fail_closed_and_restores_parent():
    budget = ComputeBudget(max_layer_apps=10_000, wall_clock_s=30.0)
    harness = _Harness(budget, [0.9])
    root = harness.root

    def fail(_action: str, _parent: int, _child: int):
        harness.budget.charge(
            2,
            1,
            operation="test_failed_tree_transition",
            attention_pairs=4,
        )
        harness.kv_ordinals.append(len(harness.kv_ordinals))
        raise RuntimeError("forced")

    receipt = run_latent_tree_search(
        episode_id="episode-tree",
        objective_sha256="a" * 64,
        action_step=3,
        root_snapshot=root,
        root_state_sha256=root["state"],
        root_kv_boundary_sha256=root["kv"],
        root_branch_boundaries=root["boundaries"],
        root_observation=_exact(0.2, "root"),
        root_evaluation=_root_evaluation(budget),
        actions=("decompose",),
        config=LatentTreeSearchConfig(max_nodes=2, branching_factor=1),
        budget=budget,
        restore_snapshot=harness.restore,
        expand=fail,
        recurrent_call_inventory=lambda: harness.kv_ordinals,
    )

    assert receipt["status"] == "restored"
    assert receipt["reason"] == "expansion_failed:RuntimeError"
    assert receipt["nodes"] == [receipt["nodes"][0]]
    assert receipt["expansions"][0]["spent_layer_apps"] == 2
    assert receipt["recurrent_kv_new_call_ordinals"] == [0]
    assert receipt["committed_recurrent_kv_call_ordinals"] == []
    assert receipt["discarded_recurrent_kv_call_ordinals"] == [0]
    assert harness.live == root


def _rehash(receipt: dict) -> None:
    payload = dict(receipt)
    payload.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = canonical_sha256(payload)


def test_validator_rejects_selection_resource_and_topology_tampering():
    receipt, _, _ = _run()
    for mutate in (
        lambda value: value["selections"][0].__setitem__("parent", 99),
        lambda value: value["expansions"][0]["resource_delta"].__setitem__(
            "transformer_layer_apps", 0
        ),
        lambda value: value["nodes"][1].__setitem__("parent", 1),
        lambda value: value["discarded_recurrent_kv_call_ordinals"].append(999),
    ):
        tampered = copy.deepcopy(receipt)
        mutate(tampered)
        _rehash(tampered)
        with pytest.raises(ValueError):
            validate_latent_tree_transaction(tampered)


def test_aggregate_receipt_binds_order_identity_and_config():
    transaction, _, config = _run()
    aggregate = build_empty_latent_tree_receipt(
        episode_id="episode-tree",
        objective_sha256="a" * 64,
        config=config,
    )
    append_transaction(aggregate, transaction)

    assert aggregate["schema"] == LATENT_TREE_SEARCH_SCHEMA
    assert aggregate["status"] == "executed"
    assert (
        validate_latent_tree_receipt(
            aggregate,
            episode_id="episode-tree",
            objective_sha256="a" * 64,
            expected_config=config,
        )["receipt_sha256"]
        == aggregate["receipt_sha256"]
    )


def test_external_validator_binds_kv_tree_action_trace_and_global_resources():
    transaction, harness, config = _run()
    aggregate = build_empty_latent_tree_receipt(
        episode_id="episode-tree",
        objective_sha256="a" * 64,
        config=config,
    )
    append_transaction(aggregate, transaction)
    kv_nodes = [
        {"node_sha256": boundary["kv_boundary_sha256"]}
        for node in transaction["nodes"]
        for boundary in node["branch_boundaries"]
    ]
    action_trace = [
        {
            "decision": {"action": "branch"},
            "transition": {
                "step_index": transaction["action_step"],
                "outcome": f"latent_tree_{transaction['status']}",
            },
        }
    ]
    loop_stability = {
        "kv_bound": {
            "calls": [
                {"ordinal": ordinal, "persist": False}
                for ordinal in transaction["recurrent_kv_new_call_ordinals"]
            ]
        },
        "excluded_speculative_kv_call_ordinals": transaction[
            "discarded_recurrent_kv_call_ordinals"
        ],
    }

    validate_latent_tree_receipt(
        aggregate,
        episode_id="episode-tree",
        objective_sha256="a" * 64,
        expected_config=config,
        kv_state_tree={"nodes": kv_nodes},
        cognitive_action_trace=action_trace,
        resource_accounting={"totals": harness.budget.resource_ledger.totals()},
        loop_stability=loop_stability,
        require_external_bindings=True,
    )

    with pytest.raises(ValueError, match="unknown KV"):
        validate_latent_tree_receipt(
            aggregate,
            episode_id="episode-tree",
            objective_sha256="a" * 64,
            expected_config=config,
            kv_state_tree={"nodes": []},
            cognitive_action_trace=action_trace,
            resource_accounting={"totals": harness.budget.resource_ledger.totals()},
            loop_stability=loop_stability,
            require_external_bindings=True,
        )


def test_worker_config_and_episode_receipt_expose_tree_search_contract():
    from core.brain.llm.latent_cortex.types import EpisodeReceipt
    from core.brain.llm.latent_cortex.worker_handler import config_from_job

    config = config_from_job({"latent_tree_search": {"strategy": "beam", "max_nodes": 7}})
    assert config.latent_tree_search == {"strategy": "beam", "max_nodes": 7}
    assert config.validate() == []
    assert EpisodeReceipt(episode_id="episode").to_dict()["latent_tree_search"] == {}


def test_config_rejects_unknown_unbounded_and_invalid_values():
    with pytest.raises(ValueError):
        LatentTreeSearchConfig.from_value({"unknown": True})
    with pytest.raises(ValueError):
        LatentTreeSearchConfig(max_nodes=33)
    with pytest.raises(ValueError):
        LatentTreeSearchConfig(max_depth=0)
    with pytest.raises(ValueError):
        LatentTreeSearchConfig(strategy="random")
