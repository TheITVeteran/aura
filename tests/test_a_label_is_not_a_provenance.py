"""A string the caller typed decided what got compiled into the weights.

Cognitive context accepts a free-form ``source``. The literal labels "memory"
and "world_model" were then read as proof that the text had been epistemically
admitted, and admitted retrieval is the only class allowed to seed fast-weight
columns. So `{"source": "memory", "text": <anything>}` — which passes ingress
as untyped context and carries no content commitment, no retrieval receipt and
no epistemic-state digest — reached the adaptation subspace.

The export side had no data policy at all. A delta seeded from a slot encodes
that slot's content, and an accepted delta is written to the consolidation
queue as a durable artifact. Hashing the text in the receipt does not undo
that: the weights are the copy.

Three wall-clock and budget facts in the same lane:

**The sentence grace was never reserved.** Admission priced `decode_limit +
contract_grace` passes, but `_decode` runs up to 48 more sentence-grace passes
on a non-contract task and charges every one, out of the margin the admission
message says it preserves.

**The cleanup reserve was six seconds.** A number, not a measurement — slack on
a fast device, short on a slow one, and being short means the erase proof
starts work it cannot finish.

**A mid-clause wall stop was a success.** `wall_reserve` fired wherever the
reserve was crossed and `reason()` listed it among the successful
terminations, so a known fragment came back ok.
"""
from __future__ import annotations

import ast
import inspect

import pytest

import core.brain.llm.latent_cortex.engine as engine_mod
from core.brain.llm.latent_cortex.cognitive_context import (
    is_admitted_retrieval,
    is_exportable_provenance,
)


# ─────────────────────────── the label is not the provenance


def _typed_memory(**overrides):
    item = {
        "source": "memory",
        "text": "she said the fox was in the garden",
        "context_role": "memory_observation",
        "instruction_authority": False,
        "evidence_id": "memory-1",
        "content_sha256": "a" * 64,
        "scope_sha256": "b" * 64,
        "retrieval_receipt_sha256": "c" * 64,
        "epistemic_state_sha256": "d" * 64,
        "memory_tier": "episodic",
        "memory_source_id": "conversation-9",
        "memory_source_version": "1",
    }
    item.update(overrides)
    return item


def _typed_evidence(kind="offline_reference", **overrides):
    item = {
        "source": "reference",
        "text": "water boils at 100C at one atmosphere",
        "context_role": "evidence_observation",
        "instruction_authority": False,
        "evidence_id": "evidence-1",
        "content_sha256": "a" * 64,
        "retrieval_receipt_sha256": "c" * 64,
        "evidence_kind": kind,
        "evidence_origin": "handbook",
        "source_version": "2",
    }
    item.update(overrides)
    return item


def test_typed_memory_is_admitted_retrieval():
    assert is_admitted_retrieval(_typed_memory()) is True


def test_typed_evidence_is_admitted_retrieval():
    assert is_admitted_retrieval(_typed_evidence()) is True


def test_an_untyped_item_labelled_memory_is_not_admitted_retrieval():
    """This is the whole defect: the label without the contract."""
    assert is_admitted_retrieval({"source": "memory", "text": "anything at all"}) is False


@pytest.mark.parametrize("label", ["memory", "world_model", "reference", "one_shot_memory"])
def test_no_bare_label_buys_admission(label):
    assert is_admitted_retrieval({"source": label, "text": "x"}) is False


def test_live_organ_state_is_not_retrieval():
    """world_model is live runtime state, not something that was retrieved
    and epistemically admitted."""
    assert is_admitted_retrieval({"source": "world_model", "text": "x"}) is False


def test_the_source_label_no_longer_selects_compilable_slots():
    source = inspect.getsource(engine_mod)

    assert "_RETRIEVAL_SLOT_SOURCES" not in source
    assert 'row.get("admitted_retrieval") is True' in source


def test_the_slot_receipt_records_the_typed_verdict():
    source = inspect.getsource(engine_mod)

    assert '"admitted_retrieval": is_admitted_retrieval(' in source
    assert '"exportable_provenance": is_exportable_provenance(' in source


# ─────────────────────────── what may outlive the episode


def test_episodic_memory_never_becomes_a_durable_adapter():
    assert is_exportable_provenance(_typed_memory()) is False


def test_an_offline_reference_may():
    assert is_exportable_provenance(_typed_evidence("offline_reference")) is True


def test_a_live_world_observation_may():
    assert is_exportable_provenance(_typed_evidence("live_world_observation")) is True


@pytest.mark.parametrize(
    "kind", ["governed_tool_observation", "one_shot_nonparametric_memory"]
)
def test_tool_output_and_the_persons_own_turn_may_not(kind):
    assert is_exportable_provenance(_typed_evidence(kind)) is False


def test_untyped_context_is_not_exportable():
    assert is_exportable_provenance({"source": "reference", "text": "x"}) is False


def _receipt(rows):
    from core.brain.llm.latent_cortex.types import EpisodeReceipt

    receipt = EpisodeReceipt(episode_id="e")
    receipt.cognitive_slots = rows
    return receipt


def _permits(rows) -> bool:
    from core.brain.llm.latent_cortex.engine import LatentCortexEngine

    engine = LatentCortexEngine.__new__(LatentCortexEngine)
    return engine._export_provenance_permits(_receipt(rows))


def test_an_episode_that_compiled_nothing_may_export():
    assert _permits([]) is True


def test_an_episode_that_compiled_only_references_may_export():
    rows = [
        {
            "slot": 1,
            "admitted_retrieval": True,
            "exportable_provenance": True,
            "knowledge_class": "offline_reference",
        }
    ]

    assert _permits(rows) is True


def test_an_episode_that_compiled_a_memory_may_not():
    rows = [
        {
            "slot": 1,
            "admitted_retrieval": True,
            "exportable_provenance": False,
            "knowledge_class": "memory.episodic",
        }
    ]

    assert _permits(rows) is False


def test_a_mixed_episode_is_refused_wholesale():
    """The delta is one object. Nothing can say which slot a weight came
    from after the fact, so there is no partial export to permit."""
    rows = [
        {
            "slot": 1,
            "admitted_retrieval": True,
            "exportable_provenance": True,
            "knowledge_class": "offline_reference",
        },
        {
            "slot": 2,
            "admitted_retrieval": True,
            "exportable_provenance": False,
            "knowledge_class": "memory.episodic",
        },
    ]

    assert _permits(rows) is False


def test_the_refusal_names_the_class_that_caused_it():
    receipt = _receipt(
        [
            {
                "slot": 1,
                "admitted_retrieval": True,
                "exportable_provenance": False,
                "knowledge_class": "memory.episodic",
            }
        ]
    )
    from core.brain.llm.latent_cortex.engine import LatentCortexEngine

    engine = LatentCortexEngine.__new__(LatentCortexEngine)
    engine._export_provenance_permits(receipt)

    assert any(
        flag.startswith("fast_weight_export_refused_private_provenance:")
        and "memory.episodic" in flag
        for flag in receipt.honest_flags
    )


def test_a_slot_that_was_never_compiled_does_not_block_export():
    """Only slots that seeded the adaptation subspace are in the delta."""
    rows = [
        {
            "slot": 1,
            "admitted_retrieval": False,
            "exportable_provenance": False,
            "knowledge_class": "live_organ_state",
        }
    ]

    assert _permits(rows) is True


def test_the_export_gate_actually_calls_the_policy():
    source = inspect.getsource(engine_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        rendered = ast.get_source_segment(source, node.test) or ""
        if "export_candidates" not in rendered:
            continue
        assert "_export_provenance_permits(receipt)" in rendered
        return
    raise AssertionError("the consolidation export gate was not found")


# ─────────────────────────── the grace the admission forgot


def test_the_sentence_grace_is_priced_at_admission():
    source = inspect.getsource(engine_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "decode_extension"
                for target in node.targets
            )
        ):
            continue
        rendered = ast.get_source_segment(source, node) or ""
        assert "sentence_grace" in rendered
        assert "contract_grace" in rendered
        return
    raise AssertionError("the admission decode extension was not found")


def test_the_admission_grace_matches_what_decode_actually_runs():
    """_decode runs limit + extension passes and picks the extension the
    same way. If the two ever disagree the reserve is fiction again."""
    source = inspect.getsource(engine_mod)

    assert (
        "extension = contract_grace if contract_required else grace_tokens" in source
    )
    assert (
        "decode_extension = int(contract_grace) if contract_required else sentence_grace"
        in source
    )


def test_a_caller_supplied_grace_is_the_one_reserved():
    source = inspect.getsource(engine_mod)

    assert "if decode_sentence_grace_tokens is None" in source
    assert "else int(decode_sentence_grace_tokens)" in source


# ─────────────────────────── the reserve is measured, not chosen


def test_the_six_second_reserve_is_gone():
    source = inspect.getsource(engine_mod)

    assert "wall_reserve_s=(6.0 if fast_weights is not None else 0.0)" not in source


def test_the_reserve_is_stated_as_the_work_it_protects():
    source = inspect.getsource(engine_mod)

    assert "_FW_ERASE_PROBE_TOKENS + 1 if fast_weights is not None else 0" in source


def test_the_probe_width_is_one_quantity_not_three_literals():
    """The affordability check, the charge and the reserve all price the same
    probe. Three copies of 8 drift."""
    source = inspect.getsource(engine_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "_fw_probe"):
            continue
        rendered = ast.get_source_segment(source, node) or ""
        assert "budget.can_afford(_FW_ERASE_PROBE_TOKENS, self.n_layers)" in rendered
        assert "tokens=_FW_ERASE_PROBE_TOKENS," in rendered
        assert "range(1, _FW_ERASE_PROBE_TOKENS + 1)" in rendered
        return
    raise AssertionError("_fw_probe was not found")


def test_the_reserve_takes_the_largest_of_the_measurements():
    source = inspect.getsource(engine_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "wall_reserve_s"
                for target in node.targets
            )
            and isinstance(node.value, ast.Call)
        ):
            continue
        rendered = ast.get_source_segment(source, node) or ""
        if "max(" not in rendered:
            continue
        assert "wall_reserve_forwards * rate_s" in rendered
        assert "_fw_probe_seconds_high_water" in rendered
        assert "_fw_cleanup_seconds_high_water" in rendered
        return
    raise AssertionError("the derived wall reserve was not found")


def test_a_cold_engine_still_reserves_a_real_number():
    """The pre-attach probe runs the same computation as the erase proof, so
    timing it gives the first episode a measurement rather than a guess."""
    source = inspect.getsource(engine_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "_fw_probe"):
            continue
        rendered = ast.get_source_segment(source, node) or ""
        assert "probe_started = time.monotonic()" in rendered
        assert "self._fw_probe_seconds_high_water = max(" in rendered
        return
    raise AssertionError("_fw_probe was not found")


def test_the_cleanup_measures_itself_for_the_next_episode():
    source = inspect.getsource(engine_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.FunctionDef) and node.name == "_finalize_fast_weights"
        ):
            continue
        rendered = ast.get_source_segment(source, node) or ""
        assert "cleanup_started = time.monotonic()" in rendered
        assert "self._fw_cleanup_seconds_high_water = max(" in rendered
        return
    raise AssertionError("_finalize_fast_weights was not found")


def test_the_reserve_is_checked_before_the_forward_not_after():
    source = inspect.getsource(engine_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "_decode"):
            continue
        rendered = ast.get_source_segment(source, node) or ""
        reserve = rendered.index("if wall_reserve_s > 0.0 or wall_reserve_forwards > 0:")
        forward = rendered.index('operation="autoregressive_decode"')
        assert reserve < forward
        return
    raise AssertionError("_decode was not found")
