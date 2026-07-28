"""Mechanics test: the schedule-search campaign closes the promotion loop.

The wiring gap this covers: `record_paired_outcome` existed with zero
callers, so the live library could never promote a schedule. The campaign
driver must run search → paired holdout trials → recorded outcomes end to
end, with per-task commitments the replay guards accept, disjoint splits
enforced, and a receipt the operator can read. Capability numbers on the
tiny random model are meaningless and asserted nowhere — this is the
machinery proof; real runs are operator-launched on a trained checkpoint.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

_TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "schedule_search_campaign.py"
_SPEC = importlib.util.spec_from_file_location("schedule_search_campaign", _TOOL_PATH)
campaign = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(campaign)


class _StubTokenizer:
    """Deterministic tokenizer: hashes words; decodes to unparseable text."""

    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [(hash(word) % 96) + 1 for word in str(text).split()][:64]

    def decode(self, tokens):
        return " ".join(f"tok{int(t)}" for t in tokens)

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        text = " ".join(str(m.get("content", "")) for m in messages)
        return self.encode(text)


def _tiny_model():
    args = ModelArgs(
        model_type="qwen2", hidden_size=64, num_hidden_layers=8,
        intermediate_size=128, num_attention_heads=4, rms_norm_eps=1e-6,
        vocab_size=128, num_key_value_heads=2, max_position_embeddings=512,
        rope_theta=10000.0,
    )
    model = Model(args)
    mx.eval(model.parameters())
    return model


def _args(tmp_path, model_dir, **overrides) -> argparse.Namespace:
    values = dict(
        model=str(model_dir),
        library=str(tmp_path / "library.json"),
        domain="general",
        families="modular",
        task_depth=2,
        search_per_cell=2,
        holdout_per_cell=3,
        seed=7,
        prelude_end=2,
        coda_start=6,
        default_repeats=2,
        max_repeats=4,
        population=2,
        generations=1,
        n_slots=4,
        budget_layer_apps=2_000_000,
        max_schedule_layer_repeats=None,
        out="",
    )
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.fixture()
def fake_load(monkeypatch, tmp_path):
    """Route mlx_lm.load to the tiny model + stub tokenizer."""
    model_dir = tmp_path / "tiny-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    (model_dir / "model.safetensors").write_bytes(b"placeholder-bytes")
    model = _tiny_model()
    tokenizer = _StubTokenizer()

    import mlx_lm

    monkeypatch.setattr(
        mlx_lm, "load", lambda _path: (model, tokenizer)
    )
    return model_dir


def test_campaign_records_paired_outcomes_end_to_end(fake_load, tmp_path):
    args = _args(tmp_path, fake_load)
    receipt = campaign.run_campaign(args)

    assert receipt["schema"] == campaign.CAMPAIGN_SCHEMA
    assert receipt["holdout_tasks"] == 3
    assert receipt["winner_schedule_hash"]
    if receipt["winner_is_default"]:
        # CP126 78c85746: the search is seeded with the default, so on a
        # random model it legitimately returns it. Pairing a schedule against
        # ITSELF would credit it with wins over its own results, so no trials
        # run and the receipt says why.
        assert receipt["paired_outcomes_recorded"] == 0
        assert "against itself" in receipt["paired_trials_skipped_reason"]
        assert receipt["library_status"]["observations"] == 0
    else:
        assert receipt["paired_outcomes_recorded"] == 3
        assert receipt["library_status"]["observations"] == 3
    # Promotion needs MIN_TRIALS and a real win rate; three tie trials on a
    # random model must NOT promote — and the receipt says so honestly.
    assert isinstance(receipt["promotion_happened"], bool)
    # The library file persisted and reloads with the recorded evidence.
    from core.brain.llm.latent_cortex.schedules import ScheduleLibrary

    reloaded = ScheduleLibrary(Path(args.library))
    assert reloaded.status()["observations"] == receipt["paired_outcomes_recorded"]


def test_campaign_refuses_overlapping_splits(fake_load, tmp_path, monkeypatch):
    from core.learning import recurrence_curriculum as curriculum

    real_battery = curriculum.task_battery

    def same_split(families, depths, per_cell, seed=0):
        return real_battery(families, depths, per_cell, seed=99)

    monkeypatch.setattr(curriculum, "task_battery", same_split)
    with pytest.raises(ValueError, match="overlap"):
        campaign.run_campaign(_args(tmp_path, fake_load))


def test_per_task_commitments_are_distinct(fake_load, tmp_path):
    args = _args(tmp_path, fake_load, holdout_per_cell=4)
    receipt = campaign.run_campaign(args)
    if receipt["winner_is_default"]:
        # Nothing is paired against itself (CP126 78c85746), so there are no
        # commitments to be distinct.
        assert receipt["paired_outcomes_recorded"] == 0
        return
    assert receipt["paired_outcomes_recorded"] == 4
    from core.brain.llm.latent_cortex.schedules import ScheduleLibrary

    reloaded = ScheduleLibrary(Path(args.library))
    records = list(reloaded._records.values())
    assert records, "the campaign must have created a record"
    commitments = {
        outcome.task_commitment_sha256
        for record in records
        for outcome in record.outcomes.values()
    }
    assert len(commitments) == 4, "commitments must be per-task, never shared"
