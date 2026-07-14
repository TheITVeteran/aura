"""Tests for CRSM capture ingestion into the LoRA training corpus."""
from __future__ import annotations

import json

from training.build_dataset_v3 import build_crsm_experience_examples, parse_crsm_capture_text


def test_parse_crsm_capture_accepts_chat_template_pair():
    pair = parse_crsm_capture_text(
        "<|im_start|>user\nWhat did you notice?<|im_end|>\n"
        "<|im_start|>assistant\nI noticed a mismatch and corrected it.<|im_end|>"
    )

    assert pair == ("What did you notice?", "I noticed a mismatch and corrected it.")


def test_crsm_experience_gate_rejects_internal_control_captures(tmp_path):
    dataset = tmp_path / "lora_dataset.jsonl"
    rows = [
        {
            "text": (
                "User: Will-approved self-reflection\n"
                "Aura: <thought>\nSelf-reflection accepted as a plasticity signal.\n</thought>"
            )
        },
        {
            "text": (
                "<|im_start|>user\nWhat did you learn from the failed desktop task?<|im_end|>\n"
                "<|im_start|>assistant\nI learned to verify the focused field before typing, then confirm the artifact landed in the requested app.<|im_end|>"
            )
        },
        {
            "text": (
                "<|im_start|>user\nWhat did you learn from the failed desktop task?<|im_end|>\n"
                "<|im_start|>assistant\nI learned to verify the focused field before typing, then confirm the artifact landed in the requested app.<|im_end|>"
            )
        },
    ]
    dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    examples, manifest = build_crsm_experience_examples(
        dataset,
        max_examples=10,
        system_variants=["system"],
    )

    assert len(examples) == 1
    assert manifest["source_lines"] == 3
    assert manifest["accepted"] == 1
    assert manifest["deduplicated"] == 1
    assert manifest["rejected_by_reason"]["internal_control_capture"] == 1
    assert examples[0]["messages"][1]["content"] == "What did you learn from the failed desktop task?"
    assert "verify the focused field" in examples[0]["messages"][2]["content"]


def test_train_and_fuse_marks_crsm_consumed_only_from_current_manifest(tmp_path, monkeypatch):
    import core.consciousness.crsm_loop_monitor as crsm_module
    from training import train_and_fuse

    dataset = tmp_path / "lora_dataset.jsonl"
    dataset.write_text("{}\n{}\n{}\n", encoding="utf-8")
    manifest = tmp_path / "crsm_integration_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source_lines": 3,
                "source_mtime": dataset.stat().st_mtime,
                "accepted": 2,
            }
        ),
        encoding="utf-8",
    )
    fused = tmp_path / "Aura-32B-test"
    fused.mkdir()
    calls = []

    class FakeMonitor:
        def mark_dataset_consumed(self, **kwargs):
            calls.append(kwargs)
            return True

    monkeypatch.setattr(train_and_fuse, "CRSM_DATASET", dataset)
    monkeypatch.setattr(train_and_fuse, "CRSM_INTEGRATION_MANIFEST", manifest)
    monkeypatch.setattr(crsm_module, "get_crsm_loop_monitor", lambda: FakeMonitor())

    train_and_fuse.mark_crsm_loop_consumed_after_training(fused)

    assert calls == [
        {
            "model_path": str(fused),
            "lines_consumed": 3,
            "accepted_lines": 2,
            "rejected_lines": 1,
            "manifest_path": str(manifest),
            "source": "training.train_and_fuse",
        }
    ]
