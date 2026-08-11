from __future__ import annotations

from tools import evaluate_unified_intrinsic_checkpoint as evaluator
from tools.evaluate_unified_intrinsic_checkpoint import (
    _evaluation_layout,
    _sign_test_p_value,
)


def test_sign_test_is_exact_and_refuses_ties() -> None:
    assert _sign_test_p_value([0.0, 0.0]) is None
    assert _sign_test_p_value([1.0] * 8) == 0.0078125
    assert _sign_test_p_value([1.0] * 4 + [-1.0] * 4) == 1.0


def test_evaluation_layout_supports_legacy_colocation(tmp_path) -> None:
    root = tmp_path.resolve()
    layout = _evaluation_layout(root)
    assert layout.checkpoint_dir == root
    assert layout.dataset_path == root / "dataset.json"
    assert layout.tokenized_dataset_path == root / "tokenized_dataset.json"


def test_evaluation_layout_uses_resident_frozen_paths(tmp_path, monkeypatch) -> None:
    root = tmp_path.resolve()
    inputs = root / "inputs"
    output = root / "training-output"
    inputs.mkdir()
    output.mkdir()
    dataset = inputs / "dataset.json"
    tokenized = inputs / "tokenized_dataset.json"
    dataset.write_text("{}", encoding="ascii")
    tokenized.write_text("{}", encoding="ascii")
    (root / "campaign.json").write_text("{}", encoding="ascii")
    monkeypatch.setattr(
        evaluator,
        "_load_resident_campaign_config",
        lambda _path: {
            "paths": {
                "campaign_root": str(root),
                "training_output": str(output),
                "dataset": str(dataset),
                "tokenized_dataset": str(tokenized),
            }
        },
    )
    layout = _evaluation_layout(root)
    assert layout.checkpoint_dir == output
    assert layout.dataset_path == dataset
    assert layout.tokenized_dataset_path == tokenized
