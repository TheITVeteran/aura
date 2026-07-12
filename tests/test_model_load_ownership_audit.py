from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.closeout.audit_model_load_ownership import ROOT, run_audit


def test_repository_model_load_inventory_is_complete() -> None:
    report = run_audit()

    assert report["passed"] is True
    assert report["inventory_entries"] == 27
    assert report["owned_paths"] == 27
    assert report["load_references"] == 39
    assert report["source_paths_scanned"] >= 2_000


@pytest.mark.parametrize(
    "relative_path",
    [
        "core/direct_model.py",
        "scripts/direct_model.py",
        "aura_bench/direct_model.py",
    ],
)
def test_uninventoried_model_load_fails_closed(
    tmp_path: Path,
    relative_path: str,
) -> None:
    source = tmp_path / relative_path
    source.parent.mkdir(parents=True)
    source.write_text(
        "from mlx_lm import load\n\nmodel, tokenizer = load('/models/direct')\n",
        encoding="utf-8",
    )
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps({"schema": "aura.model_load_ownership.v1", "entries": []}),
        encoding="utf-8",
    )

    report = run_audit(root=tmp_path, inventory_path=inventory)

    assert report["passed"] is False
    assert report["findings"] == [
        {
            "code": "unowned_model_load",
            "path": relative_path,
            "detail": "load references at lines [3]",
        }
    ]


def test_inventory_path_is_repository_scoped() -> None:
    assert (ROOT / "config" / "model_load_ownership.json").is_file()


def test_inline_child_program_model_load_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "scripts" / "inline_child.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "code = \"from mlx_lm import load\\nmodel, tok = load('/models/direct')\\n\"\n",
        encoding="utf-8",
    )
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps({"schema": "aura.model_load_ownership.v1", "entries": []}),
        encoding="utf-8",
    )

    report = run_audit(root=tmp_path, inventory_path=inventory)

    assert report["findings"] == [
        {
            "code": "unowned_model_load",
            "path": "scripts/inline_child.py",
            "detail": "load references at lines [1]",
        }
    ]


def test_mlx_submodule_load_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "scripts" / "submodule_loader.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "from mlx_lm.utils import load as model_load\n"
        "model, tok = model_load('/models/direct')\n",
        encoding="utf-8",
    )
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps({"schema": "aura.model_load_ownership.v1", "entries": []}),
        encoding="utf-8",
    )

    report = run_audit(root=tmp_path, inventory_path=inventory)

    assert report["findings"] == [
        {
            "code": "unowned_model_load",
            "path": "scripts/submodule_loader.py",
            "detail": "load references at lines [2]",
        }
    ]
