"""CP126 15f68852: an expert adapter must be removable, or never attached.

``_detach_expert_adapter`` can only restore modules that expose ``.linear``
or ``.embedding``. mlx_lm's ``load_adapters`` honours the adapter config's
``fine_tune_type``: a "full" (or "dora") adapter does not produce wrapper
modules at all — it writes into the resident weights. Attaching one
permanently rewrites the personality model this worker is serving, and the
old code discovered that only afterwards, via a detach that silently
restored nothing.

The type is now constrained before any weight mutation, and the module tree
is checked after the load as the authority on what actually happened.
"""
from __future__ import annotations

import json

import pytest

from core.brain.llm import mlx_worker
from core.brain.llm.mlx_worker import (
    _RESTORABLE_FINE_TUNE_TYPES,
    _unrestorable_wrapped,
    _validate_expert_adapter_dir,
)


@pytest.fixture
def adapter_root(tmp_path, monkeypatch):
    """An approved adapter root under a temp dir."""
    monkeypatch.setattr(
        mlx_worker, "_expert_adapter_approved_roots", lambda: [tmp_path],
    )
    return tmp_path


def _make_adapter(root, name: str, config: dict) -> str:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_config.json").write_text(json.dumps(config))
    (directory / "adapters.safetensors").write_text("weights")
    return str(directory)


class TestFineTuneTypeIsConstrained:
    def test_lora_is_accepted(self, adapter_root):
        path = _make_adapter(adapter_root, "lora", {"fine_tune_type": "lora"})
        assert _validate_expert_adapter_dir(path)

    def test_qlora_is_accepted(self, adapter_root):
        path = _make_adapter(adapter_root, "qlora", {"fine_tune_type": "qlora"})
        assert _validate_expert_adapter_dir(path)

    def test_an_absent_type_defaults_to_lora(self, adapter_root):
        # mlx_lm's own default when the key is missing.
        path = _make_adapter(adapter_root, "absent", {"lora_layers": 8})
        assert _validate_expert_adapter_dir(path)

    def test_a_full_finetune_is_refused(self, adapter_root):
        path = _make_adapter(adapter_root, "full", {"fine_tune_type": "full"})
        with pytest.raises(ValueError, match="fine_tune_type_not_restorable"):
            _validate_expert_adapter_dir(path)

    def test_dora_is_refused(self, adapter_root):
        path = _make_adapter(adapter_root, "dora", {"fine_tune_type": "dora"})
        with pytest.raises(ValueError, match="fine_tune_type_not_restorable"):
            _validate_expert_adapter_dir(path)

    def test_an_unknown_type_is_refused(self, adapter_root):
        path = _make_adapter(adapter_root, "weird", {"fine_tune_type": "something_new"})
        with pytest.raises(ValueError, match="fine_tune_type_not_restorable"):
            _validate_expert_adapter_dir(path)

    def test_a_non_object_config_is_refused(self, adapter_root):
        directory = adapter_root / "listcfg"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "adapter_config.json").write_text("[1, 2, 3]")
        (directory / "adapters.safetensors").write_text("weights")
        with pytest.raises(ValueError, match="config_not_an_object"):
            _validate_expert_adapter_dir(str(directory))

    def test_the_refusal_happens_before_any_weight_mutation(self):
        import inspect

        source = inspect.getsource(_validate_expert_adapter_dir)
        assert "fine_tune_type" in source
        # This function is pure validation: it must never touch the model.
        # Compare code only — comments legitimately discuss load_adapters.
        code = "\n".join(
            line for line in source.splitlines()
            if not line.lstrip().startswith("#")
        )
        assert "load_adapters(" not in code
        assert "model." not in code

    def test_the_restorable_set_is_explicit(self):
        assert "lora" in _RESTORABLE_FINE_TUNE_TYPES
        assert "full" not in _RESTORABLE_FINE_TUNE_TYPES


class _LinearWrapper:
    def __init__(self):
        self.linear = object()


class _EmbeddingWrapper:
    def __init__(self):
        self.embedding = object()


class _InPlaceMutation:
    """A module a detach could not restore — no wrapped base to put back."""


class TestUnrestorableDetection:
    def test_linear_wrappers_are_restorable(self):
        assert _unrestorable_wrapped([("a", _LinearWrapper())]) == []

    def test_embedding_wrappers_are_restorable(self):
        assert _unrestorable_wrapped([("a", _EmbeddingWrapper())]) == []

    def test_in_place_mutation_is_flagged(self):
        assert _unrestorable_wrapped([("layers.0", _InPlaceMutation())]) == ["layers.0"]

    def test_a_mixed_wrap_reports_only_the_unrestorable(self):
        wrapped = [
            ("ok.linear", _LinearWrapper()),
            ("ok.embed", _EmbeddingWrapper()),
            ("bad", _InPlaceMutation()),
        ]
        assert _unrestorable_wrapped(wrapped) == ["bad"]


class TestAttachVerifiesTheModuleTree:
    def _source(self) -> str:
        import inspect

        return inspect.getsource(mlx_worker._attach_expert_adapter)

    def test_attach_refuses_an_unrestorable_wrap(self):
        source = self._source()
        assert "expert_adapter_attach_unrestorable" in source
        assert "_unrestorable_wrapped(wrapped)" in source

    def test_attach_unwinds_before_raising(self):
        source = self._source()
        block = source.split("unrestorable = _unrestorable_wrapped(wrapped)", 1)[1][:400]
        assert block.index("_detach_expert_adapter(model, wrapped)") < block.index(
            "expert_adapter_attach_unrestorable",
        )

    def test_a_load_that_wrapped_nothing_is_refused(self):
        source = self._source()
        assert "expert_adapter_attach_wrapped_nothing" in source

    def test_the_module_tree_is_the_authority(self):
        # The config check is necessary but not sufficient; the post-load
        # verification is what makes the claim true.
        source = self._source()
        assert source.index("load_adapters(model, adapter_dir)") < source.index(
            "_unrestorable_wrapped(wrapped)",
        )
