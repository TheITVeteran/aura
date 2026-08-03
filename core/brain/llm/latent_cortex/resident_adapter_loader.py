"""Transactional loader for recurrence-native resident adapter packages.

The research campaign and the live worker must execute the same adapter
topology. This module is deliberately independent of either caller: it
reconstructs shared, depth-conditioned, and role-conditioned LoRA modules,
verifies every tensor at the byte and parameter boundaries, and restores the
original model graph on any failure.
"""

from __future__ import annotations

import hashlib
import stat
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Never

from core.brain.llm.latent_cortex import (
    recurrent_grpo_adapter_identity,
    resident_recurrent_sft_adapter_identity,
)
from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (
    MANIFEST_SCHEMA_V2,
)
from core.runtime.file_read_gateway import read_stable_bytes


class ResidentAdapterLoadError(RuntimeError):
    """The package could not be attached without weakening its identity."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise ResidentAdapterLoadError(code)


def resolve_resident_adapter_projection(
    model: Any,
    path: str,
) -> tuple[Any, str, Any]:
    parts = path.split(".")
    if len(parts) < 4 or parts[:2] != ["model", "layers"]:
        _fail("resident_adapter_projection_path_invalid")
    current = model
    for segment in parts[:-1]:
        if segment.isdecimal():
            try:
                current = current[int(segment)]
            except (IndexError, KeyError, TypeError) as exc:
                raise ResidentAdapterLoadError(
                    "resident_adapter_projection_index_invalid"
                ) from exc
        else:
            try:
                current = getattr(current, segment)
            except AttributeError as exc:
                raise ResidentAdapterLoadError(
                    "resident_adapter_projection_owner_missing"
                ) from exc
    leaf = parts[-1]
    try:
        original = getattr(current, leaf)
    except AttributeError as exc:
        raise ResidentAdapterLoadError(
            "resident_adapter_projection_missing"
        ) from exc
    return current, leaf, original


def _relative_artifact(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        _fail("resident_adapter_artifact_path_invalid")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or str(pure) != value
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        _fail("resident_adapter_artifact_path_invalid")
    current = root
    for part in pure.parts:
        current /= part
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise ResidentAdapterLoadError(
                "resident_adapter_artifact_unavailable"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            _fail("resident_adapter_artifact_symlink_forbidden")
    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ResidentAdapterLoadError(
            "resident_adapter_artifact_outside_package"
        ) from exc
    if not resolved.is_file():
        _fail("resident_adapter_artifact_not_file")
    return resolved


def _package_root(value: str | Path) -> Path:
    lexical = Path(value).expanduser().absolute()
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise ResidentAdapterLoadError(
                "resident_adapter_package_unavailable"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            _fail("resident_adapter_package_symlink_forbidden")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise ResidentAdapterLoadError(
            "resident_adapter_package_unavailable"
        ) from exc
    if not resolved.is_dir():
        _fail("resident_adapter_package_invalid")
    return resolved


def _tensor_projection(key: str) -> str:
    if key.endswith((".lora_a", ".lora_b")):
        return key.rsplit(".", 1)[0]
    prefix, separator, bank_index = key.rpartition(".")
    if separator and bank_index.isdecimal() and prefix.endswith(
        (".depth_a", ".depth_b", ".role_a", ".role_b")
    ):
        return prefix.rsplit(".", 1)[0]
    _fail("resident_adapter_tensor_key_invalid")


def _scoped_schema(schema: Any) -> bool:
    return schema in {
        MANIFEST_SCHEMA_V2,
        recurrent_grpo_adapter_identity.MANIFEST_SCHEMA,
        *resident_recurrent_sft_adapter_identity.MANIFEST_SCHEMAS,
    }


def _adapter_binding(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    if manifest.get("schema") in resident_recurrent_sft_adapter_identity.MANIFEST_SCHEMAS:
        bindings = manifest.get("bindings")
        binding = bindings.get("adapter") if isinstance(bindings, Mapping) else None
    else:
        binding = manifest.get("adapter")
    if (
        not isinstance(binding, Mapping)
        or set(binding) != {"path", "sha256", "size_bytes"}
        or not isinstance(binding.get("sha256"), str)
        or len(binding["sha256"]) != 64
        or type(binding.get("size_bytes")) is not int
        or binding["size_bytes"] <= 0
    ):
        _fail("resident_adapter_binding_invalid")
    return binding


def load_resident_adapter(
    model: Any,
    adapter_dir: str | Path,
    manifest: Mapping[str, Any],
) -> int:
    """Attach one already admitted package and return its projection count.

    This function still rechecks all load-boundary facts. A caller may have
    validated the package earlier, but bytes, topology, or the resident model
    can change between admission and attachment.
    """

    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.lora import LoRALinear

    from core.brain.llm.latent_cortex.fast_weights import _linear_dims
    from core.brain.llm.latent_cortex.recurrence_adapter import ScopedLoRALinear

    package_root = _package_root(adapter_dir)

    lora = manifest.get("lora")
    tensor_rows = manifest.get("tensors")
    if not isinstance(lora, Mapping) or not isinstance(tensor_rows, list):
        _fail("resident_adapter_manifest_invalid")
    try:
        rank = int(lora["rank"])
        targets = tuple(lora["targets"])
        expected_paths = sorted(lora["projection_paths"])
        tensor_records = {
            str(record["key"]): dict(record)
            for record in tensor_rows
            if isinstance(record, Mapping)
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ResidentAdapterLoadError(
            "resident_adapter_manifest_invalid"
        ) from exc
    if (
        rank <= 0
        or not targets
        or len(tensor_records) != len(tensor_rows)
        or not tensor_records
    ):
        _fail("resident_adapter_manifest_invalid")

    scoped = _scoped_schema(manifest.get("schema"))
    count_key = "wrapped_projections" if scoped else "wrapped_projection_count"
    try:
        expected_count = int(lora[count_key])
        scale = float(lora.get("scale", 20.0))
        dropout = float(lora.get("dropout", 0.0))
    except (KeyError, TypeError, ValueError) as exc:
        raise ResidentAdapterLoadError(
            "resident_adapter_manifest_invalid"
        ) from exc
    projections = sorted({_tensor_projection(key) for key in tensor_records})
    if projections != expected_paths or expected_count != len(projections):
        _fail("resident_adapter_topology_mismatch")

    wrapper_type = ScopedLoRALinear if scoped else LoRALinear
    originals: list[tuple[Any, str, Any]] = []
    try:
        for path in projections:
            target = path.rsplit(".", 1)[-1]
            if target not in targets:
                _fail("resident_adapter_projection_target_undeclared")
            parent, leaf, original = resolve_resident_adapter_projection(
                model,
                path,
            )
            output_features, input_features = _linear_dims(original)
            try:
                a_shape = tuple(tensor_records[f"{path}.lora_a"]["shape"])
                b_shape = tuple(tensor_records[f"{path}.lora_b"]["shape"])
            except (KeyError, TypeError) as exc:
                raise ResidentAdapterLoadError(
                    "resident_adapter_shared_tensor_missing"
                ) from exc
            if a_shape != (input_features, rank) or b_shape != (
                rank,
                output_features,
            ):
                _fail("resident_adapter_tensor_dimensions_mismatch")
            originals.append((parent, leaf, original))
            if wrapper_type is ScopedLoRALinear:
                try:
                    block_index = int(path.split(".")[2])
                except (IndexError, ValueError) as exc:
                    raise ResidentAdapterLoadError(
                        "resident_adapter_projection_layer_invalid"
                    ) from exc
                wrapped = ScopedLoRALinear.from_base(
                    original,
                    r=rank,
                    scale=scale,
                    dropout=dropout,
                    block_index=block_index,
                    site=path,
                )
            else:
                wrapped = LoRALinear.from_base(original, r=rank)
            setattr(parent, leaf, wrapped)

        depth_count = int(lora.get("depth_bank_size", 0))
        if depth_count:
            from core.learning.depth_conditioned_lora import (
                wrap_depth_conditioned,
            )

            depth_banks = wrap_depth_conditioned(model, depths=depth_count)
            if sorted(depth_banks) != projections:
                _fail("resident_adapter_depth_inventory_mismatch")
        role_count = int(lora.get("role_bank_size", 0))
        if role_count:
            from core.learning.role_conditioned_lora import (
                wrap_role_conditioned,
            )

            role_banks = wrap_role_conditioned(model, branches=role_count)
            if sorted(role_banks) != projections:
                _fail("resident_adapter_role_inventory_mismatch")

        binding = _adapter_binding(manifest)
        weights_path = _relative_artifact(package_root, binding["path"])
        expected_size = int(binding["size_bytes"])
        try:
            before = read_stable_bytes(
                weights_path,
                max_bytes=expected_size,
            )
        except (OSError, ValueError) as exc:
            raise ResidentAdapterLoadError(
                "resident_adapter_weights_unreadable"
            ) from exc
        if (
            len(before) != expected_size
            or hashlib.sha256(before).hexdigest() != binding["sha256"]
        ):
            _fail("resident_adapter_weights_identity_mismatch")
        weights = mx.load(str(weights_path))
        if set(weights) != set(tensor_records):
            _fail("resident_adapter_weight_keys_mismatch")
        parameters = dict(tree_flatten(model.parameters()))
        missing = sorted(set(tensor_records) - set(parameters))
        if missing:
            _fail("resident_adapter_parameter_not_addressable")
        model.load_weights(list(weights.items()), strict=False)
        loaded = dict(tree_flatten(model.parameters()))
        mx.eval(*(loaded[key] for key in sorted(tensor_records)))
        if any(
            not bool(mx.array_equal(loaded[key], weights[key]))
            for key in tensor_records
        ):
            _fail("resident_adapter_weight_readback_mismatch")
        try:
            after = read_stable_bytes(weights_path, max_bytes=expected_size)
        except (OSError, ValueError) as exc:
            raise ResidentAdapterLoadError(
                "resident_adapter_weights_unreadable"
            ) from exc
        if after != before:
            _fail("resident_adapter_weights_changed_during_load")
    except BaseException:  # noqa: BLE001 - rollback must cover cancellation too
        for parent, leaf, original in reversed(originals):
            setattr(parent, leaf, original)
        raise
    mx.eval(model.parameters())
    return len(originals)


__all__ = [
    "ResidentAdapterLoadError",
    "load_resident_adapter",
    "resolve_resident_adapter_projection",
]
