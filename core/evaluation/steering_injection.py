"""Residual-stream steering injection for evaluation harnesses.

Correctness notes learned the hard way:

- ``layer.__call__ = hooked`` does NOT intercept ``layer(...)``: Python
  resolves special methods on the type, not the instance. The original live
  A/B runner had exactly that bug, so its "steered" condition never injected
  anything and the reported effect was a system-prompt confound. Injection
  here uses a temporary per-instance subclass swap — the same pattern the
  production extraction script (`training/extract_steering_vectors.py`) and
  `AffectiveSteeringHook` use.
- Vectors are unit-normalized at load so ``alpha`` means the same thing
  regardless of the raw extraction norms (~40-70 for 32B layers).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("Aura.Evaluation.SteeringInjection")


def load_production_vectors(
    vectors_dir: Path,
    *,
    dimensions: tuple[str, ...] = ("valence_positive", "curiosity"),
) -> dict[int, np.ndarray]:
    """Load unit-normalized production steering vectors, averaged per layer.

    Only ``extracted=True`` vectors are eligible — the point of the behavioral
    A/B is to test the artifacts that steer live traffic, not ad-hoc rederived
    directions. Multiple requested dimensions on the same layer average then
    renormalize (matching how simultaneous affect axes compose in the engine).
    """
    per_layer: dict[int, list[np.ndarray]] = {}
    for path in sorted(Path(vectors_dir).glob("*.npz")):
        try:
            with np.load(path, allow_pickle=True) as z:
                if not bool(z.get("extracted", np.array(False)).item()):
                    continue
                dimension = str(z["dimension"].item()) if "dimension" in z else ""
                if dimension not in dimensions:
                    continue
                layer = int(z["layer"].item()) if "layer" in z else -1
                vec = np.asarray(z["v"], dtype=np.float32).flatten()
        except (OSError, KeyError, ValueError) as exc:
            logger.debug("Skipped unreadable vector %s: %s", path.name, exc)
            continue
        norm = float(np.linalg.norm(vec))
        if layer < 0 or norm <= 1e-6:
            continue
        per_layer.setdefault(layer, []).append(vec / norm)

    combined: dict[int, np.ndarray] = {}
    for layer, vecs in per_layer.items():
        mean = np.mean(np.stack(vecs), axis=0)
        norm = float(np.linalg.norm(mean))
        if norm > 1e-6:
            combined[layer] = (mean / norm).astype(np.float32)
    return combined


class ResidualSteeringInjector:
    """Toggleable residual-stream injection over a loaded MLX model."""

    def __init__(
        self,
        model: Any,
        vectors: dict[int, np.ndarray],
        *,
        alpha: float = 8.0,
    ) -> None:
        import mlx.core as mx

        self._mx = mx
        self._model = model
        self._alpha = float(alpha)
        self._vectors = {
            layer: mx.array(vec) for layer, vec in vectors.items()
        }
        self._installed: list[tuple[Any, type]] = []
        self.active = False
        self.injection_count = 0

    def _layers(self) -> Any:
        for attr_path in ("model.layers", "layers"):
            obj = self._model
            try:
                for part in attr_path.split("."):
                    obj = getattr(obj, part)
            except AttributeError:
                continue
            if hasattr(obj, "__len__") and len(obj) > 0:
                return obj
        raise RuntimeError("cannot locate transformer layers on model")

    def install(self) -> int:
        """Subclass-swap the target layers; returns the number hooked."""
        mx = self._mx
        layers = self._layers()
        injector = self

        for layer_idx, vector in self._vectors.items():
            if layer_idx >= len(layers):
                continue
            layer = layers[layer_idx]
            original_class = layer.__class__

            def _make_steered_class(orig_cls: type, vec: Any) -> type:
                class SteeredLayer(orig_cls):  # type: ignore[misc, valid-type]
                    __module__ = orig_cls.__module__

                    def __call__(self, *args: Any, **kwargs: Any) -> Any:
                        result = super().__call__(*args, **kwargs)
                        if not injector.active:
                            return result
                        hidden = result[0] if isinstance(result, tuple) else result
                        try:
                            steered = hidden + injector._alpha * vec.astype(hidden.dtype)
                            injector.injection_count += 1
                        except (TypeError, ValueError):
                            return result
                        if isinstance(result, tuple):
                            return (steered,) + result[1:]
                        return steered

                return SteeredLayer

            layer.__class__ = _make_steered_class(original_class, vector)
            self._installed.append((layer, original_class))
        del mx
        return len(self._installed)

    def remove(self) -> None:
        for layer, original_class in self._installed:
            try:
                layer.__class__ = original_class
            except TypeError as exc:
                logger.warning("Steering layer restore failed: %s", exc)
        self._installed.clear()

    def __enter__(self) -> ResidualSteeringInjector:
        self.install()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.remove()
