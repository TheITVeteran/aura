"""Private semantic directions for query-scoped neural plasticity.

The exact teacher never enters generation context.  Its token embeddings are
used only to choose the output subspace available to an episodic low-rank
update; the failed incumbent supplies the contrast.  Final capability still
requires a fresh teacher-free decode accepted by the task verifier.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def build_contrastive_semantic_seeds(
    model: Any,
    *,
    target_tokens: Sequence[int],
    contrast_tokens: Sequence[int],
    rank: int,
):
    """Return up to ``rank`` orthonormal target-minus-incumbent directions.

    Output projections in the recurrent window emit model-width residuals, so
    token embeddings provide a checkpoint-native semantic basis.  The first
    vector is the difference between sequence centroids.  Remaining vectors
    are deterministic token-level contrasts, Gram-Schmidt orthogonalized so a
    rank-r adapter does not spend several columns on the same direction.
    """

    import mlx.core as mx

    if type(rank) is not int or rank <= 0:
        raise ValueError("semantic plasticity rank must be positive")
    target = list(target_tokens)
    contrast = list(contrast_tokens)
    if not target or not contrast:
        raise ValueError("semantic plasticity requires target and contrast tokens")
    if any(type(token) is not int or token < 0 for token in (*target, *contrast)):
        raise ValueError("semantic plasticity token sequence is invalid")

    inner = model.model
    target_embeddings = inner.embed_tokens(mx.array(target))
    contrast_embeddings = inner.embed_tokens(mx.array(contrast))
    if target_embeddings.ndim == 3:
        target_embeddings = target_embeddings[0]
    if contrast_embeddings.ndim == 3:
        contrast_embeddings = contrast_embeddings[0]
    if target_embeddings.ndim != 2 or contrast_embeddings.ndim != 2:
        raise RuntimeError("semantic plasticity embedding shape is unsupported")
    if target_embeddings.shape[1] != contrast_embeddings.shape[1]:
        raise RuntimeError("semantic plasticity embedding widths differ")

    target_mean = mx.mean(target_embeddings, axis=0)
    contrast_mean = mx.mean(contrast_embeddings, axis=0)
    candidates = [target_mean - contrast_mean]
    # Spread deterministic samples across the private target rather than
    # privileging only its first token.  Subtracting the incumbent centroid
    # keeps every candidate explicitly contrastive.
    if rank > 1:
        count = int(target_embeddings.shape[0])
        for index in range(rank * 2):
            position = min(count - 1, ((index + 1) * count) // (rank * 2 + 1))
            candidates.append(target_embeddings[position] - contrast_mean)

    basis = []
    for candidate in candidates:
        vector = candidate.astype(mx.float32)
        for prior in basis:
            vector = vector - mx.sum(vector * prior) * prior
        norm = mx.linalg.norm(vector)
        mx.eval(norm)
        norm_value = float(norm)
        if not norm_value > 1e-6:
            continue
        basis.append(mx.stop_gradient(vector / norm))
        if len(basis) >= rank:
            break
    if not basis:
        raise RuntimeError("semantic plasticity target has no usable contrast direction")
    seeds = mx.stack(basis, axis=0)
    mx.eval(seeds)
    return seeds


def build_layerwise_trajectory_directions(
    target_features: Mapping[int, Any],
    contrast_features: Mapping[int, Any],
    *,
    rank: int,
) -> dict[int, Any]:
    """Build orthonormal teacher-minus-incumbent bases at each real layer."""

    import mlx.core as mx

    if type(rank) is not int or rank <= 0:
        raise ValueError("trajectory plasticity rank must be positive")
    target_layers = {int(index) for index in target_features}
    contrast_layers = {int(index) for index in contrast_features}
    if not target_layers or target_layers != contrast_layers:
        raise ValueError("trajectory feature layer inventories differ")
    directions: dict[int, Any] = {}
    for layer in sorted(target_layers):
        target = target_features[layer]
        contrast = contrast_features[layer]
        if (
            getattr(target, "ndim", 0) != 2
            or getattr(contrast, "ndim", 0) != 2
            or target.shape != contrast.shape
            or int(target.shape[0]) < rank
        ):
            raise ValueError("trajectory feature shapes differ")
        basis = []
        for row in range(int(target.shape[0])):
            vector = (target[row] - contrast[row]).astype(mx.float32)
            for prior in basis:
                vector = vector - mx.sum(vector * prior) * prior
            norm = mx.linalg.norm(vector)
            mx.eval(norm)
            if float(norm) <= 1e-6:
                continue
            basis.append(mx.stop_gradient(vector / norm))
            if len(basis) >= rank:
                break
        if len(basis) != rank:
            raise RuntimeError(
                f"trajectory correction rank collapsed at layer {layer}: "
                f"{len(basis)}/{rank}"
            )
        directions[layer] = mx.stack(basis, axis=0)
        mx.eval(directions[layer])
    return directions


__all__ = [
    "build_contrastive_semantic_seeds",
    "build_layerwise_trajectory_directions",
]
