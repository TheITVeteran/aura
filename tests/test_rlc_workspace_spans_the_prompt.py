"""The workspace must span the prompt, not replicate its centroid."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_slots_are_not_sixteen_copies_of_one_direction():
    """Seeding every slot from the global mean prompt embedding gave the
    workspace an effective rank of one: measured slot-to-slot cosine 0.9993
    against 0.0419 for the prompt's own tokens. A recurrent operator cannot
    differentiate states that begin 99.93% aligned -- which is the same
    0.9994 pass-to-pass similarity this program chased for weeks, arriving
    from the seed rather than from the recurrence."""
    mx = pytest.importorskip("mlx.core")
    from core.brain.llm.latent_cortex.types import WorkspaceConfig
    from core.brain.llm.latent_cortex.workspace import LatentWorkspace

    # A prompt whose tokens genuinely differ from one another.
    key = mx.random.key(20260807)
    prompt = mx.random.normal((1, 64, 128), key=key)
    ws = LatentWorkspace.from_prompt_embeddings(
        prompt, WorkspaceConfig(n_slots=16, seed=0)
    )
    z = ws.z[0]
    zn = z / mx.maximum(mx.linalg.norm(z, axis=-1, keepdims=True), 1e-6)
    cos = zn @ zn.T
    off = [float(cos[i, j]) for i in range(16) for j in range(16) if i < j]
    mean_alignment = sum(off) / len(off)
    assert mean_alignment < 0.90, (
        f"slots are {mean_alignment:.4f} mutually aligned; the workspace has "
        "collapsed to approximately one direction"
    )


def test_a_prompt_shorter_than_the_workspace_still_seeds():
    """Fewer tokens than slots has no span decomposition; falling back to the
    global mean is the only sensible answer and must not raise."""
    mx = pytest.importorskip("mlx.core")
    from core.brain.llm.latent_cortex.types import WorkspaceConfig
    from core.brain.llm.latent_cortex.workspace import LatentWorkspace

    prompt = mx.random.normal((1, 4, 128), key=mx.random.key(1))
    ws = LatentWorkspace.from_prompt_embeddings(
        prompt, WorkspaceConfig(n_slots=16, seed=0)
    )
    assert ws.z.shape == (1, 16, 128)


def test_seeds_stay_inside_the_embedding_manifold():
    """Each slot pools a span of real token embeddings, so it stays in their
    convex hull -- the distribution the frozen layers were trained on. A seed
    whose norm departs from the prompt's own scale would be out-of-manifold
    input no matter how well it spans the content."""
    mx = pytest.importorskip("mlx.core")
    from core.brain.llm.latent_cortex.types import WorkspaceConfig
    from core.brain.llm.latent_cortex.workspace import LatentWorkspace, per_position_rms

    prompt = mx.random.normal((1, 64, 128), key=mx.random.key(7))
    ws = LatentWorkspace.from_prompt_embeddings(
        prompt, WorkspaceConfig(n_slots=16, seed=0)
    )
    target = float(mx.mean(per_position_rms(prompt)))
    seeded = [float(v) for v in per_position_rms(ws.z)[0, :, 0]]
    for value in seeded:
        assert 0.5 * target <= value <= 2.0 * target
