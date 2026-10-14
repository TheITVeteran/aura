"""Explicit local visual-model selection with pinned source and checkpoint custody."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelSource:
    name: str
    repository: str
    revision: str
    checkpoint: str
    sha256: str
    license: str


SOURCES = {
    'cotracker3': ModelSource('cotracker3', 'https://github.com/facebookresearch/co-tracker',
        '82e02e8029753ad4ef13cf06be7f4fc5facdda4d',
        'https://huggingface.co/facebook/cotracker3/resolve/acc21f9dba34834006a50432bde26b86aa68d794/scaled_online.pth',
        '205d34789f19699d64b22cf93f9b697f15f28d4025240e31532e504109837218', 'CC-BY-NC-4.0'),
    'online_tapir': ModelSource('online_tapir', 'https://github.com/google-deepmind/tapnet',
        '730cda1c730877cfedbe01bf87fb1cadb78a565d',
        'https://storage.googleapis.com/dm-tapnet/bootstap/causal_bootstapir_checkpoint.pt',
        '87c1e752cf5ce56e3e2f7da460aeb4d40fc826d04ef2939bade86a5c7495377f', 'Apache-2.0'),
    'sam2': ModelSource('sam2', 'https://github.com/facebookresearch/sam2',
        '2b90b9f5ceec907a1c18123530e92e794ad901a4',
        'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt',
        '7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69', 'Apache-2.0'),
}


def verified_checkpoint(path: Path, expected_sha256: str) -> Path:
    """A selected checkpoint must exist and match before any model allocates memory."""
    if (not isinstance(path, Path) or not path.is_file() or len(expected_sha256) != 64
            or any(c not in '0123456789abcdef' for c in expected_sha256)):
        raise ValueError('a visual checkpoint needs a local file and its SHA-256 digest')
    with path.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    if digest != expected_sha256:
        raise ValueError('the visual checkpoint differs from its selected source')
    return path


def load_backend(name: str, weights: Path, *, device: str = 'cpu', expected_sha256: str | None = None,
                 **options: Any) -> Any:
    """Load one explicit numerical tracker; imports never download or select a model."""
    if name not in SOURCES:
        raise ValueError('the requested visual tracker has no registered source')
    verified_checkpoint(weights, expected_sha256 or SOURCES[name].sha256)
    if name == 'online_tapir':
        from core.perception.learned_tracking.tapir_backend import OnlineTapirBackend

        return OnlineTapirBackend(weights, device=device, **options)
    if name == 'cotracker3':
        from core.perception.learned_tracking.cotracker_backend import CoTrackerBackend

        return CoTrackerBackend(weights, device=device, **options)
    from core.perception.learned_tracking.sam2_backend import Sam2Backend

    return Sam2Backend(weights, device=device, **options)
