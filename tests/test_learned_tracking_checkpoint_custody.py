"""Model loading verifies explicit checkpoint custody before allocating vision state."""
import hashlib

import pytest

from core.perception.learned_tracking.models import SOURCES, load_backend, verified_checkpoint


def test_a_checkpoint_must_match_its_actual_file(tmp_path):
    path = tmp_path / 'invented-weights'
    path.write_bytes(b'invented numerical weights')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert verified_checkpoint(path, digest) == path
    for wrong in ('0' * 64, 'abc', 'Z' * 64):
        with pytest.raises(ValueError):
            verified_checkpoint(path, wrong)
    with pytest.raises(ValueError):
        verified_checkpoint(tmp_path / 'absent', digest)


def test_selecting_a_known_name_does_not_bypass_custody_or_choose_another_backend(tmp_path):
    assert set(SOURCES) == {'online_tapir', 'cotracker3', 'sam2'}
    for name in SOURCES:
        with pytest.raises(ValueError):
            load_backend(name, tmp_path / 'absent')
    with pytest.raises(ValueError):
        load_backend('unregistered', tmp_path / 'absent')
