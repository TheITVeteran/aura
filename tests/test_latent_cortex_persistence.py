from __future__ import annotations

import hashlib
import json
import stat
from types import SimpleNamespace

import pytest

from core.brain.llm.latent_cortex import persistence
from core.brain.llm.latent_cortex.persistence import (
    LatentCortexPersistence,
    StaleScheduleLibraryError,
    get_latent_cortex_persistence,
)


def test_lab_report_uses_private_transactional_persistence(tmp_path):
    path = tmp_path / "nested" / "report.json"

    receipt = get_latent_cortex_persistence().save_lab_report(path, b'{"ok":true}')

    assert path.read_bytes() == b'{"ok":true}'
    assert receipt.paths == (str(path),)
    assert receipt.transaction_id
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_verified_replay_uses_private_transactional_persistence(tmp_path):
    path = tmp_path / "private" / "verified-replay.json"
    payload = b'{"encrypted":true}'

    receipt = get_latent_cortex_persistence().save_verified_replay_buffer(
        path,
        payload,
    )

    assert path.read_bytes() == payload
    assert receipt.paths == (str(path),)
    assert receipt.transaction_id
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_canonical_persistence_rejects_tampered_gateway_receipt(tmp_path, monkeypatch):
    path = tmp_path / "report.json"
    payload = b'{"ok":true}'

    class TamperedGateway:
        def write_bytes_batch(self, entries, *, source):
            expected_path = str(path.resolve())
            return SimpleNamespace(
                transaction_id="forged",
                paths=(expected_path,),
                sha256=((expected_path, "0" * 64),),
            )

    monkeypatch.setattr(
        persistence, "get_file_write_gateway", lambda: TamperedGateway()
    )
    with pytest.raises(RuntimeError, match="does not match committed payloads"):
        LatentCortexPersistence().save_lab_report(path, payload)

    assert hashlib.sha256(payload).hexdigest() != "0" * 64


def _schedule_payload(revision: int) -> bytes:
    return json.dumps(
        {"version": 2, "revision": revision, "records": []},
        sort_keys=True,
    ).encode("utf-8")


def test_schedule_persistence_rejects_stale_compare_and_swap(tmp_path):
    path = tmp_path / "schedules.json"
    owner = LatentCortexPersistence()
    owner.save_schedule_library(path, _schedule_payload(1), expected_revision=0)

    with pytest.raises(StaleScheduleLibraryError, match="changed from 0 to 1"):
        owner.save_schedule_library(path, _schedule_payload(1), expected_revision=0)

    assert json.loads(path.read_text(encoding="utf-8"))["revision"] == 1


def test_schedule_persistence_never_overwrites_invalid_existing_evidence(tmp_path):
    path = tmp_path / "schedules.json"
    invalid = b'{"version":2,"revision":0,"records":{}}'
    path.write_bytes(invalid)

    with pytest.raises(ValueError, match="invalid schema"):
        LatentCortexPersistence().save_schedule_library(
            path,
            _schedule_payload(1),
            expected_revision=0,
        )

    assert path.read_bytes() == invalid
