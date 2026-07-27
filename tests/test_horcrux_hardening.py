import base64
import hashlib
import json
from pathlib import Path

import pytest

from core.memory.horcrux import HorcruxManager, reconstruct_secret, split_secret


def _manager(tmp_path: Path) -> HorcruxManager:
    store: dict[str, str] = {}

    def getter(key: str) -> str | None:
        return store.get(key)

    def setter(key: str, value: str) -> None:
        store[key] = value

    manager = HorcruxManager(base_dir=str(tmp_path), secret_getter=getter, secret_setter=setter)
    manager.hardware_base = b"aura-hardware-seed".ljust(32, b"0")
    return manager


def test_shamir_split_reconstructs_threshold_secret() -> None:
    secret = b"0123456789abcdef0123456789abcdef"
    shares = split_secret(secret, threshold=3, num_shares=5)

    recovered = reconstruct_secret({1: shares[1], 3: shares[3], 5: shares[5]})

    assert recovered == secret


def test_keychain_shard_uses_injected_secret_store(tmp_path: Path) -> None:
    manager = _manager(tmp_path)

    manager._save_keychain_sync(1, b"keychain-shard", service="AuraHorcrux")

    assert manager._load_keychain_sync(service="AuraHorcrux") == (1, b"keychain-shard")


def test_shard_cache_is_encrypted_and_round_trips(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    shards = {1: b"one", 2: b"two", 3: b"three"}

    manager._save_shard_cache(shards)

    cache_text = (tmp_path / "shard_cache.enc").read_text(encoding="utf-8")
    envelope = json.loads(cache_text)
    assert envelope["version"] == 2
    assert base64.b64encode(b"three").decode() not in cache_text
    assert manager._load_shard_cache() == shards


def test_shard_cache_reads_legacy_xor_payload(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    raw = json.dumps({"1": base64.b64encode(b"one").decode()}).encode()
    mask = (manager.hardware_base * (len(raw) // 32 + 1))[: len(raw)]
    encoded = bytes(a ^ b for a, b in zip(raw, mask, strict=True))
    (tmp_path / "shard_cache.enc").write_bytes(encoded)

    assert manager._load_shard_cache() == {1: b"one"}


def test_file_and_hint_shards_round_trip(tmp_path: Path) -> None:
    manager = _manager(tmp_path)

    manager._save_file_sync(3, b"file-shard")
    manager._save_hint_sync(b"hint-shard", "answer")

    assert manager._load_file_sync() == (3, b"file-shard")
    assert manager._load_hint_sync("answer") == (4, b"hint-shard")
    assert manager._load_hint_sync("wrong") != (4, b"hint-shard")


def test_domain_separated_subkeys_are_stable_distinct_and_bound_to_root(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    with pytest.raises(RuntimeError, match="not initialized"):
        manager.derive_subkey("rlc.verified_replay_sft.partition.v1")

    manager.derived_key = b"r" * 32
    partition = manager.derive_subkey(
        "rlc.verified_replay_sft.partition.v1"
    )
    dedup = manager.derive_subkey("rlc.verified_replay_sft.dedup.v1")

    assert len(partition) == 32
    assert partition == manager.derive_subkey(
        "rlc.verified_replay_sft.partition.v1"
    )
    assert partition != dedup
    assert manager.derived_key not in (partition, dedup)
    first_identity = manager.key_identity_sha256
    assert len(first_identity) == 64
    assert manager.derived_key.hex() not in first_identity

    manager.derived_key = b"s" * 32
    assert manager.key_identity_sha256 != first_identity
    assert manager.derive_subkey(
        "rlc.verified_replay_sft.partition.v1"
    ) != partition


def test_subkey_api_rejects_malformed_active_root(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.derived_key = b"too-short"

    with pytest.raises(RuntimeError, match="not initialized"):
        _ = manager.key_identity_sha256
    with pytest.raises(RuntimeError, match="not initialized"):
        manager.derive_subkey("rlc.verified_replay_sft.partition.v1")


@pytest.mark.parametrize(
    "context",
    ("", "UPPERCASE", "contains spaces", "../escape", "x" * 129),
)
def test_subkey_context_rejects_ambiguous_or_unbounded_names(
    tmp_path: Path,
    context: str,
) -> None:
    manager = _manager(tmp_path)
    manager.derived_key = hashlib.sha256(b"test-root").digest()

    with pytest.raises(ValueError, match="context is invalid"):
        manager.derive_subkey(context)
