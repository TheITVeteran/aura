"""Encrypted memory payloads round-trip, and fail closed on a wrong key.

This file defined ``def test()`` — which pytest does not collect, because the
convention is ``test_*``. So it never ran, and it counted as coverage for
memory encryption purely by filename.

It was also stale: it asserted on ``result['ratio']`` and described
"compression", while ``encode_payload`` is AES-GCM encryption. A test that
had run would have failed on that years ago.
"""

from __future__ import annotations

import base64
import os

import pytest

from core.memory.black_hole import decode_payload, encode_payload

KEY = base64.b64encode(os.urandom(32)).decode()
OTHER_KEY = base64.b64encode(os.urandom(32)).decode()
TEXT = "The universe remembers everything. " * 50


def test_round_trip_preserves_the_payload_exactly():
    encoded = encode_payload(TEXT, KEY)
    assert decode_payload(encoded["encoded"], KEY) == TEXT


def test_ciphertext_does_not_leak_the_plaintext():
    encoded = encode_payload(TEXT, KEY)
    assert "universe remembers" not in encoded["encoded"]


def test_encryption_is_nondeterministic():
    """A fresh nonce per call — identical plaintext must not produce identical
    ciphertext, or repeated memories become linkable."""
    first = encode_payload(TEXT, KEY)["encoded"]
    second = encode_payload(TEXT, KEY)["encoded"]
    assert first != second


def test_a_wrong_key_does_not_return_the_plaintext():
    encoded = encode_payload(TEXT, KEY)
    assert decode_payload(encoded["encoded"], OTHER_KEY) != TEXT


def test_a_corrupted_blob_does_not_return_the_plaintext():
    encoded = encode_payload(TEXT, KEY)["encoded"]
    corrupted = ("A" if encoded[0] != "A" else "B") + encoded[1:]
    assert decode_payload(corrupted, KEY) != TEXT


def test_bytes_input_is_accepted():
    encoded = encode_payload(TEXT.encode(), KEY)
    assert decode_payload(encoded["encoded"], KEY) == TEXT


def test_empty_payload_round_trips():
    encoded = encode_payload("", KEY)
    assert decode_payload(encoded["encoded"], KEY) == ""
