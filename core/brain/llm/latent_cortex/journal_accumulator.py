"""Compact append-only proof over a campaign journal.

Today an action-intervention envelope embeds `campaign_journal_prefix`: every
journal event from genesis to the head, re-serialized and re-hashed inside each
envelope. That is correct and it does not scale — an n-event campaign builds n
envelopes each carrying O(n) events, so the artifacts, the serialization time,
and the verification time are all quadratic in campaign length. SPARK-068 names
the replacement: a versioned compact append-only proof that preserves
independent replay.

This module is that proof. It is a Merkle Mountain Range over the journal's
event digests:

- **Append-only by construction.** Appending a leaf never rewrites an existing
  node, so a root at size n commits to the exact n events that produced it, in
  order. A journal that reordered or edited history cannot reproduce the root.
- **O(log n) inclusion.** An event's membership is proven by its Merkle path
  inside one perfect subtree plus the sibling peaks, not by shipping the
  history around it.
- **Size-bound roots.** The size is hashed into the root, so a proof valid at
  one length cannot be replayed at another — the classic way an append-only log
  gets forged backwards.

What this does *not* do is decide the journal's semantics. Inclusion proves an
event is in the log; it does not prove the attempt was live, unclaimed, and
correctly ordered. That is the fold in `journal_state.py`'s territory and the
suffix replay above it. Keeping the two separate is deliberate: a compact
membership proof that quietly claimed to be a state proof would be exactly the
defect this codebase keeps finding.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Any, Final, Never

ACCUMULATOR_SCHEMA: Final = "aura.latent_cortex.campaign_journal_accumulator.v1"
INCLUSION_SCHEMA: Final = "aura.latent_cortex.campaign_journal_inclusion.v1"

_SHA256_PATTERN: Final = re.compile(r"\A[0-9a-f]{64}\Z")
_MAX_LEAVES: Final = 1 << 32

# Domain separation. Every hash in the structure says what kind of node it is,
# so a leaf digest can never be reinterpreted as an internal node or a root.
_LEAF_TAG: Final = b"\x00"
_NODE_TAG: Final = b"\x01"
_BAG_TAG: Final = b"\x02"
_ROOT_TAG: Final = b"\x03"

LEFT: Final = "left"
RIGHT: Final = "right"


class JournalAccumulatorError(ValueError):
    """An accumulator, root, or inclusion proof is invalid."""


def _fail(code: str) -> Never:
    raise JournalAccumulatorError(str(code or "journal_accumulator_invalid"))


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_PATTERN.match(value))


def _digest_bytes(value: Any, code: str) -> bytes:
    if not _is_sha256(value):
        _fail(code)
    return bytes.fromhex(value)


def _leaf(event_sha256: str) -> bytes:
    return hashlib.sha256(
        _LEAF_TAG + _digest_bytes(event_sha256, "journal_accumulator_leaf_invalid")
    ).digest()


def _node(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(_NODE_TAG + left + right).digest()


def _bag(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(_BAG_TAG + left + right).digest()


def peak_sizes(size: int) -> list[int]:
    """Decompose a log length into perfect-subtree sizes, largest first.

    A range of 11 leaves is one subtree of 8, one of 2, and one of 1 -- the
    binary expansion of the length. This decomposition is what makes the
    structure append-only: adding leaves only ever merges peaks, never edits
    the interior of a completed subtree.
    """

    if type(size) is not int or size <= 0 or size > _MAX_LEAVES:
        _fail("journal_accumulator_size_invalid")
    sizes: list[int] = []
    remaining = size
    width = 1 << (remaining.bit_length() - 1)
    while width:
        if remaining >= width:
            sizes.append(width)
            remaining -= width
        width >>= 1
    return sizes


def _perfect_root(leaves: Sequence[bytes]) -> bytes:
    level = list(leaves)
    while len(level) > 1:
        level = [_node(level[i], level[i + 1]) for i in range(0, len(level), 2)]
    return level[0]


def _bag_peaks(peaks: Sequence[bytes]) -> bytes:
    # Fold right to left so the newest (smallest, rightmost) peak is innermost.
    accumulator = peaks[-1]
    for peak in reversed(peaks[:-1]):
        accumulator = _bag(peak, accumulator)
    return accumulator


def accumulator_root(event_digests: Sequence[str]) -> dict[str, Any]:
    """Commit to an exact ordered run of journal events."""

    if (
        not isinstance(event_digests, Sequence)
        or isinstance(event_digests, (str, bytes))
        or not event_digests
    ):
        _fail("journal_accumulator_events_invalid")
    leaves = [_leaf(value) for value in event_digests]
    size = len(leaves)

    offset = 0
    peaks: list[bytes] = []
    for width in peak_sizes(size):
        peaks.append(_perfect_root(leaves[offset : offset + width]))
        offset += width

    bagged = _bag_peaks(peaks)
    root = hashlib.sha256(
        _ROOT_TAG + size.to_bytes(8, "big") + bagged
    ).hexdigest()
    return {
        "schema": ACCUMULATOR_SCHEMA,
        "size": size,
        "root_sha256": root,
        "peak_sizes": peak_sizes(size),
    }


def inclusion_proof(event_digests: Sequence[str], index: int) -> dict[str, Any]:
    """Prove one event's membership without shipping the events around it."""

    if type(index) is not int or index < 0 or index >= len(event_digests):
        _fail("journal_accumulator_index_out_of_range")
    leaves = [_leaf(value) for value in event_digests]
    size = len(leaves)
    widths = peak_sizes(size)

    offset = 0
    subtree_index = -1
    subtree_leaves: list[bytes] = []
    local_index = -1
    peaks: list[bytes] = []
    for position, width in enumerate(widths):
        block = leaves[offset : offset + width]
        peaks.append(_perfect_root(block))
        if offset <= index < offset + width:
            subtree_index = position
            subtree_leaves = block
            local_index = index - offset
        offset += width

    path: list[dict[str, str]] = []
    level = subtree_leaves
    cursor = local_index
    while len(level) > 1:
        sibling = cursor ^ 1
        path.append(
            {
                "side": RIGHT if cursor % 2 == 0 else LEFT,
                "sha256": level[sibling].hex(),
            }
        )
        level = [_node(level[i], level[i + 1]) for i in range(0, len(level), 2)]
        cursor //= 2

    return {
        "schema": INCLUSION_SCHEMA,
        "size": size,
        "index": index,
        "event_sha256": event_digests[index],
        "subtree_index": subtree_index,
        "path": path,
        "peer_peaks": [
            peak.hex() for position, peak in enumerate(peaks) if position != subtree_index
        ],
    }


def verify_inclusion(proof: Any, *, root_sha256: str, size: int) -> bool:
    """Recompute the root from the proof alone and compare.

    Every structural expectation is checked before the hashes are folded: the
    claimed size must match the root's size, the subtree index must exist in
    that size's decomposition, and the path must be exactly as long as that
    subtree is deep. A proof with a short path cannot be padded into a
    different tree.
    """

    if not isinstance(proof, dict) or proof.get("schema") != INCLUSION_SCHEMA:
        _fail("journal_accumulator_proof_invalid")
    if type(size) is not int or proof.get("size") != size:
        _fail("journal_accumulator_proof_size_differs")
    if not _is_sha256(root_sha256):
        _fail("journal_accumulator_root_invalid")

    widths = peak_sizes(size)
    subtree_index = proof.get("subtree_index")
    if type(subtree_index) is not int or not 0 <= subtree_index < len(widths):
        _fail("journal_accumulator_proof_subtree_invalid")

    width = widths[subtree_index]
    depth = width.bit_length() - 1
    path = proof.get("path")
    if not isinstance(path, list) or len(path) != depth:
        _fail("journal_accumulator_proof_path_length_differs")

    peer_peaks = proof.get("peer_peaks")
    if not isinstance(peer_peaks, list) or len(peer_peaks) != len(widths) - 1:
        _fail("journal_accumulator_proof_peaks_differ")

    index = proof.get("index")
    offset = sum(widths[:subtree_index])
    if (
        type(index) is not int
        or not offset <= index < offset + width
    ):
        _fail("journal_accumulator_proof_index_differs")

    node = _leaf(proof.get("event_sha256"))
    for step in path:
        if (
            not isinstance(step, dict)
            or set(step) != {"side", "sha256"}
            or step["side"] not in (LEFT, RIGHT)
        ):
            _fail("journal_accumulator_proof_step_invalid")
        sibling = _digest_bytes(step["sha256"], "journal_accumulator_proof_step_invalid")
        node = (
            _node(node, sibling) if step["side"] == RIGHT else _node(sibling, node)
        )

    peaks: list[bytes] = []
    peers = iter(peer_peaks)
    for position in range(len(widths)):
        if position == subtree_index:
            peaks.append(node)
        else:
            peaks.append(
                _digest_bytes(next(peers), "journal_accumulator_proof_peaks_differ")
            )

    recomputed = hashlib.sha256(
        _ROOT_TAG + size.to_bytes(8, "big") + _bag_peaks(peaks)
    ).hexdigest()
    return recomputed == root_sha256


def proof_size_bytes(proof: Any) -> int:
    """How large a compact proof actually is, for the scaling assertion."""

    if not isinstance(proof, dict) or proof.get("schema") != INCLUSION_SCHEMA:
        _fail("journal_accumulator_proof_invalid")
    return 32 * (len(proof["path"]) + len(proof["peer_peaks"]) + 1)


__all__ = [
    "ACCUMULATOR_SCHEMA",
    "INCLUSION_SCHEMA",
    "LEFT",
    "RIGHT",
    "JournalAccumulatorError",
    "accumulator_root",
    "inclusion_proof",
    "peak_sizes",
    "proof_size_bytes",
    "verify_inclusion",
]
