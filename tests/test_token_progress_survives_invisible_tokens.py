"""Generating is progress, whether or not it is visible yet.

The worker's only progress signal to the parent was the `token` stream message,
and that message was sent ONLY when the decoded step produced visible new text:

    if emit_text:
        ipc_writer.put({... "status": "token", "text": emit_text ...})

Decoding legitimately produces no visible delta for a while — a detokenizer
holding a partial UTF-8 sequence, suppressed start ids, a stop sequence being
scanned. To the parent that is indistinguishable from a wedged worker, because
`_current_first_token_at` is set from that message alone.

Live on the desktop surface 2026-07-26, on an ~800-token prompt:

    [MLX] First-token HARD CEILING exceeded (livelocked: heartbeats but zero
          tokens) ... 107.7s elapsed, sla=240.0s
    Cortex ran past this turn's deadline (107.7s elapsed, budget 106.8s) but is
          healthy (heartbeat 0.8s ago). Cancelling the request.
    Proof/operator request requires a valid Cortex response; refusing
          lower-lane fallback.

A healthy generation was cancelled and the turn was lost — the same category
error as the rest of this pass: work in progress read as damage.

A step that yields no visible text now emits `progress` instead. It carries no
text, and unlike `token` it is essential, so it also cannot be dropped by the
IPC writer's backpressure shedding.
"""
from __future__ import annotations

import re
from pathlib import Path

WORKER = Path("core/brain/llm/mlx_worker.py")
CLIENT = Path("core/brain/llm/mlx_client.py")


def _emit_block() -> str:
    src = WORKER.read_text(encoding="utf-8")
    start = src.index("                                    emit_text = (")
    return src[start : src.index("if stop_hit:", start)]


def test_a_step_with_no_visible_text_still_reports_progress() -> None:
    block = _emit_block()
    assert 'if emit_text:' in block
    assert re.search(r"else:\s", block), "the invisible-token case must be handled"
    assert '"status": "progress"' in block, (
        "a token that adds no visible text must still signal progress"
    )


def test_the_progress_signal_carries_no_text() -> None:
    """It is a liveness ping, not a second copy of the stream."""
    block = _emit_block()
    else_branch = block[block.rindex("else:") :]
    assert '"status": "progress"' in else_branch
    assert '"text"' not in else_branch, "progress must not duplicate stream text"
    assert '"tokens_generated": token_count' in else_branch


def test_the_client_counts_progress_as_first_token_progress() -> None:
    """Both statuses must reach _mark_token_progress, or the fix is inert."""
    client = CLIENT.read_text(encoding="utf-8")
    assert 'status in {"progress", "token"}' in client
    branch = client[client.index('status in {"progress", "token"}') :][:600]
    assert "_mark_token_progress" in branch


def test_progress_is_essential_and_cannot_be_shed() -> None:
    """`token` is sheddable under IPC backpressure; the liveness signal must not be."""
    worker = WORKER.read_text(encoding="utf-8")
    match = re.search(r"return status not in \{([^}]*)\}", worker)
    assert match, "the essential-message predicate must exist"
    non_essential = match.group(1)
    assert '"token"' in non_essential, "stream text stays sheddable"
    assert '"heartbeat"' in non_essential
    assert "progress" not in non_essential, (
        "the progress signal must survive backpressure, or the livelock "
        "false-positive returns exactly when the queue is busiest"
    )
