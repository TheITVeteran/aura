"""core/voice/duplex/protocol.py — Wire format for the /ws/voice lane.

Control and captions travel as JSON text frames; audio travels as binary
frames with a small fixed header. WebSocket guarantees ordering across both
kinds on one connection, so a JSON descriptor followed by its binary payload
would also work — but a header keeps each audio frame self-describing, which
means a client that drops a frame during a barge-in flush cannot
mis-attribute the next one to the wrong utterance.

Binary frame layout (little-endian), 8-byte header then PCM:

    offset  size  field
    0       1     opcode      AudioOpcode
    1       1     flags       bit0 = last chunk of this utterance
    2       2     seq         chunk sequence within the utterance
    4       4     utterance   monotonic utterance id

Payload is 16-bit signed mono PCM at ``OUTPUT_RATE``.
"""
from __future__ import annotations

import struct
from enum import IntEnum
from typing import Final

HEADER_FORMAT: Final = "<BBHI"
HEADER_SIZE: Final = struct.calcsize(HEADER_FORMAT)  # 8

FLAG_LAST_CHUNK: Final = 0x01


class AudioOpcode(IntEnum):
    """What kind of audio this frame carries.

    The client renders these differently: speech drives the orb and the
    caption track, backchannels duck under whatever else is playing, and
    fillers are marked so the UI does not caption "uh…" as her answer.
    """

    SPEECH = 1
    BACKCHANNEL = 2
    FILLER = 3


def encode_audio(
    payload: bytes,
    *,
    opcode: AudioOpcode,
    seq: int,
    utterance_id: int,
    last: bool = False,
) -> bytes:
    header = struct.pack(
        HEADER_FORMAT,
        int(opcode),
        FLAG_LAST_CHUNK if last else 0,
        seq & 0xFFFF,
        utterance_id & 0xFFFFFFFF,
    )
    return header + payload


def decode_audio(frame: bytes) -> tuple[AudioOpcode, bool, int, int, bytes]:
    """Inverse of :func:`encode_audio`. Used by tests and the Python client."""
    if len(frame) < HEADER_SIZE:
        raise ValueError("audio frame shorter than header")
    opcode, flags, seq, utterance = struct.unpack(HEADER_FORMAT, frame[:HEADER_SIZE])
    return (
        AudioOpcode(opcode),
        bool(flags & FLAG_LAST_CHUNK),
        seq,
        utterance,
        frame[HEADER_SIZE:],
    )


# ── JSON event names, server -> client ───────────────────────────────────

EVT_READY = "voice.ready"
EVT_STATE = "voice.state"
EVT_PARTIAL = "voice.partial"          # live caption, may still be revised
EVT_FINAL = "voice.final"              # the transcript her mind received
EVT_REPLY = "voice.reply"              # her reply text, for captions
EVT_SPEAKING_CHUNK = "voice.chunk"     # the clause now being spoken
EVT_BACKCHANNEL = "voice.backchannel"
EVT_FILLER = "voice.filler"
EVT_INTERRUPTED = "voice.interrupted"
EVT_FLUSH = "voice.flush"              # drop buffered audio immediately
EVT_METRICS = "voice.metrics"
EVT_STYLE = "voice.style"              # delivery changed at the user's request
EVT_VOICES = "voice.voices"            # available voice list
EVT_ERROR = "voice.error"

# ── JSON commands, client -> server ──────────────────────────────────────

CMD_START = "start"
CMD_STOP = "stop"
CMD_MUTE = "mute"
CMD_UNMUTE = "unmute"
CMD_BARGE_IN = "barge_in"          # client-side detection, carries played_ms
CMD_PLAYBACK = "playback"          # periodic playback position report
CMD_TEXT = "text"                  # typed message while in voice mode
CMD_SET_VOICE = "set_voice"        # switch her speaking voice mid-session
CMD_LIST_VOICES = "list_voices"
