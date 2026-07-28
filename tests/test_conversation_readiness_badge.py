"""A thinner heartbeat must not read as "the conversation is not ready".

The websocket heartbeat is a lighter payload than /api/health and often omits
the conversation lane entirely. Treating that silence as not-ready appended the
blocker ``conversation_ready`` and flipped the header badge from ONLINE to the
literal string CONVERSATION_READY — measured live against an /api/health that
simultaneously reported ``conversation_ready: true``, ``lane.state: "ready"``
and ``blockers: []``.

Same defect class as the runtime-revision badge: absence of a reading is not a
failed reading. These tests run the production functions under node so the
behaviour is pinned, not just the source text.
"""

from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

AURA_JS = Path(__file__).resolve().parents[1] / "interface" / "static" / "aura.js"


def _extract(source: str, start_marker: str, end_marker: str) -> str:
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    return source[start:end].strip()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js required")
def test_thin_heartbeat_keeps_the_last_known_conversation_readiness():
    source = AURA_JS.read_text(encoding="utf-8")
    production = _extract(
        source,
        "function blockerIsConversationReadiness(",
        "function conversationPayloadBusy(",
    )
    assert "payloadCarriesConversationReadiness" in production

    script = textwrap.dedent(
        f"""
        'use strict';
        const assert = require('node:assert/strict');
        const state = {{ conversationReady: false }};

        {production}

        const readyPoll = {{
            conversation_ready: true,
            conversation_lane: {{ state: 'ready', conversation_ready: true, readiness_blockers: [] }},
        }};
        const thinHeartbeat = {{ status: 'ok', healthy: true }};

        // A payload that speaks to readiness is judged on its own contents.
        assert.equal(payloadCarriesConversationReadiness(readyPoll), true);
        assert.equal(conversationPayloadReady(readyPoll, []), true);

        // A payload that does NOT speak to readiness must not claim not-ready.
        assert.equal(payloadCarriesConversationReadiness(thinHeartbeat), false);
        state.conversationReady = true;
        assert.equal(
            conversationPayloadReady(thinHeartbeat, []),
            true,
            'a thin heartbeat must inherit the last known readiness, not contradict it'
        );

        // And it must not invent readiness the client never observed.
        state.conversationReady = false;
        assert.equal(conversationPayloadReady(thinHeartbeat, []), false);

        // A payload that DOES carry the lane and says not-ready is still not ready.
        const warmingPoll = {{
            conversation_ready: false,
            conversation_lane: {{ state: 'warming', conversation_ready: false, readiness_blockers: [] }},
        }};
        state.conversationReady = true;
        assert.equal(
            conversationPayloadReady(warmingPoll, []),
            false,
            'an explicit not-ready reading must win over the remembered one'
        );

        // A real readiness blocker still suppresses readiness.
        state.conversationReady = true;
        assert.equal(conversationPayloadReady(readyPoll, ['conversation_ready']), false);
        assert.equal(conversationPayloadReady(readyPoll, ['conversation_lane:failed']), false);

        console.log('OK');
        """
    )
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "OK" in result.stdout
