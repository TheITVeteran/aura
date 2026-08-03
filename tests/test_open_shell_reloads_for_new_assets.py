"""An open desktop window must not keep running a shell the runtime replaced.

Measured live 2026-08-03. Bryan's Aura Zenith window stayed open for hours,
survived four runtime restarts and three revision-token changes, and kept
executing the JS it had loaded at open time. A placeholder fix that was
committed, served, and verified in a fresh tab was invisible to him: he was
still looking at "Conversation lane initializing. Waiting for verified Aura
reply path..." while /api/health reported conversation_ready: true.

The shell HAS a reload path keyed on the runtime revision. It could not run,
because forming the evidence required ``health_read_model.fresh === true`` —
and the health read model serves stale-while-revalidate, a 5s refresh against
a 30s max-stale window. The live snapshot read fresh=false, stale=true,
expired=false, age=13.9s: valid by the server's own contract, and rejected by
the shell. Same class as the rest of that day's defects — a check demanding
more than the contract provides, so the action never happens.

These run the production functions under node, so the behaviour is pinned
rather than the source text.
"""
from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

AURA_JS = Path(__file__).resolve().parents[1] / "interface" / "static" / "aura.js"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js required")


def _extract(source: str, start_marker: str, end_marker: str) -> str:
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    return source[start:end].strip()


def _run(script: str) -> str:
    result = subprocess.run(
        [NODE, "-e", script], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, f"node failed:\n{result.stdout}\n{result.stderr}"
    return result.stdout


@pytest.fixture(scope="module")
def source() -> str:
    return AURA_JS.read_text(encoding="utf-8")


class TestStaleButValidSnapshotsStillCarryRevisionEvidence:
    def test_a_stale_while_revalidate_snapshot_is_accepted(self, source):
        production = "\n\n".join(
            (
                _extract(
                    source,
                    "function verifiedRuntimeRevision(",
                    "function runtimeRevisionPolicySatisfied(",
                ),
                _extract(
                    source,
                    "function healthSnapshotRevisionEvidence(",
                    "// ── Served-shell binding",
                ),
            )
        )
        assert "metadata.fresh !== true" not in production, (
            "requiring freshness is what stopped the reload path from ever running"
        )

        script = textwrap.dedent(
            f"""
            'use strict';
            const assert = require('node:assert/strict');

            {production}

            const token = 'a'.repeat(64);
            // The exact live shape: valid, unexpired, and not fresh.
            const stale = {{
                runtime_revision: {{
                    schema: 'aura.runtime_revision.v2',
                    required: true,
                    verified: true,
                    revision_token: token,
                }},
                health_read_model: {{
                    snapshot_generation: 37,
                    captured_at_unix: 1785786309.167798,
                    age_s: 13.88,
                    fresh: false,
                    stale: true,
                    expired: false,
                    serving: 'stale_while_revalidate',
                }},
            }};
            const evidence = healthSnapshotRevisionEvidence(stale);
            assert.ok(evidence, 'a valid unexpired snapshot must carry revision evidence');
            assert.equal(evidence.revision, token);

            // An EXPIRED snapshot is the server's real "do not trust this".
            const expired = JSON.parse(JSON.stringify(stale));
            expired.health_read_model.expired = true;
            assert.equal(
                healthSnapshotRevisionEvidence(expired), null,
                'an expired snapshot must still be refused'
            );

            // So is one with no verified token.
            const unverified = JSON.parse(JSON.stringify(stale));
            unverified.runtime_revision.verified = false;
            assert.equal(healthSnapshotRevisionEvidence(unverified), null);

            console.log('ok');
            """
        )
        assert "ok" in _run(script)


class TestTheWindowIsBoundToTheAssetsItIsRunning:
    """A second, launch-mode-independent binding.

    The revision token only exists when the runtime is a verified signed app
    (`required: true`). A direct/source launch reports `required: false` and no
    token, so the revision path can never reload it — but its shell goes stale
    exactly the same way. The runtime hashes what it is actually serving
    regardless of launch mode.
    """

    def _production(self, source: str) -> str:
        return _extract(
            source,
            "const SERVED_SHELL_ASSETS_KEY",
            "function healthSnapshotRevisionIsAuthoritative(",
        )

    def test_it_binds_then_reloads_only_when_the_bytes_change(self, source):
        script = textwrap.dedent(
            f"""
            'use strict';
            const assert = require('node:assert/strict');

            let reloads = 0;
            const state = {{ runtimeRevisionReloading: false }};
            const store = new Map();
            globalThis.sessionStorage = {{
                getItem: (k) => (store.has(k) ? store.get(k) : null),
                setItem: (k, v) => store.set(k, String(v)),
                removeItem: (k) => store.delete(k),
            }};
            let href = 'http://127.0.0.1:8000/';
            globalThis.window = {{ get location() {{ return {{ href }}; }} }};
            function requestGuardedShellReload({{ replaceUrl = '' }} = {{}}) {{
                reloads += 1;
                href = replaceUrl || href;
                return true;
            }}

            {self._production(source)}

            const payload = (hash) => ({{
                runtime_revision: {{
                    schema: 'aura.runtime_revision.v2',
                    actual_shell_assets_sha256: hash,
                }},
            }});
            const first = 'd'.repeat(64);
            const second = 'e'.repeat(64);

            // First sighting binds; it must never reload on arrival.
            assert.equal(reconcileServedShellAssets(payload(first)), false);
            assert.equal(reloads, 0, 'binding is not a reason to reload');

            // Same bytes, repeatedly: still no reload.
            for (let i = 0; i < 5; i += 1) {{
                assert.equal(reconcileServedShellAssets(payload(first)), false);
            }}
            assert.equal(reloads, 0);

            // The runtime now serves different bytes: reload exactly once.
            assert.equal(reconcileServedShellAssets(payload(second)), true);
            assert.equal(reloads, 1);
            assert.ok(href.includes('_aura_shell=' + second.slice(0, 16)));

            // After the navigation the page must settle, not loop.
            for (let i = 0; i < 5; i += 1) {{
                assert.equal(reconcileServedShellAssets(payload(second)), false);
            }}
            assert.equal(reloads, 1, 'a reload that reloads again is worse than a stale shell');

            console.log('ok');
            """
        )
        assert "ok" in _run(script)

    def test_the_url_marker_alone_stops_a_loop_when_storage_is_denied(self, source):
        """Storage denial must not turn one reload into an endless cycle."""

        script = textwrap.dedent(
            f"""
            'use strict';
            const assert = require('node:assert/strict');

            let reloads = 0;
            const state = {{ runtimeRevisionReloading: false }};
            globalThis.sessionStorage = {{
                getItem() {{ throw new Error('denied'); }},
                setItem() {{ throw new Error('denied'); }},
                removeItem() {{ throw new Error('denied'); }},
            }};
            const hash = 'f'.repeat(64);
            // The page already navigated once: the marker is in the URL.
            let href = 'http://127.0.0.1:8000/?_aura_shell=' + hash.slice(0, 16);
            globalThis.window = {{ get location() {{ return {{ href }}; }} }};
            function requestGuardedShellReload() {{ reloads += 1; return true; }}

            {self._production(source)}

            const payload = {{
                runtime_revision: {{
                    schema: 'aura.runtime_revision.v2',
                    actual_shell_assets_sha256: hash,
                }},
            }};
            for (let i = 0; i < 10; i += 1) {{
                assert.equal(reconcileServedShellAssets(payload), false);
            }}
            assert.equal(reloads, 0, 'the URL marker is the storage-independent guard');
            console.log('ok');
            """
        )
        assert "ok" in _run(script)

    def test_a_malformed_fingerprint_is_ignored(self, source):
        script = textwrap.dedent(
            f"""
            'use strict';
            const assert = require('node:assert/strict');
            let reloads = 0;
            const state = {{ runtimeRevisionReloading: false }};
            const store = new Map();
            globalThis.sessionStorage = {{
                getItem: (k) => (store.has(k) ? store.get(k) : null),
                setItem: (k, v) => store.set(k, String(v)),
                removeItem: (k) => store.delete(k),
            }};
            let href = 'http://127.0.0.1:8000/';
            globalThis.window = {{ get location() {{ return {{ href }}; }} }};
            function requestGuardedShellReload() {{ reloads += 1; return true; }}

            {self._production(source)}

            for (const bad of [null, '', 'not-a-hash', 'a'.repeat(63), 'A'.repeat(64) + 'z']) {{
                assert.equal(reconcileServedShellAssets({{
                    runtime_revision: {{
                        schema: 'aura.runtime_revision.v2',
                        actual_shell_assets_sha256: bad,
                    }},
                }}), false);
            }}
            // Wrong schema is also not evidence.
            assert.equal(reconcileServedShellAssets({{
                runtime_revision: {{ schema: 'aura.runtime_revision.v1',
                                     actual_shell_assets_sha256: 'a'.repeat(64) }},
            }}), false);
            assert.equal(reloads, 0);
            console.log('ok');
            """
        )
        assert "ok" in _run(script)


class TestItIsWiredIntoThePoll:
    def test_the_health_poll_consults_both_bindings(self, source):
        poll = _extract(source, "async function pollHealth(", "const recovered =")
        assert "reconcileRuntimeShellRevision(d)" in poll
        assert "reconcileServedShellAssets(d)" in poll, (
            "the served-assets binding must run on the same poll as the revision one"
        )
