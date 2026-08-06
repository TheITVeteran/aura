"""tests/test_frame_governor_behaviour.py — the governor must actually govern.

The sibling tests in test_image_is_any_image.py assert that
`auraFrameGovernor`, `BAD_SAMPLES_TO_SHED` and `body.perf-lean` appear in the
shipped assets. That is worth checking and it is not evidence that anything
works: every one of those assertions passes against a file containing the right
words in the wrong order.

So this file EXECUTES the shipped governor — the real IIFE, extracted from
interface/static/aura.js, not a copy — against synthetic frame sequences, and
asserts what it does. Slow frames must eventually shed; one slow frame must
not; a hidden tab must not be rescued; restoring must take more evidence than
shedding, or the background flickers between blurred and flat, which reads as a
fault and looks worse than the lag it is answering.

Skipped when node is unavailable, which is honest: the assertion is about
runtime behaviour and nothing here can fake it from Python.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AURA_JS = PROJECT_ROOT / "interface" / "static" / "aura.js"
GOVERNOR_MARKER = "/* ── Frame governor"

NODE = shutil.which("node")


def _governor_source() -> str:
    """The shipped governor, byte-for-byte, from the file the browser loads."""
    source = AURA_JS.read_text(encoding="utf-8")
    start = source.index(GOVERNOR_MARKER)
    return source[start:]


#: A DOM small enough to be obviously fake and complete enough that the
#: governor cannot tell. Everything it touches, and nothing it does not.
HARNESS = """
const classes = new Set();
const listeners = {};
globalThis.document = {
  hidden: false,
  body: {
    classList: {
      toggle(name, on) { on ? classes.add(name) : classes.delete(name); },
      contains(name) { return classes.has(name); },
    },
    dataset: {},
  },
  addEventListener(name, fn) { (listeners[name] ||= []).push(fn); },
};
let queued = null;
globalThis.window = {
  requestAnimationFrame(fn) { queued = fn; },
  matchMedia(query) { return { matches: globalThis.__reducedMotion === true }; },
};

__GOVERNOR__

// Drive the loop by hand: each entry advances the clock by that many ms.
function run(deltas) {
  let now = 0;
  for (const delta of deltas) {
    now += delta;
    const fn = queued;
    queued = null;
    if (fn) fn(now);
  }
}
globalThis.__run = run;
globalThis.__lean = () => classes.has('perf-lean');
"""


def _run_js(script: str) -> dict:
    program = HARNESS.replace("__GOVERNOR__", _governor_source()) + "\n" + script
    result = subprocess.run(
        [NODE, "--input-type=module", "-e", program],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0, result.stderr[-3000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


needs_node = pytest.mark.skipif(NODE is None, reason="node unavailable")


@needs_node
def test_the_governor_sheds_after_sustained_slow_frames():
    """The thing it exists to do, driven through the real code."""
    out = _run_js(
        """
        const th = auraFrameGovernor.thresholds;
        // Two priming frames, then a sustained run of 50ms frames.
        __run([0, 16].concat(Array(th.BAD_SAMPLES_TO_SHED + 2).fill(50)));
        console.log(JSON.stringify({lean: __lean(), th}));
        """
    )
    assert out["lean"] is True, "sustained slow frames did not put the surface in lean mode"


@needs_node
def test_one_slow_frame_does_not_shed():
    """Hysteresis. Shedding on a single bad frame makes the background flicker."""
    out = _run_js(
        """
        __run([0, 16, 50, 16, 16, 16]);
        console.log(JSON.stringify({lean: __lean()}));
        """
    )
    assert out["lean"] is False, "a single slow frame shed the ambient layers"


@needs_node
def test_restoring_takes_more_evidence_than_shedding():
    """Coming back eagerly produces the flicker from the other side."""
    out = _run_js(
        """
        const th = auraFrameGovernor.thresholds;
        console.log(JSON.stringify({
          shed: th.BAD_SAMPLES_TO_SHED,
          restore: th.GOOD_SAMPLES_TO_RESTORE,
        }));
        """
    )
    assert out["restore"] > out["shed"], (
        f"restore threshold {out['restore']} is not stricter than shed threshold "
        f"{out['shed']}; the surface will oscillate at the boundary"
    )


@needs_node
def test_it_recovers_when_the_frames_come_back():
    """Lean mode is a response to conditions, not a one-way door."""
    out = _run_js(
        """
        const th = auraFrameGovernor.thresholds;
        __run([0, 16].concat(Array(th.BAD_SAMPLES_TO_SHED + 2).fill(50)));
        const shed = __lean();
        __run(Array(th.GOOD_SAMPLES_TO_RESTORE + 2).fill(16));
        console.log(JSON.stringify({shed, restored: !__lean()}));
        """
    )
    assert out["shed"] is True
    assert out["restored"] is True, "the surface never came back after conditions improved"


@needs_node
def test_a_hidden_tab_is_not_rescued():
    """rAF is throttled or stopped when hidden, so every sample looks terrible.

    Treating that as lag would leave every backgrounded surface permanently
    lean for a reason that has nothing to do with the machine.
    """
    out = _run_js(
        """
        const th = auraFrameGovernor.thresholds;
        document.hidden = true;
        __run([0, 16].concat(Array(th.BAD_SAMPLES_TO_SHED * 3).fill(200)));
        console.log(JSON.stringify({lean: __lean()}));
        """
    )
    assert out["lean"] is False, "a hidden tab's throttled frames were read as lag"


@needs_node
def test_an_implausible_gap_is_not_lag():
    """A tab switch, a sleep or a breakpoint is not a slow frame."""
    out = _run_js(
        """
        const th = auraFrameGovernor.thresholds;
        __run([0, 16].concat(Array(th.BAD_SAMPLES_TO_SHED * 2).fill(5000)));
        console.log(JSON.stringify({lean: __lean()}));
        """
    )
    assert out["lean"] is False, "a multi-second gap was counted as dropped frames"


@needs_node
def test_reduced_motion_never_starts_the_sampler():
    """Those layers are already stopped by CSS. Sampling would measure nothing."""
    out = _run_js(
        """
        globalThis.__reducedMotion = true;
        auraFrameGovernor.stop();
        auraFrameGovernor.start();
        const th = auraFrameGovernor.thresholds;
        __run([0, 16].concat(Array(th.BAD_SAMPLES_TO_SHED * 2).fill(80)));
        console.log(JSON.stringify({lean: __lean()}));
        """
    )
    assert out["lean"] is False
