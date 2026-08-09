/*
 * The bubble's behaviour. Deliberately small.
 *
 * It owns three things and nothing else: poll her state, render the two
 * states, and forward three intents (open, clear, moved). Every decision
 * about WHETHER she speaks lives in core/perception/ambient_presence.py and
 * ambient_utterance.py, behind the Will — a surface that could decide to
 * show a message would be a second, quieter authority for unprompted speech,
 * and there is exactly one.
 */
(() => {
  "use strict";

  const pill = document.getElementById("pill");
  const glyph = document.getElementById("glyph");
  const say = document.getElementById("say");
  const close = document.getElementById("close");

  const IDLE_POLL_MS = 4000;
  const ACTIVE_POLL_MS = 1500;
  // A failing backend must not become a spin. Back off, cap, recover.
  const MAX_BACKOFF_MS = 60000;

  let backoffMs = 0;
  let lastRendered = "";
  let timer = null;

  async function api(path, options) {
    const response = await fetch(path, {
      cache: "no-store",
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    if (!response.ok) throw new Error(`${path} -> ${response.status}`);
    return response.json();
  }

  function render(state) {
    const text = String((state && state.utterance) || "");
    const speaking = Boolean(state && state.has_utterance) && text.length > 0;

    // Only touch the DOM when something changed. The bubble sits over other
    // windows; a repaint every poll is a flicker in the corner of someone's
    // eye all day.
    const signature = speaking ? `1:${text}` : "0";
    if (signature === lastRendered) return speaking;
    lastRendered = signature;

    if (speaking) {
      say.textContent = text;
      pill.classList.remove("dormant");
      pill.classList.add("speaking");
      pill.title = text;
    } else {
      say.textContent = "";
      pill.classList.add("dormant");
      pill.classList.remove("speaking");
      pill.removeAttribute("title");
    }
    return speaking;
  }

  async function poll() {
    let speaking = false;
    try {
      const state = await api("/api/ambient/state");
      speaking = render(state);
      backoffMs = 0;
    } catch (error) {
      // Stay on screen showing whatever she last said rather than blanking:
      // a message vanishing because the poll failed looks like she withdrew
      // it. Slow down instead.
      backoffMs = Math.min(MAX_BACKOFF_MS, (backoffMs || IDLE_POLL_MS) * 2);
    }
    const next = backoffMs || (speaking ? ACTIVE_POLL_MS : IDLE_POLL_MS);
    timer = window.setTimeout(poll, next);
  }

  function openChat() {
    // The bubble does not host a conversation. Clicking asks the host to
    // bring up the restrained chat window; if there is no host bridge we are
    // in a browser tab and the full surface is one navigation away.
    if (window.webkit?.messageHandlers?.auraBubble) {
      window.webkit.messageHandlers.auraBubble.postMessage({ action: "open" });
      return;
    }
    window.location.href = "/";
  }

  glyph.addEventListener("click", (event) => {
    event.preventDefault();
    openChat();
  });
  say.addEventListener("click", openChat);

  close.addEventListener("click", async (event) => {
    event.preventDefault();
    event.stopPropagation();
    // Optimistic: clearing must feel instant, and the server is the
    // authority on what comes NEXT, not on what was already dismissed.
    render({ has_utterance: false, utterance: "" });
    try {
      await api("/api/ambient/clear", { method: "POST", body: "{}" });
    } catch (error) {
      /* the next poll reconciles */
    }
  });

  // Report where the person parked her, so the position survives a restart.
  let moveTimer = null;
  window.addEventListener("aura-bubble-moved", (event) => {
    const detail = event.detail || {};
    window.clearTimeout(moveTimer);
    moveTimer = window.setTimeout(() => {
      api("/api/ambient/position", {
        method: "POST",
        body: JSON.stringify({ x: detail.x || 0, y: detail.y || 0 }),
      }).catch(() => {});
    }, 400);
  });

  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && timer === null) poll();
  });

  poll();
})();
