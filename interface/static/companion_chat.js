/*
 * The restrained chat window.
 *
 * It talks to the SAME /api/chat as the full desktop. Not a lighter model,
 * not a shorter context, not a companion-mode personality — a second, quieter
 * Aura reachable from the bubble would be a different assistant wearing her
 * icon, and the person clicking the bubble is asking the same one a question.
 * What is restrained here is the SURFACE, not her.
 */
(() => {
  "use strict";

  const log = document.getElementById("log");
  const input = document.getElementById("input");
  const send = document.getElementById("send");
  const thinking = document.getElementById("thinking");
  const expand = document.getElementById("expand");

  const REQUEST_TIMEOUT_MS = 240000;
  const DELIVERY_TIMEOUT_MS = 360000;
  const DELIVERY_POLL_MS = 750;
  const PENDING_KEY = "aura-companion-pending-v1";
  let inFlight = false;

  function idempotencyKey() {
    const id = window.crypto?.randomUUID?.()
      || `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 14)}`;
    return `aura-companion-${id}`;
  }

  function chatHeaders(key = "") {
    const headers = {
      "Content-Type": "application/json",
      "X-Aura-Surface": "desktop-ui",
      "X-Aura-Desktop-Request": "same-origin",
      "X-Aura-Require-CognitiveEngine": "true",
    };
    if (key) headers["X-Idempotency-Key"] = key;
    return headers;
  }

  function replyEnvelope(payload) {
    if (payload?.result && typeof payload.result === "object") return payload.result;
    return payload && typeof payload === "object" ? payload : {};
  }

  function replyText(payload) {
    const data = replyEnvelope(payload);
    return String(data.response ?? data.reply ?? data.message ?? "").trim();
  }

  function storePending(item) {
    try { localStorage.setItem(PENDING_KEY, JSON.stringify(item)); } catch (_error) {}
  }

  function clearPending(key) {
    try {
      const saved = JSON.parse(localStorage.getItem(PENDING_KEY) || "null");
      if (!saved || saved.key === key) localStorage.removeItem(PENDING_KEY);
    } catch (_error) {
      try { localStorage.removeItem(PENDING_KEY); } catch (_ignored) {}
    }
  }

  async function jsonFetch(path, options, timeoutMs) {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(path, {
        cache: "no-store",
        credentials: "same-origin",
        ...options,
        signal: controller.signal,
      });
      const payload = await response.json();
      return { response, payload };
    } finally {
      window.clearTimeout(timeout);
    }
  }

  async function deliveryStatus(key) {
    return jsonFetch(
      `/api/chat/delivery/${encodeURIComponent(key)}`,
      { headers: chatHeaders() },
      10000,
    );
  }

  async function awaitDelivery(key) {
    const deadline = Date.now() + DELIVERY_TIMEOUT_MS;
    while (Date.now() < deadline) {
      const { response, payload } = await deliveryStatus(key);
      if (response.ok && (payload.terminal || payload.delivery_status === "terminal")) {
        return payload;
      }
      if (response.status !== 202 && response.status !== 404 && response.status !== 503) {
        throw new Error(`delivery ${response.status}`);
      }
      const retry = Math.max(100, Number(payload.retry_after_ms) || DELIVERY_POLL_MS);
      await new Promise((resolve) => window.setTimeout(resolve, retry));
    }
    throw new Error("reply is still running; it will be recovered when this window opens again");
  }

  async function sendDurably(item) {
    try {
      const { response, payload } = await jsonFetch(
        "/api/chat",
        {
          method: "POST",
          headers: chatHeaders(item.key),
          body: JSON.stringify({ message: item.message }),
        },
        REQUEST_TIMEOUT_MS,
      );
      if (response.ok && response.status !== 202) return payload;
      if (response.status !== 202) {
        const detail = replyText(payload) || payload.status || `chat ${response.status}`;
        throw new Error(detail);
      }
    } catch (error) {
      // Aborting the HTTP wait does not mean the fenced backend turn stopped.
      // Resolve the durable receipt before allowing another send.
      if (error?.name !== "AbortError") {
        const probe = await deliveryStatus(item.key).catch(() => null);
        if (!probe || probe.response.status === 404) throw error;
        if (probe.response.ok && probe.payload.terminal) return probe.payload;
      }
    }
    return awaitDelivery(item.key);
  }

  function bubble(text, kind) {
    const node = document.createElement("div");
    node.className = `msg ${kind}`;
    node.textContent = text;
    log.appendChild(node);
    log.scrollTop = log.scrollHeight;
    return node;
  }

  function busy(state) {
    inFlight = state;
    send.disabled = state;
    thinking.classList.toggle("on", state);
    if (state) log.scrollTop = log.scrollHeight;
  }

  function autoGrow() {
    input.style.height = "auto";
    input.style.height = `${Math.min(120, input.scrollHeight)}px`;
  }

  async function submit() {
    const text = input.value.trim();
    if (!text || inFlight) return;
    bubble(text, "me");
    input.value = "";
    autoGrow();
    busy(true);

    const item = { message: text, key: idempotencyKey(), queuedAt: Date.now() };
    storePending(item);
    try {
      const data = await sendDurably(item);
      const reply = replyText(data);
      // An empty reply is reported as one. Rendering nothing would leave the
      // window looking like the message was never sent, and the person would
      // send it again.
      bubble(reply || "Aura completed the turn without a deliverable reply.", reply ? "her" : "err");
      clearPending(item.key);
    } catch (error) {
      // The failure is shown in the transcript rather than swallowed: a
      // message that vanishes is indistinguishable from one she ignored.
      bubble(`Could not reach her: ${error.message}`, "err");
    } finally {
      busy(false);
      input.focus();
    }
  }

  send.addEventListener("click", submit);

  input.addEventListener("keydown", (event) => {
    // Enter sends, Shift+Enter is a newline. This window is for one thing
    // said quickly; a send button you must aim at defeats that.
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
    if (event.key === "Escape") {
      window.webkit?.messageHandlers?.auraCompanion?.postMessage({ action: "close" });
    }
  });
  input.addEventListener("input", autoGrow);

  expand.addEventListener("click", () => {
    if (window.webkit?.messageHandlers?.auraCompanion) {
      window.webkit.messageHandlers.auraCompanion.postMessage({ action: "expand" });
      return;
    }
    window.location.href = "/";
  });

  // If she had something queued in the bubble, it is the reason this window
  // was opened. Show it as her first line and clear it, so the thing that
  // prompted the click is not lost behind an empty transcript.
  (async () => {
    try {
      const state = await fetch("/api/ambient/state", { cache: "no-store" }).then((r) =>
        r.json()
      );
      if (state && state.has_utterance && state.utterance) {
        bubble(String(state.utterance), "her");
        await fetch("/api/ambient/clear", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: "{}",
        }).catch(() => {});
      }
    } catch (error) {
      /* an empty transcript is a fine starting state */
    }
    try {
      const pending = JSON.parse(localStorage.getItem(PENDING_KEY) || "null");
      if (pending?.message && pending?.key) {
        busy(true);
        const recovered = await sendDurably(pending);
        const reply = replyText(recovered);
        bubble(reply || "Aura completed the recovered turn without a deliverable reply.", reply ? "her" : "err");
        clearPending(pending.key);
      }
    } catch (error) {
      bubble(`The earlier turn is not settled yet: ${error.message}`, "err");
    } finally {
      busy(false);
      input.focus();
    }
  })();
})();
