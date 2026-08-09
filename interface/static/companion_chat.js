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

  // One session for the life of the window, so a follow-up ("what about the
  // other one?") has its antecedent. A fresh id per message would make every
  // turn the first turn.
  const sessionId = `companion-${Date.now().toString(36)}`;
  let inFlight = false;

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

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, session_id: sessionId }),
      });
      if (!response.ok) throw new Error(`chat ${response.status}`);
      const data = await response.json();
      const reply = String(
        data.response ?? data.reply ?? data.message ?? ""
      ).trim();
      // An empty reply is reported as one. Rendering nothing would leave the
      // window looking like the message was never sent, and the person would
      // send it again.
      bubble(reply || "(she returned nothing for that turn)", reply ? "her" : "err");
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
    input.focus();
  })();
})();
