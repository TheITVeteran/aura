/**
 * perf_collector.js — frontend → backend performance telemetry.
 *
 * Hooks requestAnimationFrame to sample frame durations, batches them in
 * a 5-second window, and POSTs to /api/performance/frame. Reads the
 * throttle response and toggles the ``aura-throttle-motion`` class on
 * <body> so motion_design.css can degrade gracefully under pressure.
 *
 * The collector is silent: any error is swallowed (we never want
 * telemetry to take the page down). It does not run in environments
 * that respect prefers-reduced-motion.
 */
(() => {
  if (window.__auraPerfInstalled) return;
  window.__auraPerfInstalled = true;

  let lastTs = performance.now();
  const samples = [];
  let lastFlush = lastTs;
  let enabled = false;

  window.addEventListener("aura:access-profile", (event) => {
    const capabilities = event && event.detail && event.detail.capabilities;
    enabled = !!(capabilities && capabilities.performance_telemetry === true);
    if (!enabled) samples.length = 0;
  });

  function frame(ts) {
    const dur = ts - lastTs;
    lastTs = ts;
    if (enabled && dur > 0 && dur < 1000) samples.push(dur);
    if (ts - lastFlush > 5000 && samples.length) {
      flush();
      lastFlush = ts;
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  async function flush() {
    if (!enabled) {
      samples.length = 0;
      return;
    }
    const batch = samples.slice();
    samples.length = 0;
    const max = Math.max(...batch);
    try {
      const r = await fetch("/api/performance/frame", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ duration_ms: max, source: "ui_raf" }),
      });
      if (r.status === 401 || r.status === 403) {
        enabled = false;
        return;
      }
      const d = await r.json();
      if (d && typeof d.throttled === "boolean") {
        document.body.classList.toggle("aura-throttle-motion", d.throttled);
      }
    } catch {}
  }

  // Public hook for ack samples.
  window.auraRecordAck = (requestId, latencyMs) => {
    if (!enabled) return;
    try {
      fetch("/api/performance/ack", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ request_id: String(requestId || ""), latency_ms: Number(latencyMs) || 0 }),
      }).then((response) => {
        if (response.status === 401 || response.status === 403) enabled = false;
      }).catch(() => {});
    } catch {}
  };
})();
