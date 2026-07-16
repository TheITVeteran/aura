const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const staticRoot = path.resolve(__dirname, '..', '..');

function read(relativePath) {
  return fs.readFileSync(path.join(staticRoot, relativePath), 'utf8');
}

function compileLegacyScripts() {
  for (const script of ['aura.js', 'perf_collector.js', 'service-worker.js']) {
    new vm.Script(read(script), { filename: script });
  }
}

async function provePerformanceNegotiation() {
  const listeners = new Map();
  const calls = [];
  let frameCallback = null;
  const window = {
    addEventListener(name, callback) {
      listeners.set(name, callback);
    },
    dispatchProfile(profile) {
      const callback = listeners.get('aura:access-profile');
      if (callback) callback({ detail: profile });
    },
  };
  const context = {
    window,
    performance: { now: () => 0 },
    document: { body: { classList: { toggle() {} } } },
    requestAnimationFrame(callback) {
      frameCallback = callback;
    },
    async fetch(requestPath, options) {
      calls.push([requestPath, options]);
      return { status: 200, json: async () => ({ throttled: false }) };
    },
  };

  vm.runInNewContext(read('perf_collector.js'), context, {
    filename: 'perf_collector.js',
  });
  assert.strictEqual(typeof frameCallback, 'function');

  frameCallback(16);
  frameCallback(6001);
  await new Promise((resolve) => setImmediate(resolve));
  assert.strictEqual(calls.length, 0, 'collector posted before access negotiation');

  window.dispatchProfile({
    surface: 'paired_device',
    capabilities: { performance_telemetry: false },
  });
  frameCallback(6100);
  frameCallback(12000);
  window.auraRecordAck('paired', 12);
  await new Promise((resolve) => setImmediate(resolve));
  assert.strictEqual(calls.length, 0, 'paired surface emitted performance telemetry');

  window.dispatchProfile({
    surface: 'owner',
    capabilities: { performance_telemetry: true },
  });
  frameCallback(12100);
  frameCallback(18000);
  await new Promise((resolve) => setImmediate(resolve));
  assert.strictEqual(calls.length, 1);
  assert.strictEqual(calls[0][0], '/api/performance/frame');

  window.auraRecordAck('owner', 18);
  await new Promise((resolve) => setImmediate(resolve));
  assert.strictEqual(calls.length, 2);
  assert.strictEqual(calls[1][0], '/api/performance/ack');
}

function proveSourceContracts() {
  const legacy = read('aura.js');
  for (const marker of [
    'conversationOnly: true',
    'window.fetch = function auraSurfaceFetch',
    'surface_scope_denied:',
    'applyAccessProfile(payload.access)',
    "accessCapabilityAllowed('desktop_control')",
    "accessCapabilityAllowed('interaction_signals')",
    "accessCapabilityAllowed('voice_stream')",
    "accessCapabilityAllowed('tools_catalog')",
    "accessCapabilityAllowed('learning_status')",
  ]) {
    assert.ok(legacy.includes(marker), `missing legacy access marker: ${marker}`);
  }

  const reactShell = read('shell/src/App.jsx');
  assert.ok(reactShell.includes('conversation_only: true'));
  assert.ok(reactShell.includes('conversation-only'));

  const serviceWorker = read('service-worker.js');
  assert.ok(!serviceWorker.includes("fetch('/api/state')"));
  assert.strictEqual(
    serviceWorker.split("fetch('/api/health/heartbeat')").length - 1,
    2,
  );
  const installBlock = serviceWorker.slice(
    serviceWorker.indexOf("self.addEventListener('install'"),
    serviceWorker.indexOf("self.addEventListener('activate'"),
  );
  assert.ok(!installBlock.includes('self.skipWaiting()'));
  assert.ok(serviceWorker.includes("event.data.type === 'SKIP_WAITING'"));
  assert.ok(serviceWorker.includes("event.data.revision === SHELL_REVISION"));
  assert.ok(serviceWorker.includes("const CACHE_NAMESPACE = 'aura-runtime-shell-'"));
  assert.ok(serviceWorker.includes("url.searchParams.set('_aura_runtime', SHELL_REVISION)"));
  assert.ok(!serviceWorker.includes("'/icon-192.png'"));

  assert.ok(legacy.includes("'X-Idempotency-Key': item.idempotencyKey"));
  assert.ok(legacy.includes('function persistChatHandoff'));
  assert.ok(legacy.includes('function requestGuardedShellReload'));
  assert.ok(legacy.includes('swReloadTriggered = requestGuardedShellReload({'));
  assert.ok(legacy.includes('function refreshServiceWorkerRegistration'));
  assert.ok(legacy.includes('function registerRevisionServiceWorker'));
  assert.ok(legacy.includes("`/static/service-worker.js?_aura_runtime=${normalized}`"));
}

async function main() {
  compileLegacyScripts();
  proveSourceContracts();
  await provePerformanceNegotiation();
  process.stdout.write('paired frontend access contract passed\n');
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
