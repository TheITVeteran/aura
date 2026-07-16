const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const staticRoot = path.resolve(__dirname, '..', '..');
const source = fs.readFileSync(path.join(staticRoot, 'aura.js'), 'utf8');

function deliveryHarness() {
  const context = {
    CHAT_DELIVERY_POLL_BASE_MS: 400,
    CHAT_DELIVERY_POLL_MAX_MS: 5000,
    CHAT_DELIVERY_TERMINAL_STATES: new Set([
      'awaiting_approval',
      'completed',
      'failed',
      'ambiguous',
    ]),
    Promise,
    fetchChatDeliveryStatus: async () => {
      throw new Error('unexpected default status transport');
    },
    postChatDelivery: async () => {
      throw new Error('unexpected default post transport');
    },
    state: { chatHandoffPending: false },
    waitForChatDelivery: async () => {},
  };
  const envelopeStart = source.indexOf('function chatDeliveryEnvelope(');
  const envelopeEnd = source.indexOf('async function readChatDeliveryResponse(', envelopeStart);
  const resolverStart = source.indexOf('async function resolveChatDelivery(', envelopeEnd);
  const resolverEnd = source.indexOf('async function runChatRequest(', resolverStart);
  assert.ok(envelopeStart >= 0 && envelopeEnd > envelopeStart);
  assert.ok(resolverStart >= 0 && resolverEnd > resolverStart);
  vm.runInNewContext(source.slice(envelopeStart, envelopeEnd), context, {
    filename: 'aura.js#chat-delivery-envelope',
  });
  vm.runInNewContext(source.slice(resolverStart, resolverEnd), context, {
    filename: 'aura.js#chat-delivery-resolver',
  });
  return context;
}

test('terminal status envelope unwraps durable result and embedded HTTP truth', () => {
  const harness = deliveryHarness();
  const decision = harness.chatDeliveryDecision('status', 200, {
    delivery_status: 'terminal',
    state: 'failed',
    http_status: 503,
    turn_id: 'a'.repeat(32),
    idempotency_key: 'aura-chat-test-key',
    result: {
      response: 'The operation failed closed.',
      delivery_state: 'failed',
    },
  });

  assert.equal(decision.action, 'terminal');
  assert.equal(decision.envelope.effectiveStatus, 503);
  assert.equal(decision.envelope.ok, false);
  assert.equal(decision.envelope.data.response, 'The operation failed closed.');
  assert.equal(decision.envelope.data.delivery_replayed, true);
});

test('decision table distinguishes pending, missing, and unavailable authority', () => {
  const harness = deliveryHarness();
  assert.equal(
    harness.chatDeliveryDecision('post', 202, {
      delivery_status: 'pending',
      state: 'running',
      turn_id: 'b'.repeat(32),
    }).action,
    'retry_status',
  );
  assert.equal(
    harness.chatDeliveryDecision('status', 404, {
      delivery_status: 'not_found',
    }).action,
    'retry_post',
  );
  assert.equal(
    harness.chatDeliveryDecision('post', 503, {
      status: 'chat_delivery_journal_unavailable',
    }).action,
    'retry_status',
  );
  assert.equal(
    harness.chatDeliveryDecision('status', 503, {
      delivery_status: 'unavailable',
    }).action,
    'retry_status',
  );
});

test('legacy terminal response remains compatible without weakening 202 handling', () => {
  const harness = deliveryHarness();
  const legacy = harness.chatDeliveryDecision('post', 200, {
    response: 'legacy server answer',
    status: 'ok',
  });
  assert.equal(legacy.action, 'terminal');
  assert.equal(legacy.envelope.ok, true);
  assert.equal(legacy.envelope.data.response, 'legacy server answer');
});

test('HTTP timeout recovers through status without posting the turn twice', async () => {
  const harness = deliveryHarness();
  const phases = [];
  const pending = [];
  let statusCalls = 0;
  const item = {
    idempotencyKey: 'aura-chat-timeout-test',
    resumePending: false,
    resumeDeadline: 0,
    deliveryState: 'submitting',
    turnId: '',
  };
  const result = await harness.resolveChatDelivery(item, 'hello', {
    post: async () => {
      phases.push('post');
      const error = new Error('timed out');
      error.name = 'AbortError';
      throw error;
    },
    status: async () => {
      phases.push('status');
      statusCalls += 1;
      if (statusCalls === 1) {
        return {
          httpStatus: 202,
          payload: {
            delivery_status: 'pending',
            state: 'running',
            turn_id: 'c'.repeat(32),
          },
        };
      }
      return {
        httpStatus: 200,
        payload: {
          delivery_status: 'terminal',
          state: 'completed',
          http_status: 200,
          turn_id: 'c'.repeat(32),
          result: { response: 'recovered', delivery_state: 'completed' },
        },
      };
    },
    wait: async delay => pending.push(delay),
    shouldDefer: () => false,
    onPending: event => pending.push(event.error ? event.error.name : event.source),
  });

  assert.deepEqual(phases, ['post', 'status', 'status']);
  assert.equal(result.data.response, 'recovered');
  assert.equal(result.ok, true);
  assert.equal(item.turnId, 'c'.repeat(32));
  assert.equal(item.resumePending, false);
  assert.ok(pending.includes('AbortError'));
});

test('restored active turn checks status before safe resubmission', async () => {
  const harness = deliveryHarness();
  const phases = [];
  const item = {
    idempotencyKey: 'aura-chat-restored-test',
    resumePending: true,
    resumeDeadline: 0,
    deliveryState: 'pending',
    turnId: '',
  };
  const result = await harness.resolveChatDelivery(item, 'hello', {
    resumeFirst: true,
    status: async () => {
      phases.push('status');
      return { httpStatus: 404, payload: { delivery_status: 'not_found' } };
    },
    post: async () => {
      phases.push('post');
      return {
        httpStatus: 200,
        payload: {
          response: 'executed once',
          delivery_state: 'completed',
          turn_id: 'd'.repeat(32),
        },
      };
    },
    wait: async () => {},
    shouldDefer: () => false,
  });

  assert.deepEqual(phases, ['status', 'post']);
  assert.equal(result.data.response, 'executed once');
  assert.equal(item.deliveryState, 'completed');
});

test('shell transition defers without touching either transport', async () => {
  const harness = deliveryHarness();
  let calls = 0;
  const result = await harness.resolveChatDelivery(
    { idempotencyKey: 'aura-chat-defer-test' },
    'hello',
    {
      post: async () => { calls += 1; },
      status: async () => { calls += 1; },
      shouldDefer: () => true,
    },
  );

  assert.equal(result.deferred, true);
  assert.equal(calls, 0);
});

test('paired surface permits only read access to delivery status', () => {
  assert.match(
    source,
    /path\.startsWith\('\/api\/chat\/delivery\/'\)\) return method === 'GET'/,
  );
  assert.match(source, /'X-Aura-Approval-Resume'/);
  assert.match(source, /state\.activeChatRequest\s*\|\|\s*state\.chatHandoffPending/);
  const enqueueStart = source.indexOf('function enqueueChatMessage(');
  const enqueueEnd = source.indexOf('function drainQueuedChatMessages(', enqueueStart);
  assert.doesNotMatch(source.slice(enqueueStart, enqueueEnd), /chatSendQueue\.shift\(\)/);
  assert.match(source, /const CHAT_SEND_QUEUE_MAX = 32/);
});
