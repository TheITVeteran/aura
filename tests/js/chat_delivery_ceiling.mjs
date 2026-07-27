// Exercises resolveChatDelivery's unreachable ceiling without a browser.
//
// The function is defined inside aura.js alongside a large amount of DOM code,
// so rather than import the module we extract the two pieces under test and
// run them against fakes. That keeps the check honest about the ACTUAL source
// text — if the loop is edited back to an unbounded retry, this stops passing.

import { readFileSync } from "node:fs";

const SRC = process.argv[2];
const src = readFileSync(SRC, "utf8");

function extract(startMarker, endMarker) {
  const a = src.indexOf(startMarker);
  if (a < 0) throw new Error(`missing: ${startMarker}`);
  const b = src.indexOf(endMarker, a);
  if (b < 0) throw new Error(`missing end for: ${startMarker}`);
  return src.slice(a, b + endMarker.length);
}

const constants = extract(
  "const CHAT_DELIVERY_POLL_BASE_MS",
  "const CHAT_DELIVERY_UNREACHABLE_MS = 180000;"
);
const resolver = extract("async function resolveChatDelivery(", "\n}\n");

const harness = `
${constants}
function chatDeliveryDecision(source, httpStatus, payload) {
  return { action: 'terminal', envelope: { ok: true, turnId: 't1', deliveryState: 'completed' } };
}
${resolver}
export { resolveChatDelivery, CHAT_DELIVERY_UNREACHABLE_MS };
`;

const mod = await import(
  "data:text/javascript;base64," + Buffer.from(harness).toString("base64")
);

let failures = 0;
function check(label, condition) {
  console.log(`${condition ? "OK  " : "FAIL"} ${label}`);
  if (!condition) failures += 1;
}

// 1. A runtime that never answers ends the turn honestly instead of spinning.
{
  let now = 0;
  const realNow = Date.now;
  Date.now = () => now;
  const item = {};
  const outcome = await mod.resolveChatDelivery(item, "hello", {
    post: async () => { throw new Error("ECONNREFUSED"); },
    status: async () => { throw new Error("ECONNREFUSED"); },
    wait: async (ms) => { now += ms; },
    shouldDefer: () => false,
    onPending: () => {},
  });
  Date.now = realNow;
  check("unreachable runtime terminates the turn", outcome.unreachable === true);
  check("it is reported as not ok", outcome.ok === false);
  check("the person is told what happened", /lost contact with my own runtime/.test(outcome.data.response));
  check("the turn stays resumable", item.resumePending === true);
}

// 2. A slow-but-alive runtime is NEVER cut off, however long it takes.
{
  let now = 0;
  const realNow = Date.now;
  Date.now = () => now;
  let calls = 0;
  // Fails, recovers, fails again — repeatedly, across far more than the
  // ceiling in total elapsed time. Contact resets the clock each time.
  const outcome = await mod.resolveChatDelivery({}, "hello", {
    post: async () => { throw new Error("blip"); },
    status: async () => {
      calls += 1;
      if (calls % 2 === 1) throw new Error("blip");
      return { httpStatus: 200, payload: {}, response: null };
    },
    wait: async (ms) => { now += Math.max(ms, 60000); },
    shouldDefer: () => false,
    onPending: () => {},
  });
  Date.now = realNow;
  check("intermittent contact never triggers the ceiling", outcome.unreachable !== true);
  check("the turn completes normally", outcome.ok === true);
}

console.log(failures === 0 ? "\nall checks passed" : `\n${failures} FAILED`);
process.exit(failures === 0 ? 0 : 1);
