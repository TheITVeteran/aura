// A test world's own code, run outside a browser on a clock this program drives, for checking anything said of the
// world against what its code does.
//
// The world's two scripts (tests/fixtures/worlds_that_move/world.js and the game's own, inside its page) are run as
// written, in one global scope as a page runs them. What a browser would supply is stood in for: a canvas whose drawing
// goes nowhere, the page's address, its listeners, and a clock that moves only when told to. Nothing in either script
// is changed or read in place of being run.
//
// It answers one JSON line per JSON line on its input:
//   {"op": "load", "world": "paint", "query": "seed=3&start=1"}
//   {"op": "key", "type": "keydown", "key": "ArrowLeft"}
//   {"op": "mouse", "type": "mousedown", "x": 120, "y": 80}
//   {"op": "step", "dt": 0.0166667, "n": 60}
//   {"op": "state"}              the world's tally and the game's own state, as plain data
//   {"op": "eval", "js": "..."}  an expression on the world, for a check that needs one value
"use strict";

const fs = require("fs");
const path = require("path");
const readline = require("readline");
const vm = require("vm");

const WORLDS = path.resolve(__dirname, "..", "..", "tests", "fixtures", "worlds_that_move");
const W = 480, H = 320;

function aDrawingThatGoesNowhere() {
  const nothing = () => undefined;
  const gradient = { addColorStop: nothing };
  return new Proxy({}, {
    get(target, name) {
      if (name in target) return target[name];
      if (name === "createLinearGradient" || name === "createRadialGradient") return () => gradient;
      if (name === "measureText") return () => ({ width: 0 });
      return nothing;
    },
    set(target, name, value) { target[name] = value; return true; },
  });
}

function load(name, query) {
  const page = fs.readFileSync(path.join(WORLDS, name + ".html"), "utf8");
  const scripts = [...page.matchAll(/<script>([\s\S]*?)<\/script>/g)].map((m) => m[1]);
  const shared = fs.readFileSync(path.join(WORLDS, "world.js"), "utf8");
  const clock = { now: 1000 };
  const pending = [];
  const listeners = {};
  const canvasListeners = {};
  const canvas = {
    width: W, height: H, style: {},
    getContext: () => aDrawingThatGoesNowhere(),
    getBoundingClientRect: () => ({ left: 0, top: 0, width: W, height: H }),
    addEventListener: (type, fn) => { (canvasListeners[type] = canvasListeners[type] || []).push(fn); },
  };
  const context = {
    console, Math, JSON, Date, Number, String, Array, Object, Set, Map, Uint8Array, Float32Array, Promise, Error,
    URLSearchParams, Proxy, Reflect, Symbol, isNaN, isFinite, parseInt, parseFloat,
    location: { search: "?" + (query || "") },
    performance: { now: () => clock.now },
    requestAnimationFrame: (fn) => { pending.push(fn); return pending.length; },
    addEventListener: (type, fn) => { (listeners[type] = listeners[type] || []).push(fn); },
    document: {
      createElement: () => canvas,
      body: { style: {}, appendChild: () => undefined },
    },
  };
  context.window = context;
  vm.createContext(context);
  vm.runInContext(shared, context, { filename: "world.js" });
  for (const [i, script] of scripts.entries()) vm.runInContext(script, context, { filename: `${name}.html#${i}` });
  return { name, context, clock, pending, listeners, canvasListeners };
}

function fire(list, event) {
  for (const fn of list || []) fn(event);
}

function anEvent(fields) {
  return Object.assign({ repeat: false, preventDefault() {}, stopPropagation() {} }, fields);
}

function step(here, dt, n) {
  for (let i = 0; i < n; i++) {
    here.clock.now += dt * 1000;
    const due = here.pending.splice(0, here.pending.length);
    for (const fn of due) fn(here.clock.now);
  }
}

// The game's own state as plain data: numbers, words, lists and records, not its functions or its drawing.
function plain(value, depth) {
  if (value === null || value === undefined) return null;
  if (typeof value === "number") return Number.isFinite(value) ? value : String(value);
  if (typeof value === "string" || typeof value === "boolean") return value;
  if (typeof value === "function") return undefined;
  if (depth > 6) return undefined;
  if (ArrayBuffer.isView(value)) return Array.from(value);
  if (value instanceof Set) return [...value].map((v) => plain(v, depth + 1));
  if (Array.isArray(value)) return value.map((v) => plain(v, depth + 1));
  if (typeof value === "object") {
    const out = {};
    for (const key of Object.keys(value)) {
      if (key === "ctx" || key === "audio" || key === "game") continue;
      const kept = plain(value[key], depth + 1);
      if (kept !== undefined) out[key] = kept;
    }
    return out;
  }
  return undefined;
}

function state(here) {
  const world = here.context.__world;
  return { world: plain(world, 0), game: plain(world.game, 0), held: [...world.held], now: here.clock.now };
}

let here = null;
const lines = readline.createInterface({ input: process.stdin });
lines.on("line", (line) => {
  let asked, answer;
  try {
    asked = JSON.parse(line);
    if (asked.op === "load") {
      here = load(asked.world, asked.query || "");
      step(here, 1 / 60, 1);
      answer = { ok: true };
    } else if (asked.op === "key") {
      fire(here.listeners[asked.type], anEvent({ key: asked.key }));
      answer = { ok: true };
    } else if (asked.op === "mouse") {
      const event = anEvent({ clientX: asked.x, clientY: asked.y });
      if (asked.type === "mouseup") fire(here.listeners.mouseup, event);
      else fire(here.canvasListeners[asked.type], event);
      answer = { ok: true };
    } else if (asked.op === "step") {
      step(here, asked.dt, asked.n || 1);
      answer = { ok: true, state: asked.quiet ? undefined : state(here) };
    } else if (asked.op === "state") {
      answer = { ok: true, state: state(here) };
    } else if (asked.op === "eval") {
      answer = { ok: true, value: plain(vm.runInContext(asked.js, here.context), 0) };
    } else {
      answer = { ok: false, error: "no such op: " + asked.op };
    }
  } catch (error) {
    answer = { ok: false, error: String(error && error.stack || error) };
  }
  process.stdout.write(JSON.stringify(answer) + "\n");
});
