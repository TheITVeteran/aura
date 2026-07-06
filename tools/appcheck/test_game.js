// Functional smoke test for a single-file HTML board game.
// Loads it in a real DOM, simulates click-to-move by data-row/col, and checks
// that a legal move ACTUALLY moves a piece (i.e. it is playable, not a mockup).
// Usage: node test_game.js <path-to-html>   -> prints JSON verdict.
const fs = require("fs");
const { JSDOM } = require("jsdom");

const file = process.argv[2];
const html = fs.readFileSync(file, "utf8");

const consoleErrors = [];
const dom = new JSDOM(html, {
  runScripts: "dangerously",
  resources: "usable",
  pretendToBeVisual: true,
  virtualConsole: new (require("jsdom").VirtualConsole)().on("jsdomError", (e) =>
    consoleErrors.push(String(e.message || e))
  ),
});
const { window } = dom;
const doc = window.document;

function sq(r, c) {
  return doc.querySelector(`[data-row="${r}"][data-col="${c}"]`);
}
function pieceInfo() {
  // map of "r,c" -> true for squares that currently contain a piece element
  const map = {};
  doc.querySelectorAll("[data-row][data-col]").forEach((s) => {
    if (s.querySelector('[class*="piece"], .red, .black') || /piece|red|black/i.test(s.className)) {
      map[`${s.getAttribute("data-row")},${s.getAttribute("data-col")}`] = true;
    }
  });
  return map;
}
function click(el) {
  if (!el) return;
  el.dispatchEvent(new window.Event("click", { bubbles: true, cancelable: true }));
  // some games listen on the piece itself
  const inner = el.querySelector && el.querySelector('[class*="piece"], .red, .black');
  if (inner) inner.dispatchEvent(new window.Event("click", { bubbles: true, cancelable: true }));
}

setTimeout(() => {
  try {
    const squares = doc.querySelectorAll("[data-row][data-col]").length;
    const before = pieceInfo();
    const cells = Object.keys(before);
    let moved = false,
      tried = 0;
    // try every occupied square as a source; try the 4 diagonal + 2-jump targets
    for (const key of cells) {
      const [r, c] = key.split(",").map(Number);
      const src = sq(r, c);
      const targets = [
        [r + 1, c - 1], [r + 1, c + 1], [r - 1, c - 1], [r - 1, c + 1],
        [r + 2, c - 2], [r + 2, c + 2], [r - 2, c - 2], [r - 2, c + 2],
      ];
      for (const [tr, tc] of targets) {
        const dst = sq(tr, tc);
        if (!dst) continue;
        click(src);
        click(dst);
        tried++;
        const after = pieceInfo();
        if (JSON.stringify(after) !== JSON.stringify(before)) {
          moved = true;
          break;
        }
      }
      if (moved) break;
    }
    const verdict = {
      playable: moved,
      squares_with_data: squares,
      occupied_squares: cells.length,
      moves_tried: tried,
      console_errors: consoleErrors.slice(0, 3),
      reason: moved
        ? "a simulated click-move changed the board"
        : squares === 0
        ? "squares have no data-row/data-col — cannot address them for moves"
        : "no simulated click sequence moved any piece (interaction not wired to move logic)",
    };
    console.log(JSON.stringify(verdict));
  } catch (e) {
    console.log(JSON.stringify({ playable: false, reason: "test crashed: " + String(e.message || e), console_errors: consoleErrors.slice(0, 3) }));
  }
}, 300);
