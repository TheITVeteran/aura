from pathlib import Path
import json


def _to_int(value):
    return int(value)


def _execute(tokens, state):
    if not tokens:
        return
    cmd = tokens[0].upper()
    if cmd == "SET":
        state[tokens[1]] = _to_int(tokens[2])
    elif cmd == "ADD":
        state[tokens[1]] = _to_int(state.get(tokens[1], 0)) + _to_int(tokens[2])
    elif cmd == "MUL":
        state[tokens[1]] = _to_int(state.get(tokens[1], 0)) * _to_int(tokens[2])
    elif cmd == "MOVE":
        src, dst, amount = tokens[1], tokens[2], _to_int(tokens[3])
        state[src] = _to_int(state.get(src, 0)) - amount
        state[dst] = _to_int(state.get(dst, 0)) + amount
    elif cmd == "LOOP":
        count = _to_int(tokens[1])
        if len(tokens) < 4 or tokens[2].upper() != "DO":
            raise ValueError("LOOP syntax must be: LOOP N DO <cmd>")
        for _ in range(count):
            _execute(tokens[3:], state)
    elif cmd == "IFGE":
        if "THEN" not in [part.upper() for part in tokens]:
            raise ValueError("IFGE syntax must be: IFGE var threshold THEN <cmd>")
        then_index = next(i for i, part in enumerate(tokens) if part.upper() == "THEN")
        var = tokens[1]
        threshold = _to_int(tokens[2])
        if _to_int(state.get(var, 0)) >= threshold:
            _execute(tokens[then_index + 1:], state)
    else:
        raise ValueError(f"unknown command: {cmd}")


def run_rules(path) -> dict:
    state = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        _execute(line.split(), state)
    return state


def write_state(script, out):
    state = run_rules(script)
    output = Path(out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    return state