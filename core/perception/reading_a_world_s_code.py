"""Reading a world's own scripts for how it plays: the physics of her body, the ground to be gone over, the clock,
the lives, and what each kind of thing does to her (core/perception/a_world_as_its_code_makes_it.py).

A frame of a game is a function: given where everything is, the keys held and the time since the last frame, it says
where everything is next. Its statements say that as arithmetic: ``if (held.has("ArrowLeft")) me.x -= 150 * dt`` is
150 pixels a second to the left while the left arrow is held; ``me.vy += 1100 * dt`` is a fall that gathers 1100
pixels a second every second; ``me.x = Math.max(0, Math.min(W - 16, me.x))`` keeps her between 0 and W - 16. This
reads a frame's function the way a person reads it: what each quantity comes to by the end of the frame, written in
what it was at the start, the time, and which keys are held. Then the findings are read off that:

* her body is the thing whose place the keys change, and its pace under each key is what a key adds to its place in
  a second; where the keys change its going and its place follows its going, they push it; where a key sets its going
  upward at once while it stands, it lifts it, and what is added to its going each second is the pull on it;
* where it is kept is the bounds its place is clamped to;
* ground to be gone over is a grid cell marked, by a place divided by the cell's size, with a count kept of them, and
  the share of that count that ends the game won;
* a clock is a quantity that falls one a second and ends the game at nought;
* what a kind of thing does to her is what happens where its code tests it touching her.

Every quantity is read as a sum of products of named factors (the time, a key held, what a quantity was at the start
of the frame, something bounded like a sine), each with a number; what cannot be read is said to be unknown, never
guessed. Nothing is run: the scripts are parsed and their arithmetic folded. Nothing here knows a game.
"""
from __future__ import annotations

import logging
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from core.perception.a_world_as_its_code_makes_it import (
    AKindOfThing,
    AWorldAsItsCodeMakesIt,
    GroundToCover,
    HerBody,
)

logger = logging.getLogger("Aura.ReadingAWorldsCode")

__all__ = ["read_a_world_s_code"]

#: The names a frame's function and a new game's function go by.
_A_FRAME = re.compile(r"^(?:step|update|tick|frame|loop|on_?enter_?frame|advance|move)$", re.I)
_A_BEGINNING = re.compile(r"^(?:reset|init|initiali[sz]e|start|begin|setup|new_?game|restart|spawn|next|home|place|"
                          r"board|ready|serve)$", re.I)
#: How deep a function's calls of its own world's functions are read into.
_DEEPEST = 3
#: The most statements read in one script, so a huge program cannot hold her up.
_MOST_STATEMENTS = 20000
#: Keys as a script names them, as her hands name them.
_KEY_NAMES = {"arrowleft": "left", "arrowright": "right", "arrowup": "up", "arrowdown": "down", " ": "space",
              "space": "space", "spacebar": "space", "enter": "enter", "escape": "escape", "shift": "shift",
              "mouse": "mouse", "left": "left", "right": "right", "up": "up", "down": "down"}
_KEY_CODES = {32: "space", 37: "left", 38: "up", 39: "right", 40: "down", 13: "enter", 16: "shift", 27: "escape"}
#: Words of an ending that is a win, said in a call that ends the game.
_WON = re.compile(r"\bwon\b|\bwin\b|victory|complete|cleared|success", re.I)

Poly = dict[tuple[str, ...], float]


def _p(coefficient: float = 0.0, *factors: str) -> Poly:
    return {tuple(sorted(factors)): float(coefficient)} if coefficient else {}


def _add(a: Poly, b: Poly, scale: float = 1.0) -> Poly:
    out = dict(a)
    for term, value in b.items():
        out[term] = out.get(term, 0.0) + scale * value
        if abs(out[term]) < 1e-12:
            del out[term]
    return out


def _times(a: Poly, b: Poly) -> Poly:
    out: Poly = {}
    for ta, va in a.items():
        for tb, vb in b.items():
            term = tuple(sorted(ta + tb))
            out[term] = out.get(term, 0.0) + va * vb
    return {t: v for t, v in out.items() if abs(v) > 1e-12}


def _constant(poly: Poly | None) -> float | None:
    if poly is None:
        return None
    if not poly:
        return 0.0
    if set(poly) == {()}:
        return poly[()]
    return None


@dataclass
class _V:
    """What an expression comes to: a sum of products (``poly``), a reference to a thing in the world (``ref``), an
    object or list written out, a word, or nothing that can be read."""

    poly: Poly | None = None
    ref: str = ""
    fields: dict[str, _V] | None = None
    items: list[_V] | None = None
    word: str | None = None
    function: Any = None
    #: The key a test of what is held names, where this is one.
    key: str = ""

    def number(self) -> Poly | None:
        if self.poly is not None:
            return self.poly
        if self.ref:
            return _p(1.0, "@" + self.ref)
        if self.word is not None:
            return None
        return None


_UNKNOWN = _V()


def _num(value: float) -> _V:
    return _V(poly=_p(value) if value else {})


@dataclass
class _Condition:
    keys: tuple[str, ...] = ()
    not_keys: tuple[str, ...] = ()
    #: Pairs of things a hit test asks about, by their references.
    touches: tuple[tuple[str, str], ...] = ()
    #: Comparisons it makes: (left, operator, right), each a sum of products.
    compares: tuple[tuple[Poly | None, str, Poly | None], ...] = ()
    negated: bool = False
    words: str = ""


@dataclass
class _Event:
    """Something a frame does under conditions: a life lost, the game ended (with its word), the score raised."""

    what: str
    word: str
    conditions: tuple[_Condition, ...]


@dataclass
class _Mark:
    """A cell of a collection set, by an index, under conditions; and what counts were raised beside it."""

    collection: str
    index: Poly
    conditions: tuple[_Condition, ...]
    counted: list[str] = field(default_factory=list)


@dataclass
class _Reading:
    """What is being found while a world's scripts are read."""

    constants: dict[str, _V] = field(default_factory=dict)
    functions: dict[str, Any] = field(default_factory=dict)
    game_fields: dict[str, _V] = field(default_factory=dict)
    stage: tuple[float, float] = (0.0, 0.0)
    initial: dict[str, Poly] = field(default_factory=dict)
    current: dict[str, Poly] = field(default_factory=dict)
    clamps: list[tuple[Poly, str, Poly]] = field(default_factory=list)
    cells: dict[str, dict[str, Any]] = field(default_factory=dict)
    marks: list[_Mark] = field(default_factory=list)
    events: list[_Event] = field(default_factory=list)
    keys: dict[str, str] = field(default_factory=dict)
    flags: dict[str, str] = field(default_factory=dict)
    loops: dict[str, list[_Event]] = field(default_factory=dict)
    resets: dict[str, list[Poly]] = field(default_factory=dict)
    statements: int = 0
    read_at: dict[str, str] = field(default_factory=dict)


class _Reader:
    """Reads one world's scripts (see the module's words)."""

    def __init__(self, sources: list[str]) -> None:
        import tree_sitter_javascript
        from tree_sitter import Language, Parser

        self.parser = Parser(Language(tree_sitter_javascript.language()))
        self.trees = [(source.encode("utf-8", "replace"), self.parser.parse(source.encode("utf-8", "replace")))
                      for source in sources]
        self.r = _Reading()
        self.source = b""
        self.frame_conditions: list[_Condition] = []
        self.in_frame = False
        self.depth = 0
        self.loop_of: str = ""

    # -- text ------------------------------------------------------------------------------------------------------

    def text(self, node: Any) -> str:
        return node.text.decode("utf-8", "replace") if node is not None and node.text is not None else ""

    # -- the whole program -----------------------------------------------------------------------------------------

    def read(self) -> _Reading:
        for source, tree in self.trees:
            self.source = source
            self._stage_in(tree.root_node)
        for source, tree in self.trees:
            self.source = source
            self._declarations(tree.root_node)
        for source, tree in self.trees:
            self.source = source
            self._listeners(tree.root_node)
        beginnings = [f for name, f in self.r.functions.items() if _A_BEGINNING.match(name.split(".")[-1])]
        frames = [f for name, f in self.r.functions.items() if _A_FRAME.match(name.split(".")[-1])]
        for function in beginnings:
            self._run_function(function, {}, beginning=True)
        self.r.current = {}
        self.in_frame = True
        for function in frames:
            self._run_function(function, {}, beginning=False)
        self.in_frame = False
        return self.r

    def _stage_in(self, node: Any) -> None:
        for found in re.finditer(rb"canvas\.(width|height)\s*=\s*(\d+(?:\.\d+)?)", node.text or b""):
            w, h = self.r.stage
            value = float(found.group(2))
            self.r.stage = (value, h) if found.group(1) == b"width" else (w, value)
        for found in re.finditer(rb"<canvas[^>]*\bwidth=\"?(\d+)\"?[^>]*\bheight=\"?(\d+)", node.text or b""):
            self.r.stage = (float(found.group(1)), float(found.group(2)))

    def _declarations(self, root: Any) -> None:
        """Every constant, the world's object and its functions, and what is set on it, at the top of a script."""
        for node in root.children:
            if node.type in ("lexical_declaration", "variable_declaration"):
                for declarator in node.children:
                    if declarator.type != "variable_declarator":
                        continue
                    name = self.text(declarator.child_by_field_name("name"))
                    value = declarator.child_by_field_name("value")
                    if value is None:
                        continue
                    if value.type == "object":
                        self._an_object(name, value)
                    elif value.type in ("arrow_function", "function_expression", "function"):
                        self.r.functions[name] = value
                    else:
                        self.r.constants[name] = self.value(value, {})
            elif node.type == "function_declaration":
                self.r.functions[self.text(node.child_by_field_name("name"))] = node
            elif node.type == "expression_statement":
                inner = node.children[0] if node.children else None
                if inner is not None and inner.type == "assignment_expression":
                    left = self.text(inner.child_by_field_name("left"))
                    right = inner.child_by_field_name("right")
                    owner, _dot, prop = left.rpartition(".")
                    if owner in self.r.constants or owner in ("game",) or owner and owner == getattr(self, "game_name", ""):
                        if right is not None and right.type in ("arrow_function", "function_expression"):
                            self.r.functions[prop] = right
                        else:
                            self.r.game_fields[prop] = self.value(right, {})
                    elif re.fullmatch(r"[A-Z_][A-Z0-9_]*", left) and right is not None:
                        self.r.constants[left] = self.value(right, {})

    def _an_object(self, name: str, node: Any) -> None:
        """An object at the top of a script: its methods are the world's functions where it has a frame's; its numbers
        are constants either way."""
        methods = {}
        fields: dict[str, _V] = {}
        for child in node.children:
            if child.type == "method_definition":
                methods[self.text(child.child_by_field_name("name"))] = child
            elif child.type == "pair":
                key = self.text(child.child_by_field_name("key")).strip("'\"")
                value = child.child_by_field_name("value")
                if value is not None and value.type in ("arrow_function", "function_expression"):
                    methods[key] = value
                elif value is not None:
                    fields[key] = self.value(value, {})
        if any(_A_FRAME.match(m) for m in methods):
            self.game_name = name
            self.r.functions.update(methods)
            self.r.game_fields.update(fields)
        self.r.constants[name] = _V(fields=fields)

    def _listeners(self, root: Any) -> None:
        """What is done when a key goes down, wherever a listener for it is: each key tested, and what is set or
        called for it."""
        for node in self._walk(root):
            if node.type != "call_expression":
                continue
            callee = self.text(node.child_by_field_name("function"))
            if not callee.endswith("addEventListener"):
                continue
            args = [c for c in (node.child_by_field_name("arguments") or node).children if c.type not in ("(", ")", ",")]
            if len(args) < 2 or not self.text(args[0]).strip("'\"").startswith("key"):
                continue
            handler = args[1]
            body = handler.child_by_field_name("body")
            if body is not None:
                self.in_frame = False
                self._block(body, {}, [])

    def _walk(self, node: Any) -> Iterable[Any]:
        stack = [node]
        while stack:
            n = stack.pop()
            yield n
            stack.extend(reversed(n.children))

    # -- functions -------------------------------------------------------------------------------------------------

    def _run_function(self, function: Any, scope: dict[str, _V], *, beginning: bool, args: list[_V] | None = None) -> _V:
        if self.depth >= _DEEPEST:
            return _UNKNOWN
        self.depth += 1
        try:
            local = dict(scope)
            params = function.child_by_field_name("parameters")
            names = [self.text(p) for p in (params.children if params is not None else []) if p.type == "identifier"]
            for i, name in enumerate(names):
                if args is not None and i < len(args):
                    local[name] = args[i]
                elif name == "dt" or name.lower() in ("delta", "elapsed", "dt_s", "secs"):
                    local[name] = _V(poly=_p(1.0, "dt"))
                else:
                    local[name] = _V(ref=name)
            body = function.child_by_field_name("body")
            if body is None:
                return _UNKNOWN
            if body.type != "statement_block":
                return self.value(body, local)
            self.beginning = beginning
            return self._block(body, local, list(self.frame_conditions))
        finally:
            self.depth -= 1

    # -- statements ------------------------------------------------------------------------------------------------

    def _block(self, node: Any, local: dict[str, _V], conditions: list[_Condition]) -> _V:
        children = node.children if node.type == "statement_block" else [node]
        returned = _UNKNOWN
        for child in children:
            self.r.statements += 1
            if self.r.statements > _MOST_STATEMENTS:
                return _UNKNOWN
            got = self._statement(child, local, conditions)
            if got is not None:
                returned = got
        return returned

    def _statement(self, node: Any, local: dict[str, _V], conditions: list[_Condition]) -> _V | None:
        kind = node.type
        if kind in ("lexical_declaration", "variable_declaration"):
            for declarator in node.children:
                if declarator.type == "variable_declarator":
                    name_node = declarator.child_by_field_name("name")
                    value = declarator.child_by_field_name("value")
                    got = self.value(value, local) if value is not None else _num(0.0)
                    if name_node is not None and name_node.type == "identifier":
                        local[self.text(name_node)] = got
        elif kind == "expression_statement":
            for child in node.children:
                if child.type != ";":
                    self._effect(child, local, conditions)
        elif kind == "if_statement":
            condition = self.condition(node.child_by_field_name("condition"), local)
            self._branch(node.child_by_field_name("consequence"), local, conditions, condition)
            alternative = node.child_by_field_name("alternative")
            if alternative is not None:
                inner = next((c for c in alternative.children if c.type not in ("else",)), None)
                if inner is not None:
                    opposite = _Condition(keys=condition.not_keys, not_keys=condition.keys, negated=not condition.negated,
                                          compares=condition.compares, touches=(), words="not " + condition.words)
                    self._branch(inner, local, conditions, opposite)
        elif kind == "for_in_statement":
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            collection = self.value(right, local) if right is not None else _UNKNOWN
            each = _V(ref=collection.ref + "[]") if collection.ref else _UNKNOWN
            inner = dict(local)
            if left is not None and left.type == "identifier":
                inner[self.text(left)] = each
            previous, self.loop_of = self.loop_of, collection.ref
            body = node.child_by_field_name("body")
            if body is not None:
                self._block(body, inner, conditions)
            self.loop_of = previous
        elif kind in ("for_statement", "while_statement", "do_statement"):
            body = node.child_by_field_name("body")
            if body is not None:
                self._block(body, dict(local), conditions)
        elif kind == "statement_block":
            return self._block(node, local, conditions)
        elif kind == "return_statement":
            inner = next((c for c in node.children if c.type not in ("return", ";")), None)
            return self.value(inner, local) if inner is not None else _UNKNOWN
        return None

    def _branch(self, node: Any, local: dict[str, _V], conditions: list[_Condition], condition: _Condition) -> None:
        """A branch read under its condition: what it sets in this function's own names is kept as what they come to
        when its keys are held (a guard that is not a key is taken to hold: where she is free to move, she moves)."""
        if node is None:
            return
        before = {name: value for name, value in local.items()}
        inner = dict(local)
        self._block(node, inner, [*conditions, condition])
        for name, value in inner.items():
            if name not in before:
                continue
            old = before[name]
            if value is old:
                continue
            local[name] = self._under(old, value, condition)

    def _under(self, old: _V, new: _V, condition: _Condition) -> _V:
        """What a quantity is after a branch that changed it ran only when ``condition``'s keys are held."""
        a, b = old.number(), new.number()
        if a is None or b is None:
            return new if (condition.keys or condition.not_keys) is False else new
        if not condition.keys and not condition.not_keys:
            return _V(poly=b)
        indicator = {(): 1.0}
        for key in condition.keys:
            indicator = _times(indicator, _p(1.0, "key:" + key))
        for key in condition.not_keys:
            indicator = _times(indicator, _p(1.0, "nokey:" + key))
        return _V(poly=_add(a, _times(_add(b, a, -1.0), indicator)))

    def _effect(self, node: Any, local: dict[str, _V], conditions: list[_Condition]) -> None:
        kind = node.type
        if kind == "assignment_expression":
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            self._assign(left, self.value(right, local), local, conditions, right)
        elif kind == "augmented_assignment_expression":
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            op = self.text(node.child_by_field_name("operator"))
            was = self.value(left, local)
            by = self.value(right, local)
            self._assign(left, self._arith(was, op[:-1], by), local, conditions, right, raised_by=by if op == "+=" else None)
        elif kind == "update_expression":
            target = next((c for c in node.children if c.type not in ("++", "--")), None)
            op = "+" if "++" in self.text(node) else "-"
            if target is not None:
                was = self.value(target, local)
                self._assign(target, self._arith(was, op, _num(1.0)), local, conditions, None,
                             raised_by=_num(1.0) if op == "+" else None)
        elif kind == "call_expression":
            self.value(node, local, conditions=conditions)
        elif kind == "sequence_expression":
            for child in node.children:
                if child.type != ",":
                    self._effect(child, local, conditions)

    def _assign(self, left: Any, value: _V, local: dict[str, _V], conditions: list[_Condition], right: Any,
                raised_by: _V | None = None) -> None:
        if left is None:
            return
        if left.type == "identifier":
            name = self.text(left)
            if name in local:
                local[name] = value
            return
        if left.type == "subscript_expression":
            self._a_cell_set(left, value, local, conditions)
            return
        if left.type != "member_expression":
            return
        owner = self.value(left.child_by_field_name("object"), local)
        prop = self.text(left.child_by_field_name("property"))
        if owner.fields is not None and not owner.ref:
            owner.fields[prop] = value
            return
        if not owner.ref:
            return
        path = prop if owner.ref in ("this", "game", getattr(self, "game_name", "\0")) else owner.ref + "." + prop
        if self.in_frame and raised_by is not None and _constant(raised_by.number()) == 1.0:
            for mark in self.r.marks:
                if mark.conditions == tuple(conditions):
                    mark.counted.append(path)
        if self.in_frame and owner.ref and not self.depth > 1 and self.loop_of == "" and prop and \
                _constant(value.number()) is not None and conditions and conditions[-1].keys and \
                not path.endswith((".x", ".y", ".vx", ".vy")):
            for key in conditions[-1].keys:
                self.r.flags.setdefault(key, prop)
        if value.fields is not None:
            self._an_object_placed(path, value)
            return
        if value.items is not None:
            self._a_list_placed(path, value)
            return
        number = value.number()
        if getattr(self, "beginning", False) and not self.in_frame:
            if number is not None and _constant(number) is not None:
                self.r.initial.setdefault(path, number)
            return
        if not self.in_frame:
            # A key's listener: what it sets is what the key does.
            for condition in conditions:
                for key in condition.keys:
                    self.r.flags.setdefault(key, prop)
            return
        if number is None:
            return
        old = self.r.current.get(path, _p(1.0, "@" + path))
        if re.search(r"(?:^|\.)(?:score|points)$", path) and raised_by is not None:
            self.r.events.append(_Event("score", path, tuple(conditions)))
        if conditions and (conditions[-1].keys or conditions[-1].not_keys):
            number = self._under(_V(poly=old), _V(poly=number), conditions[-1]).number() or number
        elif conditions and abs(_coefficient(number, "@" + path) - 1.0) > 1e-9 and not _a_quantity_of(number, old):
            # Set under a guard that is not a key, to something not made from what it was: a reset or a rebound (a
            # thing put back at its start, a going turned round), not how it goes from frame to frame.
            self.r.resets.setdefault(path, []).append(number)
            return
        self.r.current[path] = number
        self.r.read_at.setdefault(path, self.text(left.parent)[:120] if left.parent is not None else "")

    def _an_object_placed(self, path: str, value: _V) -> None:
        for name, field_value in (value.fields or {}).items():
            number = field_value.number()
            if number is not None and _constant(number) is not None:
                self.r.initial.setdefault(path + "." + name, number)
            elif number is not None and all(all(f.startswith("~") for f in term) for term in number):
                self.r.initial.setdefault(path + "." + name, number)

    def _a_list_placed(self, path: str, value: _V) -> None:
        for item in value.items or []:
            if item.fields is not None:
                self._an_object_placed(path + "[]", item)

    def _a_cell_set(self, left: Any, value: _V, local: dict[str, _V], conditions: list[_Condition]) -> None:
        """A cell of a collection set to something: kept where its index is made of cells a place was divided into."""
        if not self.in_frame:
            return
        target = left
        indices: list[Poly] = []
        while target is not None and target.type == "subscript_expression":
            index = self.value(target.child_by_field_name("index"), local).number()
            indices.insert(0, index if index is not None else {})
            target = target.child_by_field_name("object")
        owner = self.value(target, local)
        if not owner.ref:
            return
        if len(indices) == 2:
            # Two indices, one into the other: the first is the column.
            index = _add(indices[1], _times(indices[0], _p(1.0, "column-of")))
        else:
            index = indices[0] if indices else {}
        if any(f.startswith("cell#") for term in index for f in term):
            self.r.marks.append(_Mark(owner.ref, index, tuple(conditions)))

    # -- conditions ------------------------------------------------------------------------------------------------

    def condition(self, node: Any, local: dict[str, _V]) -> _Condition:
        if node is None:
            return _Condition()
        found = _Condition(words=self.text(node)[:120])
        self._condition_into(node, local, found, negated=False)
        return found

    def _condition_into(self, node: Any, local: dict[str, _V], found: _Condition, negated: bool) -> None:
        kind = node.type
        if kind == "parenthesized_expression":
            for child in node.children:
                if child.type not in ("(", ")"):
                    self._condition_into(child, local, found, negated)
            return
        if kind == "unary_expression" and self.text(node.child_by_field_name("operator")) == "!":
            self._condition_into(node.child_by_field_name("argument"), local, found, not negated)
            return
        if kind == "binary_expression":
            op = self.text(node.child_by_field_name("operator"))
            left, right = node.child_by_field_name("left"), node.child_by_field_name("right")
            if op in ("&&",) and not negated or op == "||" and negated:
                self._condition_into(left, local, found, negated)
                self._condition_into(right, local, found, negated)
                return
            if op == "||":
                keys_before = found.keys
                self._condition_into(left, local, found, negated)
                if found.keys == keys_before:
                    self._condition_into(right, local, found, negated)
                return
            if op in ("===", "==", "!=", "!==", "<", "<=", ">", ">="):
                key = self._a_key_compared(left, right, local)
                if key and op in ("===", "==") and not negated or key and op in ("!=", "!==") and negated:
                    found.keys = tuple(dict.fromkeys((*found.keys, key)))
                    return
                lhs, rhs = self.value(left, local).number(), self.value(right, local).number()
                if negated:
                    op = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "==": "!=", "===": "!==", "!=": "==",
                          "!==": "==="}[op]
                found.compares = (*found.compares, (lhs, op, rhs))
                return
        if kind == "call_expression":
            got = self.value(node, local)
            if got.key:
                if negated:
                    found.not_keys = tuple(dict.fromkeys((*found.not_keys, got.key)))
                else:
                    found.keys = tuple(dict.fromkeys((*found.keys, got.key)))
                return
            callee = self.text(node.child_by_field_name("function"))
            if re.search(r"(?:^|\.)(?:hits?|collid\w*|overlap\w*|intersect\w*|touch\w*|hit_?test\w*)$", callee, re.I) \
                    and not negated:
                args = [self.value(c, local) for c in (node.child_by_field_name("arguments") or node).children
                        if c.type not in ("(", ")", ",")]
                refs = [a.ref or _the_thing_a_box_is_of(a) for a in args]
                refs = [ref for ref in refs if ref]
                if len(refs) >= 2:
                    found.touches = (*found.touches, (refs[0], refs[1]))
                elif len(refs) == 1:
                    found.touches = (*found.touches, (refs[0], ""))
            return

    def _a_key_compared(self, left: Any, right: Any, local: dict[str, _V]) -> str:
        """A key tested by name or code: ``e.key === "z"``, ``e.keyCode == 32``, ``k === "ArrowLeft"``."""
        for this, that in ((left, right), (right, left)):
            said = self.text(this)
            if this.type == "string" and re.search(r"(?:\.key|\.code|^k(?:ey)?$|keyName)", self.text(that)):
                return _a_key(said.strip("'\"`"))
            if this.type == "number" and re.search(r"keyCode|\.which|\.code", self.text(that)):
                return _KEY_CODES.get(int(float(said)), f"key {int(float(said))}")
        return ""

    # -- values ----------------------------------------------------------------------------------------------------

    def value(self, node: Any, local: dict[str, _V], conditions: list[_Condition] | None = None) -> _V:
        if node is None:
            return _UNKNOWN
        kind = node.type
        if kind == "number":
            try:
                return _num(float(self.text(node)))
            except ValueError:
                return _UNKNOWN
        if kind in ("string", "template_string"):
            return _V(word=self.text(node).strip("'\"`"))
        if kind in ("true", "false"):
            return _num(1.0 if kind == "true" else 0.0)
        if kind == "parenthesized_expression":
            inner = next((c for c in node.children if c.type not in ("(", ")")), None)
            return self.value(inner, local)
        if kind == "identifier":
            name = self.text(node)
            if name in local:
                return local[name]
            if name in self.r.constants:
                return self.r.constants[name]
            if name in ("W", "width") and self.r.stage[0]:
                return _num(self.r.stage[0])
            if name in ("H", "height") and self.r.stage[1]:
                return _num(self.r.stage[1])
            if name == "undefined":
                return _UNKNOWN
            return _V(ref=name)
        if kind == "this":
            return _V(ref="this")
        if kind == "member_expression":
            return self._member(node, local)
        if kind == "subscript_expression":
            owner = self.value(node.child_by_field_name("object"), local)
            index = self.value(node.child_by_field_name("index"), local)
            if owner.items is not None and _constant(index.number()) is not None:
                i = int(_constant(index.number()))
                return owner.items[i] if 0 <= i < len(owner.items) else _UNKNOWN
            if owner.items:
                return owner.items[0]
            if owner.ref:
                return _V(ref=owner.ref + "[]")
            return _UNKNOWN
        if kind == "unary_expression":
            op = self.text(node.child_by_field_name("operator"))
            inner = self.value(node.child_by_field_name("argument"), local)
            if op == "-" and inner.number() is not None:
                return _V(poly=_add({}, inner.number(), -1.0))
            if op == "+":
                return inner
            return _UNKNOWN
        if kind == "binary_expression":
            op = self.text(node.child_by_field_name("operator"))
            left = self.value(node.child_by_field_name("left"), local)
            right = self.value(node.child_by_field_name("right"), local)
            return self._arith(left, op, right)
        if kind == "ternary_expression":
            a = self.value(node.child_by_field_name("consequence"), local)
            b = self.value(node.child_by_field_name("alternative"), local)
            ca, cb = _constant(a.number()), _constant(b.number())
            if ca is not None and cb is not None:
                # One of two numbers: a factor bounded by the two, so a speed is read at its most and a sign either way.
                return _V(poly=_p(1.0, f"~pick:{ca:g}:{cb:g}")) if ca != cb else _num(ca)
            return a if a.number() is not None else b
        if kind == "object":
            fields: dict[str, _V] = {}
            for child in node.children:
                if child.type == "pair":
                    key = self.text(child.child_by_field_name("key")).strip("'\"")
                    fields[key] = self.value(child.child_by_field_name("value"), local)
                elif child.type == "shorthand_property_identifier":
                    fields[self.text(child)] = self.value(child, local)
            return _V(fields=fields)
        if kind == "array":
            items = [c for c in node.children if c.type not in ("[", "]", ",")]
            if len(items) == 1 and items[0].type == "spread_element":
                # A copy of a list is the list: what is done to its items is done to the list's.
                return self.value(next((c for c in items[0].children if c.type != "..."), None), local)
            return _V(items=[self.value(c, local) for c in items])
        if kind == "call_expression":
            return self._call(node, local, conditions or [])
        if kind in ("arrow_function", "function_expression"):
            return _V(function=node)
        if kind == "assignment_expression":
            self._effect(node, local, conditions or [])
            return self.value(node.child_by_field_name("right"), local)
        if kind == "new_expression":
            return _UNKNOWN
        return _UNKNOWN

    def _member(self, node: Any, local: dict[str, _V]) -> _V:
        if self.text(node) == "Math.PI":
            return _num(math.pi)
        owner = self.value(node.child_by_field_name("object"), local)
        prop = self.text(node.child_by_field_name("property"))
        if owner.fields is not None and prop in owner.fields:
            return owner.fields[prop]
        if owner.items is not None and prop == "length":
            return _num(float(len(owner.items)))
        if owner.ref == "this" or owner.ref == getattr(self, "game_name", "\0"):
            if prop in self.r.game_fields and prop not in self.r.current:
                field_value = self.r.game_fields[prop]
                if field_value.number() is not None and _constant(field_value.number()) is not None:
                    return field_value
            return _V(ref=prop)
        if owner.ref:
            path = owner.ref + "." + prop
            if prop in ("W", "width") and owner.ref in ("world", "canvas", "stage", "game") and self.r.stage[0]:
                return _num(self.r.stage[0])
            if prop in ("H", "height") and owner.ref in ("world", "canvas", "stage", "game") and self.r.stage[1]:
                return _num(self.r.stage[1])
            if self.in_frame and path in self.r.current:
                return _V(poly=self.r.current[path], ref=path)
            if path not in self.r.current and path in self.r.initial and _constant(self.r.initial[path]) is not None \
                    and prop in ("w", "h", "width", "height", "r", "size"):
                return _V(poly=self.r.initial[path], ref=path)
            return _V(ref=path)
        return _UNKNOWN

    def _arith(self, left: _V, op: str, right: _V) -> _V:
        a, b = left.number(), right.number()
        if op in ("&&", "||", "==", "===", "!=", "!==", "<", "<=", ">", ">=", "??"):
            return _UNKNOWN
        if a is None or b is None:
            return _UNKNOWN
        if op == "+":
            return _V(poly=_add(a, b))
        if op == "-":
            return _V(poly=_add(a, b, -1.0))
        if op == "*":
            return _V(poly=_times(a, b))
        if op == "/":
            divisor = _constant(b)
            if divisor:
                return _V(poly={t: v / divisor for t, v in a.items()})
            return _UNKNOWN
        if op == "%":
            ca, cb = _constant(a), _constant(b)
            return _num(math.fmod(ca, cb)) if ca is not None and cb else _V(poly=_p(1.0, "?mod"))
        return _UNKNOWN

    def _call(self, node: Any, local: dict[str, _V], conditions: list[_Condition]) -> _V:
        callee = node.child_by_field_name("function")
        name = self.text(callee)
        arg_nodes = [c for c in (node.child_by_field_name("arguments") or node).children if c.type not in ("(", ")", ",")]
        args = [self.value(a, local) for a in arg_nodes]
        last = name.split(".")[-1]
        if name.startswith("Math."):
            return self._math(last, args, arg_nodes, local)
        if last in ("rand", "random"):
            return _V(poly=_p(1.0, "~rand"))
        if last == "has" and args and args[0].word is not None:
            return _V(key=_a_key(args[0].word))
        if last == "isDown" and args and _constant(args[0].number()) is not None:
            return _V(key=_KEY_CODES.get(int(_constant(args[0].number())), f"key {int(_constant(args[0].number()))}"))
        if last == "map" and callee is not None and callee.type == "member_expression":
            each = self.value(callee.child_by_field_name("object"), local)
            if args and args[0].function is not None:
                function = args[0].function
                params = function.child_by_field_name("parameters")
                inner = dict(local)
                many = max(1, len(each.items or [1]))
                for p in (params.children if params is not None else []):
                    if p.type == "identifier":
                        inner[self.text(p)] = _V(poly=_p(1.0, f"~pick:0:{many - 1:g}"))
                if function.child_by_field_name("parameter") is not None:
                    inner[self.text(function.child_by_field_name("parameter"))] = _V(poly=_p(1.0, f"~pick:0:{many - 1:g}"))
                body = function.child_by_field_name("body")
                if body is not None and body.type != "statement_block":
                    one = self.value(body, inner)
                    return _V(items=[one] * max(1, len(each.items or [1])))
            return _UNKNOWN
        if last == "from" and name.startswith("Array") and len(args) >= 2 and args[1].function is not None:
            body = args[1].function.child_by_field_name("body")
            if body is not None and body.type != "statement_block":
                return _V(items=[self.value(body, dict(local))])
            return _UNKNOWN
        if last == "push" and callee is not None and callee.type == "member_expression":
            owner = self.value(callee.child_by_field_name("object"), local)
            if owner.ref and args and args[0].fields is not None:
                self._an_object_placed(owner.ref + "[]", args[0])
                # A key that makes a thing (a shot, a throw) fires it.
                for condition in conditions:
                    for key in condition.keys:
                        self.r.flags.setdefault(key, "fire " + owner.ref)
            return _UNKNOWN
        if last in ("lose", "over", "die", "kill", "gameOver", "game_over", "end", "win", "finish", "endLevel"):
            word = next((a.word for a in args if a.word is not None), last)
            if self.in_frame or self.depth:
                self.r.events.append(_Event("over" if last != "lose" else "lose", word, tuple(conditions)))
            return _UNKNOWN
        if last in ("assign",):
            return _UNKNOWN
        own = self.r.functions.get(last)
        if own is not None and (name.startswith(("this.", "game.")) or name == last):
            got = self._run_function(own, {}, beginning=getattr(self, "beginning", False), args=args)
            return got
        return _UNKNOWN

    def _math(self, name: str, args: list[_V], arg_nodes: list[Any], local: dict[str, _V]) -> _V:
        numbers = [a.number() for a in args]
        constants = [_constant(n) for n in numbers]
        if all(c is not None for c in constants) and constants:
            try:
                return _num(float(getattr(math, {"abs": "fabs", "pow": "pow"}.get(name, name))(*constants))) \
                    if name not in ("max", "min", "round", "sign", "random", "hypot") else _num(float({
                        "max": max, "min": min, "round": round,
                        "sign": lambda v: (v > 0) - (v < 0), "hypot": math.hypot}[name](*constants)))
            except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
                return _UNKNOWN
        if name in ("max", "min") and len(args) == 2 and None not in numbers:
            # Of the two, the one that is a quantity of the world as it stands is what is clamped; the other is the
            # bound, put in at what the quantities in it begin at once the frame is read.
            moving = [i for i, n in enumerate(numbers) if _a_quantity(n)]
            if len(moving) == 1:
                inner, bound = numbers[moving[0]], numbers[1 - moving[0]]
                self.r.clamps.append((inner, "low" if name == "max" else "high", bound))
                return args[moving[0]]
            held = [i for i, c in enumerate(constants) if c is not None]
            if len(held) == 1:
                self.r.clamps.append((numbers[1 - held[0]], "low" if name == "max" else "high", numbers[held[0]]))
                return args[1 - held[0]]
            return args[0]
        if name in ("cos", "sin"):
            inner = numbers[0] if numbers else None
            single = [t[0] for t in (inner or {}) if len(t) == 1 and t[0].startswith("@")]
            return _V(poly=_p(1.0, f"~{name}:{single[0]}" if len(single) == 1 else "~" + name))
        if name == "random":
            return _V(poly=_p(1.0, "~rand"))
        if name == "exp" and numbers and numbers[0] is not None:
            # A decay over one frame, to its first order in the time: e^(k dt) is 1 + k dt.
            return _V(poly=_add({(): 1.0}, numbers[0]))
        if name in ("floor", "round", "trunc") and numbers and numbers[0] is not None:
            number = numbers[0]
            key = f"cell#{len(self.r.cells)}"
            self.r.cells[key] = {"inside": number, "node": arg_nodes[0] if arg_nodes else None, "local": dict(local)}
            return _V(poly=_p(1.0, key))
        if name == "abs" and numbers and numbers[0] is not None:
            return _V(poly=_p(1.0, "?abs"))
        if name in ("hypot", "atan2", "sign", "sqrt"):
            return _V(poly=_p(1.0, "?" + name))
        return _UNKNOWN


def _the_thing_a_box_is_of(value: _V) -> str:
    """The thing a box written out is made from: ``{x: c.x, y: GROUND - c.h, w: c.w, h: c.h}`` is ``c``'s."""
    for name in ("x", "y"):
        found = (value.fields or {}).get(name)
        poly = found.number() if found is not None else None
        for term in poly or {}:
            if len(term) == 1 and term[0].startswith("@") and term[0].endswith("." + name):
                return term[0][1:].rsplit(".", 1)[0]
    return ""


def _a_quantity_of(new: Poly, old: Poly) -> bool:
    """Whether a value is made from what a quantity already was this frame (its first terms carried on)."""
    return bool(old) and all(new.get(term) == value for term, value in old.items() if len(term) == 1
                             and term[0].startswith("@"))


#: The names of quantities that are sizes and settings, not where things are or how they go: a stage's W and H, a
#: thing's w and h, a constant written in capitals.
_A_SIZE = re.compile(r"\.(?:w|h|width|height|size|r|[A-Z][A-Z0-9_]*)$")


def _a_quantity(poly: Poly) -> bool:
    """Whether a sum holds a quantity of the world as it stands (its place, its going), not only sizes and numbers."""
    return any(len(term) == 1 and term[0].startswith("@") and not _A_SIZE.search(term[0])
               and abs(value - 1.0) < 1e-9 for term, value in poly.items()) or any(
        any(f.startswith(("cell#", "key:")) for f in term) for term in poly)


def _a_key(said: str) -> str:
    lowered = said.lower()
    if lowered in _KEY_NAMES:
        return _KEY_NAMES[lowered]
    if len(said) == 1:
        return said.lower()
    return lowered


# -- what the reading comes to ----------------------------------------------------------------------------------------


def _coefficient(poly: Poly, *factors: str) -> float:
    return poly.get(tuple(sorted(factors)), 0.0)


def _delta(r: _Reading, path: str) -> Poly:
    return _add(r.current.get(path, _p(1.0, "@" + path)), _p(1.0, "@" + path), -1.0)


def _keys_in(poly: Poly) -> set[str]:
    return {f[4:] for term in poly for f in term if f.startswith("key:")}


def _paces(r: _Reading, body: str, lifts: dict[str, float] | None = None) -> dict[str, tuple[float, float]]:
    """How far each key held takes a body in a second, by what it adds to its place; not a key whose press sets its
    going (a jump), which its place then follows."""
    dx, dy = _delta(r, body + ".x"), _delta(r, body + ".y")
    out = {}
    for key in sorted((_keys_in(dx) | _keys_in(dy)) - set(lifts or {})):
        pace = (_coefficient(dx, "dt", "key:" + key), _coefficient(dy, "dt", "key:" + key))
        if any(abs(v) > 1e-9 for v in pace):
            out[key] = pace
    return out


def _pushes(r: _Reading, body: str) -> tuple[dict[str, tuple[float, float]], set[str]]:
    """What each key adds to a body's going each second, where its place follows its going; and the keys that do.
    With "" for the pull on it with no key held."""
    dx, dy = _delta(r, body + ".x"), _delta(r, body + ".y")
    follows_x = abs(_coefficient(dx, "@" + body + ".vx", "dt") - 1.0) < 1e-6
    follows_y = abs(_coefficient(dy, "@" + body + ".vy", "dt") - 1.0) < 1e-6
    if not (follows_x or follows_y):
        return {}, set()
    vx, vy = _delta(r, body + ".vx"), _delta(r, body + ".vy")
    rest = (_coefficient(vx, "dt") if follows_x else 0.0, _coefficient(vy, "dt") if follows_y else 0.0)
    keys = {key for key in (_keys_in(vx) if follows_x else set()) | (_keys_in(vy) if follows_y else set())
            if _coefficient(vx, "dt", "key:" + key) or _coefficient(vy, "dt", "key:" + key)}
    pushes = {"": rest}
    for key in sorted(keys):
        pushes[key] = (rest[0] + _coefficient(vx, "dt", "key:" + key), rest[1] + _coefficient(vy, "dt", "key:" + key))
    return pushes, keys


def _lifts(r: _Reading, body: str) -> tuple[dict[str, float], float | None]:
    """A key that sets a body's going upward at once (while it stands, as its condition says), and the pull after."""
    vy = r.current.get(body + ".vy")
    if vy is None:
        return {}, None
    lifts = {}
    for term, value in vy.items():
        keys = [f[4:] for f in term if f.startswith("key:")]
        rest = [f for f in term if not f.startswith("key:")]
        if len(keys) == 1 and not rest and value < 0:
            lifts[keys[0]] = -value
    pull = _coefficient(vy, "dt") or None
    return lifts, pull


def _turns(r: _Reading, body: str) -> tuple[dict[str, float], dict[str, float], tuple[float, float] | None]:
    """Where a body's place follows the way it faces (a sine and cosine of a quantity of its own, times a speed of
    its own): how fast each key turns that way, in radians a second; how fast each key gathers speed; and the speeds
    it is held between."""
    dx = _delta(r, body + ".x")
    heading = speed = ""
    for term in dx:
        angles = [f.split(":", 1)[1][1:] for f in term if f.startswith(("~cos:", "~sin:"))]
        paces = [f[1:] for f in term if f.startswith("@" + body + ".")]
        if angles and paces and "dt" in term:
            heading, speed = angles[0], paces[0]
            break
    if not heading:
        return {}, {}, None
    turning, gathering = _delta(r, heading), _delta(r, speed)
    turns = {k: _coefficient(turning, "dt", "key:" + k) for k in sorted(_keys_in(turning))}
    drives = {k: _coefficient(gathering, "dt", "key:" + k) for k in sorted(_keys_in(gathering))}
    low, high = _bounds_of(r, speed)
    return ({k: v for k, v in turns.items() if v}, {k: v for k, v in drives.items() if v},
            (low, high) if low is not None and high is not None else None)


def _follows_the_pointer(r: _Reading, body: str) -> tuple[bool, bool]:
    def follows(path: str) -> bool:
        return any(any(f.startswith(("@world.pointer", "@pointer", "@mouse")) for f in t) for t in r.current.get(path, {}))
    return follows(body + ".x"), follows(body + ".y")


def _bounds_of(r: _Reading, path: str) -> tuple[float | None, float | None]:
    low = high = None
    for inner, side, bound in r.clamps:
        if abs(_coefficient(inner, "@" + path) - 1.0) > 1e-6:
            continue
        value = _constant(_substituted(r, bound))
        if value is None:
            continue
        if side == "low":
            low = value if low is None else max(low, value)
        else:
            high = value if high is None else min(high, value)
    return low, high


def _substituted(r: _Reading, poly: Poly) -> Poly:
    """A sum with the quantities that are not changed in a frame put in at what they begin at."""
    out: Poly = {}
    for term, value in poly.items():
        rest: list[str] = []
        scale = value
        for factor in term:
            if factor.startswith("@") and factor[1:] in r.initial and _constant(r.initial[factor[1:]]) is not None:
                scale *= _constant(r.initial[factor[1:]])
            else:
                rest.append(factor)
        key = tuple(sorted(rest))
        out[key] = out.get(key, 0.0) + scale
    return {t: v for t, v in out.items() if abs(v) > 1e-12}


def _her_body(r: _Reading) -> str:
    """The thing whose place the keys change most ways; else one the keys push, lift or turn; else one the pointer
    places."""
    bodies = sorted({p.rsplit(".", 1)[0] for p in r.current if p.endswith((".x", ".y", ".vx", ".vy", ".a", ".v"))})
    ranked = sorted(bodies, key=lambda b: -len(_paces(r, b, _lifts(r, b)[0])))
    if ranked and _paces(r, ranked[0], _lifts(r, ranked[0])[0]):
        return ranked[0]
    for body in bodies:
        if _pushes(r, body)[1] or _lifts(r, body)[0] or _turns(r, body)[0]:
            return body
    for body in bodies:
        if any(_follows_the_pointer(r, body)):
            return body
    return ""


def _ground(r: _Reading, body: str, size: tuple[float, float] | None) -> tuple[GroundToCover | None, str]:
    for mark in r.marks:
        if not mark.counted:
            continue
        cells = [f for term in mark.index for f in term if f.startswith("cell#")]
        axes: dict[int, tuple[float, float, float]] = {}
        for key in dict.fromkeys(cells):
            inside = r.cells[key]["inside"]
            for axis, name in ((0, "x"), (1, "y")):
                per = _coefficient(inside, "@" + body + "." + name)
                if per:
                    cell = 1.0 / per
                    offset = _constant({t: v for t, v in inside.items() if t == ()}) or 0.0
                    marks, origin = _marks_and_origin(r, key, offset * cell)
                    axes[axis] = (cell, marks, origin)
                    r.cells[key]["axis"] = axis
        if 0 not in axes or 1 not in axes:
            continue
        column_key = next(k for k in dict.fromkeys(cells) if r.cells[k].get("axis") == 0)
        row_key = next(k for k in dict.fromkeys(cells) if r.cells[k].get("axis") == 1)
        columns = _coefficient(mark.index, row_key) or None
        if _coefficient(mark.index, "column-of", column_key):
            columns = None
        limits = {}
        for condition in mark.conditions:
            for lhs, op, rhs in condition.compares:
                if lhs is None or rhs is None or op not in ("<", "<="):
                    continue
                for key in (column_key, row_key):
                    if lhs == _p(1.0, key) and _constant(rhs) is not None:
                        limits[key] = _constant(rhs) + (1 if op == "<=" else 0)
        columns = columns or limits.get(column_key)
        rows = limits.get(row_key)
        count = mark.counted[0]
        wins = _wins_at(r, count, (columns or 0) * (rows or 0))
        ground = GroundToCover(cell=(axes[0][0], axes[1][0]), origin=(axes[0][2], axes[1][2]),
                               cells=(int(columns), int(rows)) if columns and rows else None,
                               marks_at=(axes[0][1], axes[1][1]), wins_at=wins, kept_in=mark.collection,
                               counted_by=count)
        return ground, f"{mark.collection}[...] = 1, counted in {count}"
    return None, ""


def _marks_and_origin(r: _Reading, key: str, offset: float) -> tuple[float, float]:
    """The point of her body a cell is marked by, and where the grid begins, from how the place divided was written:
    what is added to her place is the point of her (``me.y + 8``), what is taken away is where the grid begins
    (``- TOP``). Read from the words of the sum, not its value: the two only differ in what they mean."""
    node = r.cells[key].get("node")
    added = taken = 0.0
    if node is None:
        return offset, 0.0
    reader_text = node.text.decode("utf-8", "replace") if node.text else ""
    numerator = re.split(r"\)\s*/|/(?![^()]*\))", reader_text)[0]
    for sign, term in re.findall(r"([+-])\s*([A-Za-z_$][\w$.]*|\d+(?:\.\d+)?(?:\s*/\s*\d+)?)", numerator):
        value = _named_number(r, term, r.cells[key].get("local") or {})
        if value is None:
            continue
        if sign == "+":
            added += value
        else:
            taken += value
    if abs((added - taken) - offset) > 1e-6:
        return offset, 0.0
    return added, taken


def _named_number(r: _Reading, term: str, local: dict[str, _V]) -> float | None:
    term = term.strip()
    if re.fullmatch(r"\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?", term):
        a, b = (float(v) for v in term.split("/"))
        return a / b
    if re.fullmatch(r"\d+(?:\.\d+)?", term):
        return float(term)
    head = term.split(".")[0]
    value = local.get(head) if head in local else r.constants.get(head)
    if value is not None and "." not in term:
        return _constant(value.number())
    if value is not None and value.fields is not None:
        inner = value.fields.get(term.split(".")[1])
        return _constant(inner.number()) if inner is not None else None
    path = term.replace("this.", "")
    if path in r.initial:
        return _constant(r.initial[path])
    return None


def _wins_at(r: _Reading, count: str, cells: float) -> float | None:
    """The share of the cells counted at which the game ends won, by the comparison guarding a winning end."""
    for event in r.events:
        if event.what != "over" or not _WON.search(event.word or ""):
            continue
        for condition in event.conditions:
            for lhs, op, rhs in condition.compares:
                if lhs is None or rhs is None or op not in (">=", ">"):
                    continue
                share = _coefficient(lhs, "@" + count)
                if not share:
                    share = sum(v for t, v in lhs.items() if ("@" + count) in t and len(t) == 1)
                bound = _constant(rhs)
                if share and bound is not None:
                    counted = bound / share
                    return counted / cells if cells else counted
    return None


def _clock(r: _Reading) -> tuple[float | None, str]:
    for path in r.current:
        delta = _delta(r, path)
        if abs(_coefficient(delta, "dt") + 1.0) > 1e-6:
            continue
        for event in r.events:
            if event.what != "over":
                continue
            for condition in event.conditions:
                for lhs, op, rhs in condition.compares:
                    if lhs is not None and _coefficient(lhs, "@" + path) and op in ("<=", "<") and _constant(rhs) == 0.0:
                        start = _constant(r.initial.get(path))
                        if start:
                            return start, path
    return None, ""


def _things(r: _Reading, body: str) -> list[AKindOfThing]:
    """Each collection of things the code keeps: what touching her does, by what is done where its code tests the two
    touching; how it moves; how fast at most; and its size."""
    out = []
    collections = {p.split("[]")[0] for p in [*r.current, *r.initial] if "[]" in p}
    for collection in sorted(collections):
        element = collection + "[]"
        if element == body:
            continue
        done = set()
        for event in r.events:
            for condition in event.conditions:
                if any(element in pair and (body in pair or "" in pair) for pair in condition.touches):
                    done.add({"lose": "costs a life", "score": "scores"}.get(event.what, "ends the game"))
        touching = " or ".join(sorted(done))
        size = (_constant(r.initial.get(element + ".w")), _constant(r.initial.get(element + ".h")))
        speed, moves = _how_it_moves(r, element, body)
        out.append(AKindOfThing(collection, touching, moves, speed,
                                size if all(v is not None for v in size) else None))
    return out


#: The range each bounded factor of a sum keeps to.
def _range_of(factor: str) -> tuple[float, float] | None:
    if factor == "~rand":
        return 0.0, 1.0
    if factor.startswith(("~cos", "~sin")):
        return -1.0, 1.0
    if factor.startswith("~pick:"):
        a, b = (float(v) for v in factor.split(":")[1:3])
        return min(a, b), max(a, b)
    return None


def _interval(poly: Poly | None, given: dict[str, Poly] | None = None) -> tuple[float, float] | None:
    """The least and most a sum can come to, its bounded factors taken over their ranges; None where it holds one
    that is not bounded."""
    if poly is None:
        return None
    low = high = 0.0
    for term, value in poly.items():
        lo = hi = value
        for factor in term:
            span = _range_of(factor)
            if span is None and given is not None and factor.startswith("@") and factor[1:] in given:
                span = _interval(given[factor[1:]])
            if span is None:
                return None
            products = [lo * span[0], lo * span[1], hi * span[0], hi * span[1]]
            lo, hi = min(products), max(products)
        low, high = low + lo, high + hi
    return low, high


def _rate(r: _Reading, delta: Poly) -> tuple[float, float] | None:
    """What a frame adds to a quantity in a second, at its least and most, from the terms with the time in them."""
    timed = {tuple(f for f in t if f != "dt"): v for t, v in delta.items() if "dt" in t}
    if not timed:
        return 0.0, 0.0
    return _interval(timed, r.initial)


def _how_it_moves(r: _Reading, element: str, body: str) -> tuple[float | None, str]:
    dx, dy = _delta(r, element + ".x"), _delta(r, element + ".y")
    if not any("dt" in t for t in [*dx, *dy]):
        return None, "stands"
    if any(("@" + body + ".x") in t or ("@" + body + ".y") in t for t in [*dx, *dy]):
        return None, "chases her"
    # Along a way it faces, by a sine and a cosine of one angle: as fast as the factor the two share.
    cos = [(t, v) for t, v in dx.items() if any(f.startswith("~cos") for f in t) and "dt" in t]
    sin = [(t, v) for t, v in dy.items() if any(f.startswith("~sin") for f in t) and "dt" in t]
    if len(cos) == 1 and len(sin) == 1 and abs(abs(cos[0][1]) - abs(sin[0][1])) < 1e-9 and len(cos[0][0]) == 2:
        return abs(cos[0][1]), "wanders"
    across, down = _rate(r, dx), _rate(r, dy)
    if across is None or down is None:
        return None, "goes"
    most_x, most_y = max(abs(v) for v in across), max(abs(v) for v in down)
    speed = math.hypot(most_x, most_y) or None
    if speed is None:
        return None, "stands"
    if most_x < 1e-9 and down[0] >= 0:
        return speed, "falls"
    if most_x < 1e-9 and down[1] <= 0:
        return speed, "rises"
    if most_y < 1e-9:
        return speed, "crosses"
    return speed, "goes"


def read_a_world_s_code(sources: Iterable[str], stage: tuple[float, float] | None = None) -> AWorldAsItsCodeMakesIt:
    """What a world's scripts say of how it plays (see the module's words); an empty reading where they say nothing
    this can read, or cannot be parsed. ``stage`` is the size of what it draws on where the page says it and the
    scripts do not (a canvas's own width and height)."""
    texts = [s for s in sources if isinstance(s, str) and s.strip()]
    if not texts:
        return AWorldAsItsCodeMakesIt()
    try:
        reader = _Reader(texts)
        if stage and all(stage):
            reader.r.stage = (float(stage[0]), float(stage[1]))
        r = reader.read()
    except (ImportError, RecursionError, ValueError, TypeError, AttributeError, KeyError, IndexError) as why:
        logger.info("a world's code could not be read for how it plays: %s", why)
        return AWorldAsItsCodeMakesIt()
    body = _her_body(r)
    world = AWorldAsItsCodeMakesIt(stage=r.stage)
    if body:
        lifts, pull = _lifts(r, body)
        paces = _paces(r, body, lifts)
        size = (_constant(r.initial.get(body + ".w")), _constant(r.initial.get(body + ".h")))
        size = size if all(v is not None for v in size) else None
        start = (_constant(r.initial.get(body + ".x")), _constant(r.initial.get(body + ".y")))
        start = start if all(v is not None for v in start) else None
        (left, right), (top, bottom) = _bounds_of(r, body + ".x"), _bounds_of(r, body + ".y")
        # Along an axis her place never changes on, she is kept where she starts.
        if left is None and right is None and start is not None and not _delta(r, body + ".x"):
            left = right = start[0]
        if top is None and bottom is None and start is not None and not _delta(r, body + ".y"):
            top = bottom = start[1]
        kept_inside = (left, top, right, bottom) if None not in (left, top, right, bottom) else None
        pushes, _keys = _pushes(r, body)
        turns, drives, speeds = _turns(r, body)
        world.hers = HerBody(name=body, size=size, start=start, paces=paces, kept_inside=kept_inside,
                             pushes=pushes if len(pushes) > 1 else {}, lifts=lifts, pull=pull if lifts else None,
                             turns=turns, drives=drives, speeds=speeds, follows_pointer=_follows_the_pointer(r, body))
        world.read_at["hers"] = r.read_at.get(body + ".x", "") or r.read_at.get(body + ".y", "")
        for key in [*paces, *(k for k in pushes if k), *turns, *drives]:
            world.keys[key] = "moves"
        for key in lifts:
            world.keys[key] = "jumps"
        world.ground, where = _ground(r, body, size)
        if where:
            world.read_at["ground"] = where
        world.things = _things(r, body)
    world.clock_s, where = _clock(r)
    if where:
        world.read_at["clock"] = where
    lives = r.game_fields.get("lives")
    if lives is not None and _constant(lives.number()) is not None:
        world.lives = int(_constant(lives.number()))
    from core.perception.how_a_program_plays import _what_it_is

    for key, flag in r.flags.items():
        world.keys.setdefault(key, _what_it_is(flag) or "does something")
    return world
