"""AST-semantic rules for Aura's enterprise gate."""

from __future__ import annotations

import ast

try:
    from tools.aura_enterprise_contracts import (
        ALLOW_BLOCKING_SLEEP_IN_ASYNC,
        ALLOW_DYNAMIC_CODE,
        Finding,
        GateReport,
        is_production,
        subprocess_must_use_gateway,
    )
except ModuleNotFoundError as exc:
    if exc.name not in {"tools", "tools.aura_enterprise_contracts"}:
        raise
    from aura_enterprise_contracts import (
        ALLOW_BLOCKING_SLEEP_IN_ASYNC,
        ALLOW_DYNAMIC_CODE,
        Finding,
        GateReport,
        is_production,
        subprocess_must_use_gateway,
    )

def dotted_call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    parts: list[str] = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
        return ".".join(reversed(parts))
    return ""


def body_without_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.stmt]:
    body = list(node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(getattr(body[0], "value", None), ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def decorator_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parts = [node.attr]
        value = node.value
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            parts.append(value.id)
        return ".".join(reversed(parts))
    if isinstance(node, ast.Call):
        return decorator_name(node.func)
    return ""


def is_abstract_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    names = {decorator_name(item) for item in node.decorator_list}
    return bool(
        names
        & {"abstractmethod", "abc.abstractmethod", "abstractclassmethod", "abstractstaticmethod"}
    )


def is_deliberate_constructor_override(
    node: ast.FunctionDef | ast.AsyncFunctionDef, enclosing: ast.AST | None
) -> bool:
    """A no-op __init__ on a class that has real methods is intentional.

    A test double overrides the constructor so the real one does not run, and
    ``pass`` is the correct implementation of "do not set anything up". All
    three pass_only_function findings in this repo were exactly that.

    Judged by SHAPE, not by path: the class must define at least one other
    method with a real body. A class that is nothing but a pass-only __init__
    is still unimplemented scaffolding and is still reported — which is what
    keeps this from being "skip tests/" wearing a better name.
    """
    if node.name != "__init__":
        return False
    if not isinstance(enclosing, ast.ClassDef):
        return False
    for item in enclosing.body:
        if item is node or not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        sibling_body = body_without_docstring(item)
        if sibling_body and not (
            len(sibling_body) == 1 and isinstance(sibling_body[0], ast.Pass)
        ):
            return True
    return False


def is_not_implemented_only(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    body = body_without_docstring(node)
    if len(body) != 1 or not isinstance(body[0], ast.Raise):
        return False
    exc = body[0].exc
    if isinstance(exc, ast.Call):
        exc = exc.func
    return isinstance(exc, ast.Name) and exc.id == "NotImplementedError"


# Pickle/serialization guards are a legitimate raise-only idiom: a dunder that
# raises to declare "this live-runtime object is not serializable identity"
# (__getstate__/__setstate__/__reduce__/__reduce_ex__/__deepcopy__). These are
# intentional protection, not unimplemented debt.
_SERIALIZATION_GUARD_DUNDERS = frozenset({
    "__getstate__", "__setstate__", "__reduce__", "__reduce_ex__",
    "__deepcopy__", "__copy__",
})


def is_serialization_guard(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    if node.name not in _SERIALIZATION_GUARD_DUNDERS:
        return False
    body = body_without_docstring(node)
    if len(body) != 1 or not isinstance(body[0], ast.Raise):
        return False
    exc = body[0].exc
    if isinstance(exc, ast.Call):
        exc = exc.func
    return isinstance(exc, ast.Name) and exc.id in {"TypeError", "RuntimeError", "PicklingError"}


def raised_exception_name(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    """The name of the single exception a raise-only function raises."""
    body = body_without_docstring(node)
    if len(body) != 1 or not isinstance(body[0], ast.Raise):
        return None
    exc = body[0].exc
    if exc is None:
        return "<bare>"
    if isinstance(exc, ast.Call):
        exc = exc.func
    if isinstance(exc, ast.Name):
        return exc.id
    if isinstance(exc, ast.Attribute):
        return exc.attr
    return None


def is_deliberate_refusal(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """A raise-only function that refuses on purpose, rather than one nobody
    finished writing.

    The rule read "in product code a raise-only helper is dead scaffolding",
    and that premise was wrong here. It reported 118 functions and not one
    was unwritten: 104 were ``_fail(code)`` helpers, the codebase's standard
    way to fail closed with a named, greppable code, and the rest were
    ``reject_constant`` hooks handed to json.loads and protocol methods that
    exist to refuse a direct call. A rule that fires 118 times for the
    discipline it is supposed to protect gets read as noise.

    Two things say "deliberate", both machine-checked and neither writable by
    accident:

    * ``-> Never`` / ``-> NoReturn``. The annotation IS the contract, and a
      type checker enforces that no caller reads a return value. 103 of the
      118 already carried it; the other twelve said ``-> None`` while always
      raising, which is wrong type information, and they are fixed.
    * The exception raised is a NAMED type — a domain error, ValueError,
      AssertionError. Choosing which failure this is takes a decision.

    What stays a finding is what the rule was always after: a non-abstract
    function whose whole body is ``raise NotImplementedError`` (declared and
    unwritten) or a bare ``raise`` outside a handler.
    """
    returns = node.returns
    if isinstance(returns, ast.Name) and returns.id in {"Never", "NoReturn"}:
        return True
    if isinstance(returns, ast.Attribute) and returns.attr in {"Never", "NoReturn"}:
        return True
    raised = raised_exception_name(node)
    if raised is None or raised == "<bare>":
        return False
    return raised != "NotImplementedError"


_LOOP_STATEMENTS = (ast.While, ast.For, ast.AsyncFor)
_NESTED_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def loop_can_end(node: ast.While) -> bool:
    """Does this ``while True`` have any way out?

    The rule flagged the ``while True:`` line itself, which is not a defect —
    it is the standard Python idiom for a loop whose exit condition is
    computed inside the body. 30 of the 31 findings had a ``break`` or a
    ``return`` a few lines down. Counting those trains people to stop reading
    the list.

    A loop with none of these cannot end:

    * a ``break`` whose nearest enclosing loop is THIS one — a break inside a
      nested ``for`` leaves the inner loop and is not an exit from this one;
    * a ``return`` or ``raise``, which leave the function entirely;
    * an ``await``, which is a cancellation point: that is how every service
      loop in this runtime is actually stopped;
    * a ``yield``, which hands control to the consumer for the same reason —
      a generator does not run unless somebody pulls from it.

    Bodies of nested functions, lambdas and classes are not searched. Their
    control flow is their own and says nothing about this loop.
    """
    found = False

    def search(nodes: list[ast.AST], *, own_loop: bool) -> None:
        nonlocal found
        for child in nodes:
            if found:
                return
            if isinstance(child, _NESTED_SCOPES):
                continue
            if isinstance(child, (ast.Return, ast.Raise, ast.Await, ast.Yield, ast.YieldFrom)):
                found = True
                return
            if isinstance(child, ast.Break) and own_loop:
                found = True
                return
            inner_own_loop = own_loop and not isinstance(child, _LOOP_STATEMENTS)
            for name in ("body", "orelse", "finalbody", "handlers", "cases"):
                block = getattr(child, name, None)
                if isinstance(block, list):
                    search(list(block), own_loop=inner_own_loop)
            for name in ("value", "test", "iter", "func"):
                sub = getattr(child, name, None)
                if isinstance(sub, ast.AST):
                    search([sub], own_loop=inner_own_loop)
            if isinstance(child, ast.expr):
                search(list(ast.iter_child_nodes(child)), own_loop=inner_own_loop)

    search(list(node.body), own_loop=True)
    return found


def handler_answers_with_a_value(handler: ast.ExceptHandler) -> bool:
    """Did the handler turn the exception into an answer?

    ``except Exception: return False`` in a verification predicate is not a
    swallow — the caller is told the verification did not hold, and told it
    in the only vocabulary the predicate has. ``except Exception: pass`` is a
    swallow: control resumes as though nothing happened and nobody is told
    anything.

    Thirteen of this rule's twenty-three findings were the first kind.
    """
    return any(
        isinstance(item, ast.Return) and item.value is not None for item in handler.body
    )


#: The sanctioned degradation protocol. CLAUDE.md states the rule this gate is
#: enforcing a corner of: "never a silent ``except: pass``" — record the
#: degradation with a subsystem and an action instead.
_DEGRADATION_RECORDERS = frozenset({"record_degradation", "_record_degradation"})


def handler_records_a_degradation(handler: ast.ExceptHandler) -> bool:
    """Did the handler report the failure through the runtime's own protocol?

    ``# noqa: BLE001`` says a human looked at this handler once. A call to
    ``record_degradation(subsystem, exc, action=...)`` says the same thing and
    then keeps saying it at runtime: the record lands in
    ``runtime_health_report()["integrity"]``, opens an incident, and escalates
    to CRITICAL for every module on the fail-closed list. One is a comment that
    cannot be wrong because it cannot be checked; the other is evidence that
    the boundary actually fired, in production, on real inputs.

    So a handler that records a degradation is reviewed by the stronger of the
    two mechanisms, and the gate should say so. What stays reported is the
    handler that does none of the three: does not re-raise, does not record,
    and carries no annotation — which continues past a failure nobody
    enumerated and tells nobody it happened.
    """
    for node in ast.walk(handler):
        if (
            isinstance(node, ast.Call)
            and dotted_call_name(node).rsplit(".", 1)[-1] in _DEGRADATION_RECORDERS
        ):
            return True
    return False


def handler_always_reraises(handler: ast.ExceptHandler) -> bool:
    """Does control leave this handler only by the exception path?

    ``except Exception: rollback(); raise`` catches broadly on purpose. The
    breadth decides WHICH failures get cleaned up — and the answer should be
    "all of them", because a KeyError between two SQL statements leaves the
    transaction just as open as an sqlite3.Error does. The exception then
    propagates unchanged: the caller learns everything it would have learned
    without the handler, and nothing has been decided on its behalf.

    That is not the debt this rule exists to surface. The debt is
    ``except Exception: logger.warning(...)`` with no re-raise — a handler that
    DECIDES to continue, across a set of failures nobody enumerated, including
    the ones nobody thought of. Breadth is dangerous exactly when it is paired
    with a decision, and harmless when it is paired with cleanup.

    Ninety of this rule's one hundred and fifty-four findings re-raise.

    The check is deliberately conservative: the last top-level statement must
    be a ``raise``, so a handler that re-raises on one branch and falls through
    on another still counts as a decision and is still reported.
    """
    if not handler.body:
        return False
    return isinstance(handler.body[-1], ast.Raise)


class AstGate(ast.NodeVisitor):
    def __init__(self, rel: str, report: GateReport, source_lines: list[str] | None = None):
        self.rel = rel
        self.report = report
        self.async_depth = 0
        self.source_lines = source_lines or []
        #: Innermost enclosing scope, so a no-op ``__init__`` can be told from
        #: unimplemented scaffolding by looking at its class. This used to be
        #: an id -> parent map filled by overriding ``visit``, which walked
        #: every one of the repo's 9.2M nodes a second time and cost the gate
        #: about a third of its running time. A ``None`` is pushed for a
        #: function so a nested def does not inherit the class above it.
        self._scopes: list[ast.AST | None] = []
        #: Depth inside a ``__del__``. A finalizer runs while the interpreter
        #: is tearing modules out from under it, so recording a degradation
        #: there can fail on the way to reporting the failure. Silence is the
        #: correct behaviour, and the one place it is.
        self._finalizer_depth = 0

    @property
    def _in_finalizer(self) -> bool:
        return self._finalizer_depth > 0

    @property
    def _enclosing(self) -> ast.AST | None:
        return self._scopes[-1] if self._scopes else None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scopes.append(node)
        self.generic_visit(node)
        self._scopes.pop()

    def _line_has_marker(self, node: ast.AST, marker: str) -> bool:
        lineno = int(getattr(node, "lineno", 0) or 0)
        if 0 < lineno <= len(self.source_lines):
            return marker in self.source_lines[lineno - 1]
        return False

    def _line_has_reviewed_broad_except(self, node: ast.AST) -> bool:
        """True when the handler line carries an explicit BLE001 review marker.

        `# noqa: BLE001` is the ecosystem-standard annotation for a broad
        except that a human reviewed and justified (last-resort floors,
        liveness paths). The gate's job is surfacing UNREVIEWED debt.
        """
        return self._line_has_marker(node, "noqa: BLE001")

    def _line_has_reviewed_dynamic_exec(self, node: ast.AST) -> bool:
        """True when the call line carries an explicit S102 review marker.

        Same principle as BLE001 above, for the same reason: the gate exists to
        surface UNREVIEWED debt, and `# noqa: S102` is the ecosystem-standard
        annotation for an exec/eval/compile a human reviewed.

        This is deliberately per-line rather than another ALLOW_DYNAMIC_CODE
        entry: allowlisting a whole file also blesses every exec added to it
        later, which is precisely the debt this gate is meant to catch.
        """
        return self._line_has_marker(node, "noqa: S102")

    def add(self, severity: str, kind: str, node: ast.AST, detail: str = "") -> None:
        self.report.findings.append(
            Finding(severity, kind, self.rel, getattr(node, "lineno", 0), detail)
        )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if any(alias.name == "*" for alias in node.names):
            self.add("medium", "wildcard_import", node, node.module or "")
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        broad = node.type is None or (
            isinstance(node.type, ast.Name) and node.type.id in {"BaseException", "Exception"}
        )
        if broad:
            severity = "high" if is_production(self.rel) else "medium"
            if node.type is None:
                self.add(severity, "bare_except", node)
            elif (
                any(isinstance(item, ast.Pass) for item in node.body)
                or all(
                    isinstance(item, (ast.Break, ast.Continue, ast.Pass, ast.Return))
                    for item in node.body
                )
            ) and not (
                handler_answers_with_a_value(node) or self._in_finalizer
            ):
                # A silent swallow is debt even when annotated; a swallow that
                # at least logs (non-trivial body) may be a reviewed floor.
                self.add(severity, "swallowed_broad_exception", node)
            elif not (
                self._line_has_reviewed_broad_except(node)
                or handler_always_reraises(node)
                or handler_records_a_degradation(node)
            ):
                self.add(
                    "medium" if is_production(self.rel) else "low",
                    "broad_exception_review",
                    node,
                )
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        if (
            isinstance(node.test, ast.Constant)
            and node.test.value is True
            and not loop_can_end(node)
        ):
            self.add("medium" if is_production(self.rel) else "low", "unbounded_loop_review", node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        body = body_without_docstring(node)
        if (
            len(body) == 1
            and isinstance(body[0], ast.Pass)
            and not is_abstract_function(node)
            and not is_deliberate_constructor_override(node, self._enclosing)
        ):
            self.add(
                "high" if is_production(self.rel) else "medium",
                "pass_only_function",
                node,
                node.name,
            )
        if (
            len(body) == 1
            and isinstance(body[0], ast.Raise)
            and not (is_abstract_function(node) and is_not_implemented_only(node))
            and not is_serialization_guard(node)
            and not is_deliberate_refusal(node)
            and not self.rel.startswith("tests/")
        ):
            self.add(
                "high" if is_production(self.rel) else "medium",
                "raise_only_function",
                node,
                node.name,
            )
        self.async_depth += 1
        self._scopes.append(None)
        self.generic_visit(node)
        self._scopes.pop()
        self.async_depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        body = body_without_docstring(node)
        if (
            len(body) == 1
            and isinstance(body[0], ast.Pass)
            and not is_abstract_function(node)
            and not is_deliberate_constructor_override(node, self._enclosing)
        ):
            self.add(
                "high" if is_production(self.rel) else "medium",
                "pass_only_function",
                node,
                node.name,
            )
        if (
            len(body) == 1
            and isinstance(body[0], ast.Raise)
            and not (is_abstract_function(node) and is_not_implemented_only(node))
            and not is_serialization_guard(node)
            and not is_deliberate_refusal(node)
            and not self.rel.startswith("tests/")
        ):
            self.add(
                "high" if is_production(self.rel) else "medium",
                "raise_only_function",
                node,
                node.name,
            )
        previous_async_depth = self.async_depth
        self.async_depth = 0
        self._scopes.append(None)
        if node.name == "__del__":
            self._finalizer_depth += 1
        try:
            self.generic_visit(node)
        finally:
            if node.name == "__del__":
                self._finalizer_depth -= 1
            self._scopes.pop()
            self.async_depth = previous_async_depth

    def visit_Call(self, node: ast.Call) -> None:
        name = dotted_call_name(node)
        if (
            name in {"compile", "eval", "exec"}
            and self.rel not in ALLOW_DYNAMIC_CODE
            and not self._line_has_reviewed_dynamic_exec(node)
        ):
            self.add(
                "critical" if is_production(self.rel) else "medium",
                "dynamic_code_execution",
                node,
                name,
            )
        if name in {
            "os.system",
            "subprocess.Popen",
            "subprocess.call",
            "subprocess.check_call",
            "subprocess.check_output",
            "subprocess.run",
        }:
            shell_true = any(
                keyword.arg == "shell"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in node.keywords
            )
            if shell_true:
                self.add("critical", "subprocess_shell_true", node, name)
            elif subprocess_must_use_gateway(self.rel) and not self._line_has_marker(
                node, "noqa: S603"
            ):
                # Per-line, like the exec and broad-except markers above, and
                # for the same reason: blessing a whole file also blesses
                # every spawn added to it later.
                self.add("high", "subprocess_usage_review", node, name)
        if name in {"dill.load", "dill.loads", "pickle.load", "pickle.loads"}:
            self.add(
                "critical" if is_production(self.rel) else "high",
                "unsafe_deserialization",
                node,
                name,
            )
        if (
            name == "time.sleep"
            and self.async_depth
            and self.rel not in ALLOW_BLOCKING_SLEEP_IN_ASYNC
        ):
            self.add("high", "blocking_sleep_in_async", node)
        self.generic_visit(node)

__all__ = [
    "AstGate",
    "body_without_docstring",
    "decorator_name",
    "dotted_call_name",
    "handler_always_reraises",
    "handler_answers_with_a_value",
    "handler_records_a_degradation",
    "is_abstract_function",
    "is_deliberate_constructor_override",
    "is_deliberate_refusal",
    "is_not_implemented_only",
    "is_serialization_guard",
    "loop_can_end",
    "raised_exception_name",
]
