"""Neuro-symbolic reasoning bridge for exact subproblems."""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SymbolicResult:
    ok: bool
    engine: str
    result: Any
    proof_trace: str

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "engine": self.engine, "result": str(self.result), "proof_trace": self.proof_trace}


def _safe_arith(expr: str) -> float | None:
    """Evaluate a pure-numeric arithmetic expression safely (no names/calls)."""
    try:
        tree = ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError):
        return None

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = ev(node.operand)
            return v if isinstance(node.op, ast.UAdd) else -v
        if isinstance(node, ast.BinOp):
            a, b = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.Div):
                return a / b if b != 0 else float("nan")
            if isinstance(node.op, ast.Pow):
                return a ** b
            if isinstance(node.op, ast.Mod):
                return a % b if b != 0 else float("nan")
        raise ValueError("unsupported expression")

    try:
        val = ev(tree)
        return val if val == val else None  # reject NaN
    except (ValueError, TypeError, ZeroDivisionError, OverflowError):
        return None


class SymbolicBridge:
    """Routes formalizable work to exact solvers when available."""

    def simplify_math(self, expression: str) -> SymbolicResult:
        try:
            import sympy as sp

            expr = sp.sympify(expression)
            simplified = sp.simplify(expr)
            return SymbolicResult(True, "sympy", simplified, f"sympy.simplify({expression!r})")
        except (ImportError, AttributeError, RuntimeError) as exc:
            return SymbolicResult(False, "sympy", repr(exc), "solver_error")

    def evaluate(self, expression: str) -> SymbolicResult:
        """Exactly evaluate a math expression (sympy) — no LLM guessing.

        Handles arithmetic, fractions, powers, roots, and constants symbolically, then
        gives a numeric value. The exact engine for tool-augmented reasoning.
        """
        try:
            import sympy as sp

            expr = sp.sympify(expression, evaluate=True)
            if expr.free_symbols:
                value: Any = expr               # symbolic — leave it exact
            else:
                # Exact value: keep integers/rationals clean; float only if irrational.
                value = expr if (expr.is_Integer or expr.is_Rational) else sp.N(expr)
            return SymbolicResult(True, "sympy", value, f"sympy.evaluate({expression!r}) = {value}")
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, SyntaxError) as exc:
            # Fall back to the sandboxed numeric evaluator for pure arithmetic.
            val = _safe_arith(str(expression))
            if val is not None:
                return SymbolicResult(True, "numeric_ast", val, f"numeric({expression!r})")
            return SymbolicResult(False, "sympy", repr(exc), "solver_error")

    def solve_equation(self, equation: str, symbol: str = "x") -> SymbolicResult:
        """Solve an equation/expression for a symbol, exactly (sympy)."""
        try:
            import sympy as sp

            sym = sp.Symbol(symbol)
            if "=" in equation:
                lhs, rhs = equation.split("=", 1)
                expr = sp.sympify(lhs) - sp.sympify(rhs)
            else:
                expr = sp.sympify(equation)
            roots = sp.solve(expr, sym)
            return SymbolicResult(True, "sympy", roots, f"sympy.solve({equation!r}, {symbol}) = {roots}")
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            return SymbolicResult(False, "sympy", repr(exc), "solver_error")

    def check_python_boolean(self, expression: str) -> SymbolicResult:
        try:
            tree = ast.parse(expression, mode="eval")
            value = _evaluate_boolean_ast(tree)
            return SymbolicResult(True, "python_ast", bool(value), "restricted_ast_eval")
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            return SymbolicResult(False, "python_ast", repr(exc), "solver_error")

    def check_arithmetic_claims(self, text: str) -> list[dict[str, Any]]:
        """Verify numeric ``expr = value`` claims in text; return the wrong ones.

        Conservative: only evaluates pure-numeric arithmetic (no variables), so it
        never mis-flags algebra or rhetorical "=" usage. Catches a confidently
        stated calculation error in Aura's own reasoning.
        """
        import re as _re

        errors: list[dict[str, Any]] = []
        # "<arithmetic> = <number>" with at least one operator on the left.
        pattern = _re.compile(r"(?<![\w.])((?:\d[\d,]*(?:\.\d+)?\s*[-+*/]\s*)+\d[\d,]*(?:\.\d+)?)\s*=\s*(-?\d[\d,]*(?:\.\d+)?)")
        for m in pattern.finditer(str(text or "")):
            lhs_raw, rhs_raw = m.group(1), m.group(2)
            lhs_val = _safe_arith(lhs_raw.replace(",", ""))
            if lhs_val is None:
                continue
            try:
                rhs_val = float(rhs_raw.replace(",", ""))
            except ValueError:
                continue
            if abs(lhs_val - rhs_val) > 1e-6 * max(1.0, abs(rhs_val)):
                errors.append({
                    "claim": m.group(0).strip(),
                    "stated": rhs_val,
                    "correct": lhs_val,
                })
        return errors

    def audit_reasoning(self, text: str) -> dict[str, Any]:
        """Active-reasoning gateway: route logic to the prover, arithmetic to sympy.

        The single live entry point that exercises the bridge on Aura's own output —
        deductive non-sequiturs via the natural-deduction prover and calculation
        errors via numeric evaluation. Returns both findings.
        """
        non_sequiturs: list[dict[str, Any]] = []
        try:
            from core.reasoning.inference_audit import find_non_sequiturs

            non_sequiturs = [v.to_dict() for v in find_non_sequiturs(text)]
        except (ImportError, ValueError, RuntimeError, TypeError, AttributeError):
            non_sequiturs = []
        arithmetic_errors = self.check_arithmetic_claims(text)
        return {
            "non_sequiturs": non_sequiturs,
            "arithmetic_errors": arithmetic_errors,
            "clean": not non_sequiturs and not arithmetic_errors,
        }

    def prove_logic(self, premises: list[str], goal: str) -> SymbolicResult:
        """Exact propositional deduction, kernel-checked (de Bruijn criterion).

        Routes a ``premises ⊢ goal`` query through :func:`prove_certified`: the
        tableau search produces a certificate and the independent proof kernel
        re-verifies it. A search-claimed proof the kernel rejects is **not**
        reported as proved — the bridge fails closed on unsoundness. The trace
        carries the kernel verdict and the axiom audit (premises actually used).
        """
        try:
            from core.reasoning.proof_kernel import prove_certified_text

            cp = prove_certified_text(premises, goal)
            proof = cp.proof
            if not proof.provable:
                return SymbolicResult(
                    True, "natural_deduction", False, f"countermodel: {proof.countermodel}"
                )
            if not cp.verified:
                reason = cp.verdict.reason if cp.verdict else "no certificate"
                return SymbolicResult(
                    False, "natural_deduction", "kernel_rejected", f"kernel refused proof: {reason}"
                )
            assert cp.verdict is not None
            trace = (
                " ; ".join(proof.trace)
                + f" ; kernel: verified ({cp.verdict.nodes} nodes)"
                + f" ; axioms: {list(cp.verdict.used_premises)}"
            )
            return SymbolicResult(True, "natural_deduction", True, trace)
        except (ValueError, RuntimeError, AttributeError, TypeError) as exc:
            return SymbolicResult(False, "natural_deduction", repr(exc), "solver_error")

    def solve_constraints(self, constraints: list[str]) -> SymbolicResult:
        try:
            import z3  # type: ignore

            solver = z3.Solver()
            names: dict[str, Any] = {}
            for raw in constraints:
                tree = ast.parse(raw, mode="eval")
                solver.add(_z3_from_ast(tree, names, z3))
            status = solver.check()
            return SymbolicResult(True, "z3", status, str(solver.model()) if status == z3.sat else str(status))
        except (ImportError, AttributeError, RuntimeError) as exc:
            return SymbolicResult(False, "z3", repr(exc), "solver_unavailable_or_error")


def _evaluate_boolean_ast(tree: ast.Expression) -> bool:
    def value(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return value(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (bool, int, float, str)):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not bool(value(node.operand))
        if isinstance(node, ast.BoolOp):
            vals = [bool(value(item)) for item in node.values]
            if isinstance(node.op, ast.And):
                return all(vals)
            if isinstance(node.op, ast.Or):
                return any(vals)
        if isinstance(node, ast.Compare):
            left = value(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                right = value(comparator)
                if isinstance(op, ast.Eq):
                    ok = left == right
                elif isinstance(op, ast.NotEq):
                    ok = left != right
                elif isinstance(op, ast.Lt):
                    ok = left < right
                elif isinstance(op, ast.LtE):
                    ok = left <= right
                elif isinstance(op, ast.Gt):
                    ok = left > right
                elif isinstance(op, ast.GtE):
                    ok = left >= right
                else:
                    raise ValueError(f"unsupported comparator: {type(op).__name__}")
                if not ok:
                    return False
                left = right
            return True
        raise ValueError(f"unsupported boolean AST node: {type(node).__name__}")

    return bool(value(tree))


def _z3_from_ast(tree: ast.Expression, names: dict[str, Any], z3: Any) -> Any:
    def value(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return value(node.body)
        if isinstance(node, ast.Name):
            return names.setdefault(node.id, z3.Real(node.id))
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool)):
            return node.value
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return -value(node.operand)
            if isinstance(node.op, ast.Not):
                return z3.Not(value(node.operand))
        if isinstance(node, ast.BinOp):
            left = value(node.left)
            right = value(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
        if isinstance(node, ast.BoolOp):
            values = [value(item) for item in node.values]
            if isinstance(node.op, ast.And):
                return z3.And(*values)
            if isinstance(node.op, ast.Or):
                return z3.Or(*values)
        if isinstance(node, ast.Compare):
            clauses = []
            left = value(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                right = value(comparator)
                if isinstance(op, ast.Eq):
                    clauses.append(left == right)
                elif isinstance(op, ast.NotEq):
                    clauses.append(left != right)
                elif isinstance(op, ast.Lt):
                    clauses.append(left < right)
                elif isinstance(op, ast.LtE):
                    clauses.append(left <= right)
                elif isinstance(op, ast.Gt):
                    clauses.append(left > right)
                elif isinstance(op, ast.GtE):
                    clauses.append(left >= right)
                else:
                    raise ValueError(f"unsupported comparator: {type(op).__name__}")
                left = right
            return z3.And(*clauses) if len(clauses) > 1 else clauses[0]
        raise ValueError(f"unsupported constraint AST node: {type(node).__name__}")

    return value(tree)


__all__ = ["SymbolicBridge", "SymbolicResult"]
