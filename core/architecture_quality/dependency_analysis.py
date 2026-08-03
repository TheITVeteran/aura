"""Context-aware Python import extraction for architecture scoring."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

DependencyKind = Literal["runtime", "type_only", "optional", "conditional", "deferred"]


@dataclass(frozen=True)
class ImportInventory:
    """Immutable import evidence extracted from one module."""

    runtime: frozenset[str]
    type_only: frozenset[str]
    optional: frozenset[str]
    conditional: frozenset[str]
    deferred: frozenset[str]
    dynamic: frozenset[str]
    required_runtime_modules: frozenset[str]
    runtime_from_aliases: frozenset[tuple[str, str]]
    unresolved_dynamic: int
    invalid_relative: int


@dataclass
class _MutableImportInventory:
    runtime: set[str] = field(default_factory=set)
    type_only: set[str] = field(default_factory=set)
    optional: set[str] = field(default_factory=set)
    conditional: set[str] = field(default_factory=set)
    deferred: set[str] = field(default_factory=set)
    dynamic: set[str] = field(default_factory=set)
    required_runtime_modules: set[str] = field(default_factory=set)
    runtime_from_aliases: set[tuple[str, str]] = field(default_factory=set)
    unresolved_dynamic: int = 0
    invalid_relative: int = 0

    def bucket(self, kind: DependencyKind) -> set[str]:
        return getattr(self, kind)

    def freeze(self) -> ImportInventory:
        return ImportInventory(
            runtime=frozenset(self.runtime),
            type_only=frozenset(self.type_only),
            optional=frozenset(self.optional),
            conditional=frozenset(self.conditional),
            deferred=frozenset(self.deferred),
            dynamic=frozenset(self.dynamic),
            required_runtime_modules=frozenset(self.required_runtime_modules),
            runtime_from_aliases=frozenset(self.runtime_from_aliases),
            unresolved_dynamic=self.unresolved_dynamic,
            invalid_relative=self.invalid_relative,
        )


class _DependencyVisitor(ast.NodeVisitor):
    def __init__(self, current_module: str, *, current_is_package: bool) -> None:
        self.current_module = current_module
        self.current_is_package = current_is_package
        self.inventory = _MutableImportInventory()
        self._contexts: list[DependencyKind] = ["runtime"]
        self._dynamic_modules: dict[str, str] = {}
        self._dynamic_names: set[str] = {"__import__"}

    @property
    def kind(self) -> DependencyKind:
        return self._contexts[-1]

    def _visit_under(self, kind: DependencyKind, nodes: Iterable[ast.AST]) -> None:
        self._contexts.append(kind)
        try:
            for node in nodes:
                self.visit(node)
        finally:
            self._contexts.pop()

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        self.inventory.bucket(self.kind).update(alias.name for alias in node.names)
        if self.kind == "runtime":
            self.inventory.required_runtime_modules.update(alias.name for alias in node.names)
        for alias in node.names:
            if alias.name in {"importlib", "pydoc", "pkgutil"}:
                self._dynamic_modules[alias.asname or alias.name] = alias.name

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        base = _resolve_import_from_base(
            node,
            self.current_module,
            current_is_package=self.current_is_package,
        )
        if not base:
            if node.level > 0:
                self.inventory.invalid_relative += 1
            return
        bucket = self.inventory.bucket(self.kind)
        bucket.add(base)
        bucket.update(f"{base}.{alias.name}" for alias in node.names if alias.name != "*")
        if self.kind == "runtime":
            self.inventory.required_runtime_modules.add(base)
            self.inventory.runtime_from_aliases.update(
                (base, alias.name) for alias in node.names if alias.name != "*"
            )
        if node.level == 0 and node.module in {"importlib", "pydoc", "pkgutil"}:
            expected = {
                "importlib": "import_module",
                "pydoc": "locate",
                "pkgutil": "resolve_name",
            }[node.module]
            for alias in node.names:
                if alias.name == expected:
                    self._dynamic_names.add(alias.asname or alias.name)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if self._is_dynamic_import_call(node.func):
            if (
                node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                target = node.args[0].value.strip()
                if target:
                    self.inventory.bucket(self.kind).add(target)
                    self.inventory.dynamic.add(target)
                    if self.kind == "runtime":
                        self.inventory.required_runtime_modules.add(target)
                else:
                    self.inventory.unresolved_dynamic += 1
            else:
                self.inventory.unresolved_dynamic += 1
        self.generic_visit(node)

    def _is_dynamic_import_call(self, func: ast.expr) -> bool:
        if isinstance(func, ast.Name):
            return func.id in self._dynamic_names
        if not isinstance(func, ast.Attribute) or not isinstance(func.value, ast.Name):
            return False
        module = self._dynamic_modules.get(func.value.id)
        return (module, func.attr) in {
            ("importlib", "import_module"),
            ("pydoc", "locate"),
            ("pkgutil", "resolve_name"),
        }

    def visit_If(self, node: ast.If) -> None:  # noqa: N802
        if _is_type_checking_guard(node.test):
            self._visit_under("type_only", node.body)
            self._visit_under(self.kind, node.orelse)
            return
        if _is_platform_guard(node.test):
            self.visit(node.test)
            self._visit_under("conditional", node.body)
            self._visit_under("conditional", node.orelse)
            return
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:  # noqa: N802
        if _handles_import_error(node.handlers):
            self._visit_under("optional", node.body)
            for handler in node.handlers:
                self._visit_under(self.kind, handler.body)
            self._visit_under(self.kind, node.orelse)
            self._visit_under(self.kind, node.finalbody)
            return
        self.generic_visit(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)
        saved_names = set(self._dynamic_names)
        parameters = {
            argument.arg
            for argument in (
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            )
        }
        if node.args.vararg:
            parameters.add(node.args.vararg.arg)
        if node.args.kwarg:
            parameters.add(node.args.kwarg.arg)
        self._dynamic_names.difference_update(parameters)
        try:
            self._visit_under("deferred", node.body)
        finally:
            self._dynamic_names = saved_names
        if self.kind == "runtime":
            self._dynamic_names.discard(node.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._visit_function(node)


def extract_import_inventory(
    tree: ast.AST,
    current_module: str,
    *,
    current_is_package: bool,
) -> ImportInventory:
    visitor = _DependencyVisitor(current_module, current_is_package=current_is_package)
    visitor.visit(tree)
    return visitor.inventory.freeze()


def _resolve_import_from_base(
    node: ast.ImportFrom,
    current_module: str,
    *,
    current_is_package: bool = False,
) -> str | None:
    if node.level <= 0:
        return node.module
    package_parts = current_module.split(".") if current_is_package else current_module.split(".")[:-1]
    if node.level > len(package_parts):
        return None
    base_parts = package_parts[: len(package_parts) - (node.level - 1)]
    if node.module:
        base_parts.extend(node.module.split("."))
    return ".".join(part for part in base_parts if part)


def _is_type_checking_guard(node: ast.AST) -> bool:
    return (isinstance(node, ast.Name) and node.id == "TYPE_CHECKING") or (
        isinstance(node, ast.Attribute)
        and node.attr == "TYPE_CHECKING"
        and isinstance(node.value, ast.Name)
        and node.value.id == "typing"
    )


def _is_platform_guard(node: ast.AST) -> bool:
    for item in ast.walk(node):
        if isinstance(item, ast.Attribute) and isinstance(item.value, ast.Name):
            if (item.value.id, item.attr) in {
                ("sys", "platform"),
                ("os", "name"),
                ("platform", "system"),
                ("platform", "machine"),
            }:
                return True
        if isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute):
            if isinstance(item.func.value, ast.Name) and item.func.value.id == "platform":
                return True
    return False


def _handles_import_error(handlers: Iterable[ast.ExceptHandler]) -> bool:
    names: set[str] = set()
    for handler in handlers:
        if handler.type is None:
            return True
        for item in ast.walk(handler.type):
            if isinstance(item, ast.Name):
                names.add(item.id)
    return bool(names & {"ImportError", "ModuleNotFoundError"})
