"""Parsed public response contracts for latent-cortex task execution.

Frontier tasks expose a small type DSL such as
``{"count":int,"witness":list[int]}``.  The contract contains no answer
values; it is safe to use inside candidate generation and branch arbitration.
This module parses that DSL once and validates JSON payload shape exactly.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, NoReturn

MAX_CONTRACT_BYTES = 4_096
MAX_CONTRACT_DEPTH = 12
MAX_CONTRACT_FIELDS = 128
MAX_VALIDATION_ERRORS = 16

_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]*\Z")


class ResponseContractError(ValueError):
    """Stable fail-closed response-contract error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> NoReturn:
    raise ResponseContractError(code)


@dataclass(frozen=True, slots=True)
class PrimitiveType:
    name: str


@dataclass(frozen=True, slots=True)
class LiteralType:
    values: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RationalStringType:
    """The registry's quoted ``p/q`` rational-string shape."""


@dataclass(frozen=True, slots=True)
class ListType:
    item: ContractType


@dataclass(frozen=True, slots=True)
class TupleType:
    items: tuple[ContractType, ...]


@dataclass(frozen=True, slots=True)
class ObjectType:
    fields: tuple[tuple[str, ContractType], ...]


ContractType = PrimitiveType | LiteralType | RationalStringType | ListType | TupleType | ObjectType


class _Parser:
    def __init__(self, source: str) -> None:
        if (
            not isinstance(source, str)
            or not source
            or len(source.encode("utf-8")) > MAX_CONTRACT_BYTES
            or "\x00" in source
        ):
            _fail("response_contract_invalid")
        self.source = source
        self.index = 0
        self.field_count = 0

    def parse(self) -> ContractType:
        result = self._type(depth=0)
        self._space()
        if self.index != len(self.source):
            _fail("response_contract_trailing_material")
        return result

    def _space(self) -> None:
        while self.index < len(self.source) and self.source[self.index].isspace():
            self.index += 1

    def _take(self, token: str) -> None:
        self._space()
        if not self.source.startswith(token, self.index):
            _fail("response_contract_syntax_invalid")
        self.index += len(token)

    def _string(self) -> str:
        self._space()
        if self.index >= len(self.source) or self.source[self.index] != '"':
            _fail("response_contract_string_expected")
        try:
            value, end = json.JSONDecoder().raw_decode(self.source, self.index)
        except json.JSONDecodeError as exc:
            raise ResponseContractError("response_contract_string_invalid") from exc
        if not isinstance(value, str) or not value:
            _fail("response_contract_string_invalid")
        self.index = end
        return value

    def _identifier(self) -> str:
        self._space()
        start = self.index
        while self.index < len(self.source) and (
            self.source[self.index].isalnum() or self.source[self.index] == "_"
        ):
            self.index += 1
        value = self.source[start : self.index]
        if _IDENTIFIER.fullmatch(value) is None:
            _fail("response_contract_identifier_invalid")
        return value

    def _type(self, *, depth: int) -> ContractType:
        if depth > MAX_CONTRACT_DEPTH:
            _fail("response_contract_too_deep")
        self._space()
        if self.index >= len(self.source):
            _fail("response_contract_type_missing")
        current = self.source[self.index]
        if current == "{":
            return self._object(depth=depth + 1)
        if current == "[":
            return self._tuple(depth=depth + 1)
        if current == '"':
            values = [self._string()]
            self._space()
            while self.index < len(self.source) and self.source[self.index] == "|":
                self.index += 1
                values.append(self._string())
                self._space()
            if len(set(values)) != len(values):
                _fail("response_contract_literal_duplicate")
            if values == ["p/q"]:
                return RationalStringType()
            return LiteralType(tuple(values))
        identifier = self._identifier()
        if identifier in {"int", "str", "bool"}:
            return PrimitiveType(identifier)
        if identifier == "list":
            self._take("[")
            item = self._type(depth=depth + 1)
            self._take("]")
            return ListType(item)
        _fail("response_contract_type_invalid")

    def _object(self, *, depth: int) -> ObjectType:
        self._take("{")
        fields: list[tuple[str, ContractType]] = []
        names: set[str] = set()
        self._space()
        if self.index < len(self.source) and self.source[self.index] == "}":
            self.index += 1
            return ObjectType(())
        while True:
            name = self._string()
            if name in names:
                _fail("response_contract_field_duplicate")
            names.add(name)
            self.field_count += 1
            if self.field_count > MAX_CONTRACT_FIELDS:
                _fail("response_contract_too_many_fields")
            self._take(":")
            fields.append((name, self._type(depth=depth)))
            self._space()
            if self.index < len(self.source) and self.source[self.index] == ",":
                self.index += 1
                continue
            self._take("}")
            return ObjectType(tuple(fields))

    def _tuple(self, *, depth: int) -> TupleType:
        self._take("[")
        items: list[ContractType] = []
        while True:
            items.append(self._type(depth=depth))
            self._space()
            if self.index < len(self.source) and self.source[self.index] == ",":
                self.index += 1
                continue
            self._take("]")
            if not items:
                _fail("response_contract_tuple_empty")
            return TupleType(tuple(items))


def parse_response_contract(source: str) -> ContractType:
    """Parse one complete response-contract DSL document."""

    return _Parser(source).parse()


def _validate(
    value: Any,
    contract: ContractType,
    *,
    path: str,
    errors: list[str],
) -> None:
    if len(errors) >= MAX_VALIDATION_ERRORS:
        return
    if isinstance(contract, PrimitiveType):
        valid = (
            (contract.name == "int" and type(value) is int)
            or (contract.name == "str" and isinstance(value, str))
            or (contract.name == "bool" and type(value) is bool)
        )
        if not valid:
            errors.append(f"{path}:expected_{contract.name}")
        return
    if isinstance(contract, LiteralType):
        if not isinstance(value, str) or value not in contract.values:
            errors.append(f"{path}:literal_mismatch")
        return
    if isinstance(contract, RationalStringType):
        if (
            not isinstance(value, str)
            or re.fullmatch(r"-?(?:0|[1-9][0-9]*)/[1-9][0-9]*", value) is None
        ):
            errors.append(f"{path}:expected_rational_string")
        return
    if isinstance(contract, ListType):
        if not isinstance(value, list):
            errors.append(f"{path}:expected_list")
            return
        for index, item in enumerate(value):
            _validate(item, contract.item, path=f"{path}[{index}]", errors=errors)
        return
    if isinstance(contract, TupleType):
        if not isinstance(value, list) or len(value) != len(contract.items):
            errors.append(f"{path}:tuple_shape_mismatch")
            return
        for index, (item, item_contract) in enumerate(zip(value, contract.items, strict=True)):
            _validate(item, item_contract, path=f"{path}[{index}]", errors=errors)
        return
    if not isinstance(value, dict):
        errors.append(f"{path}:expected_object")
        return
    expected = {name for name, _contract in contract.fields}
    observed = set(value)
    for name in sorted(expected - observed):
        errors.append(f"{path}.{name}:missing")
    for name in sorted(observed - expected):
        errors.append(f"{path}.{name}:unexpected")
    by_name = dict(contract.fields)
    for name in sorted(expected & observed):
        _validate(value[name], by_name[name], path=f"{path}.{name}", errors=errors)


def validate_response_payload(
    payload: Any,
    contract: str | ContractType,
) -> dict[str, Any]:
    """Validate one decoded JSON value against an exact public contract."""

    parsed = parse_response_contract(contract) if isinstance(contract, str) else contract
    errors: list[str] = []
    _validate(payload, parsed, path="$", errors=errors)
    return {
        "valid": not errors,
        "errors": errors,
        "error_count": len(errors),
    }


__all__ = [
    "ContractType",
    "ResponseContractError",
    "parse_response_contract",
    "validate_response_payload",
]
