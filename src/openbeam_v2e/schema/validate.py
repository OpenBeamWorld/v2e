# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""A small JSON Schema validator for V2E documents (standard library only).

V2E publishes its document formats as JSON Schema (draft 2020-12 vocabulary) so that other
tools can validate them with any conforming library. The base install must not depend on a
third-party validator, so this module implements the subset of keywords the V2E schemas use:

``type``, ``enum``, ``const``, ``properties``, ``required``, ``additionalProperties``,
``patternProperties``, ``propertyNames``, ``minProperties``, ``items``, ``prefixItems``,
``minItems``, ``maxItems``, ``uniqueItems``, ``minimum``, ``maximum``, ``exclusiveMinimum``,
``exclusiveMaximum``, ``minLength``, ``maxLength``, ``pattern``, ``allOf``, ``anyOf``,
``oneOf``, ``not``, ``if``/``then``/``else`` and local ``$ref`` (``#/$defs/...``).

Unknown keywords are ignored, as the specification requires. Booleans are never numbers.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

__all__ = ["Issue", "SchemaError", "iter_issues", "validate", "check"]

_TYPE_CHECKS = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: (
        (isinstance(v, int) and not isinstance(v, bool))
        or (isinstance(v, float) and math.isfinite(v) and v.is_integer())
    ),
}


@dataclass(frozen=True)
class Issue:
    """One validation problem: a JSON pointer into the instance and a message."""

    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path or '/'}: {self.message}"


class SchemaError(ValueError):
    """Raised by :func:`check` when a document does not validate."""

    def __init__(self, issues: list[Issue], what: str = "document") -> None:
        self.issues = list(issues)
        lines = "\n".join(f"  {issue}" for issue in self.issues[:20])
        more = f"\n  ... and {len(self.issues) - 20} more" if len(self.issues) > 20 else ""
        super().__init__(f"{what} is invalid:\n{lines}{more}")


def _pointer(path: str, key: Any) -> str:
    token = str(key).replace("~", "~0").replace("/", "~1")
    return f"{path}/{token}"


def _resolve(ref: str, root: Mapping[str, Any]) -> Mapping[str, Any]:
    if not ref.startswith("#/"):
        raise ValueError(f"only local $ref values are supported, got {ref!r}")
    node: Any = root
    for raw in ref[2:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        node = node[token]
    return node


def _equal(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    return a == b


def iter_issues(instance: Any, schema: Mapping[str, Any] | bool, root: Mapping[str, Any] | None = None,
                path: str = "") -> Iterator[Issue]:
    """Yield every issue of ``instance`` against ``schema``."""
    if schema is True:
        return
    if schema is False:
        yield Issue(path, "no value is allowed here")
        return
    root = schema if root is None else root

    if "$ref" in schema:
        yield from iter_issues(instance, _resolve(schema["$ref"], root), root, path)

    expected = schema.get("type")
    if expected is not None:
        names = [expected] if isinstance(expected, str) else list(expected)
        if not any(_TYPE_CHECKS[name](instance) for name in names):
            yield Issue(path, f"expected {' or '.join(names)}, got {type(instance).__name__}")
            return

    if "enum" in schema and not any(_equal(instance, option) for option in schema["enum"]):
        yield Issue(path, f"must be one of {schema['enum']!r}, got {instance!r}")
    if "const" in schema and not _equal(instance, schema["const"]):
        yield Issue(path, f"must be {schema['const']!r}, got {instance!r}")

    if _TYPE_CHECKS["number"](instance):
        if "minimum" in schema and instance < schema["minimum"]:
            yield Issue(path, f"must be >= {schema['minimum']}, got {instance}")
        if "maximum" in schema and instance > schema["maximum"]:
            yield Issue(path, f"must be <= {schema['maximum']}, got {instance}")
        if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
            yield Issue(path, f"must be > {schema['exclusiveMinimum']}, got {instance}")
        if "exclusiveMaximum" in schema and instance >= schema["exclusiveMaximum"]:
            yield Issue(path, f"must be < {schema['exclusiveMaximum']}, got {instance}")
        if isinstance(instance, float) and not math.isfinite(instance):
            yield Issue(path, "must be a finite number")

    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            yield Issue(path, f"must have at least {schema['minLength']} characters")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            yield Issue(path, f"must have at most {schema['maxLength']} characters")
        if "pattern" in schema and re.search(schema["pattern"], instance) is None:
            yield Issue(path, f"{instance!r} does not match {schema['pattern']!r}")

    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            yield Issue(path, f"must have at least {schema['minItems']} items, got {len(instance)}")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            yield Issue(path, f"must have at most {schema['maxItems']} items, got {len(instance)}")
        if schema.get("uniqueItems"):
            seen: list[Any] = []
            for index, item in enumerate(instance):
                if any(_equal(item, other) for other in seen):
                    yield Issue(_pointer(path, index), "duplicates an earlier item")
                seen.append(item)
        prefix = schema.get("prefixItems", [])
        for index, (item, sub) in enumerate(zip(instance, prefix)):
            yield from iter_issues(item, sub, root, _pointer(path, index))
        if "items" in schema:
            for index in range(len(prefix), len(instance)):
                yield from iter_issues(instance[index], schema["items"], root, _pointer(path, index))

    if isinstance(instance, dict):
        for key in schema.get("required", []):
            if key not in instance:
                yield Issue(path, f"missing required member {key!r}")
        if "minProperties" in schema and len(instance) < schema["minProperties"]:
            yield Issue(path, f"must have at least {schema['minProperties']} members")
        properties = schema.get("properties", {})
        patterns = schema.get("patternProperties", {})
        names = schema.get("propertyNames")
        for key, value in instance.items():
            sub_path = _pointer(path, key)
            if names is not None:
                for issue in iter_issues(key, names, root, sub_path):
                    yield Issue(sub_path, f"member name: {issue.message}")
            matched = False
            if key in properties:
                matched = True
                yield from iter_issues(value, properties[key], root, sub_path)
            for pattern, sub in patterns.items():
                if re.search(pattern, key):
                    matched = True
                    yield from iter_issues(value, sub, root, sub_path)
            if not matched and "additionalProperties" in schema:
                extra = schema["additionalProperties"]
                if extra is False:
                    yield Issue(sub_path, "is not an allowed member")
                else:
                    yield from iter_issues(value, extra, root, sub_path)

    for sub in schema.get("allOf", []):
        yield from iter_issues(instance, sub, root, path)
    if "anyOf" in schema:
        if not any(not list(iter_issues(instance, sub, root, path)) for sub in schema["anyOf"]):
            yield Issue(path, "matches none of the allowed forms")
    if "oneOf" in schema:
        matches = sum(1 for sub in schema["oneOf"] if not list(iter_issues(instance, sub, root, path)))
        if matches != 1:
            yield Issue(path, f"must match exactly one allowed form, matches {matches}")
    if "not" in schema and not list(iter_issues(instance, schema["not"], root, path)):
        yield Issue(path, "matches a form that is not allowed")
    if "if" in schema:
        branch = "then" if not list(iter_issues(instance, schema["if"], root, path)) else "else"
        if branch in schema:
            yield from iter_issues(instance, schema[branch], root, path)


def validate(instance: Any, schema: Mapping[str, Any]) -> list[Issue]:
    """Return the list of issues (empty when ``instance`` is valid)."""
    return list(iter_issues(instance, schema))


def check(instance: Any, schema: Mapping[str, Any], what: str = "document") -> None:
    """Raise :class:`SchemaError` listing every issue when ``instance`` is invalid."""
    issues = validate(instance, schema)
    if issues:
        raise SchemaError(issues, what)
