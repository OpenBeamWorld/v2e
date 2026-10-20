# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Per-engine content dialects used by the compiler.

A dialect maps the compiler's engine-neutral ``v2e.package/1`` request to what one connector
loads (for example JBeam vehicle and prop definitions and level archives) and returns that
connector's compiled scene. :func:`build_compiled` finds the dialect for a connector: first a
plugin registered in the ``openbeam_v2e.dialect_packs`` entry point group under the
connector's name, then a built-in one. The first release ships the dialect of the ``fake``
test connector (inside that connector); dialects for engine connectors arrive with them.
"""

from __future__ import annotations

import importlib
import inspect
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Mapping

__all__ = ["ENTRY_POINT_GROUP", "BUILTIN_DIALECTS", "DialectUnavailable", "find_dialect", "build_compiled"]

ENTRY_POINT_GROUP = "openbeam_v2e.dialect_packs"
#: Connector name -> ``module:function`` taking ``(request_path, ..., world=...)``.
BUILTIN_DIALECTS: dict[str, str] = {"fake": "openbeam_v2e.backends.fake:compile_package"}


class DialectUnavailable(LookupError):
    """No dialect is installed for a connector."""


def _entry_points() -> list[Any]:
    try:
        found = metadata.entry_points()
    except Exception:  # noqa: BLE001 - a broken environment must not break compilation
        return []
    if hasattr(found, "select"):
        return list(found.select(group=ENTRY_POINT_GROUP))
    return list(found.get(ENTRY_POINT_GROUP, []))  # Python 3.10 / older importlib.metadata


def find_dialect(connector: str) -> Callable[..., Any]:
    """The dialect callable for ``connector`` (plugin first, then built-in)."""
    for ep in _entry_points():
        if ep.name == connector:
            return ep.load()
    target = BUILTIN_DIALECTS.get(connector)
    if target is None:
        known = sorted({ep.name for ep in _entry_points()} | set(BUILTIN_DIALECTS))
        raise DialectUnavailable(f"no dialect is installed for connector {connector!r} (installed: {', '.join(known)})")
    module_name, _, attr = target.partition(":")
    return getattr(importlib.import_module(module_name), attr)


def build_compiled(package: str | Path | Mapping[str, Any], connector: str, out: str | Path | None = None,
                   world: Any = None, scene: Mapping[str, Any] | None = None) -> Any:
    """Run ``connector``'s dialect on a ``v2e.package/1`` request (a path or a parsed document).

    ``world`` (a world model or directory) lets a dialect place reference points where the
    reconstruction's hulls have their origins; ``scene`` is passed through when a dialect wants
    it. The return value is the connector's compiled scene.
    """
    dialect = find_dialect(connector)
    offered = {"world": world, "scene": scene, "out": out}
    try:
        params = inspect.signature(dialect).parameters
    except (TypeError, ValueError):
        params = {}
    takes_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    kwargs = {k: v for k, v in offered.items() if v is not None and (takes_any or k in params)}
    return dialect(package, **kwargs)
