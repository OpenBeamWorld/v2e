# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Find connectors by name: the built-in ones and those registered by plugins.

Built in: ``fake`` (the offline test double) and ``openbeam`` (documented customer MCP tools).
A plugin registers another connector with an entry point in the ``openbeam_v2e.backends`` group
whose value is a ``Backend`` subclass with a ``from_config(config)`` class method::

    [project.entry-points."openbeam_v2e.backends"]
    myengine = "my_package.connector:MyEngineBackend"

Configuration (endpoints, credentials files) is passed when a connector is created, never through
a scene or a tool argument, so an agent driving V2E cannot point a connector somewhere new.
"""

from __future__ import annotations

import importlib
from importlib import metadata
from typing import Any, Mapping

from openbeam_v2e.backends.base import Backend, BackendError

__all__ = ["BUILTIN", "ENTRY_POINT_GROUP", "available", "backend_class", "create", "audit_policy_for"]

ENTRY_POINT_GROUP = "openbeam_v2e.backends"
BUILTIN = {
    "fake": "openbeam_v2e.backends.fake:FakeBackend",
    "openbeam": "openbeam_v2e.backends.openbeam.backend:OpenBeamBackend",
}


def _plugin_entry_points() -> dict[str, Any]:
    try:
        eps = metadata.entry_points(group=ENTRY_POINT_GROUP)
    except TypeError:   # pragma: no cover - Python < 3.10 API shape
        eps = metadata.entry_points().get(ENTRY_POINT_GROUP, [])
    return {ep.name: ep for ep in eps}


def available() -> dict[str, str]:
    """Connector name -> ``module:Class`` for every built-in and plugin connector."""
    names = dict(BUILTIN)
    for name, ep in _plugin_entry_points().items():
        names.setdefault(name, ep.value)
    return names


def backend_class(name: str) -> type[Backend]:
    if name in BUILTIN:
        module, _, attr = BUILTIN[name].partition(":")
        cls = getattr(importlib.import_module(module), attr)
    else:
        ep = _plugin_entry_points().get(name)
        if ep is None:
            raise BackendError(f"unknown connector {name!r}; available: {sorted(available())}")
        cls = ep.load()
    if not (isinstance(cls, type) and issubclass(cls, Backend)):
        raise BackendError(f"connector {name!r} does not provide a Backend subclass")
    return cls


def create(name: str, config: Mapping[str, Any] | None = None) -> Backend:
    """Create connector ``name`` from its configuration mapping."""
    cls = backend_class(name)
    factory = getattr(cls, "from_config", None)
    if callable(factory):
        return factory(dict(config or {}))
    return cls(**dict(config or {}))


def audit_policy_for(name: str) -> Any:
    return backend_class(name).audit_policy()
