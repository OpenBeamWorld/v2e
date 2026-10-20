# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Opt-in live checks against your own OpenBeam game or cloud session (read-only).

Run with ``python -m pytest -m openbeam`` and ``OPENBEAM_V2E_LIVE_CONFIG`` pointing at a connector
configuration file (``{"openbeam": {...}}``, see docs/openbeam.md). These checks call only
read-only documented tools (``status``, ``list_mods``, ``actor_poses``) through the connector's
guard; they install, spawn and step nothing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from openbeam_v2e.backends import registry
from openbeam_v2e.backends.openbeam.tools import GAME_TOOLS

pytestmark = pytest.mark.openbeam


@pytest.fixture(scope="module")
def connector():
    path = os.environ.get("OPENBEAM_V2E_LIVE_CONFIG")
    if not path:
        pytest.skip("set OPENBEAM_V2E_LIVE_CONFIG to a connector configuration file to run live checks")
    config = json.loads(Path(path).read_text(encoding="utf-8"))["openbeam"]
    backend = registry.create("openbeam", config)
    yield backend
    backend.close()


def test_status_answers(connector) -> None:
    status = connector.guard.call("status", {}).data
    assert status.get("app_state")


def test_documented_read_tools_answer_in_the_documented_shape(connector) -> None:
    assert isinstance(connector.guard.call("list_mods", {}).data.get("mods", []), list)
    for actor in connector.guard.call("actor_poses", {}).data.get("actors", []):
        assert {"instance_id", "position", "velocity"} <= set(actor)


def test_the_session_offers_every_game_tool_the_connector_uses(connector) -> None:
    client = connector.client
    listed = getattr(getattr(client, "mcp", None), "list_tools", None)
    if listed is None:
        pytest.skip("this transport cannot list tools")
    names = {tool.get("name") for tool in listed()}
    assert GAME_TOOLS <= names, f"missing: {sorted(GAME_TOOLS - names)}"
