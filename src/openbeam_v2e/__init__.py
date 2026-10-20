# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""OpenBeam V2E (Video-to-Engine): a general framework for recreating filmed physical events.

V2E reconstructs a video of a physical event (camera, static geometry, moving objects with
measured dimensions and uncertainty), compiles that reconstruction into packages a physics
engine can load from authored content, runs it through a connector with actuator-only control,
exports the run in the same world format and scores it against the reconstruction.

Subpackages, in pipeline order:

* ``reconstruct`` and ``perception``: video -> world package with uncertainty, using pluggable
  perception models;
* ``fourdcb``: the world-package format;
* ``scene``: ``v2e.scene/1`` and ``v2e.templates/1``;
* ``generators`` and ``compile``: content and engine packages;
* ``controllers``: actor-scoped controller programs and reference driver models;
* ``backends``: connectors (OpenBeam and the fake test connector) behind one interface;
* ``run``: run directories, row format and export;
* ``score`` and ``search``: scoring and optional parameter search;
* ``media`` and ``bench``: media helpers and benchmark runners;
* ``agents``: the tool surface shared by the CLI and the MCP server.

This is a pre-alpha scaffold: the subpackages define the layout and contracts that the
implementation fills in.
"""

__version__ = "0.1.0.dev0"

__all__ = ["__version__"]
