# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""The OpenBeam connector.

Talks to an OpenBeam game (local or cloud session) only through the documented
customer-facing MCP tools and loads only authored content compiled by V2E. The set of
tools it may call is fixed in code (``PUBLIC_TOOLS``) and checked against the published
tool reference; ``run_script`` bodies must be renderings of audited snippets.

Modules: ``backend`` (the connector), ``tools`` (allow list and call guard), ``scripts``
(snippets and the engine-script host interface), ``transport`` (a standard-library MCP
client and the documented device-code sign-in) and ``cassette`` (record and replay for
offline tests).
"""

from openbeam_v2e.backends.openbeam.backend import OpenBeamBackend
from openbeam_v2e.backends.openbeam.tools import PUBLIC_TOOLS

__all__ = ["OpenBeamBackend", "PUBLIC_TOOLS"]
