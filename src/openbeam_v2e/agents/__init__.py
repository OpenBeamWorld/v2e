# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Step 7: using V2E from AI agents.

One tool surface for agents and scripts: JSON-schema tool specifications (``reconstruct``,
``compile``, ``run``, ``export``, ``score``, ``inspect``, ``bench``) in :mod:`.tools`, shared by
the ``openbeam-v2e`` command line and the stdio MCP server in :mod:`.mcp_server`, so any
MCP-capable model can drive the pipeline. The server uses only the standard library. Pipeline
steps owned by other subpackages are found through the entry points listed in :mod:`.steps`.
"""
