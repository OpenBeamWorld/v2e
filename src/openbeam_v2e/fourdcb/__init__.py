# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""World packages in the 4DCodeBench format.

Reads, writes and checks the 4DCodeBench world-package format (camera, static meshes,
dynamic objects, render) in :mod:`.format`. It is an independent implementation written
from the published format description; it is not affiliated with or endorsed by the
format's authors.

V2E keeps its own data in sidecars under ``v2e/`` next to the base layout (frame rate, the
per-frame camera track, object classes with measured dimensions and sigma, rigid pose
tracks, uncertainty and events); :mod:`openbeam_v2e.world` reads and writes the whole
package, and :mod:`openbeam_v2e.schema.world` holds the sidecar schemas.
"""

from openbeam_v2e.fourdcb.format import (
    FormatError,
    MeshSequence,
    MotionArchive,
    Package,
    Violation,
    check_package,
    read_package,
    write_package,
)

__all__ = [
    "FormatError",
    "MeshSequence",
    "MotionArchive",
    "Package",
    "Violation",
    "check_package",
    "read_package",
    "write_package",
]
