# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Step 3: the compiler (scene + templates + content catalog -> engine packages).

Compiles a scene, its template mapping and a connector's content catalog into packages a
target engine can load: a level with collision geometry, vehicles and props built from
authored content, and a spawn plan in which every actor starts at rest at a pose. The
compiler is engine-free: :func:`compile_scene` writes an engine-neutral ``v2e.package/1``
request (level collision as OBJ files, content-origin records, spawns at rest, the frame-0
timeline and each controlled actor's reference path) or refuses with reasons; per-engine
details live in ``compile.dialects``.
"""

from openbeam_v2e.compile.compiler import CompileRefused, CompileResult, compile_files, compile_scene
from openbeam_v2e.compile.content import ContentRecord, Param, Refusal

__all__ = [
    "CompileRefused",
    "CompileResult",
    "ContentRecord",
    "Param",
    "Refusal",
    "compile_files",
    "compile_scene",
]
