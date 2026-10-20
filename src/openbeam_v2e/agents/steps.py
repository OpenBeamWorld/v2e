# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Where the agent tools find each pipeline step.

The agent surface (CLI and MCP server) calls the pipeline subpackages through one entry point per
step. A step is available when its subpackage exposes the function named here; otherwise the tool
answers ``step_unavailable`` and names what is missing, so an agent can tell an incomplete
installation from a failed run.

=============  ==============================================  ==================================
step           entry point                                     returns
=============  ==============================================  ==================================
reconstruct    ``openbeam_v2e.reconstruct.pipeline:reconstruct``  a JSON summary of the world package
compile        ``openbeam_v2e.compile:compile_scene``             writes a ``v2e.package/1`` request
dialect        ``openbeam_v2e.compile.dialects:build_compiled``   a ``CompiledScene`` for one connector
score          ``openbeam_v2e.score:score_run``                   a JSON score
export         ``openbeam_v2e.world.export:export_run``           the world package directory
synth          ``openbeam_v2e.bench.runner:run_bench``            a benchmark report
=============  ==============================================  ==================================

Entry points are called with keyword arguments; only the keywords the function accepts are
passed, so a step can grow optional parameters without breaking the agent surface. The tools
offer each input under the common spellings (for example ``out`` and ``out_dir``) and the entry
point takes the ones it declares.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any, Callable, Mapping

__all__ = ["STEP_ENTRY_POINTS", "StepUnavailable", "resolve", "available_steps", "accepts", "call_step"]

STEP_ENTRY_POINTS: Mapping[str, tuple[tuple[str, str], ...]] = {
    "reconstruct": (("openbeam_v2e.reconstruct.pipeline", "reconstruct"), ("openbeam_v2e.reconstruct", "reconstruct")),
    "compile": (("openbeam_v2e.compile", "compile_scene"), ("openbeam_v2e.compile.compiler", "compile_scene")),
    "dialect": (("openbeam_v2e.compile.dialects", "build_compiled"),),
    "score": (("openbeam_v2e.score", "score_run"), ("openbeam_v2e.score.compare", "score_run")),
    "export": (("openbeam_v2e.world.export", "export_run"), ("openbeam_v2e.run.runlog", "export_run")),
    "synth": (("openbeam_v2e.bench.runner", "run_bench"), ("openbeam_v2e.bench.synth", "run")),
}


class StepUnavailable(RuntimeError):
    def __init__(self, step: str) -> None:
        needs = " or ".join(f"{m}:{f}" for m, f in STEP_ENTRY_POINTS[step])
        super().__init__(f"the {step!r} step is not available in this installation (needs {needs})")
        self.step = step
        self.needs = needs


def resolve(step: str) -> Callable[..., Any] | None:
    for module_name, attr in STEP_ENTRY_POINTS[step]:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        fn = getattr(module, attr, None)
        if callable(fn):
            return fn
    return None


def available_steps() -> dict[str, bool]:
    return {step: resolve(step) is not None for step in STEP_ENTRY_POINTS}


def accepts(step: str, name: str) -> bool:
    """Whether the step's entry point declares keyword ``name`` (or takes ``**kwargs``)."""
    fn = resolve(step)
    if fn is None:
        return False
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def call_step(step: str, **kwargs: Any) -> Any:
    fn = resolve(step)
    if fn is None:
        raise StepUnavailable(step)
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fn(**kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(**kwargs)
    accepted = {k: v for k, v in kwargs.items() if k in params and v is not None}
    missing = [name for name, p in params.items()
               if p.default is inspect.Parameter.empty and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
               and name not in accepted]
    if missing:
        raise TypeError(f"{step}: the entry point needs {missing}, which the tool did not supply")
    return fn(**accepted)
