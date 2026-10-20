# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""The V2E tools for agents and scripts: one set of JSON-schema tools shared by the CLI and MCP.

Tools: ``reconstruct``, ``compile``, ``run``, ``export``, ``score``, ``inspect`` and ``bench``.
Each takes a JSON object that is validated against its ``input_schema`` and returns a JSON object.
Failures come back as :class:`ToolFailure` with a short machine-readable ``code``
(``bad_arguments``, ``refused``, ``step_unavailable``, ``connector_not_configured``, ``failed``).

Safety rules the tools enforce:

* **Paths stay in the workspace.** With a sandboxed context (the MCP server's default), every path
  argument is resolved inside the workspace root; absolute paths elsewhere and ``..`` escapes are
  refused. Output directories must be new or empty.
* **Connectors come from configuration.** A tool can name a connector, but only one the operator
  configured when starting the CLI or server; endpoints and credentials are never tool arguments.
* **No network I/O of their own.** The tools reach an engine only through a configured connector.
"""

from __future__ import annotations

import json
import math
import os
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from openbeam_v2e import __version__
from openbeam_v2e.agents import steps
from openbeam_v2e.backends import registry
from openbeam_v2e.backends.base import (
    CORE_RECORDINGS,
    NO_RECORDING,
    SCHEMA_COMPILED,
    SCHEMA_PROGRAM,
    BackendError,
    CompiledScene,
    Lockstep,
    RunSpec,
    to_jsonable,
)
from openbeam_v2e.backends.harness import RunDirectoryError, audit_run, execute, load_run, negotiate, repeat_runs

__all__ = ["ToolSpec", "ToolFailure", "AgentContext", "TOOLS", "tool_specs", "call_tool", "validate_arguments"]


class ToolFailure(Exception):
    def __init__(self, code: str, message: str, data: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = dict(data or {})

    def to_json(self) -> dict[str, Any]:
        return {"ok": False, "error": {"code": self.code, "message": self.message}, **self.data}


@dataclass
class AgentContext:
    """Where tools may read and write, and which connectors they may use."""

    root: Path = field(default_factory=Path.cwd)
    sandbox: bool = True
    connectors: Mapping[str, Mapping[str, Any]] = field(default_factory=lambda: {"fake": {}})

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()

    def path(self, value: Any, *, must_exist: bool = False, output: bool = False, name: str = "path") -> Path:
        if not isinstance(value, str) or not value:
            raise ToolFailure("bad_arguments", f"{name} must be a non-empty string")
        candidate = Path(value).expanduser()
        resolved = (candidate if candidate.is_absolute() else self.root / candidate).resolve()
        if self.sandbox and resolved != self.root and self.root not in resolved.parents:
            raise ToolFailure("bad_arguments", f"{name} {value!r} is outside the workspace")
        if must_exist and not resolved.exists():
            raise ToolFailure("bad_arguments", f"{name} {value!r} does not exist")
        if output and resolved.exists() and (not resolved.is_dir() or any(resolved.iterdir())):
            raise ToolFailure("bad_arguments", f"{name} {value!r} must be a new or empty directory")
        return resolved

    def show(self, path: Path | None) -> str | None:
        if path is None:
            return None
        try:
            return str(Path(path).resolve().relative_to(self.root))
        except ValueError:
            return str(path)

    def connector(self, name: str) -> Any:
        if name not in self.connectors:
            raise ToolFailure("connector_not_configured",
                              f"connector {name!r} is not configured for this session; configured: "
                              f"{sorted(self.connectors)}")
        try:
            return registry.create(name, self.connectors[name])
        except (BackendError, ValueError, OSError) as exc:
            raise ToolFailure("connector_not_configured", f"connector {name!r}: {exc}") from None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    title: str
    description: str
    input_schema: Mapping[str, Any]
    handler: Callable[[AgentContext, Mapping[str, Any]], Mapping[str, Any]]
    read_only: bool = False
    open_world: bool = False

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "inputSchema": dict(self.input_schema),
            "outputSchema": {"type": "object"},
            "annotations": {"title": self.title, "readOnlyHint": self.read_only, "destructiveHint": False,
                            "idempotentHint": self.read_only, "openWorldHint": self.open_world},
        }


# --------------------------------------------------------------------------- argument validation

_TYPES = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v),
    "null": lambda v: v is None,
}


def _issues(value: Any, schema: Mapping[str, Any], path: str) -> list[str]:
    out: list[str] = []
    types = schema.get("type")
    if types is not None:
        types = [types] if isinstance(types, str) else list(types)
        if not any(_TYPES[t](value) for t in types):
            return [f"{path or '/'}: expected {' or '.join(types)}"]
    if "enum" in schema and value not in schema["enum"]:
        out.append(f"{path or '/'}: must be one of {schema['enum']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            out.append(f"{path}: must be >= {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            out.append(f"{path}: must be <= {schema['maximum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            out.append(f"{path}: must be > {schema['exclusiveMinimum']}")
    if isinstance(value, str) and "minLength" in schema and len(value) < schema["minLength"]:
        out.append(f"{path}: must not be empty")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                out.append(f"{path}/{key}: required")
        extra = schema.get("additionalProperties", True)
        for key, item in value.items():
            if key in props:
                out += _issues(item, props[key], f"{path}/{key}")
            elif extra is False:
                out.append(f"{path}/{key}: unknown argument")
            elif isinstance(extra, Mapping):
                out += _issues(item, extra, f"{path}/{key}")
    if isinstance(value, list) and isinstance(schema.get("items"), Mapping):
        for index, item in enumerate(value):
            out += _issues(item, schema["items"], f"{path}/{index}")
    return out


def validate_arguments(schema: Mapping[str, Any], arguments: Any) -> list[str]:
    """Problems with ``arguments`` against a tool's input schema (empty when valid)."""
    return _issues(arguments, schema, "")


# --------------------------------------------------------------------------- helpers


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ToolFailure("bad_arguments", f"cannot read JSON from {path.name}: {exc}") from None


def _jsonable(result: Any) -> Any:
    if hasattr(result, "to_json") and callable(result.to_json):
        return result.to_json()
    return to_jsonable(result)


def _programs(ctx: AgentContext, raw: Any) -> dict[str, dict]:
    if raw is None:
        return {}
    if isinstance(raw, str):
        raw = _load_json(ctx.path(raw, must_exist=True, name="programs"))
    if isinstance(raw, Mapping) and "programs" in raw:
        raw = raw["programs"]
    if isinstance(raw, list):
        raw = {str(p.get("actor")): p for p in raw if isinstance(p, Mapping)}
    if isinstance(raw, Mapping) and raw.get("schema") == SCHEMA_PROGRAM:
        raw = {str(raw.get("actor")): raw}
    if not isinstance(raw, Mapping):
        raise ToolFailure("bad_arguments", "programs must be a path, a program, a list of programs or a mapping "
                                           "actor -> program")
    return {str(k): dict(v) for k, v in raw.items()}


def _scene_and_compiled(ctx: AgentContext, args: Mapping[str, Any], connector: str) -> tuple[dict, CompiledScene]:
    scene_path = ctx.path(args["scene"], must_exist=True, name="scene") if "scene" in args else None
    compiled = None
    if "compiled" in args:
        compiled_path = ctx.path(args["compiled"], must_exist=True, name="compiled")
        if compiled_path.is_dir():
            compiled_path = compiled_path / "compiled.json"
        try:
            compiled = CompiledScene.from_json(_load_json(compiled_path), base=compiled_path.parent)
        except (TypeError, ValueError, KeyError) as exc:
            raise ToolFailure("bad_arguments", f"compiled: {exc}") from None
        if scene_path is None and compiled.scene:
            scene_path = ctx.path(str((compiled_path.parent / compiled.scene)), must_exist=True, name="scene")
    if scene_path is None:
        raise ToolFailure("bad_arguments", "give 'scene' (a v2e.scene/1 file) or 'compiled' that names its scene")
    scene = _load_json(scene_path)
    if compiled is None:
        if connector != "fake":
            raise ToolFailure("compile_first", f"compile the scene for {connector!r} first (tool 'compile') and pass "
                                               f"'compiled'")
        from openbeam_v2e.backends.fake import compile_scene
        try:
            compiled = compile_scene(scene, scene_path.parent)
        except ValueError as exc:
            raise ToolFailure("refused", f"the fake compiler cannot compile this scene: {exc}") from None
    if compiled.connector != connector:
        raise ToolFailure("bad_arguments", f"compiled scene targets {compiled.connector!r}, not {connector!r}")
    return scene, compiled


def _plan_and_spec(backend: Any, scene: Mapping, compiled: CompiledScene, ctx: AgentContext,
                   args: Mapping[str, Any]) -> tuple[Any, RunSpec]:
    programs = dict(compiled.programs)
    programs.update(_programs(ctx, args.get("programs")))
    opt_in = dict(args.get("opt_in") or {})
    if "regime" in args:
        opt_in.setdefault("regime", args["regime"])
    plan = negotiate(scene, backend.manifest(), opt_in=opt_in, programs=programs,
                     lockstep=isinstance(backend, Lockstep))
    if not plan.ok:
        raise ToolFailure("refused", f"{plan.backend} cannot run this scene: {plan}", plan.to_json())
    spec = RunSpec(duration_s=float(args.get("duration_s", 5.0)), host=plan.host, regime=plan.regime,
                   telemetry_hz=args.get("telemetry_hz"), control_hz=args.get("control_hz"),
                   seed=int(args.get("seed", 0)), record=args.get("record", NO_RECORDING))
    return plan, spec


def _count_rows(path: Path) -> int:
    """Rows of a ``v2e-runlog/1`` file (every non-empty line after the header)."""
    with Path(path).open("rb") as fh:
        return max(0, sum(1 for line in fh if line.strip()) - 1)


def _run_summary(ctx: AgentContext, result: Any, run_dir: Path) -> dict[str, Any]:
    return {
        "ok": True,
        "run": ctx.show(run_dir),
        "connector": result.manifest.backend,
        "engine": result.manifest.engine,
        "valid": result.valid(),
        "invalid_reasons": result.invalid_reasons(),
        "claim_class": result.claim_class(),
        "physics_statement": result.physics_statement(),
        "provenance_signature": result.provenance_signature(),
        "audit": result.attestation.audit,
        "regime": result.plan.regime if result.plan else None,
        "notes": list(result.plan.notes) if result.plan else [],
        "events": dict(result.events),
        "t0_s": result.t0_s,
        "sim_s": result.sim_s,
        "wall_s": result.wall_s,
        "rows": ctx.show(result.rows),
        "row_count": _count_rows(result.rows),
    }


# --------------------------------------------------------------------------- handlers


def _reconstruct(ctx: AgentContext, args: Mapping[str, Any]) -> Mapping[str, Any]:
    video = ctx.path(args["video"], must_exist=True, name="video")
    out = ctx.path(args["out"], output=True, name="out")
    files: dict[str, Path] = {}
    for key in ("annotations", "config"):          # one spelling per input: <key>_path when declared
        if key in args:
            path = ctx.path(args[key], must_exist=True, name=key)
            files[f"{key}_path" if steps.accepts("reconstruct", f"{key}_path") else key] = path
    result = steps.call_step("reconstruct", video=video, out=out, out_dir=out, models=args.get("models"), **files)
    return {"ok": True, "out": ctx.show(out), "result": _jsonable(result)}


def _compile(ctx: AgentContext, args: Mapping[str, Any]) -> Mapping[str, Any]:
    """Scene -> ``package.json`` (general compiler) -> ``compiled.json`` (the connector's dialect).

    With the fake connector and no world package next to the scene, the fake's own compiler
    takes the scene's explicit poses instead.
    """
    connector = args.get("connector", "fake")
    scene_path = ctx.path(args["scene"], must_exist=True, name="scene")
    out = ctx.path(args["out"], output=True, name="out")
    templates = ctx.path(args["templates"], must_exist=True, name="templates") if "templates" in args else None
    scene = _load_json(scene_path)
    world_dir = (scene_path.parent / str(scene.get("world", "world/"))).resolve()
    package_path: Path | None = None
    if steps.resolve("compile") is not None and world_dir.is_dir():
        out.mkdir(parents=True, exist_ok=True)
        try:
            steps.call_step("compile", scene=scene_path, scene_path=scene_path, templates=templates,
                            templates_path=templates, connector=connector, out=out, out_dir=out)
        except Exception as exc:  # noqa: BLE001 - a compiler refusal is an answer, not a crash
            refusals = getattr(exc, "refusals", None)
            if refusals is None:
                raise
            raise ToolFailure("refused", "the compiler refused the scene", {"refusals": [str(r) for r in refusals]}) \
                from None
        package_path = out / "package.json"
    if connector == "fake":
        from openbeam_v2e.backends.fake import compile_package, compile_scene
        try:
            if package_path is not None and package_path.is_file():
                compiled = compile_package(package_path, scene=scene, world=world_dir)
            else:
                compiled = compile_scene(scene, scene_path.parent)
        except ValueError as exc:
            raise ToolFailure("refused", f"the fake connector cannot build this scene: {exc}") from None
    elif package_path is None:
        if steps.resolve("compile") is None:
            raise steps.StepUnavailable("compile")
        raise ToolFailure("bad_arguments", f"the scene's world package {ctx.show(world_dir)!r} does not exist")
    else:
        try:
            compiled = steps.call_step("dialect", package=package_path, package_path=package_path,
                                       connector=connector, out=out, out_dir=out, world=world_dir, scene=scene)
        except LookupError as exc:     # no dialect installed for this connector
            raise ToolFailure("step_unavailable", f"package.json is written, but {exc}") from None
    out.mkdir(parents=True, exist_ok=True)
    target = out / "compiled.json"
    if isinstance(compiled, CompiledScene):
        data = compiled.to_json()
        data["scene"] = _relpath(scene_path, out)
        target.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif isinstance(compiled, Mapping) and compiled.get("schema") == SCHEMA_COMPILED and not target.exists():
        target.write_text(json.dumps(to_jsonable(compiled), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not target.exists():
        raise ToolFailure("failed", "the compiler wrote no compiled.json")
    data = _load_json(target)
    return {"ok": True, "compiled": ctx.show(target), "package": ctx.show(package_path) if package_path else None,
            "connector": data.get("connector"), "dialect": data.get("dialect"), "t0_s": data.get("t0_s"),
            "packages": [p.get("name") for p in data.get("packages", [])],
            "spawns": [s.get("actor") for s in data.get("spawns", [])], "notes": data.get("notes", [])}


def _relpath(path: Path, start: Path) -> str:
    return os.path.relpath(Path(path).resolve(), Path(start).resolve())


def _run(ctx: AgentContext, args: Mapping[str, Any]) -> Mapping[str, Any]:
    connector = args.get("connector", "fake")
    out = ctx.path(args["out"], output=True, name="out")
    scene, compiled = _scene_and_compiled(ctx, args, connector)
    backend = ctx.connector(connector)
    try:
        plan, spec = _plan_and_spec(backend, scene, compiled, ctx, args)
        result = execute(backend, compiled, plan, spec, out)
    except (BackendError, RunDirectoryError, ValueError) as exc:
        raise ToolFailure("failed", f"run failed: {exc}") from None
    finally:
        backend.close()
    return _run_summary(ctx, result, out)


def _export(ctx: AgentContext, args: Mapping[str, Any]) -> Mapping[str, Any]:
    run_dir = ctx.path(args["run"], must_exist=True, name="run")
    reference = ctx.path(args["reference"], must_exist=True, name="reference")
    out = ctx.path(args["out"], output=True, name="out")
    try:
        run = load_run(run_dir)
    except RunDirectoryError as exc:
        raise ToolFailure("bad_arguments", str(exc)) from None
    result = steps.call_step("export", run_dir=run_dir, runlog_path=run.rows, reference_dir=reference,
                             reference=reference, out=out, out_dir=out, t0_s=run.t0_s)
    return {"ok": True, "out": ctx.show(out), "claim_class": run.claim_class(), "valid": run.valid(),
            "result": _jsonable(result)}


def _score(ctx: AgentContext, args: Mapping[str, Any]) -> Mapping[str, Any]:
    run_dir = ctx.path(args["run"], must_exist=True, name="run")
    scene = ctx.path(args["scene"], must_exist=True, name="scene") if "scene" in args else None
    reconstruction = (ctx.path(args["reconstruction"], must_exist=True, name="reconstruction")
                      if "reconstruction" in args else None)
    out = ctx.path(args["out"], name="out") if "out" in args else None
    result = steps.call_step("score", run_dir=run_dir, run=run_dir, scene=scene, scene_path=scene,
                             reconstruction=reconstruction, reference=reconstruction,
                             reference_dir=reconstruction, out=out, out_path=out)
    return {"ok": True, "run": ctx.show(run_dir), "score": _jsonable(result)}


def _inspect_scene(ctx: AgentContext, data: Mapping[str, Any]) -> dict[str, Any]:
    actors = []
    for a in data.get("actors", []):
        state = a.get("initial_state") or {}
        actors.append({"id": a.get("id"), "class": a.get("class"), "control": a.get("control"),
                       "mode": state["mode"] if "mode" in state else None})
    feasibility = {}
    for name in ctx.connectors:
        try:
            backend = registry.create(name, ctx.connectors[name])
            plan = negotiate(data, backend.manifest(), lockstep=isinstance(backend, Lockstep),
                             programs={a["id"]: {"schema": SCHEMA_PROGRAM, "actor": a["id"], "model": "passive"}
                                       for a in actors if a["control"] not in (None, "passive")
                                       and a["class"] != "static"})
            feasibility[name] = {"ok": plan.ok, "reasons": [] if plan.ok else [to_jsonable(r) for r in plan.reasons],
                                 "notes": list(plan.notes) if plan.ok else []}
        except Exception as exc:  # noqa: BLE001 - report, do not fail the inspection
            feasibility[name] = {"ok": False, "reasons": [{"code": "error", "detail": str(exc)}]}
    return {"kind": "scene", "actors": actors, "requires": data.get("requires", {}),
            "events": [e.get("name") for e in data.get("events", [])],
            "phases": [p.get("name") for p in data.get("phases", [])], "targets": len(data.get("targets", [])),
            "connectors": feasibility}


def _inspect(ctx: AgentContext, args: Mapping[str, Any]) -> Mapping[str, Any]:
    if "connector" in args:
        name = args["connector"]
        if name not in registry.available():
            raise ToolFailure("bad_arguments", f"unknown connector {name!r}; available: {sorted(registry.available())}")
        cls = registry.backend_class(name)
        out: dict[str, Any] = {"ok": True, "kind": "connector", "connector": name,
                               "configured": name in ctx.connectors}
        if name in ctx.connectors:
            backend = ctx.connector(name)
            try:
                out["manifest"] = to_jsonable(backend.manifest())
            finally:
                backend.close()
        policy = cls.audit_policy()
        out["allowed_calls"] = sorted(policy.methods)
        return out
    if "path" not in args:
        return {"ok": True, "kind": "overview", "version": __version__, "workspace": str(ctx.root),
                "connectors": {n: {"configured": n in ctx.connectors} for n in sorted(registry.available())},
                "steps": {**steps.available_steps(), "run": True, "inspect": True, "bench.repeat": True}}
    path = ctx.path(args["path"], must_exist=True)
    if path.is_dir() and (path / "run.json").is_file():
        result = load_run(path)
        summary = _run_summary(ctx, result, path)
        summary["kind"] = "run"
        summary["audit_replay"] = audit_run(path)
        return summary
    if path.is_dir() and (path / "compiled.json").is_file():
        path = path / "compiled.json"
    if path.is_file() and path.suffix == ".json":
        data = _load_json(path)
        schema = data.get("schema") if isinstance(data, Mapping) else None
        if schema == "v2e.scene/1":
            return {"ok": True, "path": ctx.show(path), **_inspect_scene(ctx, data)}
        if schema == SCHEMA_COMPILED:
            return {"ok": True, "kind": "compiled", "path": ctx.show(path), "connector": data.get("connector"),
                    "dialect": data.get("dialect"), "packages": [p.get("name") for p in data.get("packages", [])],
                    "spawns": data.get("spawns", []), "notes": data.get("notes", [])}
        if schema == "v2e.bench.repeat/1":
            return {"ok": True, "kind": "bench.repeat", "path": ctx.show(path), "report": data}
        return {"ok": True, "kind": "json", "path": ctx.show(path), "schema": schema,
                "keys": sorted(data) if isinstance(data, Mapping) else None}
    if path.is_dir():
        files = sorted(str(p.relative_to(path)) for p in path.rglob("*") if p.is_file())
        return {"ok": True, "kind": "directory", "path": ctx.show(path), "files": files[:200],
                "file_count": len(files)}
    return {"ok": True, "kind": "file", "path": ctx.show(path), "bytes": path.stat().st_size}


def _bench(ctx: AgentContext, args: Mapping[str, Any]) -> Mapping[str, Any]:
    out = ctx.path(args["out"], output=True, name="out")
    if args["kind"] == "synth":
        result = steps.call_step("synth", out=out, connector=args.get("connector"), subset=args.get("subset"),
                                 families=args.get("families"))
        return {"ok": True, "kind": "synth", "out": ctx.show(out), "result": _jsonable(result)}
    connector = args.get("connector", "fake")
    scene, compiled = _scene_and_compiled(ctx, args, connector)
    probe = ctx.connector(connector)
    try:
        plan, spec = _plan_and_spec(probe, scene, compiled, ctx, args)
    finally:
        probe.close()
    try:
        report = repeat_runs(lambda: ctx.connector(connector), compiled, plan, spec, out, int(args.get("n", 9)))
    except (BackendError, RunDirectoryError, ValueError) as exc:
        raise ToolFailure("failed", f"bench failed: {exc}") from None
    return {"ok": True, "kind": "repeat", "out": ctx.show(out), "report": report}


# --------------------------------------------------------------------------- specs

_PATH = {"type": "string", "minLength": 1}
_OPT_IN = {
    "type": "object", "additionalProperties": False,
    "description": "Explicit opt-ins for negotiation.",
    "properties": {
        "regime": {"type": "string", "description": "Timing regime, e.g. 'stepped' or 'render_paced'."},
        "host": {"type": "string", "enum": ["engine_script", "vehicle_script", "client_lockstep"]},
        "add_anchor": {"type": "boolean", "description": "Allow an off-camera anchor vehicle for regimes that need "
                                                         "a player vehicle."},
        "substitutes": {"type": "object", "additionalProperties": {"type": "string"},
                        "description": "effect -> substitute mechanism the connector declares; the run is then "
                                       "labelled 'emulated'."},
        "conditions": {"type": "array", "items": {"type": "string"}},
    },
}
_RUN_PROPS = {
    "scene": {**_PATH, "description": "A v2e.scene/1 file. With the fake connector the scene is compiled on the fly."},
    "compiled": {**_PATH, "description": "A v2e.compiled/1 file (or the directory holding compiled.json)."},
    "connector": {"type": "string", "description": "A configured connector (default 'fake')."},
    "programs": {"type": ["string", "object", "array"],
                 "description": "Controller programs (v2e.program/1): a JSON file, one program, a list, or a mapping "
                                "actor -> program. Merged over the compiled scene's programs."},
    "duration_s": {"type": "number", "exclusiveMinimum": 0, "maximum": 600, "description": "Simulated seconds."},
    "regime": {"type": "string"},
    "telemetry_hz": {"type": "number", "exclusiveMinimum": 0, "maximum": 1000},
    "control_hz": {"type": "number", "exclusiveMinimum": 0, "maximum": 2000},
    "seed": {"type": "integer"},
    "record": {"type": "string", "enum": list(CORE_RECORDINGS)},
    "opt_in": _OPT_IN,
}

TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "reconstruct", "Reconstruct a video",
        "Step 1. Reconstruct a filmed event into a world package: camera track (intrinsics, distortion, pan/tilt/zoom "
        "per frame), static geometry, and each moving object with measured dimensions, a rigid pose track and events, "
        "all with uncertainty. Perception models are pluggable ('models' maps a stage to a model id).",
        {"type": "object", "additionalProperties": False, "required": ["video", "out"], "properties": {
            "video": {**_PATH, "description": "Input video file."},
            "out": {**_PATH, "description": "New or empty directory for the world package."},
            "annotations": {**_PATH, "description": "Keyframe outlines or masks for assisted reconstruction."},
            "config": {**_PATH, "description": "Clip configuration (frame range, objects, camera hints)."},
            "models": {"type": "object", "additionalProperties": {"type": "string"},
                       "description": "Perception model per stage, for example {\"masks\": \"baseline\"}."},
        }},
        _reconstruct),
    ToolSpec(
        "compile", "Compile a scene",
        "Step 3. Compile a v2e.scene/1 scene (plus a v2e.templates/1 mapping) into packages for one connector and "
        "write compiled.json. Every actor spawns at rest; controllers bring it to its observed state.",
        {"type": "object", "additionalProperties": False, "required": ["scene", "out"], "properties": {
            "scene": {**_PATH, "description": "A v2e.scene/1 file."},
            "templates": {**_PATH, "description": "A v2e.templates/1 file."},
            "connector": {"type": "string", "description": "Target connector (default 'fake')."},
            "out": {**_PATH, "description": "New or empty output directory."},
        }},
        _compile),
    ToolSpec(
        "run", "Run a scene through a connector",
        "Steps 4-5. Negotiate a plan (or an itemised refusal) from the scene's requirements and the connector's "
        "declared capabilities, then install, spawn every actor at rest, run one controller program per controlled "
        "actor, and write a run directory: plan, replayable call ledger, v2e-runlog/1 rows and run.json with the "
        "computed claim class. No tool can set forces, poses or velocities.",
        {"type": "object", "additionalProperties": False, "required": ["out"],
         "properties": {**_RUN_PROPS, "out": {**_PATH, "description": "New or empty run directory."}}},
        _run, open_world=True),
    ToolSpec(
        "export", "Export a run as a world package",
        "Step 5. Export a run directory as a world package in the reconstruction's format, resampled onto the "
        "footage's frames and seen through the reconstructed camera.",
        {"type": "object", "additionalProperties": False, "required": ["run", "reference", "out"], "properties": {
            "run": {**_PATH, "description": "A run directory."},
            "reference": {**_PATH, "description": "The reconstructed world package the run recreates."},
            "out": {**_PATH, "description": "New or empty output directory."},
        }},
        _export),
    ToolSpec(
        "score", "Score a run",
        "Step 6. Score a run against its reconstruction and scene: per-object trajectory error, event timing, "
        "silhouette and attitude match with sigmas, per phase, with the claim class.",
        {"type": "object", "additionalProperties": False, "required": ["run"], "properties": {
            "run": {**_PATH, "description": "A run directory."},
            "scene": {**_PATH, "description": "The v2e.scene/1 file (targets, phases)."},
            "reconstruction": {**_PATH, "description": "The reconstructed world package."},
            "out": {**_PATH, "description": "Where to write the score JSON."},
        }},
        _score),
    ToolSpec(
        "inspect", "Inspect V2E files and connectors",
        "Summarise a run directory (validity, claim class, ledger audit replay), a scene (actors, requirements and "
        "whether each configured connector can run it), a compiled scene or a report; or describe a connector's "
        "declared capabilities. With no arguments, list connectors and which pipeline steps this installation has.",
        {"type": "object", "additionalProperties": False, "properties": {
            "path": {**_PATH, "description": "A file or directory in the workspace."},
            "connector": {"type": "string", "description": "A connector name."},
        }},
        _inspect, read_only=True),
    ToolSpec(
        "bench", "Benchmark repeatability",
        "Step 8. 'repeat': run the same plan N times in fresh run directories and report measured repeatability "
        "(bit_identical, or statistical with robust intervals). 'synth': run the synthetic benchmark if installed.",
        {"type": "object", "additionalProperties": False, "required": ["kind", "out"],
         "properties": {**_RUN_PROPS, "kind": {"type": "string", "enum": ["repeat", "synth"]},
                        "n": {"type": "integer", "minimum": 2, "maximum": 50},
                        "subset": {"type": "string", "description": "synth: which benchmark subset to run."},
                        "families": {"type": "array", "items": {"type": "string"},
                                     "description": "synth: limit the run to these scene families."},
                        "out": {**_PATH, "description": "New or empty output directory."}}},
        _bench, open_world=True),
)

_BY_NAME = {spec.name: spec for spec in TOOLS}


def tool_specs() -> list[dict[str, Any]]:
    return [spec.describe() for spec in TOOLS]


def call_tool(ctx: AgentContext, name: str, arguments: Mapping[str, Any] | None,
              *, on_error: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Validate and run one tool. Always returns a JSON object; failures have ``ok: false``."""
    spec = _BY_NAME.get(name)
    if spec is None:
        return ToolFailure("unknown_tool", f"unknown tool {name!r}; tools: {sorted(_BY_NAME)}").to_json()
    arguments = {} if arguments is None else arguments
    problems = validate_arguments(spec.input_schema, arguments)
    if problems:
        return ToolFailure("bad_arguments", "; ".join(problems)).to_json()
    try:
        return dict(spec.handler(ctx, arguments))
    except ToolFailure as failure:
        return failure.to_json()
    except steps.StepUnavailable as exc:
        return ToolFailure("step_unavailable", str(exc), {"needs": exc.needs}).to_json()
    except Exception as exc:  # noqa: BLE001 - tools report failures as data, never crash the server
        if on_error is not None:
            on_error(traceback.format_exc())
        return ToolFailure("failed", f"{type(exc).__name__}: {exc}").to_json()
