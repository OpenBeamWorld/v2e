# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""The ``openbeam-v2e`` command.

Each pipeline step is a subcommand. The step subcommands are thin wrappers around the agent
tools in :mod:`openbeam_v2e.agents.tools` (the same tools the MCP server exposes): they build the
tool's JSON arguments from the command line, run the tool and print its JSON answer on stdout.

Exit status: 0 on success; 1 when a tool fails or a connector refuses the scene; 2 for usage
errors and for steps that are not available in this installation.

Connectors other than ``fake`` are configured with ``--connector-config FILE``, a JSON object
mapping a connector name to its configuration, for example
``{"openbeam": {"session_id": "YOUR_SESSION_ID", "training": true}}``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from openbeam_v2e import __version__

# (name, one-line help) in pipeline order.
COMMANDS: tuple[tuple[str, str], ...] = (
    ("reconstruct", "video -> world package with camera, geometry, objects and uncertainty"),
    ("annotate", "add keyframe outlines or masks for assisted reconstruction"),
    ("scene", "build or check a v2e.scene/1 description and its v2e.templates/1 mapping"),
    ("compile", "compile scene + templates + a connector's content catalog into engine packages"),
    ("run", "run compiled packages through a connector and write a run directory"),
    ("export", "export a run as a world package in the reconstruction's format"),
    ("score", "score a run against its reconstruction"),
    ("search", "search controller and initial-condition parameters against the score"),
    ("inspect", "summarise a world package, scene, run directory, report or connector"),
    ("bench", "run the repeatability bench or a benchmark"),
    ("audit", "replay a run's call ledger against the connector's allow lists"),
    ("check", "validate files against their schemas"),
    ("connectors", "list connectors, show one, or sign in to OpenBeam"),
    ("mcp", "serve the V2E tools to AI agents over MCP (stdio)"),
)

NOT_YET = ("annotate", "scene", "search", "check")


def _add_run_options(p: argparse.ArgumentParser) -> None:
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--scene", help="a v2e.scene/1 file (compiled on the fly for the fake connector)")
    src.add_argument("--compiled", help="a v2e.compiled/1 file or the directory holding compiled.json")
    p.add_argument("--out", required=True, help="new or empty output directory")
    p.add_argument("--connector", default="fake", help="connector name (default: fake)")
    p.add_argument("--programs", help="controller programs (v2e.program/1) as a JSON file")
    p.add_argument("--duration", type=float, dest="duration_s", help="simulated seconds (default 5)")
    p.add_argument("--regime", help="timing regime, for example stepped")
    p.add_argument("--telemetry-hz", type=float)
    p.add_argument("--control-hz", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--record", help="recording class, for example stepped_capture")
    p.add_argument("--add-anchor", action="store_true",
                   help="allow an off-camera anchor vehicle for regimes that need a player vehicle")
    p.add_argument("--substitute", action="append", default=[], metavar="EFFECT=MECHANISM",
                   help="use a declared substitute mechanism (the run is then labelled emulated)")


def _add_connector_config(p: argparse.ArgumentParser) -> None:
    p.add_argument("--connector-config", help="JSON file: {connector name: configuration}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openbeam-v2e",
        description="OpenBeam V2E (Video-to-Engine): recreate a filmed physical event in a physics "
        "engine and score the run against the footage.",
    )
    parser.add_argument("--version", action="version", version=f"openbeam-v2e {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    helps = dict(COMMANDS)
    for name in NOT_YET:
        sub.add_parser(name, help=helps[name], description=helps[name])

    p = sub.add_parser("reconstruct", help=helps["reconstruct"], description=helps["reconstruct"])
    p.add_argument("video")
    p.add_argument("--out", required=True)
    p.add_argument("--annotations")
    p.add_argument("--config")
    p.add_argument("--model", action="append", default=[], metavar="STAGE=MODEL",
                   help="perception model for a stage, for example masks=baseline")

    p = sub.add_parser("compile", help=helps["compile"], description=helps["compile"])
    p.add_argument("scene")
    p.add_argument("--out", required=True)
    p.add_argument("--templates")
    p.add_argument("--connector", default="fake")
    _add_connector_config(p)

    p = sub.add_parser("run", help=helps["run"], description=helps["run"])
    _add_run_options(p)
    _add_connector_config(p)

    p = sub.add_parser("export", help=helps["export"], description=helps["export"])
    p.add_argument("run")
    p.add_argument("--reference", required=True, help="the reconstructed world package the run recreates")
    p.add_argument("--out", required=True)

    p = sub.add_parser("score", help=helps["score"], description=helps["score"])
    p.add_argument("run")
    p.add_argument("--scene")
    p.add_argument("--reconstruction")
    p.add_argument("--out")

    p = sub.add_parser("inspect", help=helps["inspect"], description=helps["inspect"])
    p.add_argument("path", nargs="?")
    p.add_argument("--connector")
    _add_connector_config(p)

    p = sub.add_parser("bench", help=helps["bench"], description=helps["bench"])
    bench = p.add_subparsers(dest="bench_kind", metavar="KIND", required=True)
    r = bench.add_parser("repeat", help="run one plan N times and measure repeatability")
    _add_run_options(r)
    r.add_argument("-n", type=int, default=9)
    _add_connector_config(r)
    s = bench.add_parser("synth", help="run the synthetic benchmark (if installed)")
    s.add_argument("--out", required=True)
    s.add_argument("--connector")
    s.add_argument("--subset")
    s.add_argument("--family", action="append", dest="families")
    _add_connector_config(s)

    p = sub.add_parser("audit", help=helps["audit"], description=helps["audit"])
    p.add_argument("run")

    p = sub.add_parser("connectors", help=helps["connectors"], description=helps["connectors"])
    conn = p.add_subparsers(dest="connectors_action", metavar="ACTION")
    conn.add_parser("list", help="list connectors")
    show = conn.add_parser("show", help="show a connector's declared capabilities")
    show.add_argument("name")
    _add_connector_config(show)
    login = conn.add_parser("login", help="sign in with the documented OpenBeam device-code flow")
    login.add_argument("name", choices=["openbeam"])
    login.add_argument("--token-file", help="where to store the token (default: the user config directory)")
    login.add_argument("--base-url", help="OpenBeam Cloud address (default: the public service)")

    p = sub.add_parser("mcp", help=helps["mcp"], description=helps["mcp"])
    p.add_argument("--root", default=".", help="workspace directory the tools may read and write (default: .)")
    _add_connector_config(p)
    return parser


def _connectors(path: str | None) -> dict[str, Mapping[str, Any]]:
    connectors: dict[str, Mapping[str, Any]] = {"fake": {}}
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not all(isinstance(v, dict) for v in data.values()):
            raise ValueError("--connector-config must hold a JSON object {connector name: configuration object}")
        connectors.update(data)
    return connectors


def _pairs(items: Sequence[str], what: str) -> dict[str, str]:
    out = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key or not value:
            raise ValueError(f"{what} must look like KEY=VALUE, not {item!r}")
        out[key] = value
    return out


def _run_arguments(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {"out": args.out, "connector": args.connector}
    for key in ("scene", "compiled", "programs", "duration_s", "regime", "telemetry_hz", "control_hz", "seed",
                "record"):
        value = getattr(args, key, None)
        if value is not None:
            out[key] = value
    opt_in: dict[str, Any] = {}
    if args.add_anchor:
        opt_in["add_anchor"] = True
    if args.substitute:
        opt_in["substitutes"] = _pairs(args.substitute, "--substitute")
    if opt_in:
        out["opt_in"] = opt_in
    return out


def _emit(answer: Mapping[str, Any]) -> int:
    print(json.dumps(answer, indent=2, sort_keys=True))
    if answer.get("ok", False):
        return 0
    code = (answer.get("error") or {}).get("code")
    return 2 if code in ("step_unavailable", "unknown_tool") else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command in NOT_YET:
        print(f"openbeam-v2e {args.command}: not implemented yet in {__version__}", file=sys.stderr)
        return 2

    from openbeam_v2e.agents.tools import AgentContext, call_tool

    try:
        connectors = _connectors(getattr(args, "connector_config", None))
    except (OSError, ValueError) as exc:
        print(f"openbeam-v2e: {exc}", file=sys.stderr)
        return 2

    if args.command == "mcp":
        from openbeam_v2e.agents.mcp_server import serve
        return serve(AgentContext(root=Path(args.root), sandbox=True, connectors=connectors))

    ctx = AgentContext(root=Path.cwd(), sandbox=False, connectors=connectors)
    try:
        if args.command == "reconstruct":
            tool_args: dict[str, Any] = {"video": args.video, "out": args.out}
            for key in ("annotations", "config"):
                if getattr(args, key):
                    tool_args[key] = getattr(args, key)
            if args.model:
                tool_args["models"] = _pairs(args.model, "--model")
            return _emit(call_tool(ctx, "reconstruct", tool_args))
        if args.command == "compile":
            tool_args = {"scene": args.scene, "out": args.out, "connector": args.connector}
            if args.templates:
                tool_args["templates"] = args.templates
            return _emit(call_tool(ctx, "compile", tool_args))
        if args.command == "run":
            return _emit(call_tool(ctx, "run", _run_arguments(args)))
        if args.command == "export":
            return _emit(call_tool(ctx, "export", {"run": args.run, "reference": args.reference, "out": args.out}))
        if args.command == "score":
            tool_args = {"run": args.run}
            for key in ("scene", "reconstruction", "out"):
                if getattr(args, key):
                    tool_args[key] = getattr(args, key)
            return _emit(call_tool(ctx, "score", tool_args))
        if args.command == "inspect":
            tool_args = {}
            if args.path:
                tool_args["path"] = args.path
            if args.connector:
                tool_args["connector"] = args.connector
            return _emit(call_tool(ctx, "inspect", tool_args))
        if args.command == "bench":
            if args.bench_kind == "repeat":
                tool_args = {"kind": "repeat", "n": args.n, **_run_arguments(args)}
            else:
                tool_args = {"kind": "synth", "out": args.out}
                for key in ("connector", "subset", "families"):
                    if getattr(args, key):
                        tool_args[key] = getattr(args, key)
            return _emit(call_tool(ctx, "bench", tool_args))
        if args.command == "audit":
            from openbeam_v2e.backends.harness import RunDirectoryError, audit_run
            try:
                report = audit_run(Path(args.run))
            except (RunDirectoryError, OSError, ValueError) as exc:
                return _emit({"ok": False, "error": {"code": "bad_arguments", "message": str(exc)}})
            return _emit({"ok": report["valid"], **report})
        if args.command == "connectors":
            return _connectors_command(args, ctx)
    except ValueError as exc:
        print(f"openbeam-v2e {args.command}: {exc}", file=sys.stderr)
        return 2
    parser.error(f"unknown command {args.command!r}")
    return 2


def _connectors_command(args: argparse.Namespace, ctx: Any) -> int:
    from openbeam_v2e.agents.tools import call_tool
    from openbeam_v2e.backends import registry

    action = args.connectors_action or "list"
    if action == "list":
        return _emit({"ok": True, "connectors": {name: {"class": target, "configured": name in ctx.connectors}
                                                 for name, target in sorted(registry.available().items())}})
    if action == "show":
        return _emit(call_tool(ctx, "inspect", {"connector": args.name}))
    from openbeam_v2e.backends.openbeam import transport

    store = transport.TokenStore(Path(args.token_file) if args.token_file else None)
    token = transport.device_code_login(args.base_url or transport.DEFAULT_BASE_URL,
                                        say=lambda m: print(m, file=sys.stderr))
    path = store.write(token)
    return _emit({"ok": True, "token_file": str(path)})


if __name__ == "__main__":
    sys.exit(main())
