# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Benchmark command line, reached with ``python -m openbeam_v2e.bench``.

Its ``synth`` group lists the cases of v2e-synth-1, generates chosen cases into a folder (frames,
masks, annotations, truth package), runs the full benchmark into a folder and prints the table,
or writes the case documents as JSON. The package command ``openbeam-v2e bench synth`` calls the
same benchmark code. Nothing here needs a network connection or a GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m openbeam_v2e.bench",
                                     description="v2e-synth-1: generate the synthetic benchmark and run it.")
    sub = parser.add_subparsers(dest="benchmark", metavar="BENCHMARK", required=True)
    synth = sub.add_parser("synth", help="the v2e-synth-1 benchmark")
    cmd = synth.add_subparsers(dest="command", metavar="COMMAND", required=True)

    def selection(p: argparse.ArgumentParser) -> None:
        p.add_argument("--subset", choices=("default", "full"), default="default",
                       help="'default': one case per family (tripod); 'full': the whole grid, both cameras")
        p.add_argument("--family", action="append", dest="families", help="limit to a family (repeatable)")
        p.add_argument("--camera", action="append", dest="cameras", choices=("tripod", "handheld_ptz"),
                       help="camera preset(s) (repeatable)")
        p.add_argument("--case", action="append", dest="case_ids", help="one case id (repeatable)")

    def render(p: argparse.ArgumentParser) -> None:
        p.add_argument("--width", type=int, default=320)
        p.add_argument("--height", type=int, default=180)
        p.add_argument("--physics-dt", type=float, default=0.002, help="connector physics step in seconds")

    p_list = cmd.add_parser("list", help="list the cases")
    selection(p_list)
    p_list.add_argument("--json", action="store_true", help="print the case documents as JSON")

    p_gen = cmd.add_parser("generate", help="simulate and render cases into a directory")
    selection(p_gen)
    render(p_gen)
    p_gen.add_argument("--out", required=True, type=Path)
    p_gen.add_argument("--mp4", action="store_true", help="also encode video.mp4 (needs ffmpeg)")

    p_run = cmd.add_parser("run", help="generate (or reuse) cases, reconstruct, recreate, score and tabulate")
    selection(p_run)
    render(p_run)
    p_run.add_argument("--out", required=True, type=Path)
    p_run.add_argument("--reconstruction", default="baseline,truth",
                       help="comma-separated sources: 'baseline' (CPU perception) and/or 'truth' (oracle)")
    p_run.add_argument("--connector", default="fake", help="connector for the recreation (offline: fake)")
    p_run.add_argument("--regenerate", action="store_true", help="regenerate cases even if present")
    p_run.add_argument("--model", action="append", default=[], metavar="STAGE=MODEL",
                       help="perception model for a reconstruction stage, e.g. masks=sam2 (repeatable)")

    p_exp = cmd.add_parser("export-scenes", help="write each case's scene, templates, targets and world inputs as JSON")
    selection(p_exp)
    p_exp.add_argument("--out", required=True, type=Path)
    return parser


def _cases(args: argparse.Namespace) -> list:
    from openbeam_v2e.bench import families as fam

    if args.case_ids:
        return [fam.get_case(c) for c in args.case_ids]
    return fam.cases(args.subset, families=args.families, cameras=args.cameras)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from openbeam_v2e.bench import families as fam

    if args.command == "list":
        cases = _cases(args)
        if args.json:
            print(json.dumps([c.document() for c in cases], indent=1))
        else:
            for c in cases:
                print(f"{c.case_id:44s} {c.frames:4d} frames  {c.description}")
        return 0
    if args.command == "export-scenes":
        out: Path = args.out
        for c in _cases(args):
            d = out / c.case_id
            d.mkdir(parents=True, exist_ok=True)
            for name, doc in (("case", c.document()), ("templates", c.templates_document()),
                              ("targets", c.targets_document())):
                (d / f"{name}.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {out}")
        return 0
    from openbeam_v2e.bench import synth

    settings = synth.RenderSettings(width=args.width, height=args.height,
                                    write_mp4=bool(getattr(args, "mp4", False)))
    if args.command == "generate":
        for c in _cases(args):
            g = synth.generate_case(c, args.out, settings, physics_dt_s=args.physics_dt)
            events = {k: v for k, v in g.truth.frames.items() if v is not None}
            print(f"{c.case_id}: {c.frames} frames, events {events} -> {g.directory}")
        return 0
    if args.command == "run":
        from openbeam_v2e.bench.runner import run_bench

        reconstructions = tuple(r.strip() for r in args.reconstruction.split(",") if r.strip())
        if args.case_ids:
            families = sorted({fam.get_case(c).family for c in args.case_ids})
            print("note: --case selects families for 'run'; use 'generate' for single cases", file=sys.stderr)
        else:
            families = args.families
        models = {}
        for item in args.model:
            stage, sep, name = item.partition("=")
            if not sep or not stage or not name:
                print(f"--model expects STAGE=MODEL, got {item!r}", file=sys.stderr)
                return 2
            models[stage.strip()] = name.strip()
        report = run_bench(args.out, subset=args.subset, families=families, cameras=args.cameras,
                           reconstructions=reconstructions, connector=args.connector, settings=settings,
                           physics_dt_s=args.physics_dt, regenerate=args.regenerate, models=models or None,
                           progress=lambda msg: print(msg, file=sys.stderr, flush=True))
        print(report.markdown())
        return 0 if all(r.status == "ok" or r.status.startswith("skipped:") for r in report.rows) else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
