# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Step-by-step walkthrough of one scene family, used by the scripts in ``examples/``.

:func:`walkthrough` takes one family of v2e-synth-1 through the whole pipeline and narrates each
step: the case and where its targets come from, generating the clip with truth, what the truth
package holds, reconstructing it, comparing the reconstruction with the truth, recreating the
scene on a connector, and scoring the run. Everything runs offline on a CPU with the ``fake``
connector; the reconstruction uses the baseline perception models when the reconstruction step
is installed and otherwise falls back to the truth (an oracle), saying so.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from openbeam_v2e.bench import families as fam
from openbeam_v2e.bench import metrics, runner, synth

__all__ = ["walkthrough", "main"]


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def walkthrough(family: str, out: str | Path, *, camera: str = "tripod", width: int = 320, height: int = 180,
                reconstruction: str = "auto", physics_dt_s: float = 0.002,
                echo: Callable[[str], None] = print) -> dict[str, Any]:
    """Run one family's default case through the pipeline, narrating each step.

    ``reconstruction`` is ``"baseline"`` (fail if the step is missing), ``"truth"`` (oracle) or
    ``"auto"`` (baseline when installed, otherwise the oracle). Returns a summary dictionary.
    """
    out = Path(out)
    case = fam.FAMILIES[family].default_case(camera)
    summary: dict[str, Any] = {"case": case.case_id}

    echo(f"== 1. The case: {case.case_id}")
    echo(f"   {case.description}; {case.frames} frames at {case.fps:g} fps; camera '{case.camera.name}'")
    echo(f"   parameters: {dict(case.params)}")
    for t in case.targets:
        band = "" if t.value is None else f" = {_fmt(t.value, 3)} +- {_fmt(t.tolerance, 3)} {t.unit}"
        echo(f"   target {t.metric} ({t.actor}){band}  [source: {t.source}]")

    echo("== 2. Generate the clip and its truth (fake connector, built-in ray caster)")
    g = synth.generate_case(case, out / "cases", synth.RenderSettings(width=width, height=height),
                            physics_dt_s=physics_dt_s)
    echo(f"   wrote {g.directory}")
    echo(f"   frame 0 is simulated time {g.truth.t0_s:.3f} s (the instant the subject reaches its frame-0 pose)")
    for name, interval in g.truth.frames.items():
        echo(f"   event {name}: frames {interval if interval is not None else 'not in the clip'}")
    echo(f"   claim class of the generating run: {g.truth.claim_class}")

    echo("== 3. Inspect the truth package (world/, kind synthetic_truth)")
    from openbeam_v2e.world import read_world

    truth_world = read_world(g.directory / "world")
    for obj in truth_world.objects:
        dims = ", ".join(f"{k} {v.value:.2f} m" for k, v in obj.dims_m.items())
        echo(f"   object {obj.id} ({obj.cls}): {dims}")
    echo(f"   {len(truth_world.statics)} static meshes; camera {truth_world.image_size[0]}x{truth_world.image_size[1]}")

    echo("== 4. Reconstruct")
    source = reconstruction
    recon = None
    if reconstruction in ("auto", "baseline"):
        try:
            recon = runner.reconstruct_baseline(case, g.directory, out / "recon" / case.case_id)
            source = "baseline"
            echo("   baseline perception (CPU): camera from the surveyed points, objects from the outlines")
        except runner.StepUnavailable as exc:
            if reconstruction == "baseline":
                raise
            echo(f"   {exc}")
            echo("   falling back to the truth as an oracle reconstruction (measures recreation and scoring only)")
    if recon is None:
        recon = truth_world
        source = "truth"
    summary["reconstruction"] = source

    echo("== 5. Compare the reconstruction with the truth")
    truth_doc = json.loads((g.directory / "truth.json").read_text(encoding="utf-8"))
    truth_events = {k: v["t_s"] for k, v in truth_doc["bench"]["events"].items()}
    landmarks = np.asarray(truth_doc.get("landmarks", []), float)
    subjects = [a.id for a in case.subjects]
    t1 = metrics.compare_worlds(truth_world, recon, subjects, truth_events, runner.world_events(case, recon),
                                points=landmarks if len(landmarks) else None)
    if t1.camera is not None:
        echo(f"   camera: reprojection RMS {_fmt(t1.camera.rms_px)} px, focal error {_fmt(t1.camera.focal_err_pct, 1)} %")
    for oid, o in t1.objects.items():
        echo(f"   {oid}: path RMS {_fmt(o.traj_rms_m)} m, attitude {_fmt(o.att_mean_deg, 1)} deg, "
             f"largest dimension error {_fmt(None if o.dims_err_max_m is None else 100 * o.dims_err_max_m, 1)} cm")
    summary["reconstruction_errors"] = t1.to_json()

    echo("== 6. Recreate the scene on the fake connector")
    run_case = runner.case_from_world(case, recon)
    for a in run_case.actors:
        speed = f", entry speed {a.speed_mps:.2f} m/s" if a.mode == "drive_in" else ""
        echo(f"   {a.id}: spawns at rest; frame-0 pose {np.round(a.frame0_position_m, 2).tolist()}{speed}")
    run = synth.simulate_case(run_case, out / "runs" / case.case_id / source, physics_dt_s=physics_dt_s)
    echo(f"   run valid: {run.valid}; claim class {run.claim_class}; ledger and rows in {run.run_dir}")

    echo("== 7. Score the run against the truth")
    run_world = runner.run_world_from_truth(run_case, run, truth_world)
    t4 = metrics.compare_worlds(truth_world, run_world, subjects, truth_events, run.events.times, dims=False)
    for oid, o in t4.objects.items():
        echo(f"   {oid}: path RMS {_fmt(o.traj_rms_m)} m, attitude {_fmt(o.att_mean_deg, 1)} deg, "
             f"silhouette IoU {_fmt(o.iou_mean)}")
    for name, err in sorted(t4.event_frames.items()):
        echo(f"   event {name}: off by {err:.1f} frames")
    met, total, rows = runner.evaluate_targets(case, run.metrics, run.events.times)
    for r in rows:
        echo(f"   target {r['metric']} ({r['actor']}): run {_fmt(r['run'], 3)} vs {_fmt(r['target'], 3)} "
             f"+- {_fmt(r['tolerance'], 3)} -> {'met' if r['met'] else 'missed'}  [{r['source']}]")
    echo(f"   {met}/{total} sourced targets met. Label: {fam.LABEL}.")
    summary.update({"run_errors": t4.to_json(), "targets_met": met, "targets_total": total,
                    "run_valid": run.valid, "claim_class": run.claim_class, "case_dir": str(g.directory)})
    (out / f"{case.case_id}.summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n",
                                                      encoding="utf-8")
    return summary


def main(family: str, argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"Walk the '{family}' example through the V2E pipeline.")
    parser.add_argument("--out", type=Path, default=Path(f"v2e-example-{family}"),
                        help="output folder (default: ./v2e-example-<family>)")
    parser.add_argument("--camera", choices=("tripod", "handheld_ptz"), default="tripod")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--reconstruction", choices=("auto", "baseline", "truth"), default="auto")
    args = parser.parse_args(argv)
    walkthrough(family, args.out, camera=args.camera, width=args.width, height=args.height,
                reconstruction=args.reconstruction)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "barrier_crash", sys.argv[2:]))
