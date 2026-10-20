# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Rigid pose tracks with covariance from per-frame masks.

:func:`fit_rigid_track` turns one object's masks and the camera track into a
:class:`~openbeam_v2e.perception.types.PoseTrack`:

1. **Start.** Each observed frame gets a starting position from its mask
    (:func:`~openbeam_v2e.reconstruct.pose_fit.initial_position`). Ground objects take their
    heading from the direction of travel where they move and from a heading search where they do
    not (both directions are tried, so reversing is allowed). Ground fits start at the first
    image-untruncated observation and revisit earlier masks backwards. Cropped masks have a weak
    adjacent-attitude prior to resolve ambiguous outlines, never a forward-motion assumption.
    Free objects search a coarse set of attitudes on the first frame and then follow their previous solution.
2. **Per-frame fits** of the template hull (bounded robust least squares) with a covariance each.
3. **Dimensions** are refitted over a spread of fitting frames, jointly with their poses for
   ground objects, then all poses again (:mod:`openbeam_v2e.reconstruct.measure`). Ground fits
   whose outline RMS exceeds the configured noise multiple are excluded from measurement;
   passing this necessary-condition gate is not proof of ground contact or model validity.
4. **Smoothing.** A constant-acceleration Rauch-Tung-Striebel smoother per coordinate, weighted
   by each frame's variance and driven by a class-specific jerk prior, gives the final track,
   velocities and variances for every frame, including frames without a mask.
5. **Residuals.** Silhouette IoU of the final track per observed frame, kept apart for frames
   used in the fit and frames held out of it.

Attitude smoothing works on unwrapped (yaw, pitch, roll); free objects whose pitch passes
through +-90 degrees should be smoothed with ``smooth_attitude=False``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from openbeam_v2e.perception.types import CameraTrack, MaskTrack, ModelProvenance, ObjectSpec, PoseTrack
from openbeam_v2e.reconstruct.geom import (
    from_yaw_pitch_roll,
    matrix_to_quat,
    rot_z,
    yaw_pitch_roll,
)
from openbeam_v2e.reconstruct.hulls import make_hull
from openbeam_v2e.reconstruct.measure import fit_dimensions, select_frames
from openbeam_v2e.reconstruct.objects import DIM_NAMES, defaults_for
from openbeam_v2e.reconstruct.pose_fit import (
    FREE,
    GROUND,
    MaskObservation,
    PoseFit,
    fit_pose,
    initial_position,
    make_observation,
    outline_iou,
    rotation_hypotheses,
    search_rotations,
    search_yaw,
)
from openbeam_v2e.reconstruct.smoothing import jerk_density, rts_smooth


@dataclass
class TrackOptions:
    min_area_px: int = 40
    dims_rounds: int = 2
    dims_frames: int = 10
    n_outline: int = 120
    n_boundary: int = 160
    sigma_px: float = 1.0
    typical_jerk: float | None = None          # m/s^3; class default when None
    typical_angular_jerk: float | None = None  # rad/s^3; class default when None
    min_speed_for_heading: float = 1.0         # m/s
    clipped_heading_sigma_rad: float | None = math.radians(15.0)  # adjacent-pose prior; None disables
    smooth_attitude: bool = True
    position_sigma_floor_m: float = 0.01
    angle_sigma_floor_rad: float = math.radians(0.2)
    log: Callable[[str], None] | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    dims_max_rms_sigma: float | None = 2.0     # ground-model compatibility gate; None disables


def _log(opts: TrackOptions, msg: str) -> None:
    if opts.log:
        opts.log(msg)


def _heading_from_motion(times: np.ndarray, pos: np.ndarray, min_speed: float) -> tuple[np.ndarray, np.ndarray]:
    """Heading and speed per frame from a local linear fit of positions (NaN where too slow)."""
    n = len(times)
    head = np.full(n, np.nan)
    speed = np.zeros(n)
    for i in range(n):
        lo, hi = max(0, i - 3), min(n, i + 4)
        if hi - lo < 3:
            continue
        t = times[lo:hi] - times[i]
        A = np.c_[np.ones_like(t), t]
        coef, *_ = np.linalg.lstsq(A, pos[lo:hi, :2], rcond=None)
        v = coef[1]
        speed[i] = float(np.hypot(v[0], v[1]))
        if speed[i] >= min_speed:
            head[i] = math.atan2(v[1], v[0])
    return head, speed


def _ground_starts(obs: MaskObservation, hull: Any, p: np.ndarray, heading: float, ground_z: float,
                   have_heading: bool) -> list[tuple[np.ndarray, np.ndarray]]:
    if have_heading:
        yaws = [heading, heading + math.pi]
    else:
        yaws = [k * math.pi / 6 for k in range(12)]
    ranked = search_yaw(obs, hull, p, ground_z, yaws)
    best = [y for _, y in ranked[:2]]
    return [(rot_z(y), p.copy()) for y in best]


def _fit_frames(observations: dict[int, MaskObservation], hull: Any, spec: ObjectSpec, ground_z: float,
                times: dict[int, float], opts: TrackOptions,
                previous: dict[int, PoseFit] | None = None) -> dict[int, PoseFit]:
    frames = sorted(observations)
    motion = spec.motion
    fits: dict[int, PoseFit] = {}
    kw = dict(n_outline=opts.n_outline, sigma_px=opts.sigma_px)
    if motion == GROUND:
        p0 = {f: initial_position(observations[f], hull, GROUND, ground_z) for f in frames}
        t = np.array([times[f] for f in frames])
        P = np.array([p0[f] for f in frames])
        head, _ = _heading_from_motion(t, P, opts.min_speed_for_heading)
        # An entry sliver rarely identifies a vehicle's front or heading. Start at the first
        # untruncated mask, then revisit preceding masks backwards with image-fitted seeds.
        seed = next((i for i, f in enumerate(frames) if _inside_image(observations[f])), 0)
        for seq in (list(range(seed, len(frames))), list(range(seed - 1, -1, -1))):
            last_frame = frames[seed] if seq and seq[0] < seed else None
            before_last = frames[seed + 1] if last_frame is not None and seed + 1 < len(frames) else None
            for i in seq:
                f = frames[i]
                o = observations[f]
                clipped = not _inside_image(o)
                last = fits[last_frame] if last_frame is not None else None
                if previous is not None and f in previous:
                    # A size change can invalidate the formerly best front/back mode. Revisit
                    # both image hypotheses, never choose a heading from signed travel speed.
                    starts = [(previous[f].R, previous[f].position),
                              (rot_z(math.pi) @ previous[f].R, previous[f].position)]
                else:
                    have = bool(np.isfinite(head[i]))
                    starts = _ground_starts(o, hull, p0[f], float(head[i]) if have else 0.0, ground_z, have)
                    if last is not None and not clipped:
                        starts.append((last.R, p0[f].copy()))
                if clipped and last is not None:
                    p = last.position.copy()
                    if before_last is not None:
                        v = (p - fits[before_last].position) / (times[last_frame] - times[before_last])
                        p += v * (times[f] - times[last_frame])
                    starts.append((last.R, p))
                cands = []
                for R, p in starts:
                    prior = None
                    if clipped and last is not None and opts.clipped_heading_sigma_rad is not None:
                        # A cropped outline may fit several headings equally well. Prefer the
                        # adjacent fitted attitude weakly, never the sign of translational speed.
                        # Put the circular mean on this start's branch of the yaw parameter.
                        yaw = yaw_pitch_roll(R)[0]
                        near = yaw + (yaw_pitch_roll(last.R)[0] - yaw + math.pi) % (2 * math.pi) - math.pi
                        sigma = math.sqrt(opts.clipped_heading_sigma_rad ** 2 + max(last.cov[5, 5], 0.0))
                        prior = (np.array([p[0], p[1], near]), np.array([np.inf, np.inf, sigma]))
                    cands.append(fit_pose(o, hull, R, p, GROUND, ground_z, prior=prior, **kw))
                fits[f] = min(cands, key=lambda c: c.cost)
                before_last, last_frame = last_frame, f
        return {f: fits[f] for f in frames}
    # free motion: search attitudes on the first frame, then follow
    last = None
    hyps = rotation_hypotheses()
    depth_fn = opts.extra.get("depth")
    for f in frames:
        o = observations[f]
        if previous is not None and f in previous:
            starts = [(previous[f].R, previous[f].position)]
        elif last is None:
            depth_m = _mask_depth(depth_fn, f, o, hull) if depth_fn is not None else None
            p = initial_position(o, hull, FREE, ground_z, depth_m=depth_m)
            starts = [(R, initial_position(o, hull, FREE, ground_z, R, depth_m)) for R in
                      search_rotations(o, hull, p, hyps, ground_z, keep=4)]
        else:
            p = initial_position(o, hull, FREE, ground_z, last.R)
            # keep the previous depth along the new centroid ray: size-based depth is noisy
            starts = [(last.R, last.position), (last.R, p)]
        cands = [fit_pose(o, hull, R, p, FREE, ground_z, **kw) for R, p in starts]
        best = min(cands, key=lambda c: c.cost)
        if last is not None and best.iou < 0.6:
            # lost track of the attitude: search again
            p = best.position
            extra = [fit_pose(o, hull, R, p, FREE, ground_z, **kw)
                     for R in search_rotations(o, hull, p, hyps, ground_z, keep=3)]
            best = min([best] + extra, key=lambda c: c.cost)
        fits[f] = best
        last = best
    return fits


def _inside_image(obs: MaskObservation) -> bool:
    x0, y0, x1, y1 = obs.bbox
    return x0 > 0 and y0 > 0 and x1 < obs.camera.width and y1 < obs.camera.height


def _dimension_frames(fits: dict[int, PoseFit], k: int, limit: float | None) -> tuple[list[int], list[int]]:
    rejected = {f for f, ft in fits.items() if not math.isfinite(ft.iou) or not math.isfinite(ft.rms_px)
                or (limit is not None and ft.rms_px > limit)}
    selected = select_frames({f: ft.iou for f, ft in fits.items() if f not in rejected}, k)
    return selected, sorted(rejected)


def camera_systematic_cov(fits: dict[int, PoseFit], observations: dict[int, MaskObservation], hull: Any,
                          motion: str, ground_z: float, cameras: CameraTrack, n_frames: int = 3,
                          opts: TrackOptions | None = None) -> np.ndarray:
    """Pose covariance (4x4 over x, y, z, yaw) caused by the reference calibration's uncertainty.

    The calibration error moves every frame's camera the same way, so it biases the whole track
    rather than adding noise per frame. It is propagated with sigma points: the camera of a few
    representative frames is perturbed along each principal axis of the calibration covariance,
    the pose is refitted from the final solution, and the spread of the solutions is the
    covariance.
    """
    cov7 = cameras.calibration_cov
    if cov7 is None or not np.any(cov7):
        return np.zeros((4, 4))
    from openbeam_v2e.reconstruct.geom import Camera, ptz_angles, ptz_rotation

    cov7 = np.asarray(cov7, float)
    n = cov7.shape[0]
    w, V = np.linalg.eigh(0.5 * (cov7 + cov7.T))
    w = np.clip(w, 0.0, None)
    axes = [np.sqrt(n * wi) * V[:, i] for i, wi in enumerate(w) if wi > 0]
    frames = sorted(fits)
    pick = [frames[int(round(i))] for i in np.linspace(0, len(frames) - 1, min(n_frames, len(frames)))]
    f0 = float(cameras.focal[0])
    acc = np.zeros((4, 4))
    count = 0
    kw = dict(n_outline=opts.n_outline, sigma_px=opts.sigma_px) if opts else {}
    for f in pick:
        base = fits[f]
        obs = observations[f]
        cam = obs.camera
        pan, tilt, roll = ptz_angles(cam.R_cw)
        yaw0 = math.atan2(base.R[1, 0], base.R[0, 0])
        for a in axes:
            for sgn in (1.0, -1.0):
                d = sgn * a
                intr = cam.intrinsics.with_focal(cam.intrinsics.f * (1.0 + d[6] / f0))
                pc = Camera(intr, ptz_rotation(pan + d[0], tilt + d[1], roll + d[2]), cam.centre + d[3:6])
                o2 = MaskObservation(obs.frame, pc, obs.mask, obs.sdf, obs.x0, obs.y0, obs.boundary, obs.bbox,
                                     obs.area)
                ft = fit_pose(o2, hull, base.R, base.position, motion, ground_z, **kw)
                dyaw = math.atan2(ft.R[1, 0], ft.R[0, 0]) - yaw0
                dyaw = (dyaw + math.pi) % (2 * math.pi) - math.pi
                dv = np.r_[ft.position - base.position, dyaw]
                acc += np.outer(dv, dv) / (2 * len(axes))
        count += 1
    return acc / max(count, 1)


def _mask_depth(depth_fn: Callable[[int], Any], frame: int, obs: MaskObservation, hull: Any) -> float | None:
    """Depth of the object's centre from a metric depth map: median surface depth in the mask
    plus half the hull's smallest extent."""
    try:
        dm = depth_fn(frame)
    except Exception:  # a failing depth model must not stop the fit
        return None
    if dm is None or dm.kind != "metric":
        return None
    d = np.asarray(dm.depth, float)
    if d.shape != obs.mask.shape:
        return None
    vals = d[obs.mask & np.isfinite(d) & (d > 0)]
    if len(vals) < 10:
        return None
    return float(np.median(vals) + 0.5 * min(hull.dims))


def _unwrap_near(values: np.ndarray) -> np.ndarray:
    out = np.array(values, float)
    ok = np.isfinite(out)
    if ok.sum() > 1:
        out[ok] = np.unwrap(out[ok])
    return out


def fit_rigid_track(masks: MaskTrack, cameras: CameraTrack, spec: ObjectSpec, frames: Sequence[int],
                    fps: float, ground_z: float = 0.0, heldout: Sequence[int] = (),
                    options: TrackOptions | None = None,
                    provenance: Sequence[ModelProvenance] = ()) -> PoseTrack:
    """Fit a rigid pose track (see the module docstring)."""
    opts = options or TrackOptions()
    if opts.dims_max_rms_sigma is not None and (not math.isfinite(opts.dims_max_rms_sigma)
                                               or opts.dims_max_rms_sigma <= 0):
        raise ValueError("dimension RMS limit must be positive and finite, or None to disable")
    if opts.clipped_heading_sigma_rad is not None and (not math.isfinite(opts.clipped_heading_sigma_rad)
                                                       or opts.clipped_heading_sigma_rad <= 0):
        raise ValueError("clipped heading sigma must be positive and finite, or None to disable")
    frames = [int(f) for f in frames if cameras.has(int(f))]
    times = {f: f / fps for f in frames}
    held = set(int(f) for f in heldout)
    cls = defaults_for(spec.object_class)
    dims = np.array([spec.dims_prior.get(n, cls.dims_prior.get(n, (1.0, 1.0)))[0] for n in DIM_NAMES], float)
    dims_sigma = np.array([spec.dims_prior.get(n, cls.dims_prior.get(n, (1.0, 1.0)))[1] for n in DIM_NAMES], float)
    hull = make_hull(spec.template, dims, **spec.template_params)

    observations: dict[int, MaskObservation] = {}
    all_obs: dict[int, MaskObservation] = {}
    for f in frames:
        m = masks.mask(f)
        if m is None or int(m.sum()) < opts.min_area_px:
            continue
        o = make_observation(f, cameras.camera(f), m, opts.n_boundary)
        if o is None:
            continue
        all_obs[f] = o
        if f not in held:
            observations[f] = o
    if len(observations) < 2:
        raise ValueError(f"object {spec.object_id!r}: fewer than two usable masks to fit")
    _log(opts, f"{spec.object_id}: {len(observations)} frames to fit, {len(all_obs) - len(observations)} held out")

    fits = _fit_frames(observations, hull, spec, ground_z, times, opts)
    diagnostics: dict[str, Any] = {"dims_rounds": []}
    if spec.motion == GROUND:
        complete = [f for f in sorted(observations) if _inside_image(observations[f])]
        diagnostics["ground_initialization"] = {
            "seed_frame": complete[0] if complete else min(observations),
            "seed_untruncated": bool(complete),
            "image_clipped_frames": [f for f in sorted(observations) if not _inside_image(observations[f])],
            "clipped_heading_sigma_deg": (None if opts.clipped_heading_sigma_rad is None else
                                           math.degrees(opts.clipped_heading_sigma_rad)),
            "heading_prior": ("disabled" if opts.clipped_heading_sigma_rad is None else
                              "adjacent image-fitted attitude, not direction of travel"),
        }
    if spec.fit_dims and opts.dims_rounds > 0:
        prior = {n: spec.dims_prior.get(n, cls.dims_prior.get(n, (dims[i], dims[i])))
                 for i, n in enumerate(DIM_NAMES)}
        for rnd in range(opts.dims_rounds):
            limit = (opts.dims_max_rms_sigma * opts.sigma_px
                     if spec.motion == GROUND and opts.dims_max_rms_sigma is not None else None)
            selected_fits = fits
            start_dims = dims
            sel, rejected = _dimension_frames(selected_fits, opts.dims_frames, limit)
            record = {"frames": sel, "rejected_frames": rejected, "rms_limit_px": limit,
                      "motion": spec.motion}
            if not sel and limit is not None:
                # A bad starting size can reject a valid ground object too. Try the original
                # conditional-size step provisionally, refit poses, and retry the SAME gate.
                # Do not publish this bootstrap size/uncertainty as a compatible measurement.
                seed, _ = _dimension_frames(fits, opts.dims_frames, None)
                if seed:
                    boot = fit_dimensions(spec.template, spec.template_params, dims,
                                          [observations[f] for f in seed],
                                          [(fits[f].R, fits[f].position) for f in seed], prior, ground_z,
                                          sigma_px=opts.sigma_px)
                    record["bootstrap"] = {"frames": seed, "dims": boot.dims.tolist(),
                                           "success": boot.success, "kind": "conditional_size_proposal"}
                    if boot.success:
                        proposed = make_hull(spec.template, boot.dims, **spec.template_params)
                        selected_fits = _fit_frames(observations, proposed, spec, ground_z, times,
                                                    opts, previous=fits)
                        start_dims = boot.dims
                        sel, rejected = _dimension_frames(selected_fits, opts.dims_frames, limit)
                        record.update(frames=sel, rejected_frames=rejected)
            if not sel:
                record.update(updated=False, reason="no compatible fitting frames")
                diagnostics["dims_rounds"].append(record)
                break
            dfit = fit_dimensions(spec.template, spec.template_params, start_dims, [observations[f] for f in sel],
                                  [(selected_fits[f].R, selected_fits[f].position) for f in sel], prior, ground_z,
                                  motion=spec.motion, sigma_px=opts.sigma_px)
            record.update(dims=dfit.dims.tolist(), sigma=dfit.sigma.tolist(), rms_px=dfit.rms_px,
                          covariance_kind=dfit.covariance_kind, success=dfit.success, updated=dfit.success)
            diagnostics["dims_rounds"].append(record)
            if not dfit.success:
                record["reason"] = "dimension solver did not converge; previous dimensions retained"
                break
            dims, dims_sigma = dfit.dims, dfit.sigma
            hull = make_hull(spec.template, dims, **spec.template_params)
            fits = _fit_frames(observations, hull, spec, ground_z, times, opts, previous=selected_fits)
            _log(opts, f"{spec.object_id}: dims round {rnd + 1}: {np.round(dims, 3).tolist()} "
                       f"+- {np.round(dims_sigma, 3).tolist()}")

    # ---- smoothing
    T = len(frames)
    t = np.array([times[f] for f in frames])
    dt = 1.0 / fps
    jerk = opts.typical_jerk if opts.typical_jerk is not None else cls.typical_jerk
    ajerk = opts.typical_angular_jerk if opts.typical_angular_jerk is not None else cls.typical_angular_jerk
    q_pos = jerk_density(jerk, dt)
    q_ang = jerk_density(ajerk, dt)
    meas_pos = np.full((T, 3), np.nan)
    var_pos = np.full((T, 3), np.nan)
    meas_ang = np.full((T, 3), np.nan)   # yaw, pitch, roll
    var_ang = np.full((T, 3), np.nan)
    pf = opts.position_sigma_floor_m ** 2
    af = opts.angle_sigma_floor_rad ** 2
    for i, f in enumerate(frames):
        if f not in fits:
            continue
        ft = fits[f]
        meas_pos[i] = ft.position
        var_pos[i] = np.maximum(np.diag(ft.cov)[:3], pf)
        yaw, pitch, roll = yaw_pitch_roll(ft.R)
        meas_ang[i] = (yaw, pitch, roll)
        # small world rotations map approximately onto (yaw about z, pitch, roll)
        var_ang[i] = np.maximum(np.diag(ft.cov)[[5, 4, 3]], af)
    for k in range(3):
        meas_ang[:, k] = _unwrap_near(meas_ang[:, k])

    position = np.zeros((T, 3))
    velocity = np.zeros((T, 3))
    pvar = np.zeros((T, 3))
    for k in range(3):
        if spec.motion == GROUND and k == 2:
            position[:, 2] = ground_z
            continue
        s = rts_smooth(t, meas_pos[:, k], var_pos[:, k], q_pos)
        position[:, k], velocity[:, k], pvar[:, k] = s.value, s.rate, s.var
    angles = np.zeros((T, 3))
    avar = np.zeros((T, 3))
    for k in range(3):
        if spec.motion == GROUND and k > 0:
            continue
        if opts.smooth_attitude:
            s = rts_smooth(t, meas_ang[:, k], var_ang[:, k], q_ang)
            angles[:, k], avar[:, k] = s.value, s.var
        else:
            angles[:, k] = np.interp(t, t[np.isfinite(meas_ang[:, k])], meas_ang[np.isfinite(meas_ang[:, k]), k])
            avar[:, k] = np.where(np.isfinite(var_ang[:, k]), var_ang[:, k], np.nanmax(var_ang[:, k]))

    systematic = camera_systematic_cov(fits, observations, hull, spec.motion, ground_z, cameras, opts=opts)
    diagnostics["camera_systematic_sigma"] = {"position_m": np.sqrt(np.diag(systematic)[:3]).tolist(),
                                              "yaw_deg": math.degrees(math.sqrt(systematic[3, 3]))}
    if np.any(systematic):
        # a calibration error scales the scene: carry the range error into the dimensions
        rng = float(np.median([np.linalg.norm(fits[f].position - cameras.centre) for f in fits]))
        rel = math.sqrt(float(np.trace(systematic[:3, :3]))) / max(rng, 1e-6)
        dims_sigma = np.sqrt(dims_sigma ** 2 + (dims * rel) ** 2)

    quat = np.zeros((T, 4))
    cov = np.zeros((T, 6, 6))
    observed = np.zeros(T, bool)
    fitted = np.zeros(T, bool)
    held_arr = np.zeros(T, bool)
    ious = np.full(T, np.nan)
    for i, f in enumerate(frames):
        R = from_yaw_pitch_roll(*angles[i])
        quat[i] = matrix_to_quat(R)
        cov[i, 0, 0], cov[i, 1, 1], cov[i, 2, 2] = pvar[i]
        cov[i, 5, 5], cov[i, 4, 4], cov[i, 3, 3] = avar[i]
        cov[i, :3, :3] += systematic[:3, :3]
        cov[i, :3, 5] += systematic[:3, 3]
        cov[i, 5, :3] += systematic[3, :3]
        cov[i, 5, 5] += systematic[3, 3]
        if f in all_obs:
            observed[i] = True
            fitted[i] = f in observations
            held_arr[i] = f in held
            ious[i] = outline_iou(hull, R, position[i], all_obs[f])
    diagnostics["per_frame_fit_iou"] = {int(f): float(fits[f].iou) for f in fits}
    diagnostics["per_frame_fit_rms_px"] = {int(f): float(fits[f].rms_px) for f in fits}
    return PoseTrack(spec.object_id, spec.object_class, spec.template, dict(spec.template_params), dims,
                     dims_sigma, hull.symmetry, np.asarray(frames), position, quat, cov, velocity, observed,
                     fitted, held_arr, ious, spec.motion, list(provenance), diagnostics)
