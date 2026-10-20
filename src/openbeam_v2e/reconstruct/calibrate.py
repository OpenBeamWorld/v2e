# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Camera calibration math: intrinsics, lens distortion and pose of one camera.

Two kinds of evidence are supported, and both end in the same bounded least-squares refinement
(:func:`refine_camera`) that reports a sigma for every parameter:

* **point correspondences** between surveyed or otherwise known world points and their pixels.
  Six or more points in general position start from a direct linear transform (DLT); four or
  more points on the ground plane start from a plane homography with the focal length solved
  from the rotation constraints;
* **vanishing points** of two perpendicular ground directions plus one metric fact (camera
  height above the ground, or one known ground distance). Vanishing points can come from
  annotated line segments or from :func:`detect_vanishing_points`, which finds converging
  straight edges (:func:`line_segments`) without a model.

Nothing here knows about a particular scene: world coordinates come from the caller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy import linalg, ndimage

from openbeam_v2e.reconstruct.geom import Camera, Intrinsics, ptz_angles, ptz_rotation
from openbeam_v2e.reconstruct.solve import LsqResult, bounded_lsq

CAMERA_PARAMS = ("pan", "tilt", "roll", "cx_w", "cy_w", "cz_w", "f", "ppx", "ppy", "k1", "k2")


@dataclass
class CalibrationResult:
    camera: Camera
    sigmas: dict[str, float]          # 1-sigma per parameter (angles in degrees, lengths in m, px)
    rms_px: float
    residuals_px: np.ndarray          # (N, 2) per correspondence, pixels
    method: str
    n_points: int
    cov: np.ndarray = field(repr=False, default=None)
    names: tuple[str, ...] = CAMERA_PARAMS

    def to_dict(self) -> dict[str, Any]:
        return {"camera": self.camera.to_dict(), "sigmas": self.sigmas, "rms_px": self.rms_px,
                "method": self.method, "n_points": self.n_points}


# --------------------------------------------------------------------------- linear starts

def _normalise(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = pts.mean(axis=0)
    d = np.sqrt(np.mean(np.sum((pts - c) ** 2, axis=1)))
    s = math.sqrt(pts.shape[1]) / max(d, 1e-12)
    T = np.eye(pts.shape[1] + 1)
    T[:-1, :-1] *= s
    T[:-1, -1] = -s * c
    h = np.c_[pts, np.ones(len(pts))] @ T.T
    return h, T


def dlt_projection(world: Any, image: Any) -> np.ndarray:
    """3x4 projection matrix from >= 6 correspondences in general position (normalised DLT)."""
    X, Tx = _normalise(np.asarray(world, float))
    x, Ti = _normalise(np.asarray(image, float))
    rows = []
    for Xi, xi in zip(X, x):
        u, v, w = xi
        rows.append(np.r_[np.zeros(4), -w * Xi, v * Xi])
        rows.append(np.r_[w * Xi, np.zeros(4), -u * Xi])
    _, _, Vt = np.linalg.svd(np.array(rows))
    Pn = Vt[-1].reshape(3, 4)
    return np.linalg.inv(Ti) @ Pn @ Tx


def decompose_projection(P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """P = K R [I | -C] -> (K, R_cw, C) with a positive diagonal K and det(R) = +1."""
    P = np.asarray(P, float)
    if np.linalg.det(P[:, :3]) < 0:
        P = -P
    K, R = linalg.rq(P[:, :3])
    S = np.diag(np.sign(np.diag(K)))
    K, R = K @ S, S @ R
    C = -np.linalg.solve(P[:, :3], P[:, 3])
    return K / K[2, 2], R, C


def homography_dlt(src: Any, dst: Any) -> np.ndarray:
    """Homography H (dst ~ H src) from >= 4 point pairs (normalised DLT)."""
    s, Ts = _normalise(np.asarray(src, float))
    d, Td = _normalise(np.asarray(dst, float))
    rows = []
    for (x, y, w), (u, v, z) in zip(s, d):
        rows.append([0, 0, 0, -z * x, -z * y, -z * w, v * x, v * y, v * w])
        rows.append([z * x, z * y, z * w, 0, 0, 0, -u * x, -u * y, -u * w])
    _, _, Vt = np.linalg.svd(np.array(rows, float))
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2]


def focal_from_plane_homography(H: np.ndarray, cx: float, cy: float) -> float | None:
    """Focal length (square pixels, known principal point) from a world-plane-to-image homography."""
    T = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]])
    Hc = T @ H
    h1, h2 = Hc[:, 0], Hc[:, 1]
    # h1' W h2 = 0 and h1' W h1 = h2' W h2 with W = diag(1/f^2, 1/f^2, 1): linear in 1/f^2
    a = np.array([h1[0] * h2[0] + h1[1] * h2[1], h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2])
    b = np.array([h1[2] * h2[2], h1[2] ** 2 - h2[2] ** 2])
    denom = float(a @ a)
    if denom <= 0:
        return None
    inv_f2 = -float(a @ b) / denom
    if inv_f2 <= 0:
        return None
    return 1.0 / math.sqrt(inv_f2)


def pose_from_plane_homography(H: np.ndarray, K: np.ndarray, plane_z: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """(R_cw, C) of a camera that sees the plane z = plane_z through homography H (plane xy -> px)."""
    B = np.linalg.inv(K) @ H
    lam = 2.0 / (np.linalg.norm(B[:, 0]) + np.linalg.norm(B[:, 1]))
    if B[2, 2] * lam < 0:  # the plane origin must be in front of the camera
        lam = -lam
    r1, r2, t = lam * B[:, 0], lam * B[:, 1], lam * B[:, 2]
    R = np.c_[r1, r2, np.cross(r1, r2)]
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
    C = -R.T @ t
    C[2] += plane_z
    return R, C


# --------------------------------------------------------------------------- refinement

def _camera_from_params(x: np.ndarray, width: int, height: int) -> Camera:
    pan, tilt, roll, cxw, cyw, czw, f, ppx, ppy, k1, k2 = x
    intr = Intrinsics(width, height, f, ppx, ppy, k1, k2)
    return Camera(intr, ptz_rotation(pan, tilt, roll), (cxw, cyw, czw))


def camera_params(camera: Camera) -> np.ndarray:
    pan, tilt, roll = ptz_angles(camera.R_cw)
    i = camera.intrinsics
    return np.array([pan, tilt, roll, *camera.centre, i.f, i.cx, i.cy, i.k1, i.k2], float)


def refine_camera(world: Any, image: Any, init: Camera, free: Sequence[str] = ("pan", "tilt", "roll",
                  "cx_w", "cy_w", "cz_w", "f"), sigma_px: float | Sequence[float] = 1.0,
                  loss: str = "soft_l1", bounds: dict[str, tuple[float, float]] | None = None,
                  centre_prior: tuple[Sequence[float], float] | None = None,
                  method: str = "refine") -> CalibrationResult:
    """Bounded least-squares refinement of a camera against point correspondences.

    ``free`` names the parameters to solve (see ``CAMERA_PARAMS``); the others stay at the
    values of ``init``. ``centre_prior = (centre, sigma_m)`` adds a soft prior on the camera
    position (useful with few points).
    """
    W = np.asarray(world, float)
    obs = np.asarray(image, float)
    sig = np.broadcast_to(np.asarray(sigma_px, float), (len(W),))
    x0 = camera_params(init)
    width, height = init.width, init.height
    lower = {"f": 0.05 * width, "k1": -1.0, "k2": -1.0, "tilt": -math.pi / 2, "roll": -math.pi / 4}
    upper = {"f": 50.0 * width, "k1": 1.0, "k2": 1.0, "tilt": math.pi / 2, "roll": math.pi / 4}
    for k, (lo, hi) in (bounds or {}).items():
        lower[k], upper[k] = lo, hi
    fixed = [n for n in CAMERA_PARAMS if n not in free]

    def residuals(x: np.ndarray) -> np.ndarray:
        cam = _camera_from_params(x, width, height)
        uv, z = cam.project(W)
        r = (uv - obs) / sig[:, None]
        r = np.where(np.isfinite(r), r, 1e3)
        r[z <= 0] = 1e3
        out = r.ravel()
        if centre_prior is not None:
            c0, s = centre_prior
            out = np.r_[out, (x[3:6] - np.asarray(c0, float)) / s]
        return out

    res: LsqResult = bounded_lsq(residuals, x0, CAMERA_PARAMS, lower, upper, fixed=fixed, loss=loss,
                                 f_scale=3.0, x_scale=np.r_[0.01, 0.01, 0.01, 1, 1, 1, 0.01 * x0[6],
                                                            1, 1, 0.01, 0.01])
    cam = _camera_from_params(res.x, width, height)
    uv, _ = cam.project(W)
    rpx = uv - obs
    sig_out = {}
    for name in CAMERA_PARAMS:
        s = res.sigma(name)
        sig_out[name] = math.degrees(s) if name in ("pan", "tilt", "roll") else s
    return CalibrationResult(cam, sig_out, float(np.sqrt(np.mean(np.sum(rpx ** 2, axis=1)))), rpx,
                             method, len(W), res.cov)


def is_coplanar(world: Any, rel_tol: float = 1e-3) -> bool:
    W = np.asarray(world, float)
    s = np.linalg.svd(W - W.mean(axis=0), compute_uv=False)
    return bool(s[-1] <= rel_tol * max(s[0], 1e-12))


def _plane_start(W: np.ndarray, x: np.ndarray, width: int, height: int, pp: Sequence[float],
                 focal_guess: float | None) -> Camera | None:
    """Initial camera from the points on the most populated horizontal plane (>= 4 points)."""
    z = np.round(W[:, 2], 6)
    values, counts = np.unique(z, return_counts=True)
    zc = values[np.argmax(counts)]
    sel = z == zc
    if sel.sum() < 4:
        return None
    H = homography_dlt(W[sel, :2], x[sel])
    f = focal_guess or focal_from_plane_homography(H, pp[0], pp[1])
    if f is None:
        f = 0.5 * width / math.tan(math.radians(30.0))
    K = np.array([[f, 0, pp[0]], [0, f, pp[1]], [0, 0, 1.0]])
    R, C = pose_from_plane_homography(H, K, float(zc))
    return Camera(Intrinsics(width, height, f, pp[0], pp[1]), R, C)


def calibrate_from_points(world: Any, image: Any, width: int, height: int,
                          principal_point: Sequence[float] | None = None,
                          free_distortion: bool = False, free_principal_point: bool = False,
                          sigma_px: float = 1.0, focal_guess: float | None = None) -> CalibrationResult:
    """Calibrate one camera from world/pixel correspondences.

    Needs at least 4 points on one horizontal plane or 6 points in general position. Every
    applicable linear start (DLT, plane homography) is refined and the best fit is returned.
    """
    W = np.asarray(world, float)
    x = np.asarray(image, float)
    if len(W) != len(x):
        raise ValueError("world and image point counts differ")
    pp = (width / 2.0, height / 2.0) if principal_point is None else tuple(principal_point)
    starts: list[tuple[str, Camera]] = []
    if len(W) >= 6 and not is_coplanar(W):
        try:
            K, R, C = decompose_projection(dlt_projection(W, x))
            f = float(0.5 * (K[0, 0] + K[1, 1]))
            if np.isfinite(f) and f > 0:
                starts.append(("dlt+refine", Camera(Intrinsics(width, height, f, pp[0], pp[1]), R, C)))
        except (np.linalg.LinAlgError, ValueError):
            pass
    plane = _plane_start(W, x, width, height, pp, focal_guess) if len(W) >= 4 else None
    if plane is not None:
        starts.append(("plane-homography+refine", plane))
    if not starts:
        raise ValueError("calibration needs at least 4 points on one horizontal plane or 6 in general position")
    free = ["pan", "tilt", "roll", "cx_w", "cy_w", "cz_w", "f"]
    if free_distortion:
        free.append("k1")
    if free_principal_point:
        free += ["ppx", "ppy"]
    best: CalibrationResult | None = None
    for method, init in starts:
        # first without distortion, then with it (a robust two-stage start)
        res = refine_camera(W, x, init, [p for p in free if p != "k1"], sigma_px, method=method)
        if free_distortion:
            res = refine_camera(W, x, res.camera, free, sigma_px, method=method)
        if best is None or res.rms_px < best.rms_px:
            best = res
    assert best is not None
    return best


# --------------------------------------------------------------------------- vanishing points

def fit_vanishing_point(segments: Any) -> tuple[np.ndarray, float]:
    """Least-squares intersection of line segments ``(x1, y1, x2, y2)`` weighted by length.

    Returns the homogeneous vanishing point (unit 3-vector) and the RMS angular residual (deg).
    """
    S = np.asarray(segments, float).reshape(-1, 4)
    p1 = np.c_[S[:, :2], np.ones(len(S))]
    p2 = np.c_[S[:, 2:], np.ones(len(S))]
    lines = np.cross(p1, p2)
    lines /= np.linalg.norm(lines[:, :2], axis=1, keepdims=True)
    w = np.linalg.norm(S[:, 2:] - S[:, :2], axis=1)
    _, _, Vt = np.linalg.svd(lines * w[:, None])
    v = Vt[-1]
    v = v / np.linalg.norm(v)
    ang = _segment_vp_angles(S, v)
    return v, float(np.degrees(np.sqrt(np.mean(ang ** 2))))


def _segment_vp_angles(S: np.ndarray, v: np.ndarray) -> np.ndarray:
    mid = 0.5 * (S[:, :2] + S[:, 2:])
    d = S[:, 2:] - S[:, :2]
    if abs(v[2]) > 1e-12:
        to_vp = v[:2] / v[2] - mid
    else:
        to_vp = np.tile(v[:2], (len(S), 1))
    cosang = np.abs(np.sum(d * to_vp, axis=1)) / (np.linalg.norm(d, axis=1) * np.linalg.norm(to_vp, axis=1) + 1e-12)
    return np.arccos(np.clip(cosang, 0.0, 1.0))


def camera_from_vanishing_points(vp_x: Any, vp_y: Any, width: int, height: int,
                                 camera_height: float, ground_z: float = 0.0,
                                 principal_point: Sequence[float] | None = None,
                                 origin_pixel: Sequence[float] | None = None) -> Camera:
    """Camera from the vanishing points of the world X and Y ground directions.

    ``vp_x`` and ``vp_y`` are pixels (2,) or homogeneous 3-vectors. The focal length follows from
    their orthogonality, the rotation from their directions; world +X points towards ``vp_x``
    (in front of the camera) and +Y is chosen so that +Z is up. The camera sits ``camera_height``
    above the ground; the world origin is the ground point below the camera, or the ground point
    seen at ``origin_pixel``.
    """
    pp = np.array([width / 2.0, height / 2.0] if principal_point is None else principal_point, float)

    def h(v: Any) -> np.ndarray:
        v = np.asarray(v, float)
        return np.r_[v, 1.0] if v.shape == (2,) else v

    vx, vy = h(vp_x), h(vp_y)
    if abs(vx[2]) < 1e-9 or abs(vy[2]) < 1e-9:
        raise ValueError("both vanishing points must be finite to recover the focal length")
    a = vx[:2] / vx[2] - pp
    b = vy[:2] / vy[2] - pp
    f2 = -float(a @ b)
    if f2 <= 0:
        raise ValueError("vanishing points are not consistent with perpendicular directions")
    f = math.sqrt(f2)
    r1 = np.r_[a / f, 1.0]
    r2 = np.r_[b / f, 1.0]
    r1 /= np.linalg.norm(r1)
    r2 /= np.linalg.norm(r2)
    r2 = r2 - (r2 @ r1) * r1
    r2 /= np.linalg.norm(r2)
    r3 = np.cross(r1, r2)
    if r3[1] > 0:  # world up must point up the image (camera -y)
        r2, r3 = -r2, -r3
    R = np.c_[r1, r2, r3]   # columns: world axes in camera coordinates -> this is R_cw
    intr = Intrinsics(width, height, f, pp[0], pp[1])
    C = np.array([0.0, 0.0, ground_z + camera_height])
    cam = Camera(intr, R, C)
    if origin_pixel is not None:
        d = cam.rays(np.asarray(origin_pixel, float)[None])[0]
        if d[2] >= 0:
            raise ValueError("origin_pixel is above the horizon")
        t = -camera_height / d[2]
        g = C + t * d
        cam = Camera(intr, R, C - np.array([g[0], g[1], 0.0]))
    return cam


def camera_height_from_distance(cam: Camera, uv_a: Sequence[float], uv_b: Sequence[float],
                                distance_m: float, ground_z: float = 0.0) -> float:
    """Camera height that makes two ground pixels ``distance_m`` apart (scales a unit-height camera)."""
    rays = cam.rays(np.array([uv_a, uv_b], float))
    h0 = cam.centre[2] - ground_z
    pts = cam.centre + rays * (-h0 / rays[:, 2])[:, None]
    d0 = float(np.linalg.norm(pts[0] - pts[1]))
    return h0 * distance_m / d0


@dataclass
class VanishingPoint:
    point: np.ndarray        # homogeneous unit 3-vector
    inliers: int
    rms_deg: float

    def pixel(self) -> np.ndarray | None:
        return None if abs(self.point[2]) < 1e-12 else self.point[:2] / self.point[2]


def _edge_maxima(mag: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Non-maximum suppression across the edge (along the gradient), so edges stay connected."""
    ang = np.mod(np.degrees(np.arctan2(gy, gx)), 180.0)
    q = (np.floor((ang + 22.5) / 45.0).astype(int)) % 4   # 0: x, 1: diagonal, 2: y, 3: anti-diagonal
    p = np.pad(mag, 1, mode="edge")
    c = p[1:-1, 1:-1]
    shifts = {0: ((0, 1), (0, -1)), 1: ((1, 1), (-1, -1)), 2: ((1, 0), (-1, 0)), 3: ((1, -1), (-1, 1))}
    keep = np.zeros(mag.shape, bool)
    h, w = mag.shape
    for k, ((dy1, dx1), (dy2, dx2)) in shifts.items():
        a = p[1 + dy1:1 + dy1 + h, 1 + dx1:1 + dx1 + w]
        b = p[1 + dy2:1 + dy2 + h, 1 + dx2:1 + dx2 + w]
        keep |= (q == k) & (c >= a) & (c >= b)
    return keep


def line_segments(gray: np.ndarray, min_length: float = 10.0, min_magnitude: float = 0.04,
                  exclude: np.ndarray | None = None, max_rms: float = 0.8) -> np.ndarray:
    """Straight edge segments ``(x1, y1, x2, y2)`` (N, 4) found without a model.

    Edge pixels (local gradient maxima) are grouped into connected components of similar
    gradient orientation; each component that a straight line fits within ``max_rms`` pixels
    becomes one segment. Segment directions are far more accurate than per-pixel gradients.
    """
    g = ndimage.gaussian_filter(np.asarray(gray, float), 1.0)
    gx = ndimage.sobel(g, axis=1) / 8.0
    gy = ndimage.sobel(g, axis=0) / 8.0
    mag = np.hypot(gx, gy)
    sel = _edge_maxima(mag, gx, gy) & (mag > min_magnitude)
    if exclude is not None:
        sel &= ~ndimage.binary_dilation(exclude, iterations=2)
    theta = np.mod(np.arctan2(gy, gx), np.pi)
    nb = 12
    bins = np.floor(theta / (np.pi / nb)).astype(int) % nb
    segs = []
    for b in range(nb):
        for layer in (sel & (bins == b), sel & ((bins == b) | (bins == (b + 1) % nb))):
            lab, n = ndimage.label(layer, structure=np.ones((3, 3), bool))
            for k, sl in enumerate(ndimage.find_objects(lab), start=1):
                if sl is None:
                    continue
                rows, cols = np.nonzero(lab[sl] == k)
                if len(rows) < min_length * 0.7:
                    continue
                P = np.stack([cols + sl[1].start + 0.5, rows + sl[0].start + 0.5], axis=1)
                c = P.mean(axis=0)
                U, S, Vt = np.linalg.svd(P - c, full_matrices=False)
                d = Vt[0]
                rms = S[1] / math.sqrt(len(P)) if len(S) > 1 else 0.0
                t = (P - c) @ d
                length = float(t.max() - t.min())
                if length < min_length or rms > max_rms:
                    continue
                segs.append(np.r_[c + t.min() * d, c + t.max() * d])
    if not segs:
        return np.zeros((0, 4))
    S = np.array(segs)
    # the two passes (one bin, two bins) find many segments twice: keep one of each
    key = np.round(S / 2.0).astype(int)
    _, first = np.unique(key, axis=0, return_index=True)
    return S[np.sort(first)]


def detect_vanishing_points(gray: np.ndarray, n: int = 3, angle_tol_deg: float = 1.5,
                            iterations: int = 2000, seed: int = 0, min_inliers: int = 6,
                            exclude: np.ndarray | None = None, segments: np.ndarray | None = None
                            ) -> list[VanishingPoint]:
    """Find up to ``n`` dominant vanishing points from straight edge segments (RANSAC, no model).

    Pairs of segments propose vanishing points; the proposal that most segment length points at
    (within ``angle_tol_deg``, each segment's vote capped at a tenth of the image diagonal so one
    long line cannot outvote many converging ones) wins, its inliers are removed and the search
    repeats. Each result is refined by length-weighted least squares over its inliers.
    """
    rng = np.random.default_rng(seed)
    S = line_segments(gray, exclude=exclude) if segments is None else np.asarray(segments, float)
    found: list[VanishingPoint] = []
    if len(S) < 2:
        return found
    p1 = np.c_[S[:, :2], np.ones(len(S))]
    p2 = np.c_[S[:, 2:], np.ones(len(S))]
    lines = np.cross(p1, p2)
    lines /= np.linalg.norm(lines[:, :2], axis=1, keepdims=True)
    length = np.linalg.norm(S[:, 2:] - S[:, :2], axis=1)
    vote = np.minimum(length, 0.1 * math.hypot(*gray.shape))
    tol = math.radians(angle_tol_deg)
    alive = np.ones(len(S), bool)
    for _ in range(n):
        idx = np.nonzero(alive)[0]
        if len(idx) < min_inliers:
            break
        best, best_score = None, 0.0
        prob = vote[idx] / vote[idx].sum()
        for _ in range(iterations):
            i, j = rng.choice(idx, 2, replace=False, p=prob)
            v = np.cross(lines[i], lines[j])
            nv = np.linalg.norm(v)
            if nv < 1e-12:
                continue
            v = v / nv
            a = _segment_vp_angles(S[idx], v)
            score = float(np.sum(vote[idx] * (a < tol)))
            if score > best_score:
                best, best_score = v, score
        if best is None:
            break
        inl = idx[_segment_vp_angles(S[idx], best) < tol]
        if len(inl) < min_inliers:
            break
        _, _, Vt = np.linalg.svd(lines[inl] * length[inl, None])
        v = Vt[-1] / np.linalg.norm(Vt[-1])
        a2 = _segment_vp_angles(S[inl], v)
        found.append(VanishingPoint(v, int(len(inl)), float(np.degrees(np.sqrt(np.mean(a2 ** 2))))))
        alive[inl] = False
        alive[idx[_segment_vp_angles(S[idx], v) < tol]] = False
    return found


def classify_vanishing_points(vps: Sequence[VanishingPoint], width: int, height: int
                              ) -> tuple[VanishingPoint | None, list[VanishingPoint]]:
    """Split vanishing points into a vertical one (direction near the image's vertical, far above
    or below the image) and the others (horizontal directions)."""
    vertical = None
    others = []
    centre = np.array([width / 2.0, height / 2.0])
    for vp in vps:
        p = vp.point
        if abs(p[2]) < 1e-9:
            direction = p[:2]
        else:
            direction = p[:2] / p[2] - centre
        ang = math.degrees(math.atan2(abs(direction[0]), abs(direction[1]) + 1e-12))
        far = abs(p[2]) < 1e-9 or abs(p[1] / p[2] - centre[1]) > 1.5 * height
        if vertical is None and ang < 15.0 and far:
            vertical = vp
        else:
            others.append(vp)
    return vertical, others
