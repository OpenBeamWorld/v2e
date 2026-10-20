# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Step 1: reconstruction (video -> world package with uncertainty).

Turns a video of a physical event into a world package plus V2E sidecars:

* camera: intrinsics, lens distortion and a pan/tilt/zoom track from a fixed position,
  each with sigma;
* static geometry: the ground plane (more planes or meshes from annotated structure later),
  with sigma;
* dynamic objects: class, measured dimensions with sigma, a rigid pose track with
  covariance, visibility per frame;
* events (contact, take-off, impact, rest) as frame intervals.

Entry point: :func:`openbeam_v2e.reconstruct.pipeline.reconstruct` (used by the CLI and the
agent tools), configured per clip by ``v2e.reconstruct/1`` (:mod:`.clip_config`) and assisted by
``v2e.annotations/1`` prompts on chosen frames (:mod:`.annotate`).

Perception models (calibration, camera tracking, segmentation, depth, pose) are not hard-wired
here; they come through the model-agnostic interfaces in :mod:`openbeam_v2e.perception`. The
modules of this package are the engine-free building blocks those models and the pipeline share:

* :mod:`.geom`, :mod:`.hulls`, :mod:`.silhouette`, :mod:`.render`: cameras, template hulls,
  outlines and signed distances, a small ray caster;
* :mod:`.calibrate`, :mod:`.camera_track`: calibration and fixed-position camera tracking math;
* :mod:`.pose_fit`, :mod:`.measure`, :mod:`.rigid_track`, :mod:`.smoothing`,
  :mod:`.dynamics_fit`, :mod:`.events`, :mod:`.objects`: pose tracks, dimensions,
  physics-prior smoothing, ballistic segments and events;
* :mod:`.solve`, :mod:`.cache`, :mod:`.video`: bounded least squares with covariance, the
  content-addressed stage cache, frame sources;
* :mod:`.synthetic`: synthetic clips with exact truth for tests and benchmarks.

Scope limits for v0 (fixed-position cameras, rigid pose tracks, assisted annotation) are
documented in ``docs/reconstruction.md``.
"""
