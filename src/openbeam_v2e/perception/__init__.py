# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Pluggable perception models behind one model-agnostic interface.

Reconstruction asks for capabilities, not for a specific model:

* camera calibration and camera tracking;
* detection, segmentation and mask tracking;
* point tracking;
* monocular depth;
* object pose and trajectory estimation.

Every model returns its result together with an uncertainty estimate and a provenance
record (model id, version, weights hash, parameters), so a reconstruction states which
models produced it. Models are registered through the ``openbeam_v2e.perception`` entry
point group; ``baseline`` holds lightweight implementations that need only NumPy and SciPy
and run on a CPU laptop, and ``adapters`` wraps heavier models that are installed as
optional extras.
"""
