# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Optional adapters for heavier perception models.

Wrappers for third-party models such as SAM 2 (segmentation and mask tracking), point
trackers such as CoTracker and TAPNext++, and monocular depth models. Each adapter is
installed through an optional extra (see ``pyproject.toml``), imports its model library
lazily inside the adapter so the base install never imports it, downloads weights only when
the user asks, and records the model id, version and weights hash in the result provenance.
No third-party model code is vendored here.
"""
