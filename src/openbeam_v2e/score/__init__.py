# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Step 6: scoring.

Measures how far a run is from its reconstruction, phase by phase and object by object: path
error, when events happen, how fast and in which orientation objects are at those events, how
well the run's outline overlaps the footage when drawn through the reconstructed camera, and
orientation error that ignores rotations a symmetric object cannot show. Events such as an
impact, a rollover or a bounce get their own after-event measures. Each number is reported
both raw and relative to the reconstruction's uncertainty, summarised over repeated runs, and
tagged with the run's claim class and provenance.
"""

from openbeam_v2e.score.claims import claim_class, provenance_signature
from openbeam_v2e.score.repeat import repeat_report, robust_summary
from openbeam_v2e.score.scorer import TARGET_METRICS, ScoreOptions, score_run, score_worlds

__all__ = [
    "TARGET_METRICS",
    "ScoreOptions",
    "claim_class",
    "provenance_signature",
    "repeat_report",
    "robust_summary",
    "score_run",
    "score_worlds",
]
