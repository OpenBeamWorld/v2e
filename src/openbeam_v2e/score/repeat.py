# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Robust statistics over repeated runs (``v2e.score_repeat/1``).

Engines whose runs are not bit-identical are compared on N repeats per arm. For each metric
this reports the median, the MAD scaled to a normal sigma (``1.4826 * MAD``), the robust
interval ``median +- 3 * 1.4826 * MAD``, the interquartile range, min and max, and, over the
whole set, the fraction of runs that failed (refused, crashed or invalid). Search ranks
candidates by the median.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from openbeam_v2e.schema.score import REPEAT_SCHEMA, REPEAT_SCHEMA_ID
from openbeam_v2e.schema.validate import check

__all__ = ["MAD_TO_SIGMA", "robust_summary", "flatten_metrics", "repeat_report", "within_robust_interval"]

#: Scale from the median absolute deviation to a normal standard deviation.
MAD_TO_SIGMA = 1.4826


def robust_summary(values: Iterable[float | None]) -> dict[str, float | int | None]:
    """Median, scaled MAD, robust interval, IQR, min and max of the finite values."""
    x = np.array([float(v) for v in values if v is not None and math.isfinite(float(v))], dtype=float)
    if len(x) == 0:
        return {"n": 0, "median": None, "mad": None, "low": None, "high": None, "iqr": None, "min": None, "max": None}
    median = float(np.median(x))
    mad = float(MAD_TO_SIGMA * np.median(np.abs(x - median)))
    q1, q3 = np.percentile(x, [25.0, 75.0])
    return {"n": int(len(x)), "median": median, "mad": mad, "low": median - 3.0 * mad, "high": median + 3.0 * mad,
            "iqr": float(q3 - q1), "min": float(x.min()), "max": float(x.max())}


def within_robust_interval(truth: float, values: Sequence[float], floor: float = 0.0) -> bool:
    """``|truth - median| <= max(3 * 1.4826 * MAD, floor)`` over ``values``."""
    s = robust_summary(values)
    if s["n"] == 0:
        return False
    return abs(float(truth) - float(s["median"])) <= max(3.0 * float(s["mad"]), float(floor))


def flatten_metrics(report: Mapping[str, Any]) -> dict[str, float]:
    """Numeric metrics of one ``v2e.score/1`` report as ``dotted.name -> value``."""
    out: dict[str, float] = {}

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, (int, float)):
            if math.isfinite(float(value)):
                out[prefix] = float(value)
        elif isinstance(value, Mapping):
            for k, v in value.items():
                walk(f"{prefix}.{k}" if prefix else str(k), v)

    walk("summary", report.get("summary", {}))
    for phase in report.get("phases", []):
        walk(f"phases.{phase['name']}", phase.get("objects", {}))
    for event in report.get("events", []):
        walk(f"events.{event['name']}", {"delta_frames": event.get("delta_frames")})
    for obj_id, obj in report.get("objects", {}).items():
        for key in ("at_events", "after_events", "rollover", "bounce", "silhouette"):
            walk(f"objects.{obj_id}.{key}", obj.get(key, {}))
    for i, target in enumerate(report.get("targets", [])):
        name = "_".join(str(target.get(k)) for k in ("metric", "actor", "event") if target.get(k))
        walk(f"targets.{name or i}", {"run": target.get("run")})
    return out


def repeat_report(reports: Sequence[Mapping[str, Any] | None], failures: int = 0) -> dict[str, Any]:
    """Summarise N score reports of the same scene; ``None`` entries count as failed runs."""
    done = [r for r in reports if r is not None]
    failed = failures + sum(1 for r in reports if r is None)
    total = len(done) + failed
    flat = [flatten_metrics(r) for r in done]
    names = sorted({k for f in flat for k in f})
    warnings = []
    classes = sorted({r.get("claim", {}).get("class", "unknown") for r in done})
    signatures = sorted({r.get("claim", {}).get("provenance_signature") for r in done}, key=lambda s: s or "")
    if len(classes) > 1:
        warnings.append(f"runs mix claim classes {classes}; compare like with like")
    if len(signatures) > 1:
        warnings.append("runs have different provenance signatures (different effects or content)")
    out = {
        "schema": REPEAT_SCHEMA_ID,
        "runs": total,
        "failures": failed,
        "failure_fraction": (failed / total) if total else 0.0,
        "claim_classes": classes,
        "provenance_signatures": signatures,
        "metrics": {name: robust_summary(f.get(name) for f in flat) for name in names},
        "warnings": warnings,
    }
    check(out, REPEAT_SCHEMA, "repeat report")
    return out
