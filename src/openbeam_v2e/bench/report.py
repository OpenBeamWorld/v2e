# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Benchmark reports: one row per case, written as Markdown, CSV and JSON.

A row holds what each pipeline step produced for one case, or the step that failed and why.
Missing values are shown as ``-``: a failed or skipped step never turns into a number. Every
report carries the benchmark's label, so figures from synthetic truth are never mistaken for
accuracy on real footage.
"""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = ["Column", "COLUMNS", "BenchRow", "BenchReport", "format_value"]


@dataclass(frozen=True)
class Column:
    key: str
    header: str
    unit: str = ""
    digits: int = 2


COLUMNS: tuple[Column, ...] = (
    Column("case", "case"),
    Column("recon", "recon"),
    Column("status", "status"),
    Column("cam_rms_px", "camera RMS", "px"),
    Column("focal_err_pct", "focal err", "%", 1),
    Column("dims_err_cm", "dims err", "cm", 1),
    Column("recon_traj_rms_m", "recon traj RMS", "m"),
    Column("recon_att_deg", "recon attitude", "deg", 1),
    Column("recon_event_frames", "recon events", "frames", 1),
    Column("claim", "claim"),
    Column("run_traj_rms_m", "run traj RMS", "m"),
    Column("run_event_frames", "run events", "frames", 1),
    Column("silhouette_iou", "IoU", "", 2),
    Column("targets", "targets met"),
    Column("seconds", "time", "s", 1),
)


def format_value(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "-"
        return f"{value:.{digits}f}"
    return str(value)


@dataclass
class BenchRow:
    """Results for one case. ``values`` holds the column values; ``detail`` everything else."""

    case: str
    family: str
    status: str = "ok"           # "ok" | "failed:<step>" | "skipped:<step>"
    error: str = ""
    values: dict[str, Any] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str) -> Any:
        if key == "case":
            return self.case
        if key == "status":
            return self.status
        return self.values.get(key)

    def to_json(self) -> dict[str, Any]:
        return {"case": self.case, "family": self.family, "status": self.status, "error": self.error,
                "values": self.values, "detail": self.detail}


@dataclass
class BenchReport:
    benchmark: str
    label: str
    settings: Mapping[str, Any]
    rows: list[BenchRow] = field(default_factory=list)

    def add(self, row: BenchRow) -> None:
        self.rows.append(row)

    @property
    def ok(self) -> bool:
        return all(r.status == "ok" for r in self.rows)

    def markdown(self, columns: Sequence[Column] = COLUMNS) -> str:
        head = [c.header + (f" ({c.unit})" if c.unit else "") for c in columns]
        lines = [f"# {self.benchmark}: {self.label}", ""]
        settings = ", ".join(f"{k}={v}" for k, v in self.settings.items())
        if settings:
            lines += [f"Settings: {settings}", ""]
        lines.append("| " + " | ".join(head) + " |")
        lines.append("| " + " | ".join("---" for _ in columns) + " |")
        for row in self.rows:
            cells = [format_value(row.get(c.key), c.digits) for c in columns]
            lines.append("| " + " | ".join(cell.replace("|", "/") for cell in cells) + " |")
        failures = [r for r in self.rows if r.status != "ok"]
        if failures:
            lines += ["", "Not completed:", ""]
            for r in failures:
                lines.append(f"- `{r.case}`: {r.status}: {r.error}")
        lines += ["", f"Label: {self.label}. Truth comes from a simulated run; these figures are not "
                  "accuracy on real footage."]
        return "\n".join(lines) + "\n"

    def csv(self, columns: Sequence[Column] = COLUMNS) -> str:
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow([c.key for c in columns] + ["family", "error"])
        for row in self.rows:
            writer.writerow([format_value(row.get(c.key), 6) for c in columns] + [row.family, row.error])
        return buf.getvalue()

    def to_json(self) -> dict[str, Any]:
        return {"schema": "v2e.bench-report/1", "benchmark": self.benchmark, "label": self.label,
                "settings": dict(self.settings), "rows": [r.to_json() for r in self.rows]}

    def write(self, out_dir: str | Path, stem: str = "report") -> dict[str, Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {
            "markdown": out / f"{stem}.md",
            "csv": out / f"{stem}.csv",
            "json": out / f"{stem}.json",
        }
        paths["markdown"].write_text(self.markdown(), encoding="utf-8")
        paths["csv"].write_text(self.csv(), encoding="utf-8")
        paths["json"].write_text(json.dumps(self.to_json(), indent=2, sort_keys=True, default=_jsonable) + "\n",
                                 encoding="utf-8")
        return paths


def _jsonable(value: Any) -> Any:
    try:
        import numpy as np  # base dependency

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialise {type(value).__name__}")


def merge(reports: Iterable[BenchReport]) -> BenchReport:
    reports = list(reports)
    if not reports:
        raise ValueError("nothing to merge")
    out = BenchReport(reports[0].benchmark, reports[0].label, dict(reports[0].settings))
    for r in reports:
        out.rows.extend(r.rows)
    return out
