# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by the model adapters: lazy imports, library versions and weight hashes."""

from __future__ import annotations

import hashlib
import importlib
from importlib import metadata
from pathlib import Path
from typing import Any


def require(module: str, extra: str) -> Any:
    """Import ``module`` or raise an ImportError that names the pip extra to install."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(f"this model needs the '{module}' package: pip install 'openbeam-v2e[{extra}]' "
                          f"({exc})") from exc


def versions(*packages: str) -> dict[str, str]:
    out = {}
    for p in packages:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            continue
    return out


def sha256_of(path: str | Path | None) -> str | None:
    """sha256 of a local weights file (``None`` when there is no local file)."""
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pick_device(requested: str | None) -> str:
    if requested:
        return requested
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"
