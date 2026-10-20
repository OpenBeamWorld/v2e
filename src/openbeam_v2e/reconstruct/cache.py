# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Content-addressed cache for expensive reconstruction stages.

A stage result is stored under a key that hashes everything it depends on: the stage name, the
source code of the modules that compute it, the package version, the model name and parameters,
a fingerprint of the input video and the keys of the upstream stages. A cached result is used
only when the key recomputed now matches, so editing a module, changing a parameter, swapping a
model or pointing at another video always recomputes, and two variants never overwrite each
other (one file per key).

Values are written as canonical JSON (floats in their shortest round-trip form) or ``.npz``, so a
result read back from the cache is bit-identical to the one computed.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from openbeam_v2e import __version__


def _plain(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "to_dict"):
        return o.to_dict()
    raise TypeError(f"not fingerprintable: {type(o)!r}")


def canonical(obj: Any) -> str:
    """Canonical JSON text: sorted keys, exact floats, tuples as lists."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_plain, allow_nan=True)


def digest(obj: Any) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def file_digest(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def code_digest(objects: Iterable[ModuleType | Callable[..., Any] | type]) -> dict[str, str]:
    """sha256 of the source of each module, class or function (with the package version)."""
    out = {"openbeam_v2e": __version__}
    for obj in objects:
        try:
            src = inspect.getsource(obj)
        except (OSError, TypeError):
            src = repr(obj)
        name = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", repr(obj))
        mod = getattr(obj, "__module__", None) or getattr(obj, "__name__", "")
        out[f"{mod}:{name}"] = hashlib.sha256(src.encode()).hexdigest()
    return out


def video_fingerprint(video: Any, sample: int = 8) -> str:
    """A fingerprint of a frame source: its file when it has one, otherwise sampled frames."""
    path = getattr(video, "path", None)
    if path is not None and Path(path).is_file():
        return "file:" + file_digest(path)
    files = getattr(video, "files", None)
    if files:
        h = hashlib.sha256()
        for p in files:
            h.update(Path(p).name.encode())
            h.update(file_digest(p).encode())
        return "files:" + h.hexdigest()
    n = len(video)
    h = hashlib.sha256(f"{n}:{video.size}:{getattr(video, 'fps', 0)}".encode())
    for i in np.unique(np.linspace(0, n - 1, min(sample, n)).astype(int)):
        h.update(np.ascontiguousarray(video.read(int(i))).tobytes())
    return "frames:" + h.hexdigest()


class StageCache:
    """``root=None`` disables caching (every stage computes)."""

    def __init__(self, root: str | Path | None, log: Callable[[str], None] | None = None) -> None:
        self.root = Path(root) if root else None
        self.log = log or (lambda m: None)
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)
        self.used: dict[str, str] = {}   # stage -> "cache" | "computed"

    def key(self, stage: str, code: Mapping[str, str], inputs: Any) -> str:
        return digest({"stage": stage, "code": dict(code), "inputs": inputs})

    def _path(self, stage: str, key: str, ext: str) -> Path:
        assert self.root is not None
        return self.root / f"{stage}-{key[:20]}.{ext}"

    def get_json(self, stage: str, key: str) -> Any | None:
        if not self.root:
            self.used[stage] = "computed"
            return None
        p = self._path(stage, key, "json")
        if p.is_file():
            z = json.loads(p.read_text())
            if z.get("key") == key:
                self.used[stage] = "cache"
                self.log(f"{stage}: loaded from cache ({key[:12]})")
                return z["value"]
        self.used[stage] = "computed"
        return None

    def put_json(self, stage: str, key: str, value: Any) -> None:
        if not self.root:
            return
        p = self._path(stage, key, "json")
        tmp = p.with_suffix(".tmp")
        tmp.write_text(canonical({"key": key, "stage": stage, "value": value}))
        tmp.replace(p)

    def get_npz(self, stage: str, key: str) -> dict[str, np.ndarray] | None:
        if not self.root:
            self.used[stage] = "computed"
            return None
        p = self._path(stage, key, "npz")
        if p.is_file():
            with np.load(p, allow_pickle=False) as z:
                if str(z["__key__"]) == key:
                    self.used[stage] = "cache"
                    self.log(f"{stage}: loaded from cache ({key[:12]})")
                    return {k: z[k] for k in z.files if k != "__key__"}
        self.used[stage] = "computed"
        return None

    def put_npz(self, stage: str, key: str, arrays: Mapping[str, np.ndarray]) -> None:
        if not self.root:
            return
        p = self._path(stage, key, "npz")
        tmp = p.with_name(p.stem + ".tmp.npz")
        np.savez_compressed(tmp, __key__=np.array(key), **arrays)
        tmp.replace(p)
