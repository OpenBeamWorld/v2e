# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Frame sources: videos, image sequences and in-memory arrays behind one small interface.

Every perception model reads frames through :class:`FrameSource`: ``len(src)``, ``src.fps``,
``src.size`` (width, height) and ``src.read(i)`` returning an ``(H, W, 3)`` ``uint8`` RGB array.

* :class:`ArrayFrames` wraps frames already in memory (tests, synthetic clips).
* :class:`ImageSequence` reads a directory of numbered images. PNG is decoded with the standard
  library (no Pillow needed); other formats use Pillow when it is installed.
* :class:`FFmpegVideo` decodes a video file with an external ``ffmpeg`` program (not bundled),
  caching decoded frames in memory up to a limit.

:func:`open_frames` picks the right one for a path.
"""

from __future__ import annotations

import json
import re
import shutil
import struct
import subprocess
import zlib
from pathlib import Path
from typing import Any, Iterator, Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class FrameSource(Protocol):
    fps: float

    def __len__(self) -> int: ...

    @property
    def size(self) -> tuple[int, int]: ...

    def read(self, index: int) -> np.ndarray: ...


def iter_frames(src: FrameSource, frames: Sequence[int] | None = None) -> Iterator[tuple[int, np.ndarray]]:
    for i in (range(len(src)) if frames is None else frames):
        yield i, src.read(i)


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """Grey level in [0, 1] as float32 ((R + 2 G + B) / 4)."""
    a = np.asarray(rgb)
    if a.ndim == 2:
        g = a.astype(np.float32)
    else:
        g = 0.25 * a[..., 0] + 0.5 * a[..., 1] + 0.25 * a[..., 2]
        g = g.astype(np.float32)
    return g / 255.0 if np.asarray(rgb).dtype == np.uint8 else g


class ArrayFrames:
    """Frames held in memory as an array (T, H, W, 3) or a list of arrays."""

    def __init__(self, frames: Any, fps: float = 30.0) -> None:
        self._frames = frames
        self.fps = float(fps)

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def size(self) -> tuple[int, int]:
        h, w = np.asarray(self._frames[0]).shape[:2]
        return int(w), int(h)

    def read(self, index: int) -> np.ndarray:
        f = np.asarray(self._frames[index])
        if f.ndim == 2:
            f = np.repeat(f[..., None], 3, axis=2)
        return f


_NUM = re.compile(r"(\d+)(?!.*\d)")


class ImageSequence:
    """A directory of numbered frames (``000000.png``, ``frame_12.jpg``, ...), sorted by number."""

    EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy")

    def __init__(self, directory: str | Path, fps: float = 30.0) -> None:
        self.directory = Path(directory)
        files = [p for p in self.directory.iterdir() if p.suffix.lower() in self.EXTENSIONS]
        if not files:
            raise FileNotFoundError(f"no image frames in {self.directory}")

        def key(p: Path) -> tuple[int, str]:
            m = _NUM.search(p.stem)
            return (int(m.group(1)) if m else -1, p.name)

        self.files = sorted(files, key=key)
        self.fps = float(fps)
        self._size: tuple[int, int] | None = None

    def __len__(self) -> int:
        return len(self.files)

    @property
    def size(self) -> tuple[int, int]:
        if self._size is None:
            h, w = self.read(0).shape[:2]
            self._size = (int(w), int(h))
        return self._size

    def read(self, index: int) -> np.ndarray:
        img = read_image(self.files[index])
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)
        return img[..., :3]


class FFmpegVideo:
    """A video file decoded by an external ``ffmpeg`` program into RGB frames."""

    def __init__(self, path: str | Path, fps: float | None = None, ffmpeg: str = "ffmpeg",
                 ffprobe: str = "ffprobe", cache_frames: int = 2000) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.ffmpeg = shutil.which(ffmpeg) or ffmpeg
        self.ffprobe = shutil.which(ffprobe) or ffprobe
        info = self._probe()
        self._w, self._h = info["width"], info["height"]
        self.fps = float(fps or info["fps"])
        self._n = info["frames"]
        self._cache: dict[int, np.ndarray] = {}
        self._cache_limit = int(cache_frames)
        self._all: np.ndarray | None = None

    def _probe(self) -> dict[str, Any]:
        if shutil.which(self.ffprobe) is None and not Path(self.ffprobe).is_file():
            raise RuntimeError("reading video files needs ffmpeg and ffprobe on PATH "
                               "(or pass a directory of image frames instead)")
        out = subprocess.run(
            [self.ffprobe, "-v", "error", "-select_streams", "v:0", "-count_packets",
             "-show_entries", "stream=width,height,r_frame_rate,nb_read_packets", "-of", "json",
             str(self.path)], check=True, capture_output=True, text=True).stdout
        s = json.loads(out)["streams"][0]
        num, den = (float(x) for x in s["r_frame_rate"].split("/"))
        return {"width": int(s["width"]), "height": int(s["height"]),
                "fps": num / den if den else 30.0, "frames": int(s.get("nb_read_packets", 0))}

    def __len__(self) -> int:
        return self._n

    @property
    def size(self) -> tuple[int, int]:
        return self._w, self._h

    def _decode_all(self) -> np.ndarray:
        proc = subprocess.run(
            [self.ffmpeg, "-v", "error", "-i", str(self.path), "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-vsync", "0", "-"], check=True, capture_output=True)
        frame_bytes = self._w * self._h * 3
        n = len(proc.stdout) // frame_bytes
        self._n = n
        return np.frombuffer(proc.stdout[: n * frame_bytes], np.uint8).reshape(n, self._h, self._w, 3)

    def read(self, index: int) -> np.ndarray:
        if self._all is None and self._n <= self._cache_limit:
            self._all = self._decode_all()
        if self._all is not None:
            return self._all[index]
        if index not in self._cache:
            t = index / self.fps
            proc = subprocess.run(
                [self.ffmpeg, "-v", "error", "-ss", f"{t:.6f}", "-i", str(self.path), "-frames:v", "1",
                 "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], check=True, capture_output=True)
            self._cache[index] = np.frombuffer(proc.stdout, np.uint8).reshape(self._h, self._w, 3)
        return self._cache[index]


def open_frames(path: str | Path, fps: float | None = None) -> FrameSource:
    """Open a video file (via ffmpeg), a directory of frames, or a ``.npy``/``.npz`` frame array."""
    p = Path(path)
    if p.is_dir():
        return ImageSequence(p, fps or 30.0)
    if p.suffix.lower() == ".npy":
        return ArrayFrames(np.load(p, mmap_mode="r"), fps or 30.0)
    if p.suffix.lower() == ".npz":
        z = np.load(p)
        return ArrayFrames(z["frames"], float(fps or z.get("fps", 30.0)))
    return FFmpegVideo(p, fps)


# --------------------------------------------------------------------------- images

def read_image(path: str | Path) -> np.ndarray:
    """Read an image as a uint8 array (H, W) or (H, W, 3|4)."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".png":
        return read_png(p)
    if suffix == ".npy":
        return np.load(p)
    try:
        from PIL import Image  # optional
    except ImportError:  # pragma: no cover - depends on the environment
        raise RuntimeError(f"reading {suffix} images needs Pillow; PNG frames work without it") from None
    with Image.open(p) as im:
        return np.asarray(im.convert("RGB") if im.mode not in ("L", "RGB") else im)


def read_mask(path: str | Path, threshold: int = 127) -> np.ndarray:
    img = read_image(path)
    if img.ndim == 3:
        img = img[..., :3].max(axis=2)
    return img > threshold


def write_png(path: str | Path, image: np.ndarray) -> None:
    """Write an 8-bit grey (H, W), RGB (H, W, 3) or RGBA (H, W, 4) PNG with the standard library."""
    a = np.ascontiguousarray(image)
    if a.dtype == bool:
        a = a.astype(np.uint8) * 255
    if a.dtype != np.uint8:
        raise ValueError("write_png needs uint8 or bool data")
    if a.ndim == 2:
        colour_type, channels = 0, 1
    elif a.shape[2] == 3:
        colour_type, channels = 2, 3
    elif a.shape[2] == 4:
        colour_type, channels = 6, 4
    else:
        raise ValueError(f"unsupported image shape {a.shape}")
    h, w = a.shape[:2]
    raw = np.empty((h, 1 + w * channels), np.uint8)
    raw[:, 0] = 0  # filter type None
    raw[:, 1:] = a.reshape(h, w * channels)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, colour_type, 0, 0, 0)
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw.tobytes(), 6)) \
        + chunk(b"IEND", b"")
    Path(path).write_bytes(data)


def read_png(path: str | Path) -> np.ndarray:
    """Decode a non-interlaced 8-bit PNG (grey, grey+alpha, RGB, RGBA or palette)."""
    data = Path(path).read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG file")
    pos = 8
    idat = []
    palette = None
    w = h = depth = ctype = interlace = None
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        tag = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            w, h, depth, ctype, _, _, interlace = struct.unpack(">IIBBBBB", body)
        elif tag == b"PLTE":
            palette = np.frombuffer(body, np.uint8).reshape(-1, 3)
        elif tag == b"IDAT":
            idat.append(body)
        elif tag == b"IEND":
            break
    if depth != 8 or interlace:
        raise ValueError(f"{path}: only 8-bit non-interlaced PNG is supported without Pillow")
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[ctype]
    raw = np.frombuffer(zlib.decompress(b"".join(idat)), np.uint8)
    stride = w * channels
    raw = raw.reshape(h, stride + 1)
    out = np.zeros((h, stride), np.uint8)
    bpp = channels
    prev = np.zeros(stride, np.int32)
    for r in range(h):
        ftype = raw[r, 0]
        line = raw[r, 1:].astype(np.int32)
        if ftype == 0:
            cur = line
        elif ftype == 1:
            cur = (np.cumsum(line.reshape(-1, bpp), axis=0) & 0xFF).reshape(-1)
        elif ftype == 2:
            cur = (line + prev) & 0xFF
        elif ftype == 3:
            cur = line.copy()
            for i in range(stride):
                left = cur[i - bpp] if i >= bpp else 0
                cur[i] = (cur[i] + ((left + prev[i]) >> 1)) & 0xFF
        elif ftype == 4:
            cur = line.copy()
            for i in range(stride):
                a = cur[i - bpp] if i >= bpp else 0
                b = prev[i]
                c = prev[i - bpp] if i >= bpp else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                cur[i] = (cur[i] + pred) & 0xFF
        else:
            raise ValueError(f"{path}: bad PNG filter type {ftype}")
        out[r] = cur
        prev = cur
    img = out.reshape(h, w, channels)
    if ctype == 3:
        return palette[img[..., 0]]
    if channels == 1:
        return img[..., 0]
    if channels == 2:
        return img[..., 0]
    return img
