"""Turn the scene's OBJ meshes and the vehicles' STL meshes into flat vertex
buffers a browser can draw directly: float32 [x, y, z, u, v] per triangle corner."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np


def _obj(path: Path) -> np.ndarray:
    pos, uv, corners = [], [], []
    with open(path, "r") as fh:
        for line in fh:
            if line.startswith("v "):
                pos.append(line.split()[1:4])
            elif line.startswith("vt "):
                uv.append(line.split()[1:3])
            elif line.startswith("f "):
                face = [c.split("/") for c in line.split()[1:]]
                idx = [(int(c[0]), int(c[1]) if len(c) > 1 and c[1] else 0) for c in face]
                for k in range(1, len(idx) - 1):            # triangles as they are, polygons as a fan
                    corners += [idx[0], idx[k], idx[k + 1]]
    pos = np.asarray(pos, np.float32).reshape(-1, 3)
    uv = np.asarray(uv, np.float32).reshape(-1, 2)
    c = np.asarray(corners, np.int64).reshape(-1, 2)
    out = np.zeros((len(c), 5), np.float32)
    vi = np.where(c[:, 0] > 0, c[:, 0] - 1, c[:, 0] + len(pos))          # OBJ indices are 1-based or negative
    out[:, :3] = pos[vi]
    has_uv = c[:, 1] != 0
    if len(uv) and has_uv.any():
        ti = np.where(c[:, 1] > 0, c[:, 1] - 1, c[:, 1] + len(uv))
        out[has_uv, 3:] = uv[ti[has_uv]]
    return out


def _stl(path: Path, scale: float) -> np.ndarray:
    raw = path.read_bytes()
    n = int(np.frombuffer(raw, "<u4", 1, 80)[0])
    if len(raw) != 84 + 50 * n:
        raise ValueError(f"{path} is not a binary STL file")
    tri = np.frombuffer(raw, np.dtype([("n", "<f4", 3), ("v", "<f4", 9), ("a", "<u2")]), n, 84)
    out = np.zeros((3 * n, 5), np.float32)
    out[:, :3] = tri["v"].reshape(-1, 3) * scale
    return out


@lru_cache(maxsize=64)
def _load(path: str, mtime: float, scale: float) -> bytes:
    p = Path(path)
    data = _stl(p, scale) if p.suffix.lower() == ".stl" else _obj(p)
    return data.tobytes()


def vertex_buffer(path: str | Path, scale: float = 1.0) -> bytes:
    """Bytes of the float32 [x, y, z, u, v] buffer for an .obj or binary .stl file (cached)."""
    p = Path(path).resolve()
    return _load(str(p), p.stat().st_mtime, scale)
