"""Time resampling shared by training and the browser (``web/src/ml/features.ts``).

Both sides must produce the same frames, so the arithmetic is spelled out and
kept in the same order as the TypeScript port: positions are ``k * src / dst``
in float64, values are linearly interpolated, and a missing (NaN) neighbour
makes the output take the nearer of the two source frames as is.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating]


def resampled_length(T: int, src_fps: float, dst_fps: float) -> int:
    """Number of output frames covering the same duration as ``T`` input frames."""
    if T <= 1:
        return T
    return math.floor((T - 1) * dst_fps / src_fps + 1e-9) + 1


def interpolate(seq: FloatArray, positions: npt.NDArray[np.float64]) -> npt.NDArray[np.float32]:
    """Sample ``seq`` (time on axis 0) at fractional frame ``positions``."""
    T = seq.shape[0]
    src = seq.astype(np.float64)
    i0 = np.minimum(np.floor(positions).astype(np.int64), T - 1)
    i1 = np.minimum(i0 + 1, T - 1)
    w = (positions - i0).reshape((-1,) + (1,) * (seq.ndim - 1))
    a = src[i0]
    b = src[i1]
    both = np.isfinite(a) & np.isfinite(b)
    nearest = np.where(w < 0.5, a, b)
    out: npt.NDArray[np.float64] = np.where(both, a + (b - a) * w, nearest)
    return out.astype(np.float32)


def resample(seq: FloatArray, src_fps: float, dst_fps: float = 30.0) -> npt.NDArray[np.float32]:
    """Resample a sequence recorded at ``src_fps`` to ``dst_fps``.

    The first output frame is the first input frame and the output never runs
    past the last input frame.
    """
    T = seq.shape[0]
    n = resampled_length(T, src_fps, dst_fps)
    if n == T and src_fps == dst_fps:
        return seq.astype(np.float32, copy=False)
    if T <= 1:
        return seq.astype(np.float32, copy=False)
    positions = np.arange(n, dtype=np.float64) * src_fps / dst_fps
    return interpolate(seq, positions)


def fit_length(seq: FloatArray, max_len: int) -> FloatArray:
    """Squeeze sequences longer than ``max_len`` onto exactly ``max_len`` frames.

    Shorter sequences are returned unchanged (the same object). Squeezing keeps
    the whole sign instead of cropping part of it away.
    """
    T = seq.shape[0]
    if max_len >= T:
        return seq
    positions = np.arange(max_len, dtype=np.float64) * (T - 1) / (max_len - 1)
    return interpolate(seq, positions)
