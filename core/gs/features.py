"""Landmark features shared by training and the browser (``web/src/ml/features.ts``).

Per frame:

1. Pick the hands, the upper-body pose and, for ``hands_pose_lips``, 40 lip points.
2. Centre on the shoulder midpoint and divide x by the shoulder width, using
   one anchor per sequence (the median over frames where both shoulders are
   seen). Divide y by the shoulder width times the face's vertical proportion
   (eye-to-mouth over ear-to-ear, from pose points) relative to its typical
   value in the training data. MediaPipe normalises x by image width and y by
   image height, so without this a landscape webcam would stretch every
   vertical distance relative to the portrait phone videos of asl-signs; the
   face proportion measures that stretch and cancels it.
3. Canonicalise handedness: when the left hand is the signing hand (seen in
   more frames; on a tie, moving more), mirror x and swap left and right so the
   model always sees the signing hand in the "dominant" slot. A mirrored webcam
   image therefore produces the same features as an unmirrored one.
4. Add each hand's shape relative to its wrist, scaled by the wrist to
   middle-knuckle length, so hand shape does not depend on where the hand is.
5. Replace NaN with 0 and append a presence flag per part.

Per sequence, frame-to-frame velocity of the position block is appended. Only
x and y are used: MediaPipe's z is estimated differently by the legacy Holistic
graph that produced the training data and the browser landmarkers.

Bump ``FEATURE_VERSION`` whenever the output changes; the browser refuses a
model whose ``featureVersion`` it does not implement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np
import numpy.typing as npt

from core.gs import landmarks as lm
Frames = npt.NDArray[np.float32]

FeatureSet = Literal["hands_pose", "hands_pose_lips"]
FEATURE_SETS: tuple[FeatureSet, ...] = get_args(FeatureSet)
FEATURE_VERSION = 2


@dataclass(frozen=True)
class Anchor:
    cx: float
    cy: float
    scale: float
    y_scale: float


# Used when no frame has both shoulders: image centre and a typical shoulder
# width for someone framed from the chest up.
DEFAULT_ANCHOR = Anchor(0.5, 0.5, 0.25, 0.25)
MIN_SCALE = 1e-3
# Median eye-to-mouth / ear-to-ear distance in asl-signs' normalised
# coordinates (63 clips from all 21 signers; spread about 8%).
FACE_RATIO_REF = 0.43
FACE_RATIO_RANGE = (0.2, 1.2)


@dataclass(frozen=True)
class FeatureLayout:
    """Where each block sits in the per-frame and per-sequence feature vectors."""

    dominant_hand: slice
    other_hand: slice
    pose: slice
    lips: slice | None
    positions: slice
    dominant_local: slice
    other_local: slice
    presence: slice
    frame_dim: int
    velocity: slice
    sequence_dim: int


def feature_layout(feature_set: FeatureSet) -> FeatureLayout:
    hand = lm.N_HAND * 2
    pose = len(lm.POSE_UPPER) * 2
    lips = len(lm.LIPS) * 2 if feature_set == "hands_pose_lips" else 0
    n_pos = 2 * hand + pose + lips
    n_parts = 4 if lips else 3
    frame_dim = n_pos + 2 * hand + n_parts
    return FeatureLayout(
        dominant_hand=slice(0, hand),
        other_hand=slice(hand, 2 * hand),
        pose=slice(2 * hand, 2 * hand + pose),
        lips=slice(2 * hand + pose, n_pos) if lips else None,
        positions=slice(0, n_pos),
        dominant_local=slice(n_pos, n_pos + hand),
        other_local=slice(n_pos + hand, n_pos + 2 * hand),
        presence=slice(n_pos + 2 * hand, frame_dim),
        frame_dim=frame_dim,
        velocity=slice(frame_dim, frame_dim + n_pos),
        sequence_dim=frame_dim + n_pos,
    )


def frame_feature_dim(feature_set: FeatureSet) -> int:
    return feature_layout(feature_set).frame_dim


def sequence_feature_dim(feature_set: FeatureSet) -> int:
    return feature_layout(feature_set).sequence_dim


def face_ratio(frames: Frames) -> float | None:
    """Median eye-to-mouth over ear-to-ear distance (pose points), or None if never seen."""
    p = frames[:, :, :2].astype(np.float64)
    eyes = (p[:, lm.pose(2)] + p[:, lm.pose(5)]) / 2
    mouth = (p[:, lm.pose(9)] + p[:, lm.pose(10)]) / 2
    ears = p[:, lm.pose(7)] - p[:, lm.pose(8)]
    vertical = np.hypot(eyes[:, 0] - mouth[:, 0], eyes[:, 1] - mouth[:, 1])
    horizontal = np.hypot(ears[:, 0], ears[:, 1])
    valid = np.isfinite(vertical) & np.isfinite(horizontal) & (horizontal > 1e-4)
    if not valid.any():
        return None
    ratio = float(np.median(vertical[valid] / horizontal[valid]))
    return min(max(ratio, FACE_RATIO_RANGE[0]), FACE_RATIO_RANGE[1])


def compute_anchor(frames: Frames) -> Anchor:
    """Shoulder midpoint and width (medians over frames showing both shoulders) and y scale."""
    left = frames[:, lm.pose(lm.POSE_LEFT_SHOULDER), :2].astype(np.float64)
    right = frames[:, lm.pose(lm.POSE_RIGHT_SHOULDER), :2].astype(np.float64)
    valid = np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1)
    if valid.any():
        left, right = left[valid], right[valid]
        centre = (left + right) / 2
        width = np.hypot(left[:, 0] - right[:, 0], left[:, 1] - right[:, 1])
        scale = float(np.median(width))
        if not scale > MIN_SCALE:
            scale = DEFAULT_ANCHOR.scale
        cx, cy = float(np.median(centre[:, 0])), float(np.median(centre[:, 1]))
    else:
        cx, cy, scale = DEFAULT_ANCHOR.cx, DEFAULT_ANCHOR.cy, DEFAULT_ANCHOR.scale
    ratio = face_ratio(frames)
    y_scale = scale * ratio / FACE_RATIO_REF if ratio is not None else scale
    return Anchor(cx, cy, scale, y_scale)


def _part_present(frames: Frames, start: int, size: int) -> npt.NDArray[np.bool_]:
    return np.isfinite(frames[:, start : start + size, 0]).any(axis=1)


def _wrist_travel(frames: Frames, start: int) -> float:
    """Total wrist movement in x/y between consecutive frames that both show the hand."""
    wrist = frames[:, start + lm.HAND_WRIST, :2].astype(np.float64)
    if wrist.shape[0] < 2:
        return 0.0
    step = np.abs(np.diff(wrist, axis=0)).sum(axis=1)
    return float(np.nansum(step))


def left_is_dominant(frames: Frames) -> bool:
    """True when the left hand is the signing hand.

    The signing hand is the one seen in more frames; on a tie, the one whose
    wrist travels further. Both rules swap exactly when the image is mirrored,
    so a mirrored clip always canonicalises to the same features.
    """
    left = int(_part_present(frames, lm.LEFT_HAND_START, lm.N_HAND).sum())
    right = int(_part_present(frames, lm.RIGHT_HAND_START, lm.N_HAND).sum())
    if left != right:
        return left > right
    return _wrist_travel(frames, lm.LEFT_HAND_START) > _wrist_travel(frames, lm.RIGHT_HAND_START)


def _indices(feature_set: FeatureSet, mirror: bool) -> tuple[list[int], list[int], list[int]]:
    """Source point indices for (dominant hand, other hand, pose + lips)."""
    left = list(range(lm.LEFT_HAND_START, lm.LEFT_HAND_START + lm.N_HAND))
    right = list(range(lm.RIGHT_HAND_START, lm.RIGHT_HAND_START + lm.N_HAND))
    pose = [lm.pose(i) for i in (lm.POSE_UPPER_MIRROR if mirror else lm.POSE_UPPER)]
    lips = list(lm.LIPS_MIRROR if mirror else lm.LIPS) if feature_set == "hands_pose_lips" else []
    dominant, other = (left, right) if mirror else (right, left)
    return dominant, other, pose + lips


def _hand_local(hand: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """[T, 21, 2] normalised hand points -> shape relative to the wrist."""
    rel = hand - hand[:, lm.HAND_WRIST : lm.HAND_WRIST + 1]
    ref = rel[:, lm.HAND_MIDDLE_MCP]
    size = np.hypot(ref[:, 0], ref[:, 1])
    size = np.where(np.isfinite(size) & (size > 1e-6), size, 1.0)
    out: npt.NDArray[np.float64] = rel / size[:, None, None]
    return out


def _frames_features(
    frames: Frames, feature_set: FeatureSet, anchor: Anchor, mirror: bool
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Per-frame features [T, F] (NaN already zeroed) and raw positions [T, P*2] (with NaN)."""
    T = frames.shape[0]
    dominant, other, body = _indices(feature_set, mirror)
    xy = frames[:, :, :2].astype(np.float64)
    centre = np.array([anchor.cx, anchor.cy])
    scale = np.array([anchor.scale, anchor.y_scale])
    sign = np.array([-1.0 if mirror else 1.0, 1.0])

    def norm(idx: list[int]) -> npt.NDArray[np.float64]:
        out: npt.NDArray[np.float64] = (xy[:, idx] - centre) / scale * sign
        return out

    dom, oth, bod = norm(dominant), norm(other), norm(body)
    positions = np.concatenate([dom.reshape(T, -1), oth.reshape(T, -1), bod.reshape(T, -1)], axis=1)
    local = np.concatenate(
        [_hand_local(dom).reshape(T, -1), _hand_local(oth).reshape(T, -1)], axis=1
    )
    n_pose = len(lm.POSE_UPPER)
    presence = [
        np.isfinite(dom[:, :, 0]).any(axis=1),
        np.isfinite(oth[:, :, 0]).any(axis=1),
        np.isfinite(bod[:, :n_pose, 0]).any(axis=1),
    ]
    if feature_set == "hands_pose_lips":
        presence.append(np.isfinite(bod[:, n_pose:, 0]).any(axis=1))
    flags = np.stack(presence, axis=1).astype(np.float64)
    features = np.concatenate([positions, local, flags], axis=1)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0), positions


def frame_features(
    frame: Frames,
    feature_set: FeatureSet,
    anchor: Anchor | None = None,
    mirror: bool | None = None,
) -> npt.NDArray[np.float32]:
    """Features for one [543, 3] frame.

    Without ``anchor`` and ``mirror`` the frame is treated as a sequence of one:
    its own shoulders anchor it, and it is mirrored when only the left hand shows.
    """
    frames = frame[None]
    if anchor is None:
        anchor = compute_anchor(frames)
    if mirror is None:
        mirror = left_is_dominant(frames)
    features, _ = _frames_features(frames, feature_set, anchor, mirror)
    out: npt.NDArray[np.float32] = features[0].astype(np.float32)
    return out


def sequence_features(frames: Frames, feature_set: FeatureSet) -> npt.NDArray[np.float32]:
    """Features for a [T, 543, 3] sequence: per-frame features plus velocity, [T, F2]."""
    layout = feature_layout(feature_set)
    T = frames.shape[0]
    if T == 0:
        return np.zeros((0, layout.sequence_dim), dtype=np.float32)
    anchor = compute_anchor(frames)
    mirror = left_is_dominant(frames)
    features, positions = _frames_features(frames, feature_set, anchor, mirror)
    velocity = np.zeros_like(positions)
    seen = np.isfinite(positions)
    both = seen[1:] & seen[:-1]
    velocity[1:] = np.where(both, positions[1:] - positions[:-1], 0.0)
    return np.concatenate([features, velocity], axis=1).astype(np.float32)


# Pose points 1-32 swap with their mirror partner; 0 (nose) stays.
_POSE_MIRROR = [0, 4, 5, 6, 1, 2, 3, 8, 7, 10, 9] + [
    i + 1 if i % 2 else i - 1 for i in range(11, 33)
]


def mirror_landmarks(frames: Frames) -> Frames:
    """What the same scene looks like in a horizontally flipped image.

    x becomes 1 - x, the left and right hands swap, and paired pose and lip
    points swap. The rest of the face mesh is only flipped, not re-indexed,
    because no feature reads it.
    """
    out = frames.copy()
    out[..., 0] = 1.0 - frames[..., 0]
    flipped = out.copy()
    left = slice(lm.LEFT_HAND_START, lm.LEFT_HAND_START + lm.N_HAND)
    right = slice(lm.RIGHT_HAND_START, lm.RIGHT_HAND_START + lm.N_HAND)
    out[..., left, :] = flipped[..., right, :]
    out[..., right, :] = flipped[..., left, :]
    pose_src = [lm.pose(i) for i in _POSE_MIRROR]
    out[..., lm.POSE_START : lm.POSE_START + lm.N_POSE, :] = flipped[..., pose_src, :]
    out[..., list(lm.LIPS), :] = flipped[..., list(lm.LIPS_MIRROR), :]
    return out
