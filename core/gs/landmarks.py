"""Layout of one MediaPipe Holistic frame as stored in the Kaggle ``asl-signs`` parquet files.

Each frame has 543 points in a fixed order: 468 face mesh points, 21 left-hand
points, 33 pose points and 21 right-hand points. Coordinates are normalised to
the image (x by width, y by height) and are NaN when a part was not detected.

The browser builds the same 543 x 3 array from the Hand and Pose (or Holistic)
landmarkers, so everything downstream indexes points through this module.
"""

from __future__ import annotations

from typing import Final

N_FACE: Final = 468
N_HAND: Final = 21
N_POSE: Final = 33
N_POINTS: Final = N_FACE + N_HAND + N_POSE + N_HAND  # 543

FACE_START: Final = 0
LEFT_HAND_START: Final = N_FACE
POSE_START: Final = LEFT_HAND_START + N_HAND
RIGHT_HAND_START: Final = POSE_START + N_POSE

TYPE_OFFSETS: Final[dict[str, int]] = {
    "face": FACE_START,
    "left_hand": LEFT_HAND_START,
    "pose": POSE_START,
    "right_hand": RIGHT_HAND_START,
}
TYPE_SIZES: Final[dict[str, int]] = {
    "face": N_FACE,
    "left_hand": N_HAND,
    "pose": N_POSE,
    "right_hand": N_HAND,
}

# Pose points kept as "upper body": nose, eyes, ears, shoulders, elbows, wrists.
# Hips and legs are usually out of a webcam's frame.
POSE_UPPER: Final = (0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16)
# The same points with left and right swapped, i.e. what a mirrored image shows.
POSE_UPPER_MIRROR: Final = (0, 5, 2, 8, 7, 12, 11, 14, 13, 16, 15)
POSE_LEFT_SHOULDER: Final = 11
POSE_RIGHT_SHOULDER: Final = 12
POSE_LEFT_WRIST: Final = 15
POSE_RIGHT_WRIST: Final = 16

# 40 face-mesh lip points as four runs, each listed from one mouth corner to the
# other, so reversing a run gives the mirror image of that run.
LIP_RUNS: Final = (
    (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291),  # outer lower
    (185, 40, 39, 37, 0, 267, 269, 270, 409),  # outer upper
    (78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308),  # inner lower
    (191, 80, 81, 82, 13, 312, 311, 310, 415),  # inner upper
)
LIPS: Final = tuple(i for run in LIP_RUNS for i in run)
LIPS_MIRROR: Final = tuple(i for run in LIP_RUNS for i in reversed(run))

# Hand points used for a scale reference: wrist and middle-finger MCP joint.
HAND_WRIST: Final = 0
HAND_MIDDLE_MCP: Final = 9

HAND_CONNECTIONS: Final = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)  # fmt: skip


def left_hand(i: int) -> int:
    return LEFT_HAND_START + i


def right_hand(i: int) -> int:
    return RIGHT_HAND_START + i


def pose(i: int) -> int:
    return POSE_START + i
