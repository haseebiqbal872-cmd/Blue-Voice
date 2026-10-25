"""
Video source abstraction for the Gesture-to-Text AI Copilot.

Provides a dual-input system:
  * Live hardware camera (CameraSource)
  * Mock video simulation from a local MP4 file (MockVideoSource)

The VideoSourceManager attempts to open a live webcam and, when none is
available, falls back to MP4 file mocking so hand tracking can be calibrated
without a physical camera.
"""

from __future__ import annotations

import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional


class VideoSource(ABC):
    """Abstract interface for a video frame provider."""

    @abstractmethod
    def start(self) -> bool:
        """Open the underlying capture device / file. Returns True on success."""

    @abstractmethod
    def stop(self) -> None:
        """Release the underlying capture resource."""

    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """Return the next frame as a BGR numpy array, or None at EOF / error."""

    @abstractmethod
    def is_opened(self) -> bool:
        """Return whether the source is currently open."""

    @abstractmethod
    def source_name(self) -> str:
        """Human-readable name of the active source."""


class CameraSource(VideoSource):
    """Live hardware webcam capture via OpenCV VideoCapture."""

    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self._cap: Optional[cv2.VideoCapture] = None

    def start(self) -> bool:
        self.stop()
        self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self._cap or not self._cap.isOpened():
            self._cap = None
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, 30)
        return True

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read(self) -> Optional[np.ndarray]:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return frame

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def source_name(self) -> str:
        return f"Live Webcam (index {self.camera_index})"


class MockVideoSource(VideoSource):
    """MP4 file video-mocking source for webcam-free testing.

    Feeds frames from a local MP4 file frame-by-frame directly into the
    MediaPipe processing loop. The video can loop continuously so a short
    calibration clip can be reused indefinitely.
    """

    def __init__(self, mp4_path: str, loop: bool = True) -> None:
        self.mp4_path = mp4_path
        self.loop = loop
        self._cap: Optional[cv2.VideoCapture] = None
        self._eof = False

    def start(self) -> bool:
        self.stop()
        self._cap = cv2.VideoCapture(self.mp4_path)
        if not self._cap or not self._cap.isOpened():
            self._cap = None
            self._eof = True
            return False
        self._eof = False
        return True

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read(self) -> Optional[np.ndarray]:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            if self.loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    self._eof = True
                    return None
            else:
                self._eof = True
                return None
        return frame

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def source_name(self) -> str:
        return f"Mock Video: {self.mp4_path}"

    @property
    def eof(self) -> bool:
        return self._eof


class SimulatedVideoSource(VideoSource):
    """Synthetic video source that generates a moving digital hand skeleton
    on a black screen, frame-by-frame.

    Requires no webcam, no MP4 file, and no MediaPipe model asset. Each call
    to :meth:`read` renders a fresh black frame with an animated 21-landmark
    hand skeleton (shifting coordinates driven by sine waves), making it an
    instant fallback for testing the particle animations and the gesture ->
    text response loop offline.
    """

    def __init__(self, width: int = 1280, height: int = 720, fps: float = 30.0) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self._cap = None  # kept for interface parity; not actually used
        self._started = False
        self._t = 0.0

    def start(self) -> bool:
        self.stop()
        self._started = True
        self._t = 0.0
        return True

    def stop(self) -> None:
        self._started = False
        self._cap = None

    def read(self) -> Optional[np.ndarray]:
        """Return the next synthetic blank frame.

        No skeleton or on-screen text is rendered: the user must never see
        their own hand or any digital skeleton lines. The landmarks are still
        generated and fed to the gesture recognizer so gestures continue to
        drive text input headlessly.
        """
        if not self._started:
            return None
        self._t += 1.0 / self.fps
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def is_opened(self) -> bool:
        return self._started

    def source_name(self) -> str:
        return "Simulated Test Mode"

    # ------------------------------------------------------------------ #
    # Synthetic skeleton generation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _skeleton(t: float) -> List[tuple]:
        """Return 21 (x, y, z) MediaPipe-style normalised landmarks for an
        animated hand cycling: open palm -> fist -> swipe right -> swipe left."""
        import math

        cycle = (t % 12.0) / 12.0

        cx, cy = 0.50, 0.78      # wrist home position
        curl = 0.0               # 0 = open palm, 1 = fist
        swipe_x = 0.0            # index-tip swipe offset

        if cycle < 0.25:
            curl = 0.0
            cx += 0.004 * math.sin(t * 3.0)
            cy += 0.003 * math.cos(t * 2.0)
        elif cycle < 0.375:
            curl = (cycle - 0.25) / 0.125
            cx += 0.05 * math.sin(t * 2.0)
        elif cycle < 0.50:
            curl = 1.0
            cx += 0.06 * math.sin(t * 2.0)
            cy += 0.04 * math.cos(t * 2.5)
        elif cycle < 0.625:
            curl = 1.0 - (cycle - 0.50) / 0.125
            cx += 0.05 * math.sin(t * 2.0)
        elif cycle < 0.70:
            curl = 0.0
            swipe_x = 0.17 * min(1.0, (cycle - 0.625) / 0.04)
            cx = 0.42 + 0.08 * math.sin(t * 4.0)
        elif cycle < 0.825:
            curl = 0.0
            swipe_x = -0.17 * min(1.0, (cycle - 0.75) / 0.04)
            cx = 0.58 + 0.08 * math.sin(t * 4.0)
        else:
            curl = 0.0
            cx += 0.05 * math.sin(t * 2.0)

        rot = 0.12 * math.sin(t * 1.5)

        def rotated(px: float, py: float) -> tuple:
            c, s = math.cos(rot), math.sin(rot)
            return (c * px - s * py, s * px + c * py)

        # Per-finger knuckle (mcp), fully-extended tip, and curled tip.
        finger_mcp = {
            5: (0.00, -0.18),   # index
            9: (0.06, -0.17),   # middle
            13: (-0.06, -0.19), # ring
            17: (-0.12, -0.20), # pinky
        }
        finger_tip_extended = {
            5: (0.00, -0.55),   # index
            9: (0.06, -0.56),   # middle
            13: (-0.06, -0.56), # ring
            17: (-0.12, -0.51), # pinky
        }
        finger_tip_curled = {
            5: (0.05, -0.02),   # index
            9: (0.08, -0.02),   # middle
            13: (-0.02, 0.00),  # ring
            17: (-0.06, 0.02),  # pinky
        }
        thumb = {
            1: (-0.04, -0.08),  # cmc
            2: (-0.07, -0.14),  # mcp
            3: (-0.10, -0.19),  # ip
            4: (-0.12, -0.23),  # tip (extended)
        }
        thumb_tip_curled = (-0.08, 0.12)  # across the palm (fist)

        landmarks: List[tuple] = [(cx, cy, 0.0)]  # 0 = wrist

        # Fingers 5-20 (index, middle, ring, pinky): mcp + 3 joints.
        for base in (5, 9, 13, 17):
            mcp = finger_mcp[base]
            tip_e = finger_tip_extended[base]
            tip_c = finger_tip_curled[base]
            # Lerp the fingertip between extended and curled by curl amount.
            tip = (tip_e[0] * (1.0 - curl) + tip_c[0] * curl,
                   tip_e[1] * (1.0 - curl) + tip_c[1] * curl)
            for i in range(4):
                frac = i / 3.0
                px = mcp[0] + (tip[0] - mcp[0]) * frac
                py = mcp[1] + (tip[1] - mcp[1]) * frac
                px, py = rotated(px, py)
                landmarks.append((cx + px, cy + py, curl * 0.1))

        # Thumb: 0 wrist already added; 1 cmc, 2 mcp, 3 ip, 4 tip.
        for idx, base_pt in thumb.items():
            px, py = rotated(base_pt[0], base_pt[1])
            px_t = cx + (px * (1.0 - curl) + thumb_tip_curled[0] * curl) if idx == 4 else cx + px
            py_t = cy + (py * (1.0 - curl) + thumb_tip_curled[1] * curl) if idx == 4 else cy + py
            landmarks.append((px_t, py_t, curl * 0.1))

        # Apply swipe offset to index finger tip (landmark 8) only.
        landmarks[8] = (landmarks[8][0] + swipe_x, landmarks[8][1], landmarks[8][2])

        # Clamp into [0, 1] normalised range.
        return [(min(1.0, max(0.0, x)), min(1.0, max(0.0, y)), z) for x, y, z in landmarks]

class VideoSourceManager:
    """Manages the dual-input switch and webcam availability detection."""

    def __init__(self) -> None:
        self._source: Optional[VideoSource] = None

    def detect_webcam(self, camera_index: int = 0) -> bool:
        """Probe whether a live webcam is available at the given index."""
        probe = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        available = probe is not None and probe.isOpened()
        probe.release()
        return available

    def start_live(self, camera_index: int = 0, width: int = 1280, height: int = 720) -> bool:
        """Switch to live hardware camera mode."""
        src = CameraSource(camera_index, width, height)
        if src.start():
            self._source = src
            return True
        return False

    def start_mock(self, mp4_path: str, loop: bool = True) -> bool:
        """Switch to MP4 mock-video mode."""
        src = MockVideoSource(mp4_path, loop)
        if src.start():
            self._source = src
            return True
        return False

    def start_simulated(self, width: int = 1280, height: int = 720, fps: float = 30.0) -> bool:
        """Switch to simulated test mode: a synthetic hand skeleton on a
        black screen, generated frame-by-frame with no external input."""
        src = SimulatedVideoSource(width, height, fps)
        if src.start():
            self._source = src
            return True
        return False

    def stop(self) -> None:
        if self._source is not None:
            self._source.stop()
            self._source = None

    def read(self) -> Optional[np.ndarray]:
        if self._source is None:
            return None
        return self._source.read()

    def is_opened(self) -> bool:
        return self._source is not None and self._source.is_opened()

    def source_name(self) -> str:
        return self._source.source_name() if self._source else "None"

    @property
    def source(self) -> Optional[VideoSource]:
        return self._source
