"""
Hand-gesture recognition for the Gesture-to-Text AI Copilot.

Recognises the system workflow gestures defined in the product spec:

  * Open, flat 5-finger palm  -> freeze-frame submission trigger
  * Rapid swipe left           -> instant backspace / undo
  * Rapid swipe right          -> instant spacebar / word gap

Finger state is derived from MediaPipe hand landmarks (21 landmarks per hand).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

import numpy as np


class Gesture(Enum):
    NONE = auto()
    OPEN_PALM = auto()      # 5-finger flat palm held static
    SWIPE_LEFT = auto()     # backspace / undo
    SWIPE_RIGHT = auto()    # spacebar / word gap


@dataclass
class GestureEvent:
    """A single recognised gesture emission."""
    gesture: Gesture = Gesture.NONE
    timestamp: float = 0.0
    hand_landmarks: Optional[List[tuple]] = None
    confidence: float = 0.0


class HandGestureRecognizer:
    """Stateful gesture recogniser operating on a stream of landmark sets."""

    # MediaPipe hand landmark indices
    TIP_IDS = [4, 8, 12, 16, 20]          # thumb, index, middle, ring, pinky tips
    PIP_IDS = [3, 6, 10, 14, 18]          # corresponding proximal interphalangeal joints

    # Swipe detection thresholds (in normalised image units)
    SWIPE_DISTANCE = 0.12                 # minimum index-tip travel for a swipe
    SWIPE_TIME_WINDOW = 0.6               # seconds within which the swipe must occur
    SWIPE_COOLDOWN = 0.9                  # seconds before another swipe is registered

    # Freeze-frame stability thresholds
    PALM_STABLE_DISTANCE = 0.025          # max wrist drift while palm held open
    PALM_HOLD_SECONDS = 1.5               # how long the palm must stay static

    def __init__(self) -> None:
        self._position_history: List[tuple] = []   # (timestamp, x, y) of index tip
        self._swipe_cooldown_until = 0.0
        self._palm_open_start: Optional[float] = None
        self._palm_open_wrist: Optional[tuple] = None

    # ------------------------------------------------------------------ #
    # Finger state
    # ------------------------------------------------------------------ #
    def finger_states(self, landmarks: List[tuple]) -> List[bool]:
        """Return [thumb, index, middle, ring, pinky] extended flags.

        ``landmarks`` is a list of 21 (x, y, z) tuples in normalised
        image coordinates (y increases downward).
        """
        states: List[bool] = []
        for tip, pip in zip(self.TIP_IDS, self.PIP_IDS):
            tip_pt = landmarks[tip]
            pip_pt = landmarks[pip]
            if tip == 4:  # thumb: extended when tip is further from wrist than pip
                wrist = landmarks[0]
                tip_dist = np.hypot(tip_pt[0] - wrist[0], tip_pt[1] - wrist[1])
                pip_dist = np.hypot(pip_pt[0] - wrist[0], pip_pt[1] - wrist[1])
                states.append(tip_dist > pip_dist + 0.02)
            else:  # fingers: extended when tip is above the PIP joint (smaller y)
                states.append(tip_pt[1] < pip_pt[1] - 0.015)
        return states

    def is_open_palm(self, landmarks: List[tuple]) -> bool:
        """True when all five fingers are extended (flat open palm)."""
        return all(self.finger_states(landmarks))

    # ------------------------------------------------------------------ #
    # Swipe detection
    # ------------------------------------------------------------------ #
    def _record_index_tip(self, timestamp: float, landmarks: List[tuple]) -> None:
        index_tip = landmarks[8]
        self._position_history.append((timestamp, float(index_tip[0]), float(index_tip[1])))
        cutoff = timestamp - self.SWIPE_TIME_WINDOW
        self._position_history = [p for p in self._position_history if p[0] >= cutoff]

    def _detect_swipe(self, timestamp: float) -> Optional[Gesture]:
        if timestamp < self._swipe_cooldown_until:
            return None
        if len(self._position_history) < 3:
            return None
        start = self._position_history[0]
        end = self._position_history[-1]
        dx = end[1] - start[1]
        dy = abs(end[2] - start[2])
        dt = end[0] - start[0]
        if dt <= 0:
            return None
        if abs(dx) > self.SWIPE_DISTANCE and abs(dx) > 2.0 * dy:
            if dx < 0:
                self._swipe_cooldown_until = timestamp + self.SWIPE_COOLDOWN
                self._position_history.clear()
                return Gesture.SWIPE_LEFT
            else:
                self._swipe_cooldown_until = timestamp + self.SWIPE_COOLDOWN
                self._position_history.clear()
                return Gesture.SWIPE_RIGHT
        return None

    # ------------------------------------------------------------------ #
    # Freeze-frame (open palm stability) detection
    # ------------------------------------------------------------------ #
    def _update_palm_hold(self, timestamp: float, landmarks: List[tuple]) -> Optional[Gesture]:
        if not self.is_open_palm(landmarks):
            self._palm_open_start = None
            self._palm_open_wrist = None
            return None

        wrist = landmarks[0]
        if self._palm_open_start is None:
            self._palm_open_start = timestamp
            self._palm_open_wrist = wrist
            return None

        if self._palm_open_wrist is not None:
            drift = np.hypot(wrist[0] - self._palm_open_wrist[0],
                             wrist[1] - self._palm_open_wrist[1])
            if drift > self.PALM_STABLE_DISTANCE:
                # Palm moved - reset the stability timer
                self._palm_open_start = timestamp
                self._palm_open_wrist = wrist
                return None

        held = timestamp - self._palm_open_start
        if held >= self.PALM_HOLD_SECONDS:
            gesture = Gesture.OPEN_PALM
            self._palm_open_start = None
            self._palm_open_wrist = None
            return gesture
        return None

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    def process(self, timestamp: float, landmarks: List[tuple]) -> GestureEvent:
        """Analyse one frame of landmarks and return any recognised gesture."""
        self._record_index_tip(timestamp, landmarks)

        swipe = self._detect_swipe(timestamp)
        if swipe is not None:
            return GestureEvent(gesture=swipe, timestamp=timestamp,
                                hand_landmarks=landmarks, confidence=1.0)

        palm = self._update_palm_hold(timestamp, landmarks)
        if palm is not None:
            return GestureEvent(gesture=palm, timestamp=timestamp,
                                hand_landmarks=landmarks, confidence=1.0)

        return GestureEvent(gesture=Gesture.NONE, timestamp=timestamp,
                            hand_landmarks=landmarks)

    def reset(self) -> None:
        """Clear all gesture state (call when tracking stops / restarts)."""
        self._position_history.clear()
        self._swipe_cooldown_until = 0.0
        self._palm_open_start = None
        self._palm_open_wrist = None
