"""
MediaPipe hand-tracking thread for the Gesture-to-Text AI Copilot.

Runs the camera / mock-video capture and MediaPipe Hands inference on a
background thread so the app stays responsive. Reports through callbacks:

  * processed video frames (with landmark overlay, ready to display)
  * recognised workflow gestures (open palm, swipe left, swipe right)

Supports the dual-input system: live hardware camera or MP4 file mocking.

MediaPipe backend
-----------------
Uses the MediaPipe *Tasks* Vision API (``HandLandmarker``), which is the
runtime-backed API shipped with current mediapipe wheels.  It requires a
hand landmarker model asset (``hand_landmarker.task``).  The tracker
resolves the asset from:

  1. ``assets/models/hand_landmarker.task``  (dev / source tree)
  2. ``models/hand_landmarker.task``         (next to the executable)
  3. ``hand_landmarker.task``                (next to main.py)

If none is found it attempts a one-time download of the official MediaPipe
hand landmarker model from Google Cloud Storage.  If that also fails it
emits :attr:`error_occurred` with a clear, actionable message instead of
silently failing.
"""

from __future__ import annotations

import os
import sys
import time
import urllib.request
from typing import List, Optional

import cv2
import numpy as np
import threading

import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarker,
    HandLandmarkerOptions,
)

from .video_source import SimulatedVideoSource, VideoSourceManager
from .gesture_recognizer import HandGestureRecognizer, GestureEvent, Gesture


# Official MediaPipe hand landmarker model (float16, hosted by Google).
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_FILENAME = "hand_landmarker.task"

class HandTrackerThread(threading.Thread):
    """Background thread that captures video, runs MediaPipe Hands and
    reports landmarks and workflow gestures through plain callbacks.

    Callbacks (all optional, called from the tracker thread):
      on_landmarks(timestamp_s, points_or_None)  every processed frame
      on_gesture(GestureEvent)                   palm hold / swipes
      on_state(state, source_name)               "started" / "stopped"
      on_error(message)
    """

    def __init__(self, on_landmarks=None, on_gesture=None, on_state=None, on_error=None) -> None:
        super().__init__(daemon=True)
        self.on_landmarks = on_landmarks
        self.on_gesture = on_gesture
        self.on_state = on_state
        self.on_error = on_error
        self._source_manager = VideoSourceManager()
        self._recognizer = HandGestureRecognizer()
        self._running = False
        self._simulated = False
        self._model_path = None
        self._process_every_n_frames = 1

    def _error(self, message: str) -> None:
        if self.on_error:
            self.on_error(message)

    def _state(self, state: str) -> None:
        if self.on_state:
            self.on_state(state, self.source_name)

    # ------------------------------------------------------------------ #
    # Model asset resolution
    # ------------------------------------------------------------------ #
    @staticmethod
    def _candidate_model_dirs() -> List[str]:
        dirs: List[str] = []
        # Dev / source tree
        dirs.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "models"))
        # Next to the executable / script
        if getattr(sys, "frozen", False):
            dirs.append(os.path.dirname(sys.executable))
        else:
            dirs.append(os.path.dirname(os.path.abspath(sys.argv[0] if sys.argv[0] else __file__)))
        # Current working directory
        dirs.append(os.getcwd())
        return dirs

    @classmethod
    def resolve_model_path(cls) -> Optional[str]:
        """Return an existing model path, or None."""
        for d in cls._candidate_model_dirs():
            candidate = os.path.join(d, MODEL_FILENAME)
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
        return None

    @classmethod
    def ensure_model(cls, timeout_s: float = 25.0) -> Optional[str]:
        """Ensure a model asset exists, downloading it on first use if needed.

        Returns the model path, or None if it cannot be obtained.
        """
        existing = cls.resolve_model_path()
        if existing:
            return existing

        # Best-effort one-time download of the official MediaPipe model.
        target_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "models")
        try:
            os.makedirs(target_dir, exist_ok=True)
        except OSError:
            target_dir = os.getcwd()
        target = os.path.join(target_dir, MODEL_FILENAME)

        try:
            req = urllib.request.Request(
                MODEL_URL,
                headers={"User-Agent": "BlueVoice/1.0"},
            )
            with urllib.request.urlopen(req, timeout=timeout_s) as resp, open(target, "wb") as out:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    out.write(chunk)
            if os.path.isfile(target) and os.path.getsize(target) > 100000:
                return os.path.abspath(target)
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------ #
    # Public control API
    # ------------------------------------------------------------------ #
    def start_live_camera(self, camera_index: int = 0,
                          width: int = 1280, height: int = 720) -> bool:
        """Begin tracking from the live hardware webcam."""
        model_path = self.ensure_model()
        if model_path is None:
            self._error(
                "MediaPipe hand landmarker model not found. "
                "Place 'hand_landmarker.task' in assets/models/ "
                "(official MediaPipe model) or ensure network access for "
                "auto-download."
            )
            return False
        ok = self._source_manager.start_live(camera_index, width, height)
        if ok:
            self._model_path = model_path
            self.start()
        return ok

    def start_mock_video(self, mp4_path: str, loop: bool = True) -> bool:
        """Begin tracking from an MP4 file (webcam-free mock mode)."""
        model_path = self.ensure_model()
        if model_path is None:
            self._error(
                "MediaPipe hand landmarker model not found. "
                "Place 'hand_landmarker.task' in assets/models/."
            )
            return False
        ok = self._source_manager.start_mock(mp4_path, loop)
        if ok:
            self._model_path = model_path
            self.start()
        return ok

    def start_simulated_test(self) -> bool:
        """Begin simulated test mode.

        Generates a moving digital hand skeleton on a black screen
        frame-by-frame and feeds the synthetic landmarks straight to the
        gesture recognizer. Requires no webcam, no MP4 file, and no
        MediaPipe model asset.
        """
        self._simulated = True
        self._source_manager.start_simulated()
        self.start()
        return True

    def stop(self) -> None:
        """Request the tracking thread to stop."""
        self._running = False
        if self.is_alive() and threading.current_thread() is not self:
            self.join(3.0)

    def detect_webcam(self, camera_index: int = 0) -> bool:
        """Probe whether a live webcam is available at the given index."""
        return self._source_manager.detect_webcam(camera_index)

    @property
    def is_tracking(self) -> bool:
        return self._running

    @property
    def source_name(self) -> str:
        return self._source_manager.source_name()

    # ------------------------------------------------------------------ #
    # QThread
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        self._running = True
        self._recognizer.reset()
        self._state("started")

        if getattr(self, "_simulated", False):
            try:
                self._run_simulated()
            finally:
                self._simulated = False
                self._running = False
                self._state("stopped")
            return

        model_path = getattr(self, "_model_path", None) or self.ensure_model()
        if not model_path:
            self._error("No MediaPipe model asset available.")
            self._running = False
            self._state("stopped")
            return

        base_options = BaseOptions(model_asset_path=model_path)
        options = HandLandmarkerOptions(
            base_options=base_options,
            running_mode=VisionTaskRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        detector = HandLandmarker.create_from_options(options)

        frame_index = 0
        try:
            while self._running:
                frame = self._source_manager.read()
                if frame is None:
                    self._state("stopped")
                    break

                frame_index += 1
                if frame_index % max(1, self._process_every_n_frames) != 0:
                    continue

                # Mirror for natural self-view interaction.
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                gesture_event: Optional[GestureEvent] = None
                timestamp_ms = int(time.perf_counter() * 1000)

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = detector.detect_for_video(mp_image, timestamp_ms)

                if result.hand_landmarks:
                    landmarks = result.hand_landmarks[0]
                    pts: List[tuple] = [
                        (lm.x, lm.y, lm.z) for lm in landmarks
                    ]
                    if self.on_landmarks:
                        self.on_landmarks(timestamp_ms / 1000.0, pts)
                    gesture_event = self._recognizer.process(timestamp_ms / 1000.0, pts)
                elif self.on_landmarks:
                    self.on_landmarks(timestamp_ms / 1000.0, None)

                if (gesture_event is not None
                        and gesture_event.gesture is not Gesture.NONE):
                    self._emit_gesture(gesture_event)

        finally:
            try:
                detector.close()
            except Exception:
                pass
            self._source_manager.stop()
            self._running = False
            self._state("stopped")

    # ------------------------------------------------------------------ #
    # Simulated test mode
    # ------------------------------------------------------------------ #
    def _run_simulated(self) -> None:
        """Render an animated digital hand skeleton on a black screen and
        feed the synthetic landmarks directly to the gesture recognizer."""
        import math

        width, height = 1280, 720
        target_fps = 30.0
        frame_period = 1.0 / target_fps
        t = 0.0

        while self._running:
            frame_start = time.perf_counter()

            landmarks = SimulatedVideoSource._skeleton(t)
            if self.on_landmarks:
                self.on_landmarks(t, landmarks)

            gesture_event = self._recognizer.process(t, landmarks)
            if gesture_event.gesture is not Gesture.NONE:
                self._emit_gesture(gesture_event)

            elapsed = time.perf_counter() - frame_start
            time.sleep(max(0.0, frame_period - elapsed))
            t += frame_period

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _emit_gesture(self, event: GestureEvent) -> None:
        if self.on_gesture:
            self.on_gesture(event)
