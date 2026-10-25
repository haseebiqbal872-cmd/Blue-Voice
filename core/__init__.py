"""Gesture-to-Text AI Copilot - core processing package."""

from .video_source import VideoSource, CameraSource, MockVideoSource, VideoSourceManager
from .gesture_recognizer import HandGestureRecognizer, GestureEvent
from .text_engine import TextEngine
from .hand_tracker import HandTrackerThread

__all__ = [
    "VideoSource",
    "CameraSource",
    "MockVideoSource",
    "VideoSourceManager",
    "HandGestureRecognizer",
    "GestureEvent",
    "TextEngine",
    "HandTrackerThread",
]
