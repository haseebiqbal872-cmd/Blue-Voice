"""Live word recognition + record-your-own-words, for the word page (words_app.py).

One WordSession owns: a camera (or video file) reader, the upper-body tracker, the sign cutter, the 250-word model
and the user's own saved words. It talks to the page only through `emit(dict)`; commands come in through methods.
Everything stays on this computer.
"""
from __future__ import annotations

import base64
import json
import os
import threading
import time
from typing import Callable, Optional

import cv2
import numpy as np

from core.gs.palm import PalmHold
from core.gs.segmenter import Segmenter
from core.gs.teach import MIN_TAKES, TeachStore

from core.sign_words import WordRecognizer

REC_SECONDS = 15
READY_SECONDS = 2

# points drawn in the preview: nose, ears, shoulders, elbows, wrists, then 21 points per hand
PREVIEW_POSE = [489 + i for i in (0, 7, 8, 11, 12, 13, 14, 15, 16)]
PREVIEW_IDX = PREVIEW_POSE + list(range(468, 489)) + list(range(522, 543))


def preview_payload(frames, fps: float, max_frames: int = 60) -> dict:
    f = np.asarray(frames)[:, PREVIEW_IDX, :2]
    step = max(1, int(np.ceil(len(f) / max_frames)))
    f = f[::step]
    rows = [[None if not np.isfinite(v) else round(float(v), 3) for v in row.reshape(-1)] for row in f]
    return {"pts": PREVIEW_IDX, "fps": round(float(fps) / step, 2), "frames": rows}


class WordSession:
    def __init__(self, root: str, emit: Callable[[dict], None], tracker_factory=None, source_factory=None):
        self.root, self.emit = root, emit
        self.rec = WordRecognizer(os.path.join(root, "models", "gs"))
        self.store = TeachStore(os.path.join(root, "data", "user_words.json"))
        self.tracker_factory = tracker_factory or self._default_tracker
        self.source_factory = source_factory or self._default_source
        self.thread: Optional[threading.Thread] = None
        self.stop_flag = threading.Event()
        self.teach: Optional[dict] = None
        self.state = "idle"
        self._trk = None                                   # body tracker kept loaded between recordings
        self.rec_word = ""
        self.rec_takes: list = []
        self.rec_thread: Optional[threading.Thread] = None
        self.rec_stop = threading.Event()
        self.on_word = None                                # callback(word, score): a confident word was read
        self.on_palm = None                                # callback(): an open palm was held (send the message)
        self.preview = True                                # send small camera previews to the page
        self.rec_cancelled = False
        self.rec_go = threading.Event()
        self.rec_clips: list = []
        self.rec_vid: list = []                            # (t_ms, jpeg) of the current take window; memory only, never saved

    # ------------------------------------------------------------------ helpers
    def _default_tracker(self):
        from core.gs.tracker import UpperBodyTracker, missing_models
        models = os.path.join(self.root, "assets", "models")
        miss = missing_models(models)
        if miss:
            raise RuntimeError("Missing tracker files in assets\\models: " + ", ".join(miss) +
                               ". Run test_words.bat once; it explains how to get them.")
        return UpperBodyTracker(models)

    def _default_source(self, source: str, path: str):
        """-> object with read() -> (bgr, t_ms) or None, close(), live (bool)."""
        return _Camera() if source == "camera" else _VideoFile(path)

    def words(self) -> list[dict]:
        return self.store.summary()

    def send_words(self):
        self.emit({"type": "words", "words": self.words()})

    def send_status(self, note: str = ""):
        self.emit({"type": "status", "state": self.state, "note": note})

    # ------------------------------------------------------------------ commands
    def start(self, source: str = "camera", path: str = "") -> Optional[str]:
        self.stop()
        self.stop_flag.clear()
        self.state = "loading"
        self.send_status("Loading the body tracker (first time takes a moment)...")
        self.thread = threading.Thread(target=self._run, args=(source, path), daemon=True)
        self.thread.start()
        return None

    def stop(self):
        self.stop_flag.set()
        t = self.thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=5)
        self.thread = None
        self.state = "idle"
        self.teach = None

    def record(self, word: str) -> Optional[str]:
        word = TeachStore.clean(word)
        if not word:
            return "Type the word first."
        if self.state != "tracking":
            return "Start the camera first, then record."
        self.store.delete(word)                         # re-recording replaces the old takes
        self.teach = {"word": word, "n": 0}
        self.emit({"type": "teach", "word": word, "state": "waiting", "take": 0, "of": MIN_TAKES,
                   "message": f"Sign \"{word}\" now (1 of {MIN_TAKES}). Rest your hands for a moment after each sign."})
        self.send_words()
        return None

    def cancel_record(self):
        if self.teach:
            self.store.delete(self.teach["word"]) if self.teach["n"] < MIN_TAKES else None
            self.teach = None
            self.emit({"type": "teach", "state": "cancelled", "message": "Recording cancelled."})
            self.send_words()

    def delete_word(self, word: str):
        self.store.delete(word)
        self.send_words()

    # ------------------------------------------------------------------ record a sign (15 second window)
    def model_info(self) -> dict:
        """Which recognition model the saved signs belong to (used to check sign packs)."""
        try:
            m = json.load(open(os.path.join(self.root, "models", "gs", "model-meta.json"), encoding="utf-8"))
            return {"version": str(m.get("version", "")), "dim": int(m.get("embeddingDim", 0))}
        except Exception:
            return {"version": "", "dim": 0}

    def send_signs(self):
        self.emit({"type": "mysigns", "signs": self.store.summary()})

    def _rec_emit(self, state: str, **kw):
        self.emit({"type": "rec", "state": state, "word": self.rec_word, "takes": len(self.rec_takes), **kw})

    def _rec_halt(self):
        self.rec_cancelled = True
        self.rec_stop.set()
        t = self.rec_thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=8)
        self.rec_thread = None

    def rec_arm(self, word: str = "") -> Optional[str]:
        """Open the camera and wait: the recording starts on the red button or on a held open palm."""
        self.stop()                                        # the live reader and the recorder never share the camera
        self.rec_word = TeachStore.clean(word)
        return self._rec_launch(autostart=False)

    def rec_word_set(self, word: str):
        self.rec_word = TeachStore.clean(word)

    def rec_start(self, word: str) -> Optional[str]:
        word = TeachStore.clean(word)
        if not word:
            return "Type the word first."
        self.rec_word = word
        t = self.rec_thread
        if t is not None and t.is_alive() and not self.rec_stop.is_set():
            self.rec_go.set()                              # already armed: begin the 15 seconds now
            return None
        self.stop()
        return self._rec_launch(autostart=True)

    def rec_again(self) -> Optional[str]:
        if not self.rec_word:
            return "Type the word first."
        return self._rec_launch(autostart=True)

    def _rec_launch(self, autostart: bool) -> Optional[str]:
        self._rec_halt()
        self.rec_stop = threading.Event()
        self.rec_go = threading.Event()
        self.rec_cancelled = False
        self.rec_takes, self.rec_clips, self.rec_vid = [], [], []
        self.rec_thread = threading.Thread(target=self._rec_run, args=(self.rec_stop, autostart), daemon=True)
        self.rec_thread.start()
        return None

    def rec_finish(self):
        """Stop early (the person pressed the red button again)."""
        self.rec_stop.set()

    def rec_save(self) -> Optional[str]:
        if len(self.rec_takes) < MIN_TAKES:
            return f"I need to see the sign {MIN_TAKES} times. Press Start again."
        self.store.delete(self.rec_word)
        for e in self.rec_takes:
            self.store.add_take(self.rec_word, e)
        w = self.rec_word
        self.rec_takes, self.rec_word, self.rec_vid = [], "", []
        self.emit({"type": "rec", "state": "saved", "word": w, "takes": MIN_TAKES})
        self.send_signs(); self.send_words()
        return None

    def rec_cancel(self):
        self._rec_halt()
        self.rec_takes, self.rec_word, self.rec_vid = [], "", []
        self.emit({"type": "rec", "state": "cancelled", "word": "", "takes": 0})

    def delete_sign(self, word: str):
        self.store.delete(word)
        self.send_signs(); self.send_words()

    def _rec_run(self, stop_evt: threading.Event, autostart: bool):
        src = None
        try:
            self._rec_emit("loading", left=REC_SECONDS)
            if self._trk is None:
                self._trk = self.tracker_factory()
            src = self.source_factory("camera", "")
            sg, palm = Segmenter(), PalmHold()
            phase = "recording" if autostart else "armed"
            t0 = ready0 = time.monotonic()
            last_left = None
            if phase == "armed":
                self._rec_emit("armed", left=REC_SECONDS)
            while not stop_evt.is_set():
                got = src.read()
                if got is None:
                    break
                bgr, t_ms = got
                lm = self._trk.track(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), int(t_ms))
                now = time.monotonic()
                if phase == "armed":
                    held = palm.push(lm, t_ms) if self.rec_word else False
                    if self.rec_go.is_set():
                        phase = "begin"
                    elif held:
                        phase, ready0 = "ready", now
                elif phase == "ready":
                    left = max(0, int(READY_SECONDS - (now - ready0) + 0.999))
                    if left != last_left:
                        last_left = left; self._rec_emit("ready", left=left)
                    if now - ready0 >= READY_SECONDS:
                        phase = "begin"
                if phase == "begin":
                    phase, t0, last_left = "recording", now, None
                    sg = Segmenter(); self.rec_takes, self.rec_clips, self.rec_vid = [], [], []
                    continue
                if phase != "recording":
                    continue
                el = now - t0
                self._rec_frame(bgr, t_ms)
                if el >= REC_SECONDS and (sg.state != "signing" or el >= REC_SECONDS + 1):
                    break                                  # 15 s are up; a sign already under way gets one more second
                ev = sg.push(lm, t_ms)
                if ev and ev[0] == "end" and not stop_evt.is_set():
                    self._rec_add(ev[1])
                    last_left = None                       # force an update with the new count
                left = max(0, int(REC_SECONDS - el + 0.999))
                if left != last_left:
                    last_left = left
                    self._rec_emit("recording", left=left)
            if not self.rec_cancelled and phase == "recording":
                ev = sg.flush()
                if ev and ev[0] == "end":
                    self._rec_add(ev[1])
                self._send_preview()
                self._rec_emit("review", left=0)
        except Exception as e:
            self.emit({"type": "rec", "state": "error", "word": self.rec_word, "takes": 0, "message": str(e)})
        finally:
            if src is not None:
                try: src.close()
                except Exception: pass

    def _rec_add(self, seg):
        self.rec_takes.append(self.rec.embed(seg.frames, seg.fps))
        self.rec_clips.append(seg)

    def _rec_frame(self, bgr, t_ms):
        try:
            h, w = bgr.shape[:2]
            if w > 640:
                bgr = cv2.resize(bgr, (640, max(1, int(h * 640 / w))))
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 72])
            if ok:
                self.rec_vid.append((float(t_ms), buf.tobytes(), bgr.shape[1], bgr.shape[0]))
        except Exception:
            pass

    def _video_payload(self, seg, max_frames=90):
        t0, t1 = seg.start_ms - 150, seg.start_ms + seg.duration_ms + 300
        fr = [f for f in self.rec_vid if t0 <= f[0] <= t1]
        if len(fr) < 4:
            return None
        step = max(1, int(np.ceil(len(fr) / max_frames)))
        fr = fr[::step]
        return {"w": fr[0][2], "h": fr[0][3], "t": [round(f[0] - fr[0][0]) for f in fr],
                "jpg": [base64.b64encode(f[1]).decode("ascii") for f in fr]}

    def _send_preview(self):
        """Send the longest take as a small hand-and-arm animation for the preview."""
        if not self.rec_clips:
            self.emit({"type": "recprev", "frames": []}); return
        seg = max(self.rec_clips, key=lambda c: c.duration_ms)
        self.emit({"type": "recprev", **preview_payload(seg.frames, seg.fps), "video": self._video_payload(seg)})
        self.rec_vid = []

    # ------------------------------------------------------------------ worker
    def _on_segment(self, seg):
        if self.teach:
            w = self.teach["word"]
            n = self.store.add_take(w, self.rec.embed(seg.frames, seg.fps))
            self.teach["n"] = n
            if n >= MIN_TAKES:
                self.teach = None
                self.emit({"type": "teach", "word": w, "state": "done", "take": n, "of": MIN_TAKES,
                           "message": f"Saved. \"{w}\" is now one of your words."})
            else:
                self.emit({"type": "teach", "word": w, "state": "saved", "take": n, "of": MIN_TAKES,
                           "message": f"Got it ({n} of {MIN_TAKES}). Sign \"{w}\" again ({n + 1} of {MIN_TAKES})."})
            self.send_words()
            return
        r = self.rec.read(seg.frames, seg.fps, self.store)
        if self.on_word and r["kind"] == "sign" and r["top"]:
            self.on_word(r["top"][0][0], float(r["top"][0][1]))
        self.emit({"type": "sign", "kind": r["kind"], "source": r["source"],
                   "top": [[w, round(p, 3)] for w, p in r["top"]], "t": round(seg.start_ms / 1000, 1)})

    def _run(self, source: str, path: str):
        src = tracker = None
        try:
            tracker = self.tracker_factory()
            src = self.source_factory(source, path)
            self.state = "tracking"
            self.send_status("")
            self.send_words()
            sg, palm, palm_off = Segmenter(), PalmHold(1200), 0.0
            last_prev = 0.0
            last_state = "idle"
            while not self.stop_flag.is_set():
                got = src.read()
                if got is None:
                    break
                bgr, t_ms = got
                lm = tracker.track(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), int(t_ms))
                if self.on_palm and time.time() > palm_off and sg.state != "signing" and palm.push(lm, t_ms):
                    palm.reset(); palm_off = time.time() + 2.5
                    self.on_palm()
                ev = sg.push(lm, t_ms)
                if sg.state != last_state:
                    last_state = sg.state
                    self.emit({"type": "moving", "on": sg.state == "signing"})
                if ev and ev[0] == "end":
                    self._on_segment(ev[1])
                now = time.time()
                if self.preview and now - last_prev > 0.12:
                    last_prev = now
                    small = cv2.resize(bgr, (320, int(320 * bgr.shape[0] / bgr.shape[1])))
                    if getattr(src, "live", False):
                        small = cv2.flip(small, 1)           # mirror the preview only, like a mirror
                    ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 55])
                    if ok:
                        self.emit({"type": "frame", "jpg": base64.b64encode(jpg.tobytes()).decode()})
            ev = sg.flush()
            if ev and ev[0] == "end":
                self._on_segment(ev[1])
        except Exception as e:                                # report instead of dying silently
            self.emit({"type": "error", "message": str(e)})
        finally:
            if src is not None:
                try: src.close()
                except Exception: pass
            self.state = "idle"
            self.teach = None
            self.send_status("")


class _Camera:
    """Webcam read by a small thread so the newest frame is always the one processed."""
    live = True

    def __init__(self):
        from core.video_source import CameraSource
        self.cam = CameraSource(0, 960, 540)
        if not self.cam.start():
            raise RuntimeError("No camera found. Plug in a webcam, close other apps using it, and try again.")
        self.t0 = time.monotonic(); self.frame = None; self.seq = 0; self.last = 0; self.run = True
        self.th = threading.Thread(target=self._loop, daemon=True); self.th.start()

    def _loop(self):
        while self.run:
            f = self.cam.read()
            if f is None:
                time.sleep(0.01); continue
            self.frame, self.seq = f, self.seq + 1

    def read(self):
        for _ in range(500):
            if self.seq != self.last and self.frame is not None:
                self.last = self.seq
                return self.frame, (time.monotonic() - self.t0) * 1000
            time.sleep(0.005)
        return None

    def close(self):
        self.run = False
        self.cam.stop()


class _VideoFile:
    live = False

    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open that video file.")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.i = 0

    def read(self):
        ok, f = self.cap.read()
        if not ok:
            return None
        self.i += 1
        return f, (self.i - 1) * 1000.0 / self.fps

    def close(self):
        self.cap.release()
