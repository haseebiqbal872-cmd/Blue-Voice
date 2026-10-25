import os
import numpy as np
from core.gesture_recognizer import HandGestureRecognizer, Gesture
from core.text_engine import TextEngine
from core.letter_classifier import LetterCommitter, LetterClassifier, normalise


def hand(open_=True, x=0.5, y=0.6):
    pts = [[x, y, 0.0] for _ in range(21)]
    pts[3] = [x - 0.05, y - 0.05, 0]; pts[4] = [x - 0.12, y - 0.08, 0] if open_ else [x - 0.04, y - 0.04, 0]
    for tip, pip, dx in [(8, 6, -0.03), (12, 10, 0.0), (16, 14, 0.03), (20, 18, 0.06)]:
        pts[pip] = [x + dx, y - 0.12, 0]
        pts[tip] = [x + dx, y - 0.25, 0] if open_ else [x + dx, y - 0.06, 0]
    pts[9] = [x, y - 0.1, 0]
    return [tuple(p) for p in pts]


def test_open_palm_detected():
    r = HandGestureRecognizer()
    assert all(r.finger_states(hand(True)))
    assert not r.is_open_palm(hand(False))


def test_palm_hold_triggers_after_1_5s():
    r = HandGestureRecognizer()
    assert Gesture.OPEN_PALM in [r.process(t / 30, hand(True)).gesture for t in range(60)]


def test_palm_moving_does_not_trigger():
    r = HandGestureRecognizer()
    assert Gesture.OPEN_PALM not in [r.process(t / 30, hand(True, x=0.3 + 0.01 * (t % 20))).gesture for t in range(90)]


def test_swipes():
    r = HandGestureRecognizer()
    assert Gesture.SWIPE_LEFT in [r.process(t / 30, hand(False, x=0.7 - 0.03 * t)).gesture for t in range(8)]
    r2 = HandGestureRecognizer()
    assert Gesture.SWIPE_RIGHT in [r2.process(t / 30, hand(False, x=0.3 + 0.03 * t)).gesture for t in range(8)]


def test_text_engine():
    e = TextEngine()
    e.commit_token("hello"); e.add_space(); e.commit_token("world")
    assert e.text == "Hello world"
    e.backspace()
    assert e.text == "Hello "
    e2 = TextEngine(); e2.commit_token("hi."); e2.add_space(); e2.commit_token("there")
    assert e2.text == "Hi. There"
    e3 = TextEngine(); e3.commit_token("abc"); e3.commit_token("def")
    assert e3.text == "Abcdef"


def test_text_engine_letters_space_backspace():
    e = TextEngine()
    for ch in "HI":
        e.append_character(ch)
    e.add_space(); e.append_character("A")
    assert e.text == "HI A"
    e.backspace(); assert e.text == "HI "


def test_ensure_model_does_not_crash_without_network(monkeypatch):
    from core.hand_tracker import HandTrackerThread
    monkeypatch.setattr(HandTrackerThread, "resolve_model_path", classmethod(lambda c: None))
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    assert HandTrackerThread.ensure_model(timeout_s=1) is None


def test_committer_types_once_per_hold_and_repeats_after_change():
    c = LetterCommitter(stable_frames=3, min_conf=0.8)
    out = [c.update("A", 0.9) for _ in range(10)]
    assert out.count("A") == 1
    assert [c.update("B", 0.9) for _ in range(3)].count("B") == 1
    assert [c.update("A", 0.9) for _ in range(3)].count("A") == 1
    c.update(None, 0)
    assert [c.update("A", 0.9) for _ in range(3)].count("A") == 1


def test_committer_ignores_low_confidence_and_flicker():
    c = LetterCommitter(stable_frames=3, min_conf=0.8)
    assert not any(c.update("A", 0.5) for _ in range(10))
    assert not any(c.update(l, 0.9) for l in "ABABABAB")


def test_normalise_is_position_and_scale_invariant():
    a = np.array(hand(True, 0.3, 0.4)); b = (np.array(hand(True, 0.6, 0.8)) - [0.6, 0.8, 0]) * 2 + [0.1, 0.1, 0]
    assert np.allclose(normalise(a), normalise(b), atol=1e-4)


def test_classifier_unavailable_without_model(tmp_path):
    assert not LetterClassifier(str(tmp_path / "none.joblib")).available


def test_train_pipeline_on_synthetic_data(tmp_path, monkeypatch):
    """Checks the plumbing only (synthetic shapes) - NOT real sign accuracy."""
    import sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    import _common, train
    rng = np.random.default_rng(0)
    rows = []
    for label, open_ in [("OPEN", True), ("FIST", False)]:
        for s in range(3):
            for _ in range(40):
                h = np.array(hand(open_)) + rng.normal(0, 0.004, (21, 3))
                rows.append((label, f"s{s}", [tuple(p) for p in h]))
    csv_path = str(tmp_path / "d.csv"); _common.append_rows(rows, csv_path)
    X, y, g = _common.load(csv_path)
    r = train.evaluate(train.features(X), y, g)
    assert r["session_acc"] > 0.95
    out = str(tmp_path / "m.joblib")
    monkeypatch.setattr(sys, "argv", ["train", "--csv", csv_path, "--out", out]); train.main()
    clf = LetterClassifier(out)
    assert clf.available and clf.predict(hand(True))[0] == "OPEN" and clf.predict(hand(False))[0] == "FIST"
