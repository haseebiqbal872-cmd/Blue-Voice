"""Types the signed message into the app the person was using before (ChatGPT, Claude, Gemini, any message box).

How: remember the last window that was not ours, bring it to the front when the palm is shown, paste the text
with Ctrl+V (your clipboard is put back afterwards) and leave the Send button for the person to press.
Windows is fully supported. macOS and Linux use the standard clipboard and keyboard tools and are best effort.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time


class Injector:
    def __init__(self, backend=None):
        self.b = backend or _default_backend()

    def start_watching(self):
        if hasattr(self.b, "watch"):
            self.b.watch()

    def stop_watching(self):
        if hasattr(self.b, "unwatch"):
            self.b.unwatch()

    def paste(self, text: str) -> tuple[bool, str]:
        text = (text or "").strip()
        if not text:
            return False, "There is nothing to send yet."
        try:
            return self.b.paste(text)
        except Exception as e:
            return False, f"Could not type into the other app: {e}"


# ---------------------------------------------------------------- Windows
class WindowsBackend:
    def __init__(self):
        import ctypes
        from ctypes import wintypes
        self.ct, self.wt = ctypes, wintypes
        self.u = ctypes.WinDLL("user32", use_last_error=True)
        self.k = ctypes.WinDLL("kernel32", use_last_error=True)
        self.u.GetForegroundWindow.restype = wintypes.HWND
        self.u.GetClipboardData.restype = wintypes.HANDLE
        self.k.GlobalAlloc.restype = wintypes.HGLOBAL
        self.k.GlobalLock.restype = ctypes.c_void_p
        self.k.GlobalLock.argtypes = [wintypes.HGLOBAL]
        self.k.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
        self.u.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
        self.last = None
        self._stop = threading.Event()
        self.me = os.getpid()

    def _pid(self, hwnd):
        pid = self.wt.DWORD(); self.u.GetWindowThreadProcessId(hwnd, self.ct.byref(pid)); return pid.value

    def _title(self, hwnd):
        n = self.u.GetWindowTextLengthW(hwnd)
        b = self.ct.create_unicode_buffer(n + 1); self.u.GetWindowTextW(hwnd, b, n + 1); return b.value

    def watch(self):
        self._stop.clear()
        def loop():
            while not self._stop.is_set():
                h = self.u.GetForegroundWindow()
                if h and self._pid(h) != self.me and self.u.IsWindowVisible(h) and self._title(h):
                    self.last = h
                time.sleep(0.2)
        threading.Thread(target=loop, daemon=True).start()

    def unwatch(self):
        self._stop.set()

    def _focus(self, hwnd):
        u = self.u
        if u.IsIconic(hwnd):
            u.ShowWindow(hwnd, 9)
        u.keybd_event(0x12, 0, 0, 0); u.keybd_event(0x12, 0, 2, 0)          # a tap of Alt lets us take focus
        u.SetForegroundWindow(hwnd)
        for _ in range(20):
            if u.GetForegroundWindow() == hwnd:
                return True
            time.sleep(0.05)
        return False

    def _get_clip(self):
        u, k, ct = self.u, self.k, self.ct
        if not u.OpenClipboard(None):
            return None
        try:
            h = u.GetClipboardData(13)                                          # CF_UNICODETEXT
            if not h:
                return ""
            p = k.GlobalLock(h)
            try:
                return ct.wstring_at(p)
            finally:
                k.GlobalUnlock(h)
        finally:
            u.CloseClipboard()

    def _set_clip(self, text):
        u, k, ct = self.u, self.k, self.ct
        data = (text + "\0").encode("utf-16-le")
        for _ in range(10):
            if u.OpenClipboard(None):
                break
            time.sleep(0.05)
        else:
            raise RuntimeError("the clipboard is busy")
        try:
            u.EmptyClipboard()
            h = k.GlobalAlloc(0x0042, len(data))                                # GMEM_MOVEABLE | GMEM_ZEROINIT
            p = k.GlobalLock(h); ct.memmove(p, data, len(data)); k.GlobalUnlock(h)
            u.SetClipboardData(13, h)
        finally:
            u.CloseClipboard()

    def paste(self, text):
        if not self.last or not self.u.IsWindow(self.last):
            return False, "I could not find the app you were using. Click its message box once, then show your palm again."
        title = self._title(self.last)
        if not self._focus(self.last):
            return False, f"I could not switch to “{title}”. Click it once, then show your palm again."
        time.sleep(0.15)
        old = self._get_clip()
        self._set_clip(text)
        time.sleep(0.05)
        u = self.u
        u.keybd_event(0x11, 0, 0, 0); u.keybd_event(0x56, 0, 0, 0); u.keybd_event(0x56, 0, 2, 0); u.keybd_event(0x11, 0, 2, 0)   # Ctrl+V
        time.sleep(0.25)
        if old is not None:
            try: self._set_clip(old)
            except Exception: pass
        return True, f"Typed into “{title}”. Press its Send button when you are ready."


# ---------------------------------------------------------------- macOS / Linux (best effort)
class UnixBackend:
    def __init__(self):
        self.mac = sys.platform == "darwin"

    def paste(self, text):
        if self.mac:
            subprocess.run(["pbcopy"], input=text.encode(), check=True)
            subprocess.run(["osascript", "-e", 'tell application "System Events" to keystroke "v" using command down'], check=True)
        else:
            try:
                subprocess.run(["xclip", "-selection", "clipboard"], input=text.encode(), check=True)
                subprocess.run(["xdotool", "key", "ctrl+v"], check=True)
            except FileNotFoundError:
                return False, "On Linux this needs the xclip and xdotool programs."
        return True, "Typed into the app in front. Press its Send button when you are ready."


def _default_backend():
    return WindowsBackend() if sys.platform.startswith("win") else UnixBackend()
