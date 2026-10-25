"""
Text engine for the Gesture-to-Text AI Copilot.

Manages the token buffer produced by gesture input:

  * Swipe right  -> append a word-gap space
  * Swipe left   -> backspace / undo the last token
  * Palm freeze  -> commit the currently recognised text token

Provides simple AI-copilot formatting (sentence casing, token cleanup).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class Token:
    """A single text token emitted by the copilot."""
    text: str
    kind: str = "word"          # word | space | punctuation | command
    timestamp: float = 0.0


class TextEngine:
    """Holds and mutates the document text driven by gestures."""

    def __init__(self) -> None:
        self._tokens: List[Token] = []
        self._pending: str = ""   # characters currently being composed

    # -- document access ------------------------------------------------ #
    @property
    def text(self) -> str:
        """Full assembled document text."""
        return "".join(t.text for t in self._tokens) + self._pending

    @property
    def tokens(self) -> List[Token]:
        return list(self._tokens)

    # -- mutations ------------------------------------------------------ #
    def append_character(self, ch: str) -> None:
        """Append a single composed character (used while a token forms)."""
        self._pending += ch

    def commit_token(self, text: str, kind: str = "word", timestamp: float = 0.0) -> None:
        """Commit a completed text token to the document."""
        if not text:
            return
        cleaned = self._clean(text)
        if not cleaned:
            return
        self._flush_pending()
        prior = self.text.rstrip()
        if kind == "word" and (not prior or prior[-1] in ".!?"):
            cleaned = cleaned[:1].upper() + cleaned[1:]
        self._tokens.append(Token(text=cleaned, kind=kind, timestamp=timestamp))

    def add_space(self, timestamp: float = 0.0) -> None:
        """Append a word-gap space character (swipe right)."""
        self._flush_pending()
        if self._tokens and self._tokens[-1].kind != "space" and not self.text.endswith(" "):
            self._tokens.append(Token(text=" ", kind="space", timestamp=timestamp))

    def backspace(self, timestamp: float = 0.0) -> None:
        """Delete the last typed token / character (swipe left / undo)."""
        if self._pending:
            self._pending = self._pending[:-1]
            return
        if self._tokens:
            last = self._tokens.pop()
            # If we removed a trailing space, also remove a preceding space to
            # avoid double spaces when typing resumes.
            if last.kind == "space" and self._tokens and self._tokens[-1].kind == "space":
                self._tokens.pop()

    def clear(self) -> None:
        """Clear the entire document."""
        self._tokens.clear()
        self._pending = ""

    # -- internals ------------------------------------------------------ #
    def _flush_pending(self) -> None:
        if self._pending:
            self._tokens.append(Token(text=self._pending, kind="word"))
            self._pending = ""

    @staticmethod
    def _clean(text: str) -> str:
        """Strip control characters and collapse internal whitespace."""
        text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
        return text.strip()
